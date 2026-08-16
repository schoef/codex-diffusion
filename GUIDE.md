# nef-qvf-diffusion — repository guide

A NumPy/SciPy toolkit for the six one-dimensional natural exponential families
with quadratic variance functions (NEF-QVF), and for the amplitude and
density-matrix models built on them. It is the reference implementation for the
accompanying note and reproduces its figures.

The repository has two layers, and the split is strict:

| layer | location | what it is |
| --- | --- | --- |
| **package** | `src/nefqvf/` | the installable library: families, orthogonal polynomials, product tensors. No fitting, no plotting, no research workflow. |
| **applications** | `applications/` | executables built on the package's public API: fitting studies, benchmarks, and the scripts that write the note's figures. |

Nothing in `applications/` is importable as a library dependency, and nothing in
`src/nefqvf/` knows about fitting. That is deliberate: research workflows change
faster than the mathematics beneath them.

---

## 1. Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable install points imports at `src/nefqvf`, so source edits are live.

Check the installation:

```bash
python -m pytest -q          # 221 tests, ~2 minutes
ruff check src applications tests
```

---

## 2. The package: `src/nefqvf`

### 2.1 The six families

Each family is a stateless singleton; parameters are immutable dataclasses that
may hold scalars or arrays.

| family | object | parameters | basis | support |
| --- | --- | --- | --- | --- |
| Normal | `Normal` | `NormalParams(mean, sigma)` | Hermite | ℝ |
| Poisson | `Poisson` | `PoissonParams(mean)` | Charlier | {0,1,2,…} |
| Gamma | `Gamma` | `GammaParams(mean, r)` | Laguerre | (0,∞) |
| Binomial | `Binomial` | `BinomialParams(mean, N)` | Krawtchouk | {0,…,N} |
| Negative binomial | `NegativeBinomial` | `NegativeBinomialParams(mean, r)` | Meixner | {0,1,2,…} |
| Generalized hyperbolic secant | `GHS` | `GHSParams(mean, r)` | Meixner–Pollaczek | ℝ |

The public mean is the parameter throughout; `sigma`, `r` and `N` are *fixed*
parameters that shifts do not move. Names are exported lazily from
`nefqvf/__init__.py`, so importing the package is cheap.

### 2.2 The `Family` interface

Every family implements the same interface (`src/nefqvf/_family.py`). Grouped by
what it is for:

**Probability**
- `log_prob(x, params)`, `prob(x, params)` — ordinary broadcasting
- `log_prob_grid(x, params)`, `prob_grid(x, params)` — explicit parameter × observation grid
- `sample(params, size, rng=…)` — independent variates from one member
- `is_lattice(params)` — whether the family lives on the integers

**Moments and parameters**
- `mean(params)`, `variance(params)`, `variance_slope(params)` — the latter is `V'(mean)`
- `natural_parameter(params)`, `from_natural(eta, fixed)` — the `eta` chart
- `log_affinity(p1, p2)`, `affinity(p1, p2)` — Hellinger affinity between members

**Shifts and the forward flow**
- `shifted_params(params, natural_shift)` — the member at `eta + j`
- `shift_coordinate(natural_shift, params)` → `z`, and `from_shift_coordinate(z, params)` back
- `shift_coefficients(natural_shift, n_max, params)` — the ratio coefficients `gamma_n z**n`

`z` is the coordinate the reference flow contracts, `z → e^{-t} z`. That single
fact is what makes the diffusion closed-form on these families, and it is why
`from_shift_coordinate` exists: noising a member is a rescale and an inversion,
not a kernel integral.

**Orthogonal polynomials and the product tensor**
- `jacobi_coefficients(n, params)` → `(a_n, b_n)` in the **positive off-diagonal** convention
- `basis(x, n_max, params)` — `phi_0 … phi_n_max` along a final degree axis
- `basis_dot(x, coefficients, params)` — evaluates `sum_n c_n phi_n(x)` by **Clenshaw recurrence**, without materialising the basis
- `linearization_tensor(n_max, params)` — `Lambda[m,n,k] = E_ref[phi_m phi_n phi_k]`

Use `basis_dot` rather than `basis @ c` whenever you only need the value: it is
the numerically stable path, and it is what keeps degree-50 expansions usable.

### 2.3 Internals

| module | contents |
| --- | --- |
| `_family.py` | the `Family` base class, validation helpers, the generic `linearization_tensor` |
| `params.py` | the six immutable parameter dataclasses |
| `jacobi.py` | the shared three-term recurrence: `basis`, `basis_dot`, `jacobi_coefficients` |
| `linearization.py` | `linearization_tensor_from_jacobi` — builds `Lambda` from `(a, b)` alone |
| `broadcasting.py` | parameter/observation broadcasting: paired, outer, and batch forms |
| `sampling.py` | `resolve_generator`, `inverse_cdf_sample` (used by GHS), `symmetric_grid` |

### 2.4 Two conventions worth knowing

**Terminating bases.** The Krawtchouk basis stops at degree `N`. `Family._maximum_ops_degree`
reports that cap (`None` for the other five), and `linearization_tensor` clamps
its internal workspace accordingly, so on a lattice the product `phi_m phi_n`
*folds back* into degrees `≤ N` rather than running to `m+n`. This is exact,
not an approximation — see §6.

**The tensor is built from the recurrence.** `Lambda` is never computed by
quadrature. Everything derives from `(a_n, b_n)`, which is what keeps it exact
and cheap at high degree.

---

## 3. The applications

Run everything as a module from the repository root: `python -m applications.<name>`.

| module | question it answers |
| --- | --- |
| `amplitude_fit_recovery` | Can a one-channel amplitude be recovered from a sample? **Contains the shared fitting core.** |
| `amplitude_fit_degree` | How large a truncation does a given sample size support? |
| `baseline_matching` | How should the reference law and the degree be chosen? |
| `amplitude_fit_landscape` | What does the objective look like around the truth? |
| `amplitude_fit_limits` | How far does the Born parametrisation go, and where does it stop? |
| `amplitude_fit_complex` | Does the complex amplitude reach the convex optimum? |
| `shifted_baseline_probability_modes` | Do probability modes project and damp as claimed? |
| `two_state_hmm` | The toy model: a benchmark with an exact likelihood at every diffusion time. |
| `paper_one_channel`, `paper_complex_relaxation`, `paper_bimodal_gauss` | the note's figures (§5) |

### 3.1 `amplitude_fit_recovery` — the fitting core

The one function most other modules import:

```python
fit_amplitude(phi, target, *, initial=None, weight=None, tau=0.0,
              penalty=None, k_max=None) -> dict
```

Minimises `J(c) = ½ (R(c) − R̂)ᵀ W (R(c) − R̂)` over the real unit sphere by
Riemannian Levenberg–Marquardt, with `R_k(c) = cᵀ Φ_k c`. Degree zero is excluded
from the residual: `R_0 = ‖c‖² = 1` identically, so it would only add a null row.
Returns `coefficients`, `objective`, `iterations`, `residual_norm`, and
`curvature_ratio` (the curvature Gauss–Newton discards, relative to what it keeps).

Also here: `product_matrices` (the `Φ_k` stack), `exact_amplitude` (the analytic
half-shift amplitude), `empirical_coefficients` (sample coefficients and their
covariance), and `run_recovery`, which reports the `N^{-1/2}` rate.

```bash
python -m applications.amplitude_fit_recovery --family all
python -m applications.amplitude_fit_recovery --family gamma --plot --log-y
```

### 3.2 `amplitude_fit_complex` — relaxation, complex amplitude, certificates

The largest application module. It provides:

**Matrices.** `fitting_matrices(family, baseline, degree)` replaces
`product_matrices` and is the one to use: it caps the matched coefficient band at
the terminating degree instead of demanding `2K`, which is what lets the binomial
run to `K = N`. `terminating_degree` reports the cap.

**Fitting.** `fit_complex_amplitude` is the real LM in doubled coordinates
`(a, b)`, with the norm direction `c` and the phase direction `ic` removed from
the tangent space. `continued_complex_fit` fits real, applies the eigenvalue
test, and continues into the imaginary direction.

**The convex relaxation.** `relaxed_optimum` minimises the same objective over
`ρ ⪰ 0, Tr ρ = 1` by accelerated projected gradient, with `project_to_states`
(eigenvalues onto the simplex) as the projection. It returns a **duality gap**,
`Tr(Mρ) − λ_min(M)`, so its accuracy is certified rather than assumed.

**Certificates.** `optimality_test` implements the note's eigenvalue test at a
real fit: stationarity makes `a` an eigenvector of `M = Σ_k (Wr)_k Φ_k`, and the
fit is the convex optimum exactly when its eigenvalue is `λ_min(M)`; the bottom
eigenvector is the descent direction when it is not. `certified_gap` is the
corresponding bound at a **complex** amplitude, where `Tr(Mρ_c) = aᵀMa + bᵀMb` —
the real-part-only quantity is not this, and using it is a mistake.

**Construction rather than search.** The complex problem is not convex and the
optimiser does get stuck. Two constructive routes:
- `rank_two_seed(rho)` — for a rank-≤2 state, `f² + g² = |f + ig|²` reads the
  amplitude straight off the spectrum. Exact at any degree; use this by default.
- `factorise_state(...)` — the general Markov–Lukács factorisation through the
  roots of the law. Roots come from the **comrade matrix** (the Jacobi matrix
  under a rank-one correction) and the projection back onto the OPS uses the
  reference measure's own **Gauss rule**; no power-basis coefficients are ever
  formed. It validates itself and raises rather than returning a bad seed.

**Half-line supports.** `localised_matrices(family, baseline, name, degree)`
implements `q = σ₀ + π σ₁` for a family with a support wall. The normalisation
`R_0 = 1` couples the two blocks, but the coupling matrix is `π(J)` restricted,
which is positive definite, so rescaling by its square root turns `R_0` back into
a plain norm and `Ξ_0` into the identity. **The localised problem is then the same
problem with `Ξ` in place of `Φ`**, and every routine above applies unchanged.
`split_localised` and `law_from_localised` recover and evaluate the pair.

```bash
python -m applications.amplitude_fit_complex --family all --target all
python -m applications.amplitude_fit_complex --family binomial --degree 12 --plot
python -m applications.amplitude_fit_complex --family gamma --target shape --localised
```

Targets: `shifted`, `mixture` (set by standardised gap, not raw shift),
`truncated` (a hard upper edge — a failure probe, deliberately not repaired), and
`shape` (Gamma only: the `Γ(r+1)/Γ(r)` ratio, which no global sum of squares can
represent).

### 3.3 `baseline_matching` — choosing the reference

Library-style, no CLI. `moment_matched`, `matched_to_sample`,
`shift_reaching_mean`, `largest_valid_shift`, `separation_for_gap` (the
standardised separation used by every mixture target), and
`select_degree_by_likelihood` (held-out selection of `K`). Use `separation_for_gap`
rather than a raw natural shift when comparing families: the same `δ` means
wildly different difficulty across them.

### 3.4 `amplitude_fit_degree`, `_landscape`, `_limits`

`amplitude_fit_degree` builds the target ladder every other module reuses —
`shifted_target`, `mixture_target`, `truncated_target`, plus `support_grid`,
`integrate`, `reference_coefficients` and `total_variation` — and sweeps the
truncation against sample size.

`amplitude_fit_landscape` probes great circles through the known optimum, counts
local minima and measures basin widths, for the coefficient objective and for the
chambered likelihood.

`amplitude_fit_limits` measures how the bias floor of a hard-edge target falls
with degree, and draws the best achievable fit for each family.

---

## 4. The toy model: `two_state_hmm`

A symmetric two-state Markov chain supplies the spatial dependence; the emissions
are two members of one NEF-QVF family. The family can be exchanged without
touching the latent process, and **everything stays in closed form at every
diffusion time**, so a fitted model can be scored against truth rather than
against a longer run of itself.

Three facts do the work:

1. **Family invariance.** The reference process maps a family member to a family
   member by contracting `z → e^{-t} z`. The noised emission is two rescaled
   scalars — no kernel integral — and the law at time `t` is again a two-state HMM
   with the *same* transition matrix.
2. **Exact likelihood.** The latent space has dimension two, so the likelihood is
   a transfer-matrix product, `O(L)` per sample (`log_likelihood`), cross-checked
   against the `2^L` enumeration (`log_likelihood_by_enumeration`).
3. **Closed-form moments.** Noising leaves the correlation length alone, damps the
   mean contrast as `e^{-t}`, and the one-site variance is the variance function
   read at the contracted means (`predicted_moments`).

Index 0 is the state `+1`, index 1 the state `−1`, throughout. The only
family-specific data is `BASELINES`: a baseline to relax towards and a
natural-parameter separation for the two emissions.

### 4.1 Down to a single site

`--sites 1` is supported and is the natural smallest test case. There the chain
has nothing to correlate and the model collapses exactly to an **equal mixture of
the two emissions, independent of the flip rate `ε`**:

```
log L = log ½ ( p₊(x) + p₋(x) )
```

Verified for all six families to `1e-15`, against both the transfer matrix and
the enumeration, at `ε = 0, 0.15, 0.5, 0.9`
(`test_a_single_site_is_the_two_component_mixture`). The lag-dependent tables in
the benchmark output are empty at one site, by construction; everything else runs,
including the figure.

```bash
python -m applications.two_state_hmm --family normal --sites 1 --no-plot
python -m applications.two_state_hmm --family all --sites 24
python -m applications.two_state_hmm --family poisson --variants
```

**A note on GHS.** Its marginal flow is legitimate — `z → e^{-t} z` is an identity
in the shift coordinate and the contracted member is an ordinary law. What GHS
lacks is a positivity-preserving kernel, hence any *joint* draw of a state and its
noised counterpart. This module only ever needs marginals, so GHS runs; a forward
Markov process would not.

---

## 5. Reproducing the figures

All three figure scripts take `--output`; without it they write to the location
the note uses. The note includes **PDFs only**.

### The note's figures

```bash
# Section 8, amplitude fitting: one-channel-fits-1..3.pdf and one-channel-edge.pdf
python -m applications.paper_one_channel --output /path/to/figures/one-channel

# Section 9.7, real against complex: complex-relaxation-fits-1..3.pdf
python -m applications.paper_complex_relaxation \
    --pages --no-sheet --output /path/to/figures/complex-relaxation

# Section 9.7, the reach study: bimodal-gauss-fits.pdf
python -m applications.paper_bimodal_gauss \
    --output /path/to/figures/complex-relaxation
```

Each prints the table of numbers the note quotes, so the text and the figures
cannot drift apart.

`--no-sheet` suppresses `paper_complex_relaxation`'s raster diagnostic sheet —
all six families on one tall PNG, useful for looking at them together but not part
of the note. Dropping both flags writes only the sheet. `paper_bimodal_gauss`
needs `--png` to emit a raster at all.

### Shared figure conventions

`paper_one_channel` holds what the other two import: `FAMILY_TITLES` (families are
spelled out — "Generalized hyperbolic secant distribution"), the colour palette
(`TARGET_COLOUR`, `REFERENCE_COLOUR`, `FIT_COLOUR`, `EXACT_COLOUR`,
`SAMPLE_COLOUR`), `compact_scientific` (renders `1.3·10⁻³`, not `1.3e-03`), and
`separation_for`. Change them there and all three figure sets follow.

Block headings are placed from the grid cell's own position plus a fixed physical
offset — a GridSpec cell bounds the *axes*, not their titles, so a hand-tuned
figure-fraction offset lands on the panel titles at one figure size and not
another.

### Diagnostic figures

Not part of the note; written to `artifacts/`, which is git-ignored.

```bash
python -m applications.amplitude_fit_recovery --family gamma --plot
python -m applications.amplitude_fit_complex --family all --target all --plot
python -m applications.two_state_hmm --family poisson
python -m applications.shifted_baseline_probability_modes --family all --no-plot
```

Figures elsewhere in the note — reference densities, activated laws, relaxation
kernels — come from the Mathematica notebooks and are not generated here.

---

## 6. Tests

```bash
python -m pytest -q
```

| file | covers |
| --- | --- |
| `tests/package/test_jacobi.py` | recurrence coefficients, orthonormality, Clenshaw |
| `tests/package/test_log_prob.py` | densities against closed forms, broadcasting |
| `tests/package/test_moments_affinity.py` | mean, variance, variance slope, Hellinger affinity |
| `tests/package/test_shifts_linearization.py` | shift coefficients and the product tensor |
| `tests/package/test_broadcasting.py` | paired, outer and batch parameter shapes |
| `tests/applications/test_amplitude_fit.py` | the real fit, degree selection, targets |
| `tests/applications/test_amplitude_fit_complex.py` | complex fit, relaxation, certificates, factorisation, lattice caps, half-line model |
| `tests/applications/test_two_state_hmm.py` | transfer matrix vs enumeration, flow, moments, **single site** |
| `tests/applications/test_shifted_baseline_probability_modes.py` | mode projection and damping |

---

## 7. Conventions and traps

Things that cost time to learn, in the order they bite.

**`2K` is a continuous-support requirement, not a universal one.** A degree-`K`
amplitude emits coefficients out to `R_{2K}` only where `phi_m phi_n` has degree
`m+n`. On a lattice with `N+1` points the product folds back into degrees `≤ N`,
and `R_0 … R_N` determine the law completely. Demanding `2K` there is not
stronger, it is impossible — it caps the binomial at `K = 6` when `K = 12` is both
legal and *exact*, since at `K = N` the amplitude spans every function on the
support. Use `fitting_matrices`, not `product_matrices`.

**Never form power-basis coefficients.** Root-finding on a degree-`2K` polynomial
via interpolation into the power basis is hopeless past degree ~20: the error on a
rank-3 state at `K = 12` ran from `10⁻⁵` (Normal) to `10⁷` (negative binomial).
The comrade matrix plus a Gauss rule fixes it by six to twelve orders. Clenshaw
covers *evaluation*; it does not find roots.

**Rank two needs no roots at all.** `f² + g² = |f + ig|²`. Reach for
`rank_two_seed` first and the root route only for rank ≥ 3.

**Matching coefficients is not being close to the law.** `J` and total variation
part company once `R_k` spans orders of magnitude — in several fits `J` improves
while `D` gets worse, because unweighted least squares spends itself on the
high-`k` tail rather than where the mass is. Quote both.

**The relaxation's own gap can be loose; the amplitude's is tight.** LM converges
far harder than first-order methods on the spectrahedron. To certify a claim,
compute `certified_gap` at the fitted amplitude rather than trusting agreement
between two optimisers.

**Sum of squares is conservative off ℝ.** `ρ ⪰ 0` makes `q ≥ 0` on the whole line.
On ℝ that is exactly the positive class (Markov–Lukács); on a lattice or half line
it is strictly smaller. For the binomial at `K < N` the shortfall is real and
measurable; the half-line remedy is `localised_matrices`.

**Degree scales with the square of the separation.** For a symmetric Gaussian
mixture at `±d` on a unit base, both exact amplitudes have coefficient profile
`(d/2)ⁿ/√(n!)` — Poisson in `n` with mean `λ = (d/2)²` — so the budget is
`K ≈ λ + 5√λ`. Quadratic, not exponential. But the coefficients `R_k = d^k/√(k!)`
reach `10²¹` by `d = 10`, and the degree a *sample* supports grows far more
slowly: at `d ≳ 6` the separation is not recoverable from data at any realistic
`N`. The wall is statistical, not representational.

**A GridSpec cell bounds the axes, not their titles.** Any heading placed at a
hand-tuned figure fraction will collide at some other figure size.
