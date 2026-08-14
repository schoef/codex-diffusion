# NEF-QVF numerical toolkit

`nefqvf` provides NumPy/SciPy implementations of the six one-dimensional
natural exponential families with quadratic variance functions:

| Family | Parameters | Orthogonal basis |
| --- | --- | --- |
| Normal | `NormalParams(mean, sigma)` | Hermite |
| Poisson | `PoissonParams(mean)` | Charlier |
| Gamma | `GammaParams(mean, r)` | Laguerre |
| Binomial | `BinomialParams(mean, N)` | Krawtchouk |
| Negative binomial | `NegativeBinomialParams(mean, r)` | Meixner |
| Generalized hyperbolic secant (GHS) | `GHSParams(mean, r)` | Meixner-Pollaczek |

The current implementation covers probability evaluation, moments, natural
parameters, Hellinger affinities, orthonormal-polynomial evaluation, exact
shift coefficients, product-linearization tensors, and inverse-CDF sampling.
Diffusion kernels and a JAX backend are later milestones.

This repository is the reference implementation for the accompanying note on
amplitude parametrisations of NEF-QVF laws, and reproduces its figures. See
[Reproducing the figures](#reproducing-the-figures).

## Installation

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The editable install keeps imports pointed at `src/nefqvf`, so source changes
are immediately visible without reinstalling.

## First calculation

```python
import numpy as np

from nefqvf import Gamma, GammaParams

params = GammaParams(mean=3.0, r=2.5)
x = np.linspace(0.01, 10.0, 500)

log_density = Gamma.log_prob(x, params)
basis = Gamma.basis(x, n_max=4, params=params)

# Evaluate sum_k coefficients[k] * phi_k(x) without storing another basis.
coefficients = np.array([1.0, 0.2, -0.05, 0.0, 0.01])
expansion = Gamma.basis_dot(x, coefficients, params)

# Lambda[m, n, k] = E[phi_m(X) phi_n(X) phi_k(X)] at the baseline.
linearization = Gamma.linearization_tensor(n_max=4, params=params)
```

Family objects such as `Gamma` are stateless singletons. Distribution
parameters are immutable dataclasses and may contain scalars or arrays.

## Array semantics

Paired methods use ordinary NumPy broadcasting:

```python
from nefqvf import Normal, NormalParams

params = NormalParams(mean=np.array([-1.0, 1.0]), sigma=2.0)
Normal.log_prob(np.array([0.0, 0.5]), params).shape
# (2,)
```

Grid methods explicitly evaluate every parameter batch member against every
observation:

```python
x = np.linspace(-5.0, 5.0, 101)
Normal.log_prob_grid(x, params).shape
# (2, 101)
```

For multidimensional observations, `log_prob_grid` returns
`parameter_batch_shape + x.shape`. Use `chunk_size=` when an outer grid would
otherwise create a large temporary array.

`basis(..., grid=False)` follows paired broadcasting and appends a polynomial
degree axis. With `grid=True`, its shape is
`parameter_batch_shape + observation_shape + (n_max + 1,)`.

## Polynomial convention

All six bases use the orthonormal positive-positive off-diagonal convention

```text
x phi_n(x) = a_(n+1) phi_(n+1)(x) + b_n phi_n(x) + a_n phi_(n-1)(x),
a_n > 0 for n > 0.
```

This fixes sign ambiguities in the Laguerre, Krawtchouk, and Meixner systems.
The same convention is used by `basis`, `shift_coefficients`, and
`linearization_tensor`.

For a natural-parameter shift `j`,

```python
j = 0.25
shifted = Gamma.shifted_params(params, j)
coefficients = Gamma.shift_coefficients(j, n_max=8, params=params)
```

returns the coefficients in

```text
p_(eta+j)(x) / p_eta(x) = sum_n coefficients[n] phi_n(x; eta).
```

These are probability-ratio coefficients. They are not the coherent-state
coefficients of a square-root probability amplitude.

## Repository layout

```text
src/nefqvf/    Installable numerical package
applications/ Standalone numerical studies and the figure scripts
tests/package Package-level tests
tests/applications/
               End-to-end demonstration tests
material/      Retained mathematical source material
```

See [applications/README.md](applications/README.md) for what each study does.

## Reproducing the figures

The one-channel amplitude figures of the note are written by a single script.
It fits every family, selects the truncation by held-out likelihood, and writes
four PDFs:

```bash
python -m applications.paper_one_channel --output /path/to/paper/figures/one-channel
```

This produces `one-channel-fits-1.pdf` through `-3.pdf`, two families to a page,
and `one-channel-edge.pdf` for the truncated targets. It also prints the table of
total variations and selected degrees that the note quotes. With no `--output` it
writes to the location the note uses.

The two-state hidden-Markov figure is produced separately:

```bash
python -m applications.two_state_hmm --family poisson --plot
```

Figures elsewhere in the note that show reference densities, activated laws and
relaxation kernels come from the accompanying Mathematica notebooks and are not
generated here.

## Development checks

```bash
python -m pytest -q
python -m compileall -q src applications tests
ruff check src applications tests
ruff format --check src applications tests
python -m applications.shifted_baseline_probability_modes --family all --no-plot
python -m applications.paper_one_channel --output /tmp/figures
```

Each family module also has a small protected smoke test:

```bash
python -m nefqvf.normal
python -m nefqvf.poisson
python -m nefqvf.gamma
python -m nefqvf.binomial
python -m nefqvf.negative_binomial
python -m nefqvf.ghs
```
