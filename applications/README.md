# Numerical demonstrations

Applications live outside the installable `nefqvf` package. They exercise the
public package API and provide reproducible numerical examples without turning
research workflows into library dependencies.

## Shifted-baseline probability modes

Run one family and write a diagnostic figure:

```bash
python -m applications.shifted_baseline_probability_modes --family normal
```

Run every family without plotting:

```bash
python -m applications.shifted_baseline_probability_modes \
    --family all \
    --no-plot
```

Available family names are:

```text
normal
poisson
gamma
binomial
negative-binomial
ghs
```

Figures are written to `artifacts/`, which is intentionally ignored by Git.

### What the demonstration checks

For a baseline density \(p_0\) and a naturally shifted family member \(q\), the
probability ratio is expanded as

\[
\frac{q(x)}{p_0(x)} = \sum_k c_k\phi_k(x).
\]

The application computes \(c_k\) in three independent ways:

1. From the analytic family shift kernel \(\gamma_k\xi^k\).
2. By exact summation or numerical quadrature under the shifted law.
3. From toy samples using the empirical average of \(\phi_k\).

It then damps the probability modes by

\[
c_k(\tau)=e^{-k\tau}c_k
\]

and compares the reconstructed density with the exact family member obtained
by damping the family-specific shift coordinate \(\xi\).

The second check concerns the product-linearization tensor

\[
\Lambda_{mnk}
=
\mathbb{E}_{p_0}[\phi_m(X)\phi_n(X)\phi_k(X)].
\]

The package computes this tensor analytically from the Jacobi recurrence. The
application estimates the same triple products from an independent baseline
sample and compares the difference with its Monte Carlo uncertainty. It also
checks permutation symmetry and \(\Lambda_{0nk}=\delta_{nk}\).

### Reading the output

The coefficient table reports:

```text
analytic     closed-form shift coefficient
quadrature   exact sum or numerical integral
empirical    estimate from shifted toy samples
MC error     standard error of the empirical estimate
```

Higher polynomial modes emphasize distribution tails and therefore have larger
Monte Carlo variance. Their empirical values can visibly differ from the
analytic coefficients even when the reported pull is statistically ordinary.
Changing `n_max` does not alter an already computed coefficient; truncation
only affects the reconstructed density.

The `Lambda check` reports the largest absolute sample difference and the
largest standardized difference. The demonstration deliberately samples only
degrees zero through three because high-order triple products have rapidly
growing tail variance.

### GHS sampling

The package does not yet expose sampling. For this demonstration, the five
standard families use `numpy.random.Generator`, while GHS uses an approximate
inverse-CDF sampler on a dense numerical grid. The latter is adequate for the
consistency check but is not intended as a production GHS sampler.

## Changing or adding a configuration

Family-specific choices are isolated in configuration factories near the top
of `shifted_baseline_probability_modes.py`. A configuration specifies:

- baseline parameters and natural shift;
- observation grid and plotting limits;
- truncation orders;
- sample size and random seed;
- numerical tolerances;
- a demonstration-local sampler.

The projection, damping, \(\Lambda\) checks, output, and plotting code are
family independent. To add another benchmark, create a configuration factory
and register it in `CONFIGURATION_FACTORIES`.
