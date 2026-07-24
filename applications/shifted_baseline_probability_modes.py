"""Demonstrate probability-mode projection and damping for all six families.

For a baseline law ``p_0`` and a naturally shifted member ``q``, the demo
expands the probability ratio

    q(x) / p_0(x) = sum_k c_k phi_k(x).

It compares analytic, deterministic, and sampled coefficients, damps them by
``exp(-k*tau)``, and checks the package's product-linearization tensor against
an independent Monte Carlo estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator
from scipy.integrate import cumulative_trapezoid

from nefqvf import (
    GHS,
    Binomial,
    BinomialParams,
    Gamma,
    GammaParams,
    GHSParams,
    NegativeBinomial,
    NegativeBinomialParams,
    Normal,
    NormalParams,
    Poisson,
    PoissonParams,
)

FAMILY_NAMES = (
    "normal",
    "poisson",
    "gamma",
    "binomial",
    "negative-binomial",
    "ghs",
)


def _inverse_cdf_sample(
    family: Any,
    params: Any,
    grid: np.ndarray,
    rng: Generator,
    size: int,
) -> np.ndarray:
    """Draw approximate continuous samples from a numerical inverse CDF.

    This demonstration-only fallback is used for GHS until the package gains
    a production sampler. ``grid`` must cover enough tail mass for the chosen
    parameters.
    """

    probability = family.prob(grid, params)
    cdf = cumulative_trapezoid(probability, grid, initial=0.0)
    cdf /= cdf[-1]
    keep = np.concatenate(([True], np.diff(cdf) > 0.0))
    return np.interp(rng.random(size), cdf[keep], grid[keep])


def _integrate(
    values: np.ndarray,
    grid: np.ndarray,
    *,
    discrete: bool,
) -> np.ndarray:
    """Integrate values over the leading grid axis."""

    if discrete:
        return np.sum(values, axis=0)
    return np.trapezoid(values, grid, axis=0)


def _damped_coefficients(coefficients: np.ndarray, tau: float) -> np.ndarray:
    """Apply probability-mode damping ``c_k -> exp(-k*tau) c_k``."""

    degree = np.arange(coefficients.size)
    return np.exp(-degree * tau) * coefficients


def _reconstructed_density(
    family: Any,
    grid: np.ndarray,
    baseline: Any,
    coefficients: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Reconstruct a damped density from probability-mode coefficients."""

    ratio = family.basis_dot(
        grid,
        _damped_coefficients(coefficients, tau),
        baseline,
    )
    return family.prob(grid, baseline) * ratio


def _sampled_linearization(
    family: Any,
    baseline: Any,
    samples: np.ndarray,
    n_max: int,
    *,
    chunk_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ``Lambda_mnk`` and its standard error in bounded memory."""

    shape = (n_max + 1,) * 3
    total = np.zeros(shape)
    total_square = np.zeros(shape)

    for start in range(0, samples.size, chunk_size):
        values = family.basis(
            samples[start : start + chunk_size],
            n_max,
            baseline,
        )
        products = (
            values[:, :, None, None]
            * values[:, None, :, None]
            * values[:, None, None, :]
        )
        total += products.sum(axis=0)
        total_square += np.square(products).sum(axis=0)

    estimate = total / samples.size
    sample_variance = (total_square - samples.size * estimate**2) / (samples.size - 1)
    standard_error = np.sqrt(np.maximum(sample_variance, 0.0) / samples.size)
    return estimate, standard_error


def run_demo(
    name: str,
    *,
    make_plot: bool = True,
    sample_size: int = 200_000,
    output_dir: Path | None = None,
) -> None:
    """Run the complete shifted-baseline demonstration for one family."""

    if sample_size < 2:
        raise ValueError("sample_size must be at least two")

    # These are the only family-specific choices in the application.
    n_max = 8
    linearization_n_max = 3
    taus = (0.0, 0.35, 0.8, 1.5)
    projection_atol = 2e-12
    exact_density_atol = 2e-8

    if name == "normal":
        display_name = "Normal / Hermite"
        family = Normal
        baseline = NormalParams(mean=0.0, sigma=1.0)
        natural_shift = 0.8
        grid = np.linspace(-9.0, 9.0, 12_001)
        discrete = False
        plot_limits = (-4.0, 5.0)
        seed = 17
        projection_atol = 2e-9
        exact_density_atol = 1e-8

        def sample(rng: Generator, params: NormalParams, size: int) -> np.ndarray:
            return rng.normal(
                loc=float(params.mean),
                scale=float(params.sigma),
                size=size,
            )

    elif name == "poisson":
        display_name = "Poisson / Charlier"
        family = Poisson
        baseline = PoissonParams(mean=4.0)
        natural_shift = 0.45
        grid = np.arange(80)
        discrete = True
        plot_limits = (-0.5, 17.5)
        seed = 23

        def sample(rng: Generator, params: PoissonParams, size: int) -> np.ndarray:
            return rng.poisson(float(params.mean), size=size)

    elif name == "gamma":
        display_name = "Gamma / Laguerre"
        family = Gamma
        baseline = GammaParams(mean=3.0, r=2.5)
        natural_shift = 0.25
        grid = np.linspace(1e-7, 100.0, 100_001)
        discrete = False
        plot_limits = (0.0, 15.0)
        seed = 29
        projection_atol = 1e-9

        def sample(rng: Generator, params: GammaParams, size: int) -> np.ndarray:
            return rng.gamma(
                shape=float(params.r),
                scale=float(params.mean) / float(params.r),
                size=size,
            )

    elif name == "binomial":
        display_name = "Binomial / Krawtchouk"
        family = Binomial
        baseline = BinomialParams(mean=3.6, N=12)
        natural_shift = 0.7
        grid = np.arange(int(baseline.N) + 1)
        discrete = True
        plot_limits = (-0.5, float(baseline.N) + 0.5)
        seed = 31
        n_max = int(baseline.N)
        exact_density_atol = 2e-12

        def sample(rng: Generator, params: BinomialParams, size: int) -> np.ndarray:
            return rng.binomial(
                int(params.N),
                float(params.mean) / int(params.N),
                size=size,
            )

    elif name == "negative-binomial":
        display_name = "Negative binomial / Meixner"
        family = NegativeBinomial
        baseline = NegativeBinomialParams(mean=4.0, r=3.0)
        natural_shift = 0.12
        grid = np.arange(200)
        discrete = True
        plot_limits = (-0.5, 24.5)
        seed = 37

        def sample(
            rng: Generator,
            params: NegativeBinomialParams,
            size: int,
        ) -> np.ndarray:
            success_probability = float(params.r) / (
                float(params.r) + float(params.mean)
            )
            return rng.negative_binomial(
                float(params.r),
                success_probability,
                size=size,
            )

    elif name == "ghs":
        display_name = "GHS / Meixner-Pollaczek"
        family = GHS
        baseline = GHSParams(mean=0.0, r=1.5)
        natural_shift = 0.6
        grid = np.linspace(-25.0, 25.0, 100_001)
        discrete = False
        plot_limits = (-4.0, 5.0)
        seed = 41
        projection_atol = 2e-9

        def sample(rng: Generator, params: GHSParams, size: int) -> np.ndarray:
            return _inverse_cdf_sample(family, params, grid, rng, size)

    else:
        choices = ", ".join(FAMILY_NAMES)
        raise ValueError(f"unknown family {name!r}; choose one of: {choices}")

    shifted = family.shifted_params(baseline, natural_shift)
    rng = np.random.default_rng(seed)
    shifted_samples = sample(rng, shifted, sample_size)

    # Project q/p_0 in three independent ways.
    sample_basis = family.basis(shifted_samples, n_max, baseline)
    empirical = sample_basis.mean(axis=0)
    standard_error = sample_basis.std(axis=0, ddof=1) / np.sqrt(sample_size)

    grid_basis = family.basis(grid, n_max, baseline)
    shifted_probability = family.prob(grid, shifted)
    quadrature = _integrate(
        shifted_probability[:, None] * grid_basis,
        grid,
        discrete=discrete,
    )
    analytic = family.shift_coefficients(natural_shift, n_max, baseline)

    assert np.allclose(
        quadrature,
        analytic,
        atol=projection_atol,
        rtol=projection_atol,
    ), "quadrature coefficients do not match the analytic shift kernel"
    assert np.all(np.abs(empirical - analytic) <= 6.0 * standard_error + 2e-3), (
        "empirical coefficients disagree with their Monte Carlo errors"
    )
    assert np.isclose(empirical[0], 1.0), "the constant mode must remain normalized"

    shift_coordinate = family.shift_coordinate(natural_shift, baseline)
    for tau in taus:
        density = _reconstructed_density(
            family,
            grid,
            baseline,
            quadrature,
            tau,
        )
        normalization = _integrate(density, grid, discrete=discrete)
        assert np.isclose(
            normalization,
            1.0,
            atol=max(projection_atol, 2e-10),
        ), f"damped reconstruction is not normalized at tau={tau}"

    comparison_tau = max(taus)
    exact_damped_params = family.from_shift_coordinate(
        np.exp(-comparison_tau) * shift_coordinate,
        baseline,
    )
    reconstructed = _reconstructed_density(
        family,
        grid,
        baseline,
        quadrature,
        comparison_tau,
    )
    exact = family.prob(grid, exact_damped_params)
    assert np.max(np.abs(reconstructed - exact)) <= exact_density_atol, (
        "damped coefficients do not reproduce the exact shifted family"
    )

    print(f"\n{display_name}")
    print(" k      analytic     quadrature      empirical        MC error")
    for degree, values in enumerate(
        zip(
            analytic,
            quadrature,
            empirical,
            standard_error,
            strict=True,
        )
    ):
        exact_coefficient, integrated, sampled, error = values
        print(
            f"{degree:2d}  {exact_coefficient:12.6g}  {integrated:12.6g}"
            f"  {sampled:12.6g}  {error:12.3g}"
        )

    # Independently check Lambda_mnk with samples from the baseline law.
    analytic_lambda = family.linearization_tensor(linearization_n_max, baseline)
    baseline_rng = np.random.default_rng(seed + 10_000)
    baseline_samples = sample(baseline_rng, baseline, sample_size)
    sampled_lambda, lambda_error = _sampled_linearization(
        family,
        baseline,
        baseline_samples,
        linearization_n_max,
    )

    assert np.allclose(
        analytic_lambda,
        np.transpose(analytic_lambda, (1, 0, 2)),
        atol=2e-12,
    ), "analytic Lambda is not symmetric in m and n"
    assert np.allclose(
        analytic_lambda,
        np.transpose(analytic_lambda, (2, 1, 0)),
        atol=2e-12,
    ), "analytic Lambda is not symmetric in m and k"
    assert np.allclose(
        analytic_lambda[0],
        np.eye(linearization_n_max + 1),
        atol=2e-12,
    ), "Lambda[0, n, k] must equal delta[n, k]"
    assert np.all(
        np.abs(sampled_lambda - analytic_lambda) <= 6.0 * lambda_error + 3e-3
    ), "sampled Lambda disagrees with its Monte Carlo errors"

    lambda_difference = sampled_lambda - analytic_lambda
    lambda_pull = np.divide(
        lambda_difference,
        lambda_error,
        out=np.zeros_like(lambda_difference),
        where=lambda_error > 0.0,
    )
    largest_pull = np.unravel_index(
        np.argmax(np.abs(lambda_pull)),
        lambda_pull.shape,
    )
    print(
        "Lambda check:"
        f" max |sample-analytic|={np.max(np.abs(lambda_difference)):.4g};"
        f" max |pull|={abs(lambda_pull[largest_pull]):.2f}"
        f" at {tuple(int(index) for index in largest_pull)}"
    )

    minimum = min(
        _reconstructed_density(
            family,
            grid,
            baseline,
            empirical,
            tau,
        ).min()
        for tau in taus
    )
    print(f"Minimum reconstructed probability: {minimum:.6g}")

    if not make_plot:
        return

    figure, (density_axis, coefficient_axis) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.6),
        constrained_layout=True,
    )

    if discrete:
        sample_min = int(np.min(grid))
        sample_max = int(min(np.max(shifted_samples), plot_limits[1]))
        bins = np.arange(sample_min, sample_max + 2) - 0.5
        drawstyle = "steps-mid"
    else:
        bins = 90
        drawstyle = "default"

    density_axis.hist(
        shifted_samples,
        bins=bins,
        density=True,
        color="0.82",
        edgecolor="none",
        label="toy data",
    )
    density_axis.plot(
        grid,
        shifted_probability,
        color="black",
        linewidth=2.0,
        drawstyle=drawstyle,
        label="shifted baseline",
    )

    colors = plt.colormaps["viridis"](np.linspace(0.15, 0.85, len(taus)))
    for color, tau in zip(colors, taus, strict=True):
        empirical_density = _reconstructed_density(
            family,
            grid,
            baseline,
            empirical,
            tau,
        )
        exact_params = family.from_shift_coordinate(
            np.exp(-tau) * shift_coordinate,
            baseline,
        )
        exact_density = family.prob(grid, exact_params)
        density_axis.plot(
            grid,
            empirical_density,
            color=color,
            linewidth=1.6,
            drawstyle=drawstyle,
            label=rf"spectral fit, $\tau={tau:g}$",
        )
        if tau > 0:
            density_axis.plot(
                grid,
                exact_density,
                color=color,
                linewidth=1.0,
                linestyle=":",
                drawstyle=drawstyle,
            )

    density_axis.axhline(0.0, color="0.5", linewidth=0.7)
    density_axis.set_xlim(*plot_limits)
    density_axis.set_xlabel("x")
    density_axis.set_ylabel("probability" if discrete else "probability density")
    density_axis.set_title(display_name)
    density_axis.legend(frameon=False, fontsize=8)

    degree = np.arange(n_max + 1)
    coefficient_axis.errorbar(
        degree,
        empirical,
        yerr=standard_error,
        fmt="o",
        color="#2166ac",
        capsize=2.5,
        label="empirical projection",
    )
    coefficient_axis.plot(
        degree,
        analytic,
        "x",
        color="#b2182b",
        markersize=7,
        label=r"analytic $\gamma_k\xi^k$",
    )
    coefficient_axis.axhline(0.0, color="0.5", linewidth=0.7)
    coefficient_axis.set_xlabel("mode k")
    coefficient_axis.set_ylabel(r"$c_k$")
    coefficient_axis.set_title("Probability-ratio coefficients")
    coefficient_axis.set_xticks(degree)
    coefficient_axis.legend(frameon=False, fontsize=8)

    output = (
        output_dir
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "artifacts"
    )
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / f"{name}_shifted_baseline_probability_modes.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    print(f"Figure: {figure_path}")


def _parse_args() -> argparse.Namespace:
    """Parse the family selector and plotting switch."""

    parser = argparse.ArgumentParser(
        description="Project and damp a shifted NEF-QVF probability ratio."
    )
    parser.add_argument(
        "--family",
        choices=(*FAMILY_NAMES, "all"),
        default="normal",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="run numerical checks without writing figures",
    )
    return parser.parse_args()


def main() -> None:
    """Run one family or the complete six-family demonstration."""

    args = _parse_args()
    names = FAMILY_NAMES if args.family == "all" else (args.family,)
    for name in names:
        run_demo(name, make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
