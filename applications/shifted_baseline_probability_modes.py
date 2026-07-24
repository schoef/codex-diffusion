"""Project and damp the probability ratio of a shifted NEF-QVF baseline.

This example concerns the probability-ratio expansion

    q(x) / p_0(x) = sum_k c_k phi_k(x).

For a natural shift, its analytic coefficients are the shift-kernel
coefficients gamma_k * xi(j)**k. They are distinct from the amplitude
coherent-state coefficients, which involve a half-shift and a Hellinger
factor.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class FamilyConfiguration:
    """All ingredients that differ between NEF-QVF families."""

    slug: str
    name: str
    family: Any
    baseline: Any
    shifted: Any
    grid: np.ndarray
    is_discrete: bool
    plot_limits: tuple[float, float]
    n_max: int
    linearization_n_max: int
    sample_size: int
    seed: int
    natural_shift: float
    taus: tuple[float, ...]
    projection_atol: float
    exact_density_atol: float
    sample: Callable[[Generator, Any, int], np.ndarray]


@dataclass(frozen=True)
class ProjectionResult:
    """Analytic, deterministic, and sampled probability-mode coefficients."""

    samples: np.ndarray
    analytic: np.ndarray
    quadrature: np.ndarray
    empirical: np.ndarray
    standard_error: np.ndarray


@dataclass(frozen=True)
class LinearizationResult:
    """Analytic and Monte Carlo estimates of the same ``Lambda_mnk`` tensor."""

    analytic: np.ndarray
    sampled: np.ndarray
    standard_error: np.ndarray


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
    interpolation_cdf = cdf[keep]
    interpolation_grid = grid[keep]
    return np.interp(rng.random(size), interpolation_cdf, interpolation_grid)


# ---------------------------------------------------------------------------
# Family-specific configurations. These are selected by the CLI registry.
# ---------------------------------------------------------------------------
def normal_configuration() -> FamilyConfiguration:
    """Return the Normal/Hermite benchmark configuration."""

    baseline = NormalParams(mean=0.0, sigma=1.0)
    natural_shift = 0.8
    shifted = Normal.shifted_params(baseline, natural_shift)

    def sample(rng: Generator, params: NormalParams, size: int) -> np.ndarray:
        """Draw Normal samples through NumPy's reference generator."""

        return rng.normal(
            loc=float(params.mean),
            scale=float(params.sigma),
            size=size,
        )

    return FamilyConfiguration(
        slug="normal",
        name="Normal / Hermite",
        family=Normal,
        baseline=baseline,
        shifted=shifted,
        grid=np.linspace(-9.0, 9.0, 12_001),
        is_discrete=False,
        plot_limits=(-4.0, 5.0),
        n_max=8,
        linearization_n_max=3,
        sample_size=200_000,
        seed=17,
        natural_shift=natural_shift,
        taus=(0.0, 0.35, 0.8, 1.5),
        projection_atol=2e-9,
        exact_density_atol=1e-8,
        sample=sample,
    )


def poisson_configuration() -> FamilyConfiguration:
    """Return the Poisson/Charlier benchmark configuration."""

    baseline = PoissonParams(mean=4.0)
    natural_shift = 0.45
    shifted = Poisson.shifted_params(baseline, natural_shift)

    def sample(rng: Generator, params: PoissonParams, size: int) -> np.ndarray:
        """Draw Poisson samples through NumPy's reference generator."""

        return rng.poisson(float(params.mean), size=size)

    return FamilyConfiguration(
        slug="poisson",
        name="Poisson / Charlier",
        family=Poisson,
        baseline=baseline,
        shifted=shifted,
        grid=np.arange(80),
        is_discrete=True,
        plot_limits=(-0.5, 17.5),
        n_max=8,
        linearization_n_max=3,
        sample_size=200_000,
        seed=23,
        natural_shift=natural_shift,
        taus=(0.0, 0.35, 0.8, 1.5),
        projection_atol=2e-12,
        exact_density_atol=2e-8,
        sample=sample,
    )


def gamma_configuration() -> FamilyConfiguration:
    """Return the Gamma/Laguerre benchmark configuration."""

    baseline = GammaParams(mean=3.0, r=2.5)
    natural_shift = 0.25
    shifted = Gamma.shifted_params(baseline, natural_shift)

    def sample(rng: Generator, params: GammaParams, size: int) -> np.ndarray:
        """Draw Gamma samples after converting mean to scale."""

        return rng.gamma(
            shape=float(params.r),
            scale=float(params.mean) / float(params.r),
            size=size,
        )

    return FamilyConfiguration(
        slug="gamma",
        name="Gamma / Laguerre",
        family=Gamma,
        baseline=baseline,
        shifted=shifted,
        grid=np.linspace(1e-7, 100.0, 100_001),
        is_discrete=False,
        plot_limits=(0.0, 15.0),
        n_max=8,
        linearization_n_max=3,
        sample_size=200_000,
        seed=29,
        natural_shift=natural_shift,
        taus=(0.0, 0.35, 0.8, 1.5),
        projection_atol=1e-9,
        exact_density_atol=2e-8,
        sample=sample,
    )


def binomial_configuration() -> FamilyConfiguration:
    """Return the finite Binomial/Krawtchouk benchmark configuration."""

    trials = 12
    baseline = BinomialParams(mean=3.6, N=trials)
    natural_shift = 0.7
    shifted = Binomial.shifted_params(baseline, natural_shift)

    def sample(rng: Generator, params: BinomialParams, size: int) -> np.ndarray:
        """Draw Binomial samples after converting mean to probability."""

        return rng.binomial(
            trials,
            float(params.mean) / trials,
            size=size,
        )

    return FamilyConfiguration(
        slug="binomial",
        name="Binomial / Krawtchouk",
        family=Binomial,
        baseline=baseline,
        shifted=shifted,
        grid=np.arange(trials + 1),
        is_discrete=True,
        plot_limits=(-0.5, trials + 0.5),
        n_max=trials,
        linearization_n_max=3,
        sample_size=200_000,
        seed=31,
        natural_shift=natural_shift,
        taus=(0.0, 0.35, 0.8, 1.5),
        projection_atol=2e-12,
        exact_density_atol=2e-12,
        sample=sample,
    )


def negative_binomial_configuration() -> FamilyConfiguration:
    """Return the negative-binomial/Meixner benchmark configuration."""

    baseline = NegativeBinomialParams(mean=4.0, r=3.0)
    natural_shift = 0.12
    shifted = NegativeBinomial.shifted_params(baseline, natural_shift)

    def sample(
        rng: Generator,
        params: NegativeBinomialParams,
        size: int,
    ) -> np.ndarray:
        """Draw counts using NumPy's success-probability convention."""

        success_probability = float(params.r) / (float(params.r) + float(params.mean))
        return rng.negative_binomial(
            float(params.r),
            success_probability,
            size=size,
        )

    return FamilyConfiguration(
        slug="negative-binomial",
        name="Negative binomial / Meixner",
        family=NegativeBinomial,
        baseline=baseline,
        shifted=shifted,
        grid=np.arange(200),
        is_discrete=True,
        plot_limits=(-0.5, 24.5),
        n_max=8,
        linearization_n_max=3,
        sample_size=200_000,
        seed=37,
        natural_shift=natural_shift,
        taus=(0.0, 0.35, 0.8, 1.5),
        projection_atol=2e-12,
        exact_density_atol=2e-8,
        sample=sample,
    )


def ghs_configuration() -> FamilyConfiguration:
    """Return the GHS/Meixner-Pollaczek benchmark configuration."""

    baseline = GHSParams(mean=0.0, r=1.5)
    natural_shift = 0.6
    shifted = GHS.shifted_params(baseline, natural_shift)
    grid = np.linspace(-25.0, 25.0, 100_001)

    def sample(rng: Generator, params: GHSParams, size: int) -> np.ndarray:
        """Draw approximate GHS samples from the configured numerical grid."""

        return _inverse_cdf_sample(
            GHS,
            params,
            grid,
            rng,
            size,
        )

    return FamilyConfiguration(
        slug="ghs",
        name="GHS / Meixner-Pollaczek",
        family=GHS,
        baseline=baseline,
        shifted=shifted,
        grid=grid,
        is_discrete=False,
        plot_limits=(-4.0, 5.0),
        n_max=8,
        linearization_n_max=3,
        sample_size=200_000,
        seed=41,
        natural_shift=natural_shift,
        taus=(0.0, 0.35, 0.8, 1.5),
        projection_atol=2e-9,
        exact_density_atol=2e-8,
        sample=sample,
    )


CONFIGURATION_FACTORIES: dict[str, Callable[[], FamilyConfiguration]] = {
    "normal": normal_configuration,
    "poisson": poisson_configuration,
    "gamma": gamma_configuration,
    "binomial": binomial_configuration,
    "negative-binomial": negative_binomial_configuration,
    "ghs": ghs_configuration,
}


# ---------------------------------------------------------------------------
# Family-independent projection, damping, checks, and visualization.
# ---------------------------------------------------------------------------
def integrate(config: FamilyConfiguration, values: np.ndarray) -> np.ndarray:
    """Integrate over the configured support along the leading axis."""

    if config.is_discrete:
        return np.sum(values, axis=0)
    return np.trapezoid(values, config.grid, axis=0)


def compute_projection(config: FamilyConfiguration) -> ProjectionResult:
    """Compute shifted-law coefficients analytically, exactly, and by sampling."""

    rng = np.random.default_rng(config.seed)
    samples = config.sample(rng, config.shifted, config.sample_size)

    sample_basis = config.family.basis(
        samples,
        config.n_max,
        config.baseline,
    )
    empirical = sample_basis.mean(axis=0)
    standard_error = sample_basis.std(axis=0, ddof=1) / np.sqrt(samples.size)

    grid_basis = config.family.basis(
        config.grid,
        config.n_max,
        config.baseline,
    )
    shifted_probability = config.family.prob(config.grid, config.shifted)
    quadrature = integrate(
        config,
        shifted_probability[:, None] * grid_basis,
    )
    analytic = config.family.shift_coefficients(
        config.natural_shift,
        config.n_max,
        config.baseline,
    )

    return ProjectionResult(
        samples=samples,
        analytic=analytic,
        quadrature=quadrature,
        empirical=empirical,
        standard_error=standard_error,
    )


def damped_coefficients(coefficients: np.ndarray, tau: float) -> np.ndarray:
    """Apply probability-mode damping ``c_k -> exp(-k*tau) c_k``."""

    degree = np.arange(coefficients.size)
    return np.exp(-degree * tau) * coefficients


def exact_damped_params(config: FamilyConfiguration, tau: float) -> Any:
    """Return the exact family member obtained by damping ``xi``."""

    xi = config.family.shift_coordinate(
        config.natural_shift,
        config.baseline,
    )
    return config.family.from_shift_coordinate(
        np.exp(-tau) * xi,
        config.baseline,
    )


def reconstructed_density(
    config: FamilyConfiguration,
    coefficients: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Reconstruct a damped density from a truncated probability expansion."""

    damped = damped_coefficients(coefficients, tau)
    ratio = config.family.basis_dot(
        config.grid,
        damped,
        config.baseline,
    )
    return config.family.prob(config.grid, config.baseline) * ratio


def compute_linearization(
    config: FamilyConfiguration,
    *,
    chunk_size: int = 20_000,
) -> LinearizationResult:
    """Compute analytic and independently sampled ``Lambda_mnk`` tensors.

    Samples come from the baseline, not the shifted toy law used by
    ``compute_projection``. Triple products are accumulated in chunks to avoid
    allocating a ``sample_size x degree**3`` array.
    """

    rng = np.random.default_rng(config.seed + 10_000)
    samples = config.sample(rng, config.baseline, config.sample_size)
    analytic = config.family.linearization_tensor(
        config.linearization_n_max,
        config.baseline,
    )

    shape = analytic.shape
    total = np.zeros(shape)
    total_square = np.zeros(shape)
    for start in range(0, samples.size, chunk_size):
        chunk = samples[start : start + chunk_size]
        values = config.family.basis(
            chunk,
            config.linearization_n_max,
            config.baseline,
        )
        products = (
            values[:, :, None, None]
            * values[:, None, :, None]
            * values[:, None, None, :]
        )
        total += products.sum(axis=0)
        total_square += np.square(products).sum(axis=0)

    sampled = total / samples.size
    sample_variance = (total_square - samples.size * sampled**2) / (samples.size - 1)
    standard_error = np.sqrt(np.maximum(sample_variance, 0.0) / samples.size)
    return LinearizationResult(
        analytic=analytic,
        sampled=sampled,
        standard_error=standard_error,
    )


def check_projection(
    config: FamilyConfiguration,
    result: ProjectionResult,
) -> None:
    """Raise if analytic, deterministic, or sampled projection checks fail."""

    if not np.allclose(
        result.quadrature,
        result.analytic,
        atol=config.projection_atol,
        rtol=config.projection_atol,
    ):
        raise AssertionError("quadrature coefficients do not match the shift kernel")

    uncertainty = 6.0 * result.standard_error + 2e-3
    if np.any(np.abs(result.empirical - result.analytic) > uncertainty):
        raise AssertionError("empirical coefficients disagree with their MC errors")

    if not np.isclose(result.empirical[0], 1.0):
        raise AssertionError("the constant probability mode must remain normalized")

    for tau in config.taus:
        density = reconstructed_density(config, result.quadrature, tau)
        if not np.isclose(
            integrate(config, density),
            1.0,
            atol=max(config.projection_atol, 2e-10),
        ):
            raise AssertionError(
                f"damped reconstruction is not normalized at tau={tau}"
            )

    comparison_tau = max(config.taus)
    reconstructed = reconstructed_density(
        config,
        result.quadrature,
        comparison_tau,
    )
    exact = config.family.prob(
        config.grid,
        exact_damped_params(config, comparison_tau),
    )
    if np.max(np.abs(reconstructed - exact)) > config.exact_density_atol:
        raise AssertionError("damped coefficients do not reproduce the shifted family")


def check_linearization(result: LinearizationResult) -> None:
    """Check tensor identities and agreement with Monte Carlo uncertainty."""

    analytic = result.analytic
    if not np.allclose(analytic, np.transpose(analytic, (1, 0, 2)), atol=2e-12):
        raise AssertionError("analytic Lambda is not symmetric in m and n")
    if not np.allclose(analytic, np.transpose(analytic, (2, 1, 0)), atol=2e-12):
        raise AssertionError("analytic Lambda is not symmetric in m and k")
    if not np.allclose(analytic[0], np.eye(analytic.shape[1]), atol=2e-12):
        raise AssertionError("Lambda[0, n, k] must equal delta[n, k]")

    uncertainty = 6.0 * result.standard_error + 3e-3
    if np.any(np.abs(result.sampled - analytic) > uncertainty):
        raise AssertionError("sampled Lambda disagrees with its Monte Carlo errors")


def print_linearization_summary(result: LinearizationResult) -> None:
    """Print the largest absolute and standardized Lambda differences."""

    difference = result.sampled - result.analytic
    pulls = np.divide(
        difference,
        result.standard_error,
        out=np.zeros_like(difference),
        where=result.standard_error > 0.0,
    )
    index = np.unravel_index(np.argmax(np.abs(pulls)), pulls.shape)
    index = tuple(int(component) for component in index)
    print(
        "Lambda check:"
        f" max |sample-analytic|={np.max(np.abs(difference)):.4g};"
        f" max |pull|={abs(pulls[index]):.2f} at {index}"
    )


def print_coefficient_table(result: ProjectionResult) -> None:
    """Print the three coefficient determinations side by side."""

    print(" k      analytic     quadrature      empirical        MC error")
    for degree, values in enumerate(
        zip(
            result.analytic,
            result.quadrature,
            result.empirical,
            result.standard_error,
            strict=True,
        )
    ):
        analytic, quadrature, empirical, error = values
        print(
            f"{degree:2d}  {analytic:12.6g}  {quadrature:12.6g}"
            f"  {empirical:12.6g}  {error:12.3g}"
        )


def plot_projection(
    config: FamilyConfiguration,
    result: ProjectionResult,
    output: Path,
) -> None:
    """Plot shifted toys, damped reconstructions, and mode coefficients."""

    figure, (density_axis, coefficient_axis) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.6),
        constrained_layout=True,
    )

    if config.is_discrete:
        sample_min = int(np.min(config.grid))
        sample_max = int(min(np.max(result.samples), config.plot_limits[1]))
        bins = np.arange(sample_min, sample_max + 2) - 0.5
        drawstyle = "steps-mid"
    else:
        bins = 90
        drawstyle = "default"

    density_axis.hist(
        result.samples,
        bins=bins,
        density=True,
        color="0.82",
        edgecolor="none",
        label="toy data",
    )
    density_axis.plot(
        config.grid,
        config.family.prob(config.grid, config.shifted),
        color="black",
        linewidth=2.0,
        drawstyle=drawstyle,
        label="shifted baseline",
    )

    colors = plt.colormaps["viridis"](np.linspace(0.15, 0.85, len(config.taus)))
    for color, tau in zip(colors, config.taus, strict=True):
        empirical_density = reconstructed_density(
            config,
            result.empirical,
            tau,
        )
        exact_density = config.family.prob(
            config.grid,
            exact_damped_params(config, tau),
        )
        density_axis.plot(
            config.grid,
            empirical_density,
            color=color,
            linewidth=1.6,
            drawstyle=drawstyle,
            label=rf"spectral fit, $\tau={tau:g}$",
        )
        if tau > 0:
            density_axis.plot(
                config.grid,
                exact_density,
                color=color,
                linewidth=1.0,
                linestyle=":",
                drawstyle=drawstyle,
            )

    density_axis.axhline(0.0, color="0.5", linewidth=0.7)
    density_axis.set_xlim(*config.plot_limits)
    density_axis.set_xlabel("x")
    density_axis.set_ylabel(
        "probability density" if not config.is_discrete else "probability"
    )
    density_axis.set_title(config.name)
    density_axis.legend(frameon=False, fontsize=8)

    degree = np.arange(config.n_max + 1)
    coefficient_axis.errorbar(
        degree,
        result.empirical,
        yerr=result.standard_error,
        fmt="o",
        color="#2166ac",
        capsize=2.5,
        label="empirical projection",
    )
    coefficient_axis.plot(
        degree,
        result.analytic,
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

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_configuration(
    config: FamilyConfiguration,
    *,
    make_plot: bool,
) -> ProjectionResult:
    """Execute all checks and optional plotting for one family."""

    print(f"\n{config.name}")
    result = compute_projection(config)
    check_projection(config, result)
    print_coefficient_table(result)
    linearization = compute_linearization(config)
    check_linearization(linearization)
    print_linearization_summary(linearization)

    minimum = min(
        reconstructed_density(config, result.empirical, tau).min()
        for tau in config.taus
    )
    print(f"Minimum reconstructed probability: {minimum:.6g}")

    if make_plot:
        output = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / f"{config.slug}_shifted_baseline_probability_modes.png"
        )
        plot_projection(config, result, output)
        print(f"Figure: {output}")
    return result


def _parse_args() -> argparse.Namespace:
    """Parse the family selector and plotting switch."""

    parser = argparse.ArgumentParser(
        description="Project and damp a shifted NEF-QVF probability ratio."
    )
    parser.add_argument(
        "--family",
        choices=(*CONFIGURATION_FACTORIES, "all"),
        default="normal",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="run numerical checks without writing figures",
    )
    return parser.parse_args()


def main() -> None:
    """Run one configured family or the complete six-family demonstration."""

    args = _parse_args()
    names = tuple(CONFIGURATION_FACTORIES) if args.family == "all" else (args.family,)
    for name in names:
        run_configuration(
            CONFIGURATION_FACTORIES[name](),
            make_plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
