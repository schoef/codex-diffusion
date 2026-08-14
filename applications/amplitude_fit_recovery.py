"""Test 1: can a one-channel amplitude be recovered from a sample?

This is the gate for everything downstream. If the amplitude of a law whose
answer is known analytically cannot be recovered at the statistical rate, no
multichannel construction built on the same primitive is trustworthy.

The target is a naturally shifted member of the baseline family. That choice is
what makes the truth known: for an exponential family the amplitude of
``P_{eta+j}`` relative to ``P_eta`` is the *half*-shift ratio times the
Hellinger affinity,

    h = B_eta(j) * sum_n gamma_n(eta) z_eta(j/2)^n phi_n,

because sqrt of ``exp(j T(x) - dA)`` is ``exp((j/2) T(x) - dA/2)``, which is the
ratio at shift ``j/2`` up to a constant. The constant is fixed by normalisation
and equals the affinity. So the exact coefficient vector is available in closed
form, and ``||c|| = 1`` is a free check on it.

Three error sources are separated rather than mixed, since only the first should
follow ``N^{-1/2}``:

* statistical: the fit from a sample of size ``N`` against the fit from the exact
  coefficients at the same truncation. This is the rate being tested.
* truncation: the exact amplitude is an infinite series, so a degree-``K``
  vector cannot represent it. Reported as the norm defect of the truncation.
* optimiser: the population fit against the truncated exact vector.

Reporting them together is how a plateau gets misread as a failed rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space

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

FAMILY_NAMES = ("normal", "poisson", "gamma", "binomial", "negative-binomial", "ghs")

# baseline, and a natural shift small enough that a modest degree suffices
TARGETS: dict[str, tuple[Any, Any, float]] = {
    "normal": (Normal, NormalParams(mean=0.0, sigma=1.0), 0.5),
    "poisson": (Poisson, PoissonParams(mean=6.0), 0.4),
    "gamma": (Gamma, GammaParams(mean=3.0, r=2.5), 0.2),
    "binomial": (Binomial, BinomialParams(mean=3.6, N=12), 0.4),
    "negative-binomial": (
        NegativeBinomial,
        NegativeBinomialParams(mean=4.0, r=3.0),
        0.1,
    ),
    "ghs": (GHS, GHSParams(mean=0.0, r=1.5), 0.4),
}

DEFAULT_DEGREE = 6
DEFAULT_SAMPLE_SIZES = (10**3, 10**4, 10**5, 10**6)
DEFAULT_SEED = 5
DEFAULT_REPLICATES = 48
DEFAULT_PLOT_SAMPLE_SIZE = 10**4
FIGURE_SUBDIRECTORY = "amplitude_fit"


# --------------------------------------------------------------------- truth --
def exact_amplitude(
    family: Any, baseline: Any, shift: float, degree: int
) -> np.ndarray:
    """Return the exact amplitude coefficients, truncated at ``degree``."""

    shifted = family.shifted_params(baseline, shift)
    affinity = float(family.affinity(baseline, shifted))
    half = np.asarray(
        family.shift_coefficients(0.5 * shift, degree, baseline), dtype=float
    )
    return affinity * half


def exact_ratio_coefficients(
    family: Any, baseline: Any, shift: float, k_max: int
) -> np.ndarray:
    """Return the exact density-ratio coefficients ``gamma_k z(j)^k``."""

    return np.asarray(family.shift_coefficients(shift, k_max, baseline), dtype=float)


def product_matrices(family: Any, baseline: Any, degree: int) -> np.ndarray:
    """Return ``Phi[k] = Lambda[:K+1, :K+1, k]`` for ``k = 0 ... 2K``."""

    tensor = family.linearization_tensor(2 * degree, baseline)
    return np.ascontiguousarray(
        np.transpose(tensor[: degree + 1, : degree + 1, :], (2, 0, 1))
    )


def ratio_coefficients(coefficients: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Return ``R_k(c) = c^T Phi_k c``."""

    return np.einsum("n,knm,m->k", coefficients, phi, coefficients)


# ---------------------------------------------------------------- estimation --
def empirical_coefficients(
    family: Any, baseline: Any, observations: np.ndarray, k_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical OPS coefficients and their covariance estimate."""

    features = np.asarray(family.basis(observations, k_max, baseline), dtype=float)
    mean = features.mean(axis=0)
    centred = features - mean
    covariance = centred.T @ centred / (len(observations) * (len(observations) - 1))
    return mean, covariance


# ------------------------------------------------------------------- fitting --
def fit_amplitude(
    phi: np.ndarray,
    target: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    tau: float = 0.0,
    penalty: np.ndarray | None = None,
    k_max: int | None = None,
    max_iterations: int = 200,
    tolerance: float = 1e-13,
) -> dict[str, Any]:
    """Minimise the coefficient objective on the unit sphere by Riemannian LM.

    Degree zero is excluded from the residual: ``R_0(c) = ||c||^2`` equals one
    identically on the sphere, so it carries no information and would only add a
    null row to the Jacobian.
    """
    degree = phi.shape[1] - 1
    penalty = (
        np.diag(np.arange(degree + 1, dtype=float))
        if penalty is None
        else np.diag(np.asarray(penalty, dtype=float))
    )
    highest = phi.shape[0] - 1 if k_max is None else int(k_max)
    active = np.arange(1, highest + 1)
    phi_active = phi[active]
    target_active = np.asarray(target, dtype=float)[active]
    weight_matrix = np.eye(active.size) if weight is None else np.asarray(weight)

    c = np.zeros(degree + 1) if initial is None else np.array(initial, dtype=float)
    if initial is None:
        c[0] = 1.0
    c = c / np.linalg.norm(c)

    def residual(vec: np.ndarray) -> np.ndarray:
        return np.einsum("n,knm,m->k", vec, phi_active, vec) - target_active

    def objective(vec: np.ndarray) -> float:
        r = residual(vec)
        return float(0.5 * r @ weight_matrix @ r + 0.5 * tau * vec @ penalty @ vec)

    mu = 1e-3
    value = objective(c)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        r = residual(c)
        jacobian = 2.0 * np.einsum("knm,m->kn", phi_active, c)
        tangent = null_space(c[None, :])

        jb = jacobian @ tangent
        hessian = jb.T @ weight_matrix @ jb + tau * tangent.T @ penalty @ tangent
        gradient = jb.T @ weight_matrix @ r + tau * tangent.T @ penalty @ c
        if np.linalg.norm(gradient) < tolerance:
            break

        accepted = False
        for _ in range(40):
            step = np.linalg.solve(hessian + mu * np.eye(hessian.shape[0]), -gradient)
            trial = c + tangent @ step
            trial = trial / np.linalg.norm(trial)
            if trial[0] < 0.0:
                trial = -trial
            trial_value = objective(trial)
            if trial_value < value:
                c, value, accepted = trial, trial_value, True
                mu = max(mu * 0.3, 1e-14)
                break
            mu *= 3.0
        if not accepted:
            break

    # the curvature Gauss-Newton discards, relative to the part it keeps
    r = residual(c)
    jacobian = 2.0 * np.einsum("knm,m->kn", phi_active, c)
    kept = jacobian.T @ weight_matrix @ jacobian
    discarded = 2.0 * np.einsum("k,knm->nm", weight_matrix @ r, phi_active)
    return {
        "coefficients": c,
        "objective": value,
        "iterations": iterations,
        "residual_norm": float(np.linalg.norm(r)),
        "curvature_ratio": float(
            np.linalg.norm(discarded, 2) / max(np.linalg.norm(kept, 2), 1e-300)
        ),
    }


# --------------------------------------------------------------------- driver --
def run_recovery(
    name: str,
    *,
    degree: int = DEFAULT_DEGREE,
    sample_sizes: tuple[int, ...] = DEFAULT_SAMPLE_SIZES,
    replicates: int = DEFAULT_REPLICATES,
    k_max: int | None = None,
    seed: int = DEFAULT_SEED,
) -> None:
    """Fit the shifted-member amplitude at several sample sizes."""

    if name not in TARGETS:
        raise ValueError(
            f"unknown family {name!r}; choose one of: {', '.join(FAMILY_NAMES)}"
        )

    family, baseline, shift = TARGETS[name]
    shifted = family.shifted_params(baseline, shift)
    phi = product_matrices(family, baseline, degree)

    exact = exact_amplitude(family, baseline, shift, degree)
    defect = abs(float(np.linalg.norm(exact)) - 1.0)
    exact_unit = exact / np.linalg.norm(exact)
    exact_ratio = exact_ratio_coefficients(family, baseline, shift, 2 * degree)

    print()
    print(f"{name}: baseline {baseline}, natural shift {shift:g}, degree K = {degree}")
    print(
        f"exact amplitude ||c|| - 1 = {defect:.2e}  (truncation defect)   "
        f"affinity B = {float(family.affinity(baseline, shifted)):.6f}"
    )

    # the shift machinery and the linearisation tensor must agree
    consistency = np.max(np.abs(ratio_coefficients(exact_unit, phi) - exact_ratio))
    beyond = exact_amplitude(family, baseline, shift, degree + 1)[-1]
    print(
        f"cross-check  max |R(c_exact) - gamma_k z^k| = {consistency:.2e}"
        f"   (truncation limited; 2 c_0 c_{{K+1}} = {2 * exact_unit[0] * beyond:.2e})"
    )

    # population fit: the same problem with exact coefficients in place of the
    # sample, which isolates optimiser and truncation error from statistics
    population = fit_amplitude(phi, exact_ratio, k_max=k_max)
    c_pop = population["coefficients"]
    print(
        f"population fit: ||c_pop - c_exact|| = "
        f"{np.linalg.norm(c_pop - exact_unit):.2e}   "
        f"objective = {population['objective']:.3e}   "
        f"iterations = {population['iterations']}"
    )

    # A rate cannot be read off single draws: the error at one sample size is
    # itself random, so the scan is replicated and the root mean square taken.
    print()
    print(f"{replicates} replicates per sample size")
    print(
        f"  {'N':>9} {'rms|c-c_pop|':>13} {'x sqrt(N)':>10} {'rms law err':>12} "
        f"{'iters':>6} {'curv ratio':>11}"
    )
    rng = np.random.default_rng(seed)
    sizes, errors = [], []
    for size in sample_sizes:
        amplitude, law, iterations, curvature = [], [], [], []
        for _ in range(replicates):
            observations = family.sample(shifted, size, rng=rng)
            empirical, _ = empirical_coefficients(
                family, baseline, observations, 2 * degree
            )
            fit = fit_amplitude(phi, empirical, k_max=k_max)
            c = fit["coefficients"]
            amplitude.append(np.linalg.norm(c - c_pop))
            law.append(
                np.linalg.norm(
                    ratio_coefficients(c, phi) - ratio_coefficients(c_pop, phi)
                )
            )
            iterations.append(fit["iterations"])
            curvature.append(fit["curvature_ratio"])
        rms = float(np.sqrt(np.mean(np.square(amplitude))))
        sizes.append(size)
        errors.append(rms)
        print(
            f"  {size:9d} {rms:13.3e} {rms * np.sqrt(size):10.3f} "
            f"{float(np.sqrt(np.mean(np.square(law)))):12.3e} "
            f"{float(np.median(iterations)):6.0f} {float(np.median(curvature)):11.3e}"
        )

    slope, _ = np.polyfit(np.log(sizes), np.log(errors), 1)
    print()
    print(f"fitted rate: rms error ~ N^({slope:+.3f})      expected N^(-0.500)")
    print(
        "verdict: "
        + ("PASS" if abs(slope + 0.5) < 0.08 else "does not match the expected rate")
    )


# -------------------------------------------------------------------- figures --
def plot_recovery(
    name: str,
    *,
    degree: int = DEFAULT_DEGREE,
    sample_size: int = DEFAULT_PLOT_SAMPLE_SIZE,
    shift: float | None = None,
    k_max: int | None = None,
    log_y: bool = False,
    seed: int = DEFAULT_SEED,
    output_dir: Any = None,
) -> None:
    """Draw one fit: the three laws, the amplitude, and both coefficient sets.

    The figure is meant for intuition rather than for the rate test, so it shows
    a single sample. Four panels, because the failure modes live in different
    ones: a bad law is visible in the first, a spurious node only in the second,
    and a noise-dominated coefficient only in the third.
    """
    if name not in TARGETS:
        raise ValueError(
            f"unknown family {name!r}; choose one of: {', '.join(FAMILY_NAMES)}"
        )

    family, baseline, default_shift = TARGETS[name]
    shift = default_shift if shift is None else float(shift)
    shifted = family.shifted_params(baseline, shift)
    lattice = family.is_lattice(baseline)

    phi = product_matrices(family, baseline, degree)
    exact = exact_amplitude(family, baseline, shift, degree)
    exact_unit = exact / np.linalg.norm(exact)
    exact_ratio = exact_ratio_coefficients(family, baseline, shift, 2 * degree)

    rng = np.random.default_rng(seed)
    observations = family.sample(shifted, sample_size, rng=rng)
    empirical, covariance = empirical_coefficients(
        family, baseline, observations, 2 * degree
    )
    fit = fit_amplitude(phi, empirical, k_max=k_max)
    fitted = fit["coefficients"]

    # a coordinate range covering both laws and the sample
    spread = max(
        float(np.sqrt(family.variance(member))) for member in (baseline, shifted)
    )
    centre = 0.5 * (float(family.mean(baseline)) + float(family.mean(shifted)))
    low = min(centre - 5.0 * spread, float(np.min(observations)))
    high = max(centre + 5.0 * spread, float(np.max(observations)))
    if lattice:
        low = max(low, 0.0)
        grid = np.arange(int(np.floor(low)), int(np.ceil(high)) + 1)
    else:
        grid = np.linspace(low, high, 600)

    reference_density = np.asarray(family.prob(grid, baseline), dtype=float)
    target_density = np.asarray(family.prob(grid, shifted), dtype=float)
    amplitude_exact = np.asarray(
        family.basis_dot(grid, exact_unit, baseline), dtype=float
    )
    amplitude_fitted = np.asarray(family.basis_dot(grid, fitted, baseline), dtype=float)
    fitted_density = reference_density * amplitude_fitted**2

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))

    # (a) the three laws and the sample
    axis = axes[0, 0]
    if lattice:
        counts = np.array(
            [np.count_nonzero(observations == value) for value in grid], dtype=float
        )
        axis.bar(
            grid,
            counts / sample_size,
            width=0.9,
            color="0.85",
            label=f"sample, N = {sample_size}",
        )
    else:
        axis.hist(
            observations,
            bins=60,
            range=(low, high),
            density=True,
            color="0.85",
            label=f"sample, N = {sample_size}",
        )
    axis.plot(
        grid,
        reference_density,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"baseline $p_{\rm ref}$",
    )
    axis.plot(grid, target_density, color="crimson", linewidth=1.8, label="target")
    axis.plot(
        grid,
        fitted_density,
        color="tab:blue",
        linewidth=1.5,
        linestyle=":",
        label=r"fitted $p_{\rm ref}h^2$",
    )
    axis.set_xlabel("$m$" if lattice else "$x$")
    axis.set_ylabel("density")
    if log_y:
        axis.set_yscale("log")
        positive = target_density[target_density > 0.0]
        axis.set_ylim(bottom=max(float(positive.min()) * 0.5, 1e-12))
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("laws and sample", fontsize=10)

    # (b) the amplitude, where a spurious node would show
    axis = axes[0, 1]
    axis.axhline(0.0, color="0.6", linewidth=0.8)
    axis.plot(grid, amplitude_exact, color="crimson", linewidth=1.8, label="exact $h$")
    axis.plot(
        grid,
        amplitude_fitted,
        color="tab:blue",
        linewidth=1.5,
        linestyle=":",
        label=r"fitted $h$",
    )
    axis.set_xlabel("$m$" if lattice else "$x$")
    axis.set_ylabel("$h$")
    if log_y:
        scale = max(float(np.max(np.abs(amplitude_exact))), 1e-30)
        axis.set_yscale("symlog", linthresh=1e-4 * scale)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("amplitude (a zero here is a node of the fitted law)", fontsize=10)

    # (c) the ratio coefficients that the objective actually matches
    axis = axes[1, 0]
    degrees = np.arange(2 * degree + 1)
    axis.errorbar(
        degrees,
        empirical,
        yerr=np.sqrt(np.diag(covariance)),
        fmt="o",
        markersize=3.5,
        color="0.35",
        capsize=2,
        label=r"empirical $\widehat R_k$",
    )
    axis.plot(
        degrees,
        exact_ratio,
        "s",
        markersize=4.5,
        markerfacecolor="none",
        color="crimson",
        label="exact $R_k$",
    )
    axis.plot(
        degrees,
        ratio_coefficients(fitted, phi),
        "x",
        markersize=5,
        color="tab:blue",
        label=r"fitted $R_k(\widehat c)$",
    )
    axis.axhline(0.0, color="0.6", linewidth=0.8)
    axis.set_xlabel("degree $k$")
    axis.set_ylabel("$R_k$")
    if log_y:
        axis.set_yscale(
            "symlog", linthresh=1e-5 * max(float(np.max(np.abs(exact_ratio))), 1e-30)
        )
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("ratio coefficients", fontsize=10)

    # (d) the amplitude coefficients, which are what is being recovered
    axis = axes[1, 1]
    orders = np.arange(degree + 1)
    axis.plot(
        orders,
        exact_unit,
        "s-",
        markersize=4.5,
        markerfacecolor="none",
        color="crimson",
        linewidth=1.0,
        label="exact $c_n$",
    )
    axis.plot(
        orders,
        fitted,
        "x:",
        markersize=5,
        color="tab:blue",
        linewidth=1.0,
        label=r"fitted $\widehat c_n$",
    )
    axis.axhline(0.0, color="0.6", linewidth=0.8)
    axis.set_xlabel("degree $n$")
    axis.set_ylabel("$c_n$")
    if log_y:
        axis.set_yscale("symlog", linthresh=1e-5)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("amplitude coefficients", fontsize=10)

    chi_squared = float(np.sum(exact_ratio[1:] ** 2))
    figure.suptitle(
        f"{name}: shift {shift:g}, $K$ = {degree}, "
        rf"$\chi^2 = {chi_squared:.3f}$,   "
        rf"$\|\widehat c - c\| = {np.linalg.norm(fitted - exact_unit):.2e}$",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "artifacts" / FIGURE_SUBDIRECTORY
    )
    output.mkdir(parents=True, exist_ok=True)
    suffix = "_log" if log_y else ""
    figure_path = output / f"{name}_amplitude_fit_shift{shift:g}_K{degree}{suffix}.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    print(f"Figure: {figure_path}")


def _parse_args() -> argparse.Namespace:
    """Parse the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="normal", choices=(*FAMILY_NAMES, "all"))
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument(
        "--k-max",
        type=int,
        default=None,
        help="highest ratio coefficient matched; default 2K",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--shift", type=float, default=None, help="override the natural shift"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="also draw the laws, amplitude and coefficients of a single fit",
    )
    parser.add_argument("--plot-samples", type=int, default=DEFAULT_PLOT_SAMPLE_SIZE)
    parser.add_argument("--no-scan", action="store_true", help="skip the rate scan")
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="logarithmic vertical scales, which is where a failed fit shows",
    )
    return parser.parse_args()


def main() -> None:
    """Run the recovery test from the command line."""

    args = _parse_args()
    names = FAMILY_NAMES if args.family == "all" else (args.family,)
    for name in names:
        if args.shift is not None:
            family, baseline, _ = TARGETS[name]
            TARGETS[name] = (family, baseline, args.shift)
        if args.plot:
            plot_recovery(
                name,
                degree=args.degree,
                sample_size=args.plot_samples,
                shift=args.shift,
                k_max=args.k_max,
                log_y=args.log_y,
                seed=args.seed,
            )
        if args.no_scan:
            continue
        run_recovery(
            name,
            degree=args.degree,
            replicates=args.replicates,
            k_max=args.k_max,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
