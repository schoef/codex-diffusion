"""How far the Born parametrisation goes, and where it stops.

Two questions are answered here, and they are the ones that decide whether the
one-channel fit is worth carrying into the tensor constructions.

**How fast does the bias floor fall for a target with an edge?** A degree-``K``
amplitude squared is a polynomial, hence smooth, while a truncated density has a
jump. The approximation error can then only fall algebraically in ``K``, and the
exponent decides everything: if it implies ``K`` of order fifty for a per-mille
fit, edges are a truncation matter; if it implies ``K`` of order a million, the
parametrisation cannot represent edges and something else is needed.

The scan is run on exact coefficients, so it isolates approximation error from
sampling. It also warm-starts each degree from the previous one, padded with a
zero. That is the continuation in ``K`` the notes describe, and it matters: a
cold start fails above ``K`` of about thirty on the lattice families, landing on
a much worse point and making the floor look non-monotone when it is not.

**What does the best achievable fit actually look like?** For each target the
degree is chosen by held-out log-likelihood, the fit is made on a large sample,
and the result is drawn against the data -- the laws, the amplitude, and the
coefficients -- so that the numbers above can be read as pictures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.targets import (
    FAMILY_MAX_DEGREE,
    FAMILY_NAMES,
    TARGETS,
    build_targets,
    reference_coefficients,
    target_grid,
    total_variation,
)
from nefqvf.fitting import fit_amplitude, product_matrices

EDGE_DEGREES = (2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44)
FIT_DEGREES = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
DEFAULT_SAMPLE_SIZE = 10**5
DEFAULT_EPSILON = 1e-2
DEFAULT_SEED = 7
FOLDS = 5
FIGURE_SUBDIRECTORY = "limits"
TARGET_TV = 1e-3


def continued_population_fit(
    target: Any, name: str, degrees: tuple[int, ...]
) -> dict[int, dict[str, Any]]:
    """Fit the exact coefficients at each degree, warm-starting from the last.

    The previous solution padded with a zero is an exact point of the larger
    model, and a good one, so it is a far better starting point than the
    baseline vector once the degree is large.
    """
    family, baseline = target.family, target.baseline
    cap = FAMILY_MAX_DEGREE.get(name)
    degrees = degrees if cap is None else tuple(d for d in degrees if d <= cap)
    grid = target_grid(target)

    results: dict[int, dict[str, Any]] = {}
    previous: np.ndarray | None = None
    for degree in degrees:
        phi = product_matrices(family, baseline, degree)
        truth = reference_coefficients(target, 2 * degree, grid)

        candidates = [None]
        if previous is not None:
            warm = np.zeros(degree + 1)
            warm[: len(previous)] = previous
            candidates.append(warm / np.linalg.norm(warm))

        best = None
        for start in candidates:
            fit = fit_amplitude(phi, truth, initial=start)
            if best is None or fit["objective"] < best["objective"]:
                best = fit
        coefficients = best["coefficients"]
        previous = coefficients
        results[degree] = {
            "coefficients": coefficients,
            "tv": total_variation(target, coefficients, grid),
            "objective": best["objective"],
        }
    return results


def edge_exponent(name: str, degrees: tuple[int, ...] = EDGE_DEGREES) -> dict[str, Any]:
    """Measure how the bias floor of a truncated target falls with the degree."""

    target = build_targets(name, TARGETS[name][2], (), 0.8)[-1]
    fits = continued_population_fit(target, name, degrees)
    used = sorted(fits)
    values = np.array([fits[d]["tv"] for d in used])

    # the exponent is read from the upper half, where the asymptotic rate has set in
    half = len(used) // 2
    slope, intercept = np.polyfit(
        np.log(np.asarray(used[half:], float)), np.log(values[half:]), 1
    )
    needed = (
        float(np.exp((np.log(TARGET_TV) - intercept) / slope)) if slope < 0 else np.inf
    )
    return {
        "family": name,
        "label": target.label,
        "degrees": used,
        "tv": values.tolist(),
        "exponent": float(slope),
        "degree_for_target": needed,
    }


def select_degree(
    target: Any,
    name: str,
    sample: np.ndarray,
    degrees: tuple[int, ...],
    epsilon: float,
) -> tuple[int, dict[int, np.ndarray]]:
    """Choose the degree by held-out log-likelihood and return the fits."""

    family, baseline = target.family, target.baseline
    cap = FAMILY_MAX_DEGREE.get(name)
    degrees = degrees if cap is None else tuple(d for d in degrees if d <= cap)
    band = 2 * max(degrees)
    size = len(sample)

    features = np.asarray(family.basis(sample, band, baseline), dtype=float)
    folds = np.array_split(np.arange(size), FOLDS)
    sums = features.sum(axis=0)

    scores = {degree: 0.0 for degree in degrees}
    for fold in folds:
        held = features[fold]
        inside = (sums - held.sum(axis=0)) / (size - len(fold))
        for degree in degrees:
            phi = product_matrices(family, baseline, degree)
            trial = fit_amplitude(phi, inside[: 2 * degree + 1])["coefficients"]
            amplitude = held[:, : degree + 1] @ trial
            scores[degree] += -float(
                np.mean(np.log((1.0 - epsilon) * amplitude**2 + epsilon))
            )

    whole = sums / size
    fits = {}
    for degree in degrees:
        phi = product_matrices(family, baseline, degree)
        fits[degree] = fit_amplitude(phi, whole[: 2 * degree + 1])["coefficients"]
    return min(scores, key=scores.get), fits


def plot_best_fits(
    name: str,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    epsilon: float = DEFAULT_EPSILON,
    seed: int = DEFAULT_SEED,
    output_dir: Any = None,
) -> list[dict[str, Any]]:
    """Draw the best achievable fit for each target, against the data."""

    targets = build_targets(name, TARGETS[name][2], (0.4, 0.8), 0.8)
    family, baseline = targets[0].family, targets[0].baseline
    lattice = family.is_lattice(baseline)
    rng = np.random.default_rng(seed)

    figure, axes = plt.subplots(
        len(targets), 3, figsize=(13.5, 3.2 * len(targets)), squeeze=False
    )
    summary = []
    for row, target in enumerate(targets):
        grid = target_grid(target, width=10.0)
        sample = target.sample(sample_size, rng)
        degree, fits = select_degree(target, name, sample, FIT_DEGREES, epsilon)
        coefficients = fits[degree]

        reference = np.asarray(family.prob(grid, baseline), dtype=float)
        truth = target.density(grid)
        amplitude = np.asarray(
            family.basis_dot(grid, coefficients, baseline), dtype=float
        )
        fitted = reference * amplitude**2
        distance = total_variation(target, coefficients, target_grid(target))
        summary.append({"target": target.label, "degree": degree, "tv": distance})

        axis = axes[row][0]
        if lattice:
            counts = (
                np.array([np.count_nonzero(sample == v) for v in grid], dtype=float)
                / sample_size
            )
            axis.bar(grid, counts, width=0.9, color="0.85", label="sample")
        else:
            axis.hist(
                sample,
                bins=80,
                range=(grid[0], grid[-1]),
                density=True,
                color="0.85",
                label="sample",
            )
        axis.plot(
            grid, reference, "--", color="black", linewidth=1.0, label=r"$p_{\rm ref}$"
        )
        axis.plot(grid, truth, color="crimson", linewidth=1.8, label="target")
        axis.plot(grid, fitted, ":", color="tab:blue", linewidth=1.6, label="fitted")
        axis.set_yscale("log")
        positive = truth[truth > 0]
        axis.set_ylim(bottom=max(float(positive.min()) * 0.5, 1e-8))
        axis.set_ylabel("density")
        axis.set_title(f"{target.label}:  $K$={degree},  TV={distance:.2e}", fontsize=9)
        axis.legend(frameon=False, fontsize=7)

        axis = axes[row][1]
        axis.plot(grid, amplitude, color="tab:blue", linewidth=1.4)
        axis.axhline(0.0, color="0.6", linewidth=0.8)
        axis.set_ylabel("$h$")
        axis.set_title("fitted amplitude", fontsize=9)

        axis = axes[row][2]
        orders = np.arange(len(coefficients))
        axis.plot(
            orders,
            np.abs(coefficients),
            "o-",
            color="tab:blue",
            markersize=4,
            label=f"fitted, K={degree}",
        )
        finest = max(k for k in fits)
        axis.plot(
            np.arange(finest + 1),
            np.abs(fits[finest]),
            "x:",
            color="0.6",
            markersize=4,
            label=f"K={finest}",
        )
        axis.set_yscale("log")
        axis.set_xlabel("degree $n$")
        axis.set_ylabel("$|c_n|$")
        axis.set_title("coefficients", fontsize=9)
        axis.legend(frameon=False, fontsize=7)

    for axis in axes[-1]:
        axis.set_xlabel("$m$" if lattice else "$x$")
    figure.suptitle(
        f"{name}: best achievable fit, $N$ = {sample_size}, "
        rf"degree chosen by held-out likelihood ($\epsilon$ = {epsilon:g})",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "artifacts" / FIGURE_SUBDIRECTORY
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}_best_fit.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    print(f"Figure: {path}")
    return summary


def plot_edge_exponents(results: list[dict[str, Any]], output_dir: Any = None) -> None:
    """Draw the bias floor against the degree for every family, with its fit."""

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(results)))

    for colour, result in zip(colours, results):
        degrees = np.asarray(result["degrees"], dtype=float)
        values = np.asarray(result["tv"])
        axes[0].plot(
            degrees,
            values,
            "o-",
            color=colour,
            markersize=4,
            label=f"{result['family']} ($K^{{{result['exponent']:.2f}}}$)",
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("degree $K$")
    axes[0].set_ylabel("TV of the population fit")
    axes[0].set_title("bias floor for a truncated target", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    names = [r["family"] for r in results]
    needed = [r["degree_for_target"] for r in results]
    axes[1].barh(names, needed, color=colours)
    axes[1].set_xscale("log")
    axes[1].axvline(
        44,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        label="largest degree tested",
    )
    axes[1].set_xlabel(f"degree needed for TV = {TARGET_TV:g}  (extrapolated)")
    axes[1].set_title("what the exponent implies", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)

    figure.tight_layout()
    output = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "artifacts" / FIGURE_SUBDIRECTORY
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "edge_exponent.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    print(f"Figure: {path}")


def _parse_args() -> argparse.Namespace:
    """Parse the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="all")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-edge", action="store_true")
    parser.add_argument("--no-fits", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the edge-exponent scan and draw the best achievable fits."""

    args = _parse_args()
    names = FAMILY_NAMES if args.family == "all" else (args.family,)

    if not args.no_edge:
        print("Edge exponent: bias floor of a truncated target, exact coefficients")
        print(
            f"  {'family':18s} {'K range':>12} {'TV at max K':>12} "
            f"{'exponent':>9} {'K for 1e-3':>12}"
        )
        results = []
        for name in names:
            result = edge_exponent(name)
            results.append(result)
            print(
                f"  {name:18s} {min(result['degrees']):4d}-{max(result['degrees']):<7d} "
                f"{result['tv'][-1]:12.3e} {result['exponent']:9.2f} "
                f"{result['degree_for_target']:12.3g}"
            )
        plot_edge_exponents(results)
        output = Path(__file__).resolve().parents[1] / "artifacts" / FIGURE_SUBDIRECTORY
        (output / "edge_exponent.json").write_text(json.dumps(results, indent=1))

    if not args.no_fits:
        print()
        print("Best achievable fits")
        for name in names:
            summary = plot_best_fits(
                name, sample_size=args.samples, epsilon=args.epsilon, seed=args.seed
            )
            for row in summary:
                print(
                    f"  {name:18s} {row['target']:20s} K={row['degree']:3d} "
                    f"TV={row['tv']:.3e}"
                )


if __name__ == "__main__":
    main()
