"""Test 2: how large a truncation does a given sample size support?

The error of a degree-``K`` amplitude fit splits into a bias that falls with
``K`` and a variance that grows with it. The variance is generic, of order
``K/N``. The bias is not: it is fixed by how fast the *amplitude* coefficients
``c_n`` of the particular target decay, and that rate varies enormously between
targets. So the optimal degree ``k*(N)`` is a property of the fitting problem,
and the useful output of this test is a rule that reads the sample rather than a
single number.

Three families of target are provided, chosen to span the decay regimes:

* ``shifted`` -- a shifted member of the baseline family. The amplitude is the
  half-shift amplitude, ``c_n = B gamma_n z(j/2)^n``, which decays factorially.
  This is the easiest case that exists and every earlier test used it.
* ``mixture`` -- an equal mixture of two oppositely shifted members. The density
  ratio is a sum of two exponentials and stays positive, so the amplitude is
  analytic; but its complex singularities sit at a distance set by the
  separation, so the decay is geometric at a rate that degrades as the
  separation grows.
* ``truncated`` -- the baseline conditioned on a hard upper limit. The density
  has a jump, so the amplitude coefficients decay algebraically. This is the
  regime a kinematic edge produces, and the one where ``k*`` should grow like a
  power of ``N`` rather than like a logarithm.

Reference coefficients are computed numerically for every target by integrating
``phi_k`` against the true density, so all three are handled identically and no
per-target analysis is needed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.amplitude_fit_recovery import (
    FAMILY_NAMES,
    TARGETS,
    fit_amplitude,
    product_matrices,
    ratio_coefficients,
)

DEFAULT_SAMPLE_SIZES = (10**3, 10**4, 10**5)
DEFAULT_DEGREES = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
DEFAULT_REPLICATES = 24
DEFAULT_SEED = 23
QUADRATURE_POINTS = 20001
FOLDS = 5
DEFAULT_EPSILONS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1)
# The Krawtchouk basis terminates at the binomial index, and the fit needs the
# product tensor out to 2K, so the binomial baseline with N = 12 caps at K = 6.
FAMILY_MAX_DEGREE = {"binomial": 6}
FIGURE_SUBDIRECTORY = "degree_selection"


@dataclass
class Target:
    """A law to be fitted, together with a sampler and its true density."""

    label: str
    family: Any
    baseline: Any
    sample: Callable[[int, Any], np.ndarray]
    density: Callable[[np.ndarray], np.ndarray]
    members: tuple[Any, ...] = ()


def support_grid(
    family: Any, baseline: Any, members: tuple[Any, ...] = (), width: float = 40.0
):
    """Return a grid wide enough for the integrals, not merely for the mass.

    The integrand is ``p phi_k``, and ``phi_k`` is a degree-``k`` polynomial that
    grows rapidly away from the centre. A grid that captures all but ``1e-9`` of
    the mass can therefore still miss a percent of ``R_k`` for the larger ``k``,
    which is why the width here is generous and why every component of a mixture
    is taken into account rather than the baseline alone.
    """
    parts = (baseline, *members)
    low = min(
        float(family.mean(p)) - width * float(np.sqrt(family.variance(p)))
        for p in parts
    )
    high = max(
        float(family.mean(p)) + width * float(np.sqrt(family.variance(p)))
        for p in parts
    )
    if family.is_lattice(baseline):
        return np.arange(max(int(np.floor(low)), 0), int(np.ceil(high)) + 1)
    return np.linspace(low, high, QUADRATURE_POINTS)


def target_grid(target: Target, width: float = 40.0) -> np.ndarray:
    """Return the integration grid for a target."""

    return support_grid(target.family, target.baseline, target.members, width)


def integrate(
    family: Any, baseline: Any, grid: np.ndarray, values: np.ndarray
) -> float:
    """Integrate over the grid, summing on a lattice and by trapezoid otherwise."""

    if family.is_lattice(baseline):
        return float(np.sum(values))
    return float(np.trapezoid(values, grid))


def reference_coefficients(target: Target, k_max: int, grid: np.ndarray) -> np.ndarray:
    """Return the exact ``R_k = E_target[phi_k]`` by numerical integration."""

    density = target.density(grid)
    basis = np.asarray(target.family.basis(grid, k_max, target.baseline), dtype=float)
    return np.array(
        [
            integrate(target.family, target.baseline, grid, density * basis[:, k])
            for k in range(k_max + 1)
        ]
    )


def total_variation(
    target: Target, coefficients: np.ndarray, grid: np.ndarray
) -> float:
    """Return the total-variation distance between the fitted and true laws."""

    reference = np.asarray(target.family.prob(grid, target.baseline), dtype=float)
    amplitude = np.asarray(
        target.family.basis_dot(grid, coefficients, target.baseline), dtype=float
    )
    difference = np.abs(reference * amplitude**2 - target.density(grid))
    return 0.5 * integrate(target.family, target.baseline, grid, difference)


# --------------------------------------------------------------------- targets --
def shifted_target(name: str, shift: float) -> Target:
    """Return a shifted member of the baseline family."""

    family, baseline, _ = TARGETS[name]
    member = family.shifted_params(baseline, shift)
    return Target(
        label=f"shifted j={shift:g}",
        family=family,
        baseline=baseline,
        sample=lambda size, rng: np.asarray(family.sample(member, size, rng=rng)),
        density=lambda x: np.asarray(family.prob(x, member), dtype=float),
        members=(member,),
    )


def mixture_target(name: str, separation: float) -> Target:
    """Return the equal mixture of two oppositely shifted members."""

    family, baseline, _ = TARGETS[name]
    plus = family.shifted_params(baseline, separation)
    minus = family.shifted_params(baseline, -separation)

    def sample(size: int, rng: Any) -> np.ndarray:
        picks = rng.random(size) < 0.5
        draws = np.empty(size)
        count = int(np.count_nonzero(picks))
        draws[picks] = np.asarray(family.sample(plus, count, rng=rng))
        draws[~picks] = np.asarray(family.sample(minus, size - count, rng=rng))
        return draws

    return Target(
        label=f"mixture d={separation:g}",
        family=family,
        baseline=baseline,
        sample=sample,
        density=lambda x: (
            0.5
            * (
                np.asarray(family.prob(x, plus), dtype=float)
                + np.asarray(family.prob(x, minus), dtype=float)
            )
        ),
        members=(plus, minus),
    )


def truncated_target(name: str, quantile: float = 0.8) -> Target:
    """Return the baseline conditioned on a hard upper limit.

    The limit is placed at a quantile of the baseline, so the discarded mass is
    the same whatever the family. The resulting density has a jump, which is the
    feature that makes the amplitude coefficients decay only algebraically.
    """
    family, baseline, _ = TARGETS[name]
    grid = support_grid(family, baseline)
    density = np.asarray(family.prob(grid, baseline), dtype=float)  # noqa: E501
    lattice = family.is_lattice(baseline)
    cumulative = (
        np.cumsum(density) if lattice else np.cumsum(density) * (grid[1] - grid[0])
    )
    limit = float(grid[int(np.searchsorted(cumulative, quantile))])
    mass = integrate(family, baseline, grid, np.where(grid <= limit, density, 0.0))

    def sample(size: int, rng: Any) -> np.ndarray:
        kept: list[np.ndarray] = []
        total = 0
        while total < size:
            draw = np.asarray(family.sample(baseline, 2 * size, rng=rng))
            draw = draw[draw <= limit]
            kept.append(draw)
            total += len(draw)
        return np.concatenate(kept)[:size]

    return Target(
        label=f"truncated at {limit:g}",
        family=family,
        baseline=baseline,
        sample=sample,
        density=lambda x: np.where(
            np.asarray(x) <= limit,
            np.asarray(family.prob(x, baseline), dtype=float) / mass,
            0.0,
        ),
    )


# ----------------------------------------------------------------------- sweep --
def usable_degrees(name: str, degrees: tuple[int, ...]) -> tuple[int, ...]:
    """Return the requested degrees, capped where the OPS basis terminates."""

    cap = FAMILY_MAX_DEGREE.get(name)
    return degrees if cap is None else tuple(d for d in degrees if d <= cap)


def scan_target(
    target: Target,
    name: str,
    *,
    sample_sizes: tuple[int, ...] = DEFAULT_SAMPLE_SIZES,
    degrees: tuple[int, ...] = DEFAULT_DEGREES,
    epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Sweep the truncation at each sample size and score every selection rule.

    Each replicate is fitted once per degree on the whole sample, which gives the
    realised error, and once per degree per fold, which gives the held-out scores
    the selection rules use. The folds are reused across every rule, so the
    comparison between them is paired.
    """
    family, baseline = target.family, target.baseline
    degrees = usable_degrees(name, degrees)
    grid = target_grid(target)
    band = 2 * max(degrees)
    truth = reference_coefficients(target, band, grid)
    matrices = {
        degree: product_matrices(family, baseline, degree) for degree in degrees
    }

    floor_tv, floor_l2 = {}, {}
    finest = np.zeros(max(degrees) + 1)
    for degree in degrees:
        phi = matrices[degree]
        population = fit_amplitude(phi, truth[: 2 * degree + 1])["coefficients"]
        if degree == max(degrees):
            finest = population
        floor_tv[degree] = total_variation(target, population, grid)
        predicted = np.zeros(band + 1)
        predicted[: 2 * degree + 1] = ratio_coefficients(population, phi)
        floor_l2[degree] = float(np.sum((predicted[1:] - truth[1:]) ** 2))

    rules = ["l2"] + [f"ll{eps:g}" for eps in epsilons]
    tv_by_size: dict[int, dict[int, float]] = {}
    picks: dict[str, dict[int, int]] = {rule: {} for rule in rules}

    for size in sample_sizes:
        rng = np.random.default_rng(seed)
        variations = {degree: [] for degree in degrees}
        chosen = {rule: [] for rule in rules}
        for _ in range(replicates):
            draw = target.sample(size, rng)
            features = np.asarray(family.basis(draw, band, baseline), dtype=float)
            folds = np.array_split(np.arange(size), FOLDS)
            sums = features.sum(axis=0)
            fold_sums = np.array([features[f].sum(axis=0) for f in folds])
            fold_sizes = np.array([len(f) for f in folds], dtype=float)

            scores = {rule: {degree: 0.0 for degree in degrees} for rule in rules}
            for index, fold in enumerate(folds):
                inside = (sums - fold_sums[index]) / (size - fold_sizes[index])
                outside = fold_sums[index] / fold_sizes[index]
                held = features[fold]
                for degree in degrees:
                    phi = matrices[degree]
                    trial = fit_amplitude(phi, inside[: 2 * degree + 1])["coefficients"]
                    predicted = np.zeros(band + 1)
                    predicted[: 2 * degree + 1] = ratio_coefficients(trial, phi)
                    scores["l2"][degree] += float(
                        np.sum((predicted[1:] - outside[1:]) ** 2)
                    )
                    # one amplitude evaluation serves every epsilon
                    amplitude = held[:, : degree + 1] @ trial
                    squared = amplitude**2
                    for eps in epsilons:
                        scores[f"ll{eps:g}"][degree] += -float(
                            np.mean(np.log((1.0 - eps) * squared + eps))
                        )
            for rule in rules:
                chosen[rule].append(min(scores[rule], key=scores[rule].get))

            whole = sums / size
            for degree in degrees:
                phi = matrices[degree]
                fitted = fit_amplitude(phi, whole[: 2 * degree + 1])["coefficients"]
                variations[degree].append(total_variation(target, fitted, grid))

        tv_by_size[size] = {
            degree: float(np.median(values)) for degree, values in variations.items()
        }
        for rule in rules:
            picks[rule][size] = max(set(chosen[rule]), key=chosen[rule].count)

    return {
        "label": target.label,
        "family": name,
        "chi2": float(np.sum(truth[1:] ** 2)),
        "degrees": degrees,
        "sizes": sample_sizes,
        "epsilons": epsilons,
        "rules": rules,
        "finest_amplitude": finest,
        "floor_tv": floor_tv,
        "floor_l2": floor_l2,
        "tv": tv_by_size,
        "picks": picks,
    }


def report(result: dict[str, Any]) -> None:
    """Print one target's sweep."""

    degrees, sizes = result["degrees"], result["sizes"]
    print()
    print(f"{result['family']} / {result['label']}   chi^2 = {result['chi2']:.4g}")
    print(
        "  bias floor (TV): "
        + ", ".join(f"K={d}:{result['floor_tv'][d]:.2e}" for d in degrees)
    )
    monotone = all(
        result["floor_l2"][b] <= result["floor_l2"][a] + 1e-12
        for a, b in zip(degrees, degrees[1:])
    )
    print(f"  L2 floor decreasing with K: {monotone}")
    header = f"  {'N':>8} {'orc K':>6} {'TV':>10}"
    for rule in result["rules"]:
        header += f" | {rule:>8} {'x':>5}"
    print(header)
    for size in sizes:
        medians = result["tv"][size]
        oracle = min(medians, key=medians.get)
        row = f"  {size:8d} {oracle:6d} {medians[oracle]:10.3e}"
        for rule in result["rules"]:
            picked = result["picks"][rule][size]
            row += f" | {picked:8d} {medians[picked] / medians[oracle]:5.2f}"
        print(row)


def plot_results(
    results: list[dict[str, Any]], name: str, output_dir: Any = None
) -> None:
    """Draw the bias floor, the realised error, and the epsilon dependence."""

    from pathlib import Path

    rows = len(results)
    figure, axes = plt.subplots(rows, 3, figsize=(13.5, 3.1 * rows), squeeze=False)
    colours = plt.cm.viridis(np.linspace(0.0, 0.8, len(results[0]["sizes"])))

    for row, result in enumerate(results):
        degrees = np.asarray(result["degrees"], dtype=float)
        floor = np.array([result["floor_tv"][d] for d in result["degrees"]])

        # The decay of the amplitude coefficients is the quantity that sets
        # everything else, so it is shown first: factorial decay falls off a
        # cliff, an edge in the density leaves a nearly flat tail.
        axis = axes[row][0]
        amplitude = np.abs(result["finest_amplitude"])
        orders = np.arange(len(amplitude))
        axis.plot(
            orders, np.maximum(amplitude, 1e-18), "o-", color="crimson", markersize=4
        )
        axis.set_yscale("log")
        axis.set_ylim(bottom=max(1e-16, float(np.min(amplitude[amplitude > 0])) * 0.2))
        axis.set_xlabel("degree $n$")
        axis.set_ylabel("$|c_n|$")
        axis.set_title(f"{result['label']}: amplitude decay", fontsize=9)

        axis = axes[row][1]
        for colour, size in zip(colours, result["sizes"]):
            medians = np.array([result["tv"][size][d] for d in result["degrees"]])
            axis.plot(
                degrees,
                medians,
                "o-",
                color=colour,
                label=f"N=$10^{{{int(np.log10(size))}}}$",
            )
            best = int(np.argmin(medians))
            axis.plot(
                degrees[best],
                medians[best],
                "o",
                color=colour,
                markersize=12,
                markerfacecolor="none",
            )
            picked = result["picks"][result["rules"][-1]][size]
            index = result["degrees"].index(picked)
            axis.plot(degrees[index], medians[index], "x", color=colour, markersize=11)
        axis.plot(
            degrees, floor, ":", color="crimson", linewidth=1.2, label="bias floor"
        )
        axis.set_yscale("log")
        axis.set_xlabel("degree $K$")
        axis.set_title("realised TV; o = best, x = picked", fontsize=9)
        axis.legend(frameon=False, fontsize=7)

        axis = axes[row][2]
        for colour, size in zip(colours, result["sizes"]):
            medians = result["tv"][size]
            oracle = min(medians, key=medians.get)
            penalties = [
                medians[result["picks"][f"ll{eps:g}"][size]] / medians[oracle]
                for eps in result["epsilons"]
            ]
            axis.plot(result["epsilons"], penalties, "o-", color=colour)
            axis.axhline(
                medians[result["picks"]["l2"][size]] / medians[oracle],
                color=colour,
                linestyle=":",
                linewidth=1.0,
            )
        axis.set_xscale("log")
        axis.set_xlabel(r"$\epsilon$")
        axis.set_ylabel("TV(picked) / TV(best)")
        axis.set_title(r"selection penalty; dotted = $L^2$ rule", fontsize=9)
        axis.axhline(1.0, color="0.6", linewidth=0.8)

    figure.suptitle(f"{name}: degree selection", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "artifacts" / FIGURE_SUBDIRECTORY
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}_degree_selection.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    print(f"Figure: {path}")


def save_results(
    results: list[dict[str, Any]], name: str, output_dir: Any = None
) -> None:
    """Write the sweep to disk so the figures can be redrawn without refitting."""

    import json
    from pathlib import Path

    output = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "artifacts" / FIGURE_SUBDIRECTORY
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}_degree_selection.json"

    def plain(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    path.write_text(json.dumps(plain(results), indent=1))
    print(f"Results: {path}")


def _parse_args() -> argparse.Namespace:
    """Parse the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="poisson")
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--separations", type=float, nargs="*", default=(0.4, 0.8))
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SAMPLE_SIZES)
    parser.add_argument("--degrees", type=int, nargs="*", default=DEFAULT_DEGREES)
    parser.add_argument("--epsilons", type=float, nargs="*", default=DEFAULT_EPSILONS)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def build_targets(
    name: str, shift: float, separations: tuple[float, ...], quantile: float
) -> list[Target]:
    """Return the ladder of targets, skipping shifts the family cannot represent.

    The natural parameter is constrained in several families -- the negative
    binomial and the gamma need it negative, the GHS needs it inside a strip --
    so a separation that is fine for one baseline can leave the family for
    another. Such a target is dropped with a note rather than clamped, since a
    silently shrunk separation would not be the case the ladder is meant to test.
    """
    targets = [shifted_target(name, shift)]
    for separation in separations:
        try:
            targets.append(mixture_target(name, separation))
        except (ValueError, AssertionError) as error:
            print(f"  [{name}] skipping mixture d={separation:g}: {error}")
    targets.append(truncated_target(name, quantile))
    return targets


def main() -> None:
    """Run the degree scan over the ladder of targets."""

    args = _parse_args()
    names = FAMILY_NAMES if args.family == "all" else (args.family,)
    for name in names:
        shift = TARGETS[name][2] if args.shift is None else args.shift
        targets = build_targets(name, shift, tuple(args.separations), args.quantile)
        results = []
        for target in targets:
            result = scan_target(
                target,
                name,
                sample_sizes=tuple(args.sizes),
                degrees=tuple(args.degrees),
                epsilons=tuple(args.epsilons),
                replicates=args.replicates,
                seed=args.seed,
            )
            report(result)
            results.append(result)
        save_results(results, name)
        if not args.no_plot:
            plot_results(results, name)


if __name__ == "__main__":
    main()
