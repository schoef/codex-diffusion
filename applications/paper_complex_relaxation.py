"""Publication figures for the convex relaxation and the complex amplitude.

The layout follows the one-channel figures: three targets per family across the
columns, two rows of three panels each, two families to a page. What the rows
carry is different, because the question is different.

The top row is the fitted laws, real against complex. The bottom row is the
amplitudes themselves -- ``h`` for the real fit and ``|h_c|`` for the complex
one -- because that is where the mechanism is visible rather than merely its
consequence. A real amplitude that has to make a law small over an interval can
only do it by passing through zero, and the law then has a node the target does
not. A complex amplitude does not: ``h_c(x) = 0`` asks the real and imaginary
parts to vanish at the same point, which is two conditions on one variable.

The third column is the hard-edge target, kept as a failure mode rather than
repaired. Neither class can represent a step by a polynomial ratio of degree
``2K``, so the second square buys a little and the ringing outside the support
stays. It is drawn here so the failure is comparable across the six families.

Each law panel reports the total variation of both fits; each amplitude panel
reports the two objectives and the relative eigenvalue gap of the optimality
test at the converged real fit, which is what says whether the real fit was
already the convex optimum.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.amplitude_fit_complex import (
    certified_gap,
    factorise_state,
    fit_complex_amplitude,
    fitting_matrices,
    optimality_test,
    rank_two_seed,
    relaxed_optimum,
    terminating_degree,
)
from applications.paper_one_channel import (
    FAMILY_TITLES,
    FIT_COLOUR,
    REFERENCE_COLOUR,
    SAMPLE_COLOUR,
    TARGET_COLOUR,
    compact_scientific,
    separation_for,
)
from applications.targets import (
    TARGETS,
    integrate,
    mixture_target,
    shifted_target,
    support_grid,
    truncated_target,
)
from nefqvf.fitting import fit_amplitude

PAGES = (("normal", "poisson"), ("gamma", "binomial"), ("negative-binomial", "ghs"))
DISPLAY_DEGREE = 12
# the Binomial is drawn at six so that the truncation still binds; at K = N = 12
# its amplitude spans every function on the support and every panel is exact
FAMILY_DEGREE = {"binomial": 6}
SAMPLE_SIZE = 10**5
SEED = 5
GAP = 2.0
RELAXATION_ITERATIONS = 40000

COMPLEX_COLOUR = "#1b7837"
RELAXED_COLOUR = "0.55"


# ------------------------------------------------------------------- fitting --
def continued_real_fit(
    family: Any, baseline: Any, coefficients: np.ndarray, degree: int
) -> np.ndarray:
    """Fit the real amplitude by continuation in the truncation.

    As in the one-channel figures, each degree starts from the previous solution
    padded with a zero. The matrices come from ``fitting_matrices``, so a
    terminating basis caps the matched coefficients instead of the degree.
    """
    previous: np.ndarray | None = None
    for order in range(2, degree + 1):
        phi = fitting_matrices(family, baseline, order)
        matched = phi.shape[0]
        start = None
        if previous is not None:
            warm = np.zeros(order + 1)
            warm[: len(previous)] = previous
            start = warm / np.linalg.norm(warm)
        candidates = [fit_amplitude(phi, coefficients[:matched], initial=start)]
        if start is not None:
            candidates.append(fit_amplitude(phi, coefficients[:matched]))
        previous = min(candidates, key=lambda f: f["objective"])["coefficients"]
    return previous


def best_complex_fit(
    family: Any,
    baseline: Any,
    phi: np.ndarray,
    observed: np.ndarray,
    degree: int,
    real: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the best complex amplitude found, and the convex optimum.

    Every seed the module knows is tried: the relaxation reduced to rank two,
    its root factorisation where that is available, and the real fit itself. The
    complex problem is not convex, so a single start is not evidence of anything.
    """
    relaxed = relaxed_optimum(phi, observed, max_iterations=RELAXATION_ITERATIONS)
    seeds = [rank_two_seed(relaxed["matrix"]), real.astype(complex)]
    try:
        seeds.append(factorise_state(family, baseline, relaxed["matrix"], degree))
    except (ValueError, np.linalg.LinAlgError):
        pass

    best: dict[str, Any] | None = None
    for seed in seeds:
        fit = fit_complex_amplitude(phi, observed, initial=seed)
        if best is None or fit["objective"] < best["objective"]:
            best = fit

    vector = best["coefficients"]
    warm = relaxed_optimum(
        phi,
        observed,
        initial=np.real(np.outer(vector, np.conj(vector))),
        max_iterations=RELAXATION_ITERATIONS,
    )
    return best, (warm if warm["objective"] < relaxed["objective"] else relaxed)


def build_column(name: str, index: int) -> tuple[Any, bool, str]:
    """Return the target, whether to standardise the reference, and the label."""

    _, _, shift = TARGETS[name]
    if index == 0:
        return shifted_target(name, shift), False, rf"single shift, $j={shift:g}$"
    if index == 1:
        separation = separation_for(name, GAP)
        return (
            mixture_target(name, separation),
            True,
            rf"mixture, $\delta={separation:.2f}$",
        )
    return truncated_target(name), False, "hard upper edge"


def fit_column(name: str, target: Any, standardise: bool, rng: Any) -> dict[str, Any]:
    """Fit one target both ways and return everything the panels need."""

    family, template, _ = TARGETS[name]
    degree = FAMILY_DEGREE.get(name, DISPLAY_DEGREE)
    cap = terminating_degree(family, template)
    degree = degree if cap is None else min(degree, cap)

    baseline = template
    if standardise:
        wide = support_grid(family, template, target.members, 40.0)
        density = target.density(wide)
        mean = integrate(family, template, wide, density * wide)
        baseline = dataclasses.replace(template, mean=float(mean))

    phi = fitting_matrices(family, baseline, degree)
    matched = phi.shape[0]
    sample = target.sample(SAMPLE_SIZE, rng)
    observed = np.asarray(
        family.basis(sample, matched - 1, baseline), dtype=float
    ).mean(axis=0)

    real = continued_real_fit(family, baseline, observed, degree)
    real_objective = float(fit_amplitude(phi, observed, initial=real)["objective"])
    test = optimality_test(phi, observed, real.astype(complex))
    complex_fit, relaxed = best_complex_fit(
        family, baseline, phi, observed, degree, real
    )

    grid = support_grid(family, baseline, target.members, 40.0)
    truth = target.density(grid)
    reference = np.asarray(family.prob(grid, baseline), dtype=float)

    real_amplitude = np.asarray(family.basis_dot(grid, real, baseline), dtype=float)
    vector = complex_fit["coefficients"]
    part_a = np.asarray(family.basis_dot(grid, np.real(vector), baseline), dtype=float)
    part_b = np.asarray(family.basis_dot(grid, np.imag(vector), baseline), dtype=float)
    complex_modulus = np.hypot(part_a, part_b)

    real_law = reference * real_amplitude**2
    complex_law = reference * complex_modulus**2
    return {
        "family": family,
        "baseline": baseline,
        "grid": grid,
        "truth": truth,
        "reference": reference,
        "sample": sample,
        "real_amplitude": real_amplitude,
        "complex_modulus": complex_modulus,
        "real_law": real_law,
        "complex_law": complex_law,
        "degree": degree,
        "matched": matched - 1,
        "real_objective": real_objective,
        "complex_objective": complex_fit["objective"],
        "relaxed_objective": relaxed["objective"],
        "certified": certified_gap(phi, observed, vector, relative=True),
        "relative_gap": test["relative_gap"],
        "imaginary_weight": float(np.linalg.norm(np.imag(vector))),
        "tv_real": 0.5 * integrate(family, baseline, grid, np.abs(real_law - truth)),
        "tv_complex": 0.5
        * integrate(family, baseline, grid, np.abs(complex_law - truth)),
        "lattice": family.is_lattice(baseline),
    }


# ------------------------------------------------------------------- drawing --
def _window(result: dict[str, Any]) -> tuple[float, float, np.ndarray]:
    """Return the plotting range and the mask of the grid inside it."""

    grid, truth = result["grid"], result["truth"]
    envelope = np.maximum(
        truth / max(truth.max(), 1e-300),
        result["reference"] / max(result["reference"].max(), 1e-300),
    )
    window = grid[envelope > 1e-5]
    low, high = float(window.min()), float(window.max())
    return low, high, (grid >= low) & (grid <= high)


def draw_law(axis: Any, result: dict[str, Any], title: str, leftmost: bool) -> None:
    """Draw the sample, reference, target and the two fits."""

    grid, truth = result["grid"], result["truth"]
    low, high, inside = _window(result)

    if result["lattice"]:
        counts = np.array(
            [np.count_nonzero(result["sample"] == v) for v in grid], dtype=float
        ) / len(result["sample"])
        axis.bar(
            grid,
            counts,
            width=0.8,
            color=SAMPLE_COLOUR,
            alpha=0.75,
            zorder=0,
            label="sample",
        )
    else:
        axis.hist(
            result["sample"],
            bins=70,
            range=(low, high),
            density=True,
            histtype="stepfilled",
            facecolor=SAMPLE_COLOUR,
            edgecolor="0.72",
            linewidth=0.5,
            alpha=0.55,
            zorder=0,
            label="sample",
        )
    axis.plot(
        grid,
        result["reference"],
        linestyle=(0, (5, 2)),
        color=REFERENCE_COLOUR,
        linewidth=1.2,
        zorder=1,
        label=r"reference $p_{\rm ref}$",
    )
    axis.plot(grid, truth, color=TARGET_COLOUR, linewidth=1.6, zorder=2, label="target")
    axis.plot(
        grid,
        result["real_law"],
        linestyle=(0, (1.5, 1.4)),
        color=FIT_COLOUR,
        linewidth=1.9,
        zorder=3,
        label=r"real $p_{\rm ref}h^2$",
    )
    axis.plot(
        grid,
        result["complex_law"],
        linestyle="-",
        color=COMPLEX_COLOUR,
        linewidth=1.4,
        zorder=4,
        label=r"complex $p_{\rm ref}|h_c|^2$",
    )

    axis.set_xlim(low, high)
    axis.set_yscale("log")
    top = float(
        max(
            truth[inside].max(),
            result["real_law"][inside].max(),
            result["complex_law"][inside].max(),
        )
    )
    axis.set_ylim(max(float(truth[truth > 0].min()) * 0.5, 1e-8), top * 40.0)
    axis.set_title(
        rf"{title}"
        "\n"
        rf"$D={compact_scientific(result['tv_real'])}"
        rf"\to{compact_scientific(result['tv_complex'])}$",
        fontsize=8.6,
        linespacing=1.5,
    )
    axis.tick_params(labelsize=8)
    if leftmost:
        axis.set_ylabel("density", fontsize=9)
    axis.legend(
        frameon=False,
        fontsize=6.4,
        loc="upper left",
        ncol=2,
        handlelength=2.0,
        columnspacing=0.9,
        borderpad=0.15,
        labelspacing=0.25,
    )


def draw_amplitude(axis: Any, result: dict[str, Any], leftmost: bool) -> None:
    """Draw the real amplitude and the complex modulus, and mark the nodes."""

    grid = result["grid"]
    low, high, inside = _window(result)
    real, modulus = result["real_amplitude"], result["complex_modulus"]

    axis.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    axis.plot(
        grid,
        real,
        linestyle=(0, (1.5, 1.4)),
        color=FIT_COLOUR,
        linewidth=1.9,
        zorder=2,
        label=r"real $h$",
    )
    axis.plot(
        grid,
        modulus,
        color=COMPLEX_COLOUR,
        linewidth=1.4,
        zorder=3,
        label=r"complex $|h_c|$",
    )

    # every sign change of the real amplitude is a node of its law
    sign = np.sign(real)
    crossings = np.flatnonzero((sign[:-1] * sign[1:] < 0) & inside[:-1] & inside[1:])
    if crossings.size:
        axis.plot(
            0.5 * (grid[crossings] + grid[crossings + 1]),
            np.zeros(crossings.size),
            "v",
            color=FIT_COLOUR,
            markersize=5.0,
            zorder=4,
            label=f"{crossings.size} node" + ("s" if crossings.size > 1 else ""),
        )

    axis.set_xlim(low, high)
    span = np.concatenate((real[inside], modulus[inside]))
    pad = 0.15 * max(float(span.max() - span.min()), 1e-12)
    axis.set_ylim(float(span.min()) - pad, float(span.max()) + pad)
    axis.set_xlabel("$m$" if result["lattice"] else "$x$", fontsize=9)
    axis.tick_params(labelsize=8)
    if leftmost:
        axis.set_ylabel(r"amplitude", fontsize=9)
    axis.set_title(
        rf"$\mathcal{{J}}={compact_scientific(result['real_objective'])}\to"
        rf"{compact_scientific(result['complex_objective'])}$,  "
        rf"gap $={compact_scientific(result['relative_gap'], digits=0)}$",
        fontsize=8,
    )
    axis.legend(
        frameon=False,
        fontsize=6.4,
        loc="upper left",
        ncol=2,
        handlelength=2.0,
        columnspacing=0.9,
        borderpad=0.15,
        labelspacing=0.25,
    )


def draw_family(
    figure: Any, cell: Any, name: str, seed_offset: int, *, fontsize: float = 12.0
) -> list[dict]:
    """Draw the two rows of one family into a grid cell, and label it.

    The label is placed from the cell's own position rather than from a hand
    tuned offset, so the same routine serves the two-family page and the
    six-family sheet without the heading landing on a panel title.
    """
    box = cell.get_position(figure)
    figure.text(
        0.5,
        box.y1 + 0.52 / float(figure.get_size_inches()[1]),
        FAMILY_TITLES[name],
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )
    inner = cell.subgridspec(2, 3, hspace=0.40, wspace=0.26, height_ratios=(1.3, 1.0))
    rng = np.random.default_rng(SEED + seed_offset)
    summary = []
    for column in range(3):
        target, standardise, label = build_column(name, column)
        result = fit_column(name, target, standardise, rng)
        draw_law(figure.add_subplot(inner[0, column]), result, label, column == 0)
        draw_amplitude(figure.add_subplot(inner[1, column]), result, column == 0)
        summary.append(
            {
                "family": name,
                "label": label,
                **{
                    key: result[key]
                    for key in (
                        "degree",
                        "matched",
                        "tv_real",
                        "tv_complex",
                        "real_objective",
                        "complex_objective",
                        "relaxed_objective",
                        "certified",
                        "relative_gap",
                        "imaginary_weight",
                    )
                },
            }
        )
    return summary


def make_page(names: tuple[str, ...], index: int, output: Path) -> list[dict]:
    """Draw one page of two families."""

    figure = plt.figure(figsize=(9.6, 12.0))
    outer = figure.add_gridspec(2, 1, hspace=0.32, top=0.930, bottom=0.04)
    summary = []
    for block, name in enumerate(names):
        summary += draw_family(figure, outer[block], name, block)
    path = output / f"complex-relaxation-fits-{index}.pdf"
    figure.savefig(path)
    plt.close(figure)
    print(f"Figure: {path}")
    return summary


def make_diagnostic_sheet(output: Path) -> list[dict]:
    """Draw all six families on one sheet, for looking at them together."""

    names = [name for page in PAGES for name in page]
    figure = plt.figure(figsize=(19.6, 36.0))
    outer = figure.add_gridspec(6, 1, hspace=0.30, top=0.965, bottom=0.02)
    summary = []
    for block, name in enumerate(names):
        summary += draw_family(figure, outer[block], name, block, fontsize=15.0)
    path = output / "complex-relaxation-all-families.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    print(f"Figure: {path}")
    return summary


def report(summary: list[dict]) -> None:
    """Print the table the panels are annotated from."""

    print()
    print(
        f"{'family':>18} {'target':>22} {'K':>3} {'kmax':>5} {'J_real':>11} "
        f"{'J_complex':>11} {'J_relaxed':>11} {'cert':>9} {'D_real':>10} "
        f"{'D_complex':>10} {'|Im c|':>8}"
    )
    for row in summary:
        print(
            f"{row['family']:>18} {row['label'][:22]:>22} {row['degree']:3d} "
            f"{row['matched']:5d} {row['real_objective']:11.3e} "
            f"{row['complex_objective']:11.3e} {row['relaxed_objective']:11.3e} "
            f"{abs(row['certified']):9.1e} {row['tv_real']:10.3e} "
            f"{row['tv_complex']:10.3e} {row['imaginary_weight']:8.3f}"
        )


def _parse_args() -> argparse.Namespace:
    """Parse the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=None, help="directory for the figures")
    parser.add_argument(
        "--pages", action="store_true", help="also write the paged PDF version"
    )
    parser.add_argument(
        "--no-sheet",
        action="store_true",
        help="skip the raster diagnostic sheet, which the note does not use",
    )
    return parser.parse_args()


def main() -> None:
    """Write the diagnostic sheet, and the paged version on request."""

    args = _parse_args()
    output = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parents[1] / "artifacts" / "complex_relaxation"
    )
    output.mkdir(parents=True, exist_ok=True)

    summary = [] if args.no_sheet else make_diagnostic_sheet(output)
    if args.pages:
        for index, names in enumerate(PAGES, start=1):
            summary += make_page(names, index, output)
    report(summary)


if __name__ == "__main__":
    main()
