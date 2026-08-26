"""Publication figures for the one-channel prototype section.

Three displacements per family, two rows of three panels each: the fitted laws
across the top and the amplitude coefficients directly beneath. Two families to
a page, so the six take three pages of twelve panels. A fourth target with a
hard upper edge is drawn separately.

The first column is a single shifted member fitted against the **unstandardised**
baseline. That is deliberate. Standardising the reference to the sample mean
absorbs a pure shift completely -- for the Normal it reproduces the target
exactly, leaving nothing for the amplitude to do -- so the informative version of
this test keeps the reference where it is and lets the amplitude carry the whole
displacement. Its exact coefficients are then known analytically, being the
half-shift amplitude of Section 5, and are drawn for comparison.

The second and third columns are equal mixtures, fitted against a reference
standardised to the sample mean with the shape parameter left alone. Their
separations are set by the standardised gap between the two components, so that
the same nominal difficulty is compared across families, and capped where one
component would become too narrow for the reference to resolve.
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

from applications.baseline_matching import (
    select_degree_by_likelihood,
    separation_for_gap,
)
from applications.targets import (
    FAMILY_MAX_DEGREE,
    TARGETS,
    integrate,
    mixture_target,
    shifted_target,
    support_grid,
    truncated_target,
)
from nefqvf.fitting import fit_amplitude, product_matrices

PAGES = (("normal", "poisson"), ("gamma", "binomial"), ("negative-binomial", "ghs"))
EDGE_FAMILIES = ("gamma", "ghs")
DEGREES = (2, 3, 4, 5, 6, 8, 10, 12)
DISPLAY_DEGREE = 12
SAMPLE_SIZE = 10**5
SEED = 5
# the widest standardised gap every family sustains: at 2.5 the GHS mixture
# is no longer resolvable against its reference
GAPS = (1.1, 2.0)
# beyond this ratio of component widths the narrower one is finer than a
# degree-DISPLAY_DEGREE amplitude resolves, and the separation is capped
MAX_WIDTH_RATIO = 4.5

# how the families are named in figure headings; spelled out, since the figures
# are read on their own
FAMILY_TITLES = {
    "normal": "Normal distribution",
    "poisson": "Poisson distribution",
    "gamma": "Gamma distribution",
    "binomial": "Binomial distribution",
    "negative-binomial": "Negative binomial distribution",
    "ghs": "Generalized hyperbolic secant distribution",
}

TARGET_COLOUR = "0.15"
REFERENCE_COLOUR = "#d95f02"
FIT_COLOUR = "#1b6ca8"
EXACT_COLOUR = "0.45"
SAMPLE_COLOUR = "0.86"


def compact_scientific(value: float, digits: int = 1) -> str:
    r"""Return a mathtext scientific form, for titles that must fit a panel.

    ``1.3\cdot10^{-3}`` rather than ``1.3e-03``, which sets badly next to the
    surrounding mathematics and is wider than the space a panel title has.
    """
    if not np.isfinite(value) or value <= 0.0:
        return "0"
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent
    if digits == 0:
        return (
            rf"10^{{{exponent}}}"
            if round(mantissa) == 1
            else (rf"{mantissa:.0f}{{\cdot}}10^{{{exponent}}}")
        )
    return rf"{mantissa:.{digits}f}{{\cdot}}10^{{{exponent}}}"


def width_ratio(family: Any, template: Any, separation: float) -> float:
    """Return the ratio of the two component standard deviations."""

    spreads = [
        float(
            np.sqrt(family.variance(family.shifted_params(template, sign * separation)))
        )
        for sign in (+1.0, -1.0)
    ]
    return max(spreads) / min(spreads)


def separation_for(name: str, gap: float) -> float:
    """Return the separation giving the standardised gap, capped for resolvability."""

    family, template, _ = TARGETS[name]
    wanted, _ = separation_for_gap(family, template, gap)
    if width_ratio(family, template, wanted) <= MAX_WIDTH_RATIO:
        return wanted
    low, high = 0.0, wanted
    for _ in range(80):
        middle = 0.5 * (low + high)
        low, high = (
            (middle, high)
            if width_ratio(family, template, middle) <= MAX_WIDTH_RATIO
            else (low, middle)
        )
    return low


def continued_fit(
    family: Any, baseline: Any, coefficients: np.ndarray, degree: int
) -> np.ndarray:
    """Fit by continuation in the truncation, as Eq. (K-continuation) prescribes.

    Each degree is started from the previous solution padded with a zero, which
    is an exact point of the larger model. A cold start from the baseline vector
    is reliable at small degree but occasionally fails outright at twelve, and
    the continuation costs nothing.
    """
    previous: np.ndarray | None = None
    for order in range(2, degree + 1):
        phi = product_matrices(family, baseline, order)
        start = None
        if previous is not None:
            warm = np.zeros(order + 1)
            warm[: len(previous)] = previous
            start = warm / np.linalg.norm(warm)
        candidates = [fit_amplitude(phi, coefficients[: 2 * order + 1], initial=start)]
        if start is not None:
            candidates.append(fit_amplitude(phi, coefficients[: 2 * order + 1]))
        previous = min(candidates, key=lambda f: f["objective"])["coefficients"]
    return previous


def build_column(name: str, index: int) -> tuple[Any, bool, str]:
    """Return the target, whether to standardise, and the panel label."""

    _, _, shift = TARGETS[name]
    if index == 0:
        target = shifted_target(name, shift)
        return target, False, rf"single shift, $j={shift:g}$"
    # the widest separation is set by the gap, then capped for resolvability; the
    # narrower one is a fixed fraction of it, so the two columns stay distinct in
    # the families where the cap binds
    widest = separation_for(name, GAPS[-1])
    separation = widest if index == 2 else 0.55 * widest
    return (
        mixture_target(name, separation),
        True,
        rf"mixture, $\delta={separation:.2f}$",
    )


def fit_column(name: str, target: Any, standardise: bool, rng: Any) -> dict[str, Any]:
    """Fit one target and return everything the panels need."""

    family, template, shift = TARGETS[name]
    cap = FAMILY_MAX_DEGREE.get(name, DISPLAY_DEGREE)
    degrees = tuple(d for d in DEGREES if d <= cap)
    degree = min(DISPLAY_DEGREE, cap)

    baseline = template
    if standardise:
        wide = support_grid(family, template, target.members, 40.0)
        density = target.density(wide)
        mean = integrate(family, template, wide, density * wide)
        baseline = dataclasses.replace(template, mean=float(mean))

    sample = target.sample(SAMPLE_SIZE, rng)
    selected = select_degree_by_likelihood(family, baseline, sample, degrees)

    grid = support_grid(family, baseline, target.members, 40.0)
    truth = target.density(grid)
    empirical = np.asarray(
        family.basis(sample, 2 * degree, baseline), dtype=float
    ).mean(axis=0)
    fitted = continued_fit(family, baseline, empirical, degree)

    if standardise:
        # no closed form outside the family: use the fit to exact coefficients
        exact = continued_fit(
            family,
            baseline,
            np.array(
                [
                    integrate(
                        family,
                        baseline,
                        grid,
                        truth
                        * np.asarray(
                            family.basis(grid, 2 * degree, baseline), dtype=float
                        )[:, k],
                    )
                    for k in range(2 * degree + 1)
                ]
            ),
            degree,
        )
        exact_label = "population fit"
    else:
        # the analytic half-shift amplitude of Section 5
        exact = np.asarray(
            family.shift_coefficients(0.5 * shift, degree, baseline), dtype=float
        )
        exact /= np.linalg.norm(exact)
        exact_label = r"analytic $h_{j/2}$"

    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    amplitude = np.asarray(family.basis_dot(grid, fitted, baseline), dtype=float)
    law = reference * amplitude**2
    return {
        "family": family,
        "grid": grid,
        "truth": truth,
        "reference": reference,
        "law": law,
        "sample": sample,
        "fitted": fitted,
        "exact": exact,
        "exact_label": exact_label,
        "degree": degree,
        "selected": selected,
        "tv": 0.5 * integrate(family, baseline, grid, np.abs(law - truth)),
        "lattice": family.is_lattice(baseline),
    }


def draw_law(axis: Any, result: dict[str, Any], title: str, leftmost: bool) -> None:
    """Draw the sample, reference, target and fit."""

    grid, truth = result["grid"], result["truth"]
    envelope = np.maximum(
        truth / truth.max(), result["reference"] / result["reference"].max()
    )
    window = grid[envelope > 1e-5]
    low, high = float(window.min()), float(window.max())
    inside = (grid >= low) & (grid <= high)

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
        result["law"],
        linestyle=(0, (1.5, 1.4)),
        color=FIT_COLOUR,
        linewidth=2.0,
        zorder=3,
        label=r"fit $p_{\rm ref}h_K^2$",
    )

    axis.set_xlim(low, high)
    axis.set_yscale("log")
    top = float(max(truth[inside].max(), result["law"][inside].max()))
    axis.set_ylim(max(float(truth[truth > 0].min()) * 0.5, 1e-8), top * 30.0)
    axis.set_title(rf"{title}:  $D={result['tv']:.1e}$", fontsize=9)
    axis.tick_params(labelsize=8)
    if leftmost:
        axis.set_ylabel("density", fontsize=9)
    axis.legend(
        frameon=False,
        fontsize=6.6,
        loc="upper left",
        ncol=2,
        handlelength=2.0,
        columnspacing=0.9,
        borderpad=0.15,
        labelspacing=0.25,
    )


def draw_coefficients(axis: Any, result: dict[str, Any], leftmost: bool) -> None:
    """Draw the fitted coefficients against the exact ones."""

    orders = np.arange(result["degree"] + 1)
    floor = 1e-7
    # everything past the selected degree is where sampling noise dominates
    if result["selected"] < result["degree"]:
        axis.axvspan(
            result["selected"] + 0.5, result["degree"] + 0.5, color="0.92", zorder=0
        )
        # directly under the legend, which sits in the upper right, so the note
        # reads with the entries it qualifies rather than across the shaded band
        axis.text(
            0.985,
            0.78,
            rf"noise beyond $K_\star={result['selected']}$",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.6,
            color="0.35",
        )
    axis.plot(
        orders,
        np.maximum(np.abs(result["exact"]), floor),
        "s-",
        color=EXACT_COLOUR,
        markersize=3.6,
        linewidth=1.1,
        markerfacecolor="white",
        zorder=2,
        label=result["exact_label"],
    )
    axis.plot(
        orders,
        np.maximum(np.abs(result["fitted"]), floor),
        "o:",
        color=FIT_COLOUR,
        markersize=3.6,
        linewidth=1.3,
        zorder=3,
        label=r"fitted, $N=10^5$",
    )
    axis.set_yscale("log")
    axis.set_xlabel(r"degree $n$", fontsize=9)
    axis.set_ylim(floor * 0.5, 40.0)
    axis.set_xlim(-0.5, result["degree"] + 0.5)
    axis.set_xticks(orders[:: max(1, len(orders) // 6)])
    axis.tick_params(labelsize=8)
    if leftmost:
        axis.set_ylabel(r"$|c_n|$", fontsize=9)
    axis.legend(
        frameon=False,
        fontsize=6.6,
        loc="upper right",
        handlelength=2.0,
        borderpad=0.15,
        labelspacing=0.25,
    )


def make_page(names: tuple[str, str], index: int, output: Path) -> list[dict[str, Any]]:
    """Draw one page of two families."""

    figure = plt.figure(figsize=(9.6, 12.0))
    outer = figure.add_gridspec(2, 1, hspace=0.20, top=0.955, bottom=0.04)
    summary = []
    for block, name in enumerate(names):
        inner = outer[block].subgridspec(
            2, 3, hspace=0.30, wspace=0.24, height_ratios=(1.3, 1.0)
        )
        rng = np.random.default_rng(SEED + block)
        for column in range(3):
            target, standardise, label = build_column(name, column)
            result = fit_column(name, target, standardise, rng)
            draw_law(figure.add_subplot(inner[0, column]), result, label, column == 0)
            draw_coefficients(figure.add_subplot(inner[1, column]), result, column == 0)
            summary.append(
                {
                    "family": name,
                    "label": label,
                    "tv": result["tv"],
                    "selected": result["selected"],
                    "degree": result["degree"],
                }
            )
        # a grid cell bounds the axes, not their titles, so the heading is set
        # from the cell's own position and cleared by a fixed physical distance
        box = outer[block].get_position(figure)
        figure.text(
            0.5,
            box.y1 + 0.30 / float(figure.get_size_inches()[1]),
            FAMILY_TITLES[name],
            ha="center",
            va="bottom",
            fontsize=12,
        )
    path = output / f"one-channel-fits-{index}.pdf"
    figure.savefig(path)
    plt.close(figure)
    print(f"Figure: {path}")
    return summary


def make_edge_page(output: Path) -> list[dict[str, Any]]:
    """Draw the hard-edge targets, whose fitted amplitudes ring."""

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(8.6, 6.0),
        gridspec_kw={"height_ratios": (1.3, 1.0), "hspace": 0.32, "wspace": 0.24},
    )
    summary = []
    for column, name in enumerate(EDGE_FAMILIES):
        rng = np.random.default_rng(SEED + 11 + column)
        target = truncated_target(name, 0.8)
        result = fit_column(name, target, True, rng)
        label = FAMILY_TITLES[name]
        draw_law(axes[0][column], result, label, column == 0)
        draw_coefficients(axes[1][column], result, column == 0)
        axes[0][column].set_xlabel("")
        limit = float(result["grid"][result["truth"] > 0].max())
        span = axes[0][column].get_xlim()
        axes[0][column].set_xlim(span[0], limit + 0.45 * (limit - span[0]))
        axes[0][column].axvline(limit, color="crimson", linestyle=":", linewidth=1.1)
        summary.append(
            {
                "family": name,
                "label": "truncated",
                "tv": result["tv"],
                "selected": result["selected"],
                "degree": result["degree"],
            }
        )
    path = output / "one-channel-edge.pdf"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"Figure: {path}")
    return summary


def _parse_args() -> argparse.Namespace:
    """Parse the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        default="/Users/robertschoefbeck/Library/CloudStorage/Dropbox/Apps/"
        "Overleaf/Toolkit/figures/one-channel",
    )
    return parser.parse_args()


def main() -> None:
    """Write every figure and print the numbers the table quotes."""

    output = Path(_parse_args().output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, names in enumerate(PAGES, start=1):
        rows += make_page(names, index, output)
    rows += make_edge_page(output)
    print()
    print(f"{'family':18s} {'target':26s} {'K':>3} {'K*':>3} {'D':>10}")
    for row in rows:
        label = row["label"].replace("$", "").replace("\\", "")
        print(
            f"{row['family']:18s} {label:26s} {row['degree']:3d} "
            f"{row['selected']:3d} {row['tv']:10.2e}"
        )


if __name__ == "__main__":
    main()
