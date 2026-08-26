"""The two-site latent figure of the note.

One page, three panels, from the Poisson pair study at the default
separation: the singular values of the empirical cross-moment matrix along
the schedule (the rank-two prediction and the slice time beyond which the
second mode is invisible), the latent-resolution experiment at ``t = 0``
(the same data fitted with and without the cross moments), and the members
recovered from the fitted moments by the moment-curve factorisation.

It also prints the persistence sweep the note quotes: recovered member
shifts and flip probability at ``epsilon = 0, 0.05, 0.25``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.two_site_diffusion import (
    DEFAULT_SEPARATION,
    build_pair_target,
    pair_density,
    run_pair_study,
)

EPSILONS = (0.0, 0.05, 0.25)
MAIN_EPSILON = 0.05


def draw(result: dict, path: Path) -> None:
    """Write the three-panel figure."""

    target = result["target"]
    family, baseline = target.family, target.baseline
    grid = result["grid"]
    slices = result["slices"]
    times = [entry["t"] for entry in slices]

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))

    axis = axes[0]
    for index, marker in zip(range(3), ("o", "s", "^"), strict=True):
        axis.semilogy(
            times,
            [entry["cross_singulars"][index] for entry in slices],
            marker + "-",
            ms=4,
            label=rf"$\sigma_{index + 1}$",
        )
    axis.set_xlabel(r"$t$")
    axis.legend(fontsize=8, frameon=False)
    axis.set_title("cross-moment singular values", fontsize=10)

    axis = axes[1]
    experiment = result["experiment"]
    labels = [label for label in experiment if "total_variation" in experiment[label]]
    curve = [
        max(experiment[label]["factorisation"]["curve_residuals"]) for label in labels
    ]
    tvs = [experiment[label]["total_variation"] for label in labels]
    positions = np.arange(len(labels))
    axis.bar(positions - 0.18, curve, width=0.36, label="moment-curve residual")
    axis.bar(positions + 0.18, tvs, width=0.36, label="joint total variation")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8)
    axis.legend(fontsize=8, frameon=False)
    axis.set_title(r"latent resolution at $t = 0$", fontsize=10)

    axis = axes[2]
    lattice = family.is_lattice(baseline)
    style = {"drawstyle": "steps-mid"} if lattice else {}
    for member in (target.plus, target.minus):
        axis.plot(
            grid,
            np.asarray(family.prob(grid, member), dtype=float),
            color="0.65",
            lw=2.5,
            **style,
        )
    for position, label in enumerate(["full cold", "marginal only"]):
        factor = experiment[label]["factorisation"]
        for column, shift in enumerate(factor["shifts"]):
            member = family.shifted_params(baseline, shift)
            axis.plot(
                grid,
                np.asarray(family.prob(grid, member), dtype=float),
                color=f"C{position}",
                lw=1.0,
                label=label if column == 0 else None,
                **style,
            )
    window = np.nonzero(pair_density(target, 0.0, grid).sum(axis=1) > 1e-8)[0]
    axis.set_xlim(grid[window[0]], grid[window[-1]])
    axis.legend(fontsize=8, frameon=False)
    axis.set_title("recovered members (true in grey)", fontsize=10)

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    print(f"figure: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        default="/Users/robertschoefbeck/Library/CloudStorage/Dropbox/Apps/"
        "Overleaf/Toolkit/figures/two-site",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("persistence sweep (poisson, sep 0.57):")
    print("    eps    shifts recovered      eps^   max curve res   TV(full)")
    main_result = None
    for epsilon in EPSILONS:
        result = run_pair_study("poisson", epsilon=epsilon)
        factor = result["experiment"]["full cold"]["factorisation"]
        print(
            f"  {epsilon:5.2f}   ({factor['shifts'][0]:+.3f}, {factor['shifts'][1]:+.3f})"
            f"   {factor['epsilon']:+.3f}"
            f"   {max(factor['curve_residuals']):13.3f}"
            f"   {result['experiment']['full cold']['total_variation']:.3e}"
        )
        if epsilon == MAIN_EPSILON:
            main_result = result
    target = build_pair_target("poisson", DEFAULT_SEPARATION, MAIN_EPSILON)
    half = 0.5 * DEFAULT_SEPARATION
    print(f"  true shifts ({half:+.3f}, {-half:+.3f}); baseline {target.baseline}")
    draw(main_result, output / "two-site-latent.pdf")


if __name__ == "__main__":
    main()
