"""Publication figure for a symmetric Gaussian mixture on a preset base.

Base ``N(0,1)``, fixed in advance. Target ``(1/2)N(-d,1) + (1/2)N(+d,1)``. Nothing
about the target is used to choose the base -- no component scale is estimated,
which is the point: the separation is carried entirely by the amplitude.

Everything here is analytic. For a normal base ``E_{N(m,1)}[phi_k] = m^k/sqrt(k!)``,
so the exact ratio coefficients are ``R_k = d^k/sqrt(k!)`` on even ``k``, and two
exact amplitudes exist, both with coefficient profile ``(d/2)^n/sqrt(n!)``, a
Poisson law in ``n`` with mean ``lambda = (d/2)^2``:

    real     sqrt2 e^{-d^2/4} cosh(dx/2)   even n only   gives q + e^{-d^2/2}
    complex  (h_+ + i h_-)/sqrt2           all n         gives q exactly

The cross term ``h_+ h_- = e^{-d^2/2}`` is a constant, which is the whole story:
it is the depth of the valley floor, so the real class is short by exactly that
much. It matters while the modes overlap and is ``2e-22`` by ``d = 10``.

The degree needed is therefore ``lambda`` plus a few ``sqrt(lambda)``, empirically
``(d/2)^2 + 5(d/2)`` -- quadratic in the separation, not exponential. The columns
run ``d = 1, 2, 3, 4``: the first three are reached comfortably at the display
degree and the fourth is one step past it, which is far enough to see the
mechanism of the failure without labouring it.

The two blocks separate the two limits. The upper block fits exact coefficients,
so it shows what the model class can do. The lower block fits a sample at the
degree a held-out likelihood selects, so it shows what the data supports. They
part company early, and that gap -- not the model class -- is what bounds the
separation that can be recovered.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln

from applications.amplitude_fit_complex import (
    certified_gap,
    fit_complex_amplitude,
    fitting_matrices,
    rank_two_seed,
    relaxed_optimum,
)
from applications.baseline_matching import select_degree_by_likelihood
from applications.paper_one_channel import (
    EXACT_COLOUR,
    FIT_COLOUR,
    REFERENCE_COLOUR,
    SAMPLE_COLOUR,
    TARGET_COLOUR,
    compact_scientific,
)
from applications.targets import TARGETS
from nefqvf import NormalParams
from nefqvf.fitting import fit_amplitude

SEPARATIONS = (1.0, 2.0, 3.0, 4.0)
DISPLAY_DEGREE = 12
CANDIDATE_DEGREES = (2, 3, 4, 5, 6, 8, 10, 12)
SAMPLE_SIZE = 10**5
SEED = 7
COMPLEX_COLOUR = "#1b7837"

BASELINE = NormalParams(mean=0.0, sigma=1.0)


# ----------------------------------------------------------------- analytics --
def ratio_coefficients(d: float, k_max: int) -> np.ndarray:
    """Return the exact ``R_k = d^k / sqrt(k!)``, zero on odd ``k``."""

    k = np.arange(k_max + 1)
    value = np.exp(k * np.log(d) - 0.5 * gammaln(k + 1.0))
    return np.where(k % 2 == 0, value, 0.0)


def exact_amplitudes(d: float, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact real and complex amplitudes, both normalised."""

    n = np.arange(degree + 1)
    profile = np.exp(-0.25 * d**2 + n * np.log(0.5 * d) - 0.5 * gammaln(n + 1.0))
    real = np.where(n % 2 == 0, np.sqrt(2.0) * profile, 0.0)
    complex_ = (profile / np.sqrt(2.0)) * np.where(n % 2 == 0, 1.0 + 1.0j, 1.0 - 1.0j)
    return real / np.linalg.norm(real), complex_ / np.linalg.norm(complex_)


def target_density(d: float, grid: np.ndarray) -> np.ndarray:
    """Return the symmetric mixture density."""

    return (
        0.5
        * (np.exp(-0.5 * (grid - d) ** 2) + np.exp(-0.5 * (grid + d) ** 2))
        / np.sqrt(2.0 * np.pi)
    )


def draw(d: float, size: int, rng: Any) -> np.ndarray:
    """Draw from the mixture."""

    signs = rng.integers(0, 2, size=size) * 2.0 - 1.0
    return rng.normal(loc=signs * d, scale=1.0, size=size)


# ------------------------------------------------------------------- fitting --
def fit_column(d: float, degree: int, sample: np.ndarray | None) -> dict[str, Any]:
    """Fit one separation, from exact coefficients or from a sample."""

    family = TARGETS["normal"][0]
    phi = fitting_matrices(family, BASELINE, degree)
    matched = phi.shape[0] - 1
    if sample is None:
        observed = ratio_coefficients(d, matched)
    else:
        observed = np.asarray(
            family.basis(sample, matched, BASELINE), dtype=float
        ).mean(axis=0)

    real = fit_amplitude(phi, observed)
    relaxed = relaxed_optimum(phi, observed, max_iterations=40000)
    seeds = [rank_two_seed(relaxed["matrix"]), real["coefficients"].astype(complex)]
    seeds.append(exact_amplitudes(d, degree)[1])
    best = min(
        (fit_complex_amplitude(phi, observed, initial=s) for s in seeds),
        key=lambda f: f["objective"],
    )

    grid = np.linspace(-(d + 7.0), d + 7.0, 40001)
    reference = np.asarray(family.prob(grid, BASELINE), dtype=float)
    truth = target_density(d, grid)

    def evaluate(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        re = np.asarray(family.basis_dot(grid, np.real(c), BASELINE), dtype=float)
        im = np.asarray(family.basis_dot(grid, np.imag(c), BASELINE), dtype=float)
        modulus = np.hypot(re, im)
        law = reference * modulus**2
        return modulus, law, 0.5 * float(np.trapezoid(np.abs(law - truth), grid))

    real_modulus, real_law, real_tv = evaluate(real["coefficients"].astype(complex))
    complex_modulus, complex_law, complex_tv = evaluate(best["coefficients"])
    exact_real, exact_complex = exact_amplitudes(d, degree)
    # the best this class can do at this degree: the exact amplitude truncated.
    # It is not what the coefficient objective returns, and the difference
    # between the two is a statement about the objective, not the class.
    floor_modulus, floor_law, floor_tv = evaluate(exact_complex)
    return {
        "d": d,
        "degree": degree,
        "grid": grid,
        "reference": reference,
        "truth": truth,
        "sample": sample,
        "real": real["coefficients"],
        "complex": best["coefficients"],
        "exact_real": exact_real,
        "real_modulus": real_modulus,
        "complex_modulus": complex_modulus,
        "real_law": real_law,
        "complex_law": complex_law,
        "tv_real": real_tv,
        "tv_complex": complex_tv,
        "tv_floor": floor_tv,
        "floor_law": floor_law,
        "floor_modulus": floor_modulus,
        "certified": certified_gap(phi, observed, best["coefficients"], relative=True),
        "lam": (0.5 * d) ** 2,
        "needed": (0.5 * d) ** 2 + 5.0 * (0.5 * d),
    }


# ------------------------------------------------------------------- drawing --
def _compact(value: float) -> str:
    """Return a short scientific form, for titles that must fit a narrow panel."""

    if value <= 0.0:
        return "0"
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.1f}{{\cdot}}10^{{{exponent}}}"


def draw_law(axis: Any, result: dict[str, Any], leftmost: bool) -> None:
    """Draw the base, target and the two fitted laws."""

    grid, truth = result["grid"], result["truth"]
    if result["sample"] is not None:
        axis.hist(
            result["sample"],
            bins=90,
            range=(grid.min(), grid.max()),
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
        label=r"base $N(0,1)$",
    )
    axis.plot(grid, truth, color=TARGET_COLOUR, linewidth=1.7, zorder=2, label="target")
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
        color=COMPLEX_COLOUR,
        linewidth=1.4,
        zorder=4,
        label=r"complex $p_{\rm ref}|h_c|^2$",
    )
    axis.set_yscale("log")
    axis.set_ylim(max(float(truth.max()) * 1e-9, 1e-30), float(truth.max()) * 40.0)
    axis.plot(
        grid,
        result["floor_law"],
        linestyle=(0, (4, 1.5)),
        color="0.35",
        linewidth=1.0,
        zorder=5,
        label="exact truncation",
    )
    axis.set_title(
        rf"$d={result['d']:.0f}$,  $K={result['degree']}$"
        "\n"
        rf"$D={compact_scientific(result['tv_real'])}"
        rf"\to{compact_scientific(result['tv_complex'])}$",
        fontsize=8.2,
        linespacing=1.6,
    )
    axis.tick_params(labelsize=7)
    if leftmost:
        axis.set_ylabel("density", fontsize=9)
    if leftmost:
        axis.legend(
            frameon=False,
            fontsize=6.0,
            loc="upper left",
            handlelength=1.8,
            borderpad=0.15,
            labelspacing=0.22,
        )


def draw_amplitude(axis: Any, result: dict[str, Any], leftmost: bool) -> None:
    """Draw the fitted amplitudes against the exact real one."""

    family = TARGETS["normal"][0]
    grid = result["grid"]
    exact = np.asarray(
        family.basis_dot(grid, result["exact_real"], BASELINE), dtype=float
    )
    axis.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    axis.plot(
        grid,
        exact,
        color=EXACT_COLOUR,
        linewidth=1.6,
        zorder=1,
        label=r"exact $\sqrt{2}\,e^{-d^2/4}\cosh(dx/2)$",
    )
    axis.plot(
        grid,
        result["real_modulus"],
        linestyle=(0, (1.5, 1.4)),
        color=FIT_COLOUR,
        linewidth=1.9,
        zorder=2,
        label=r"real $h$",
    )
    axis.plot(
        grid,
        result["complex_modulus"],
        color=COMPLEX_COLOUR,
        linewidth=1.4,
        zorder=3,
        label=r"complex $|h_c|$",
    )
    axis.set_yscale("log")
    finite = result["complex_modulus"][result["complex_modulus"] > 0.0]
    axis.set_ylim(
        max(float(finite.min()), float(finite.max()) * 1e-8),
        float(finite.max()) * 20.0,
    )
    axis.tick_params(labelsize=7)
    if leftmost:
        axis.set_ylabel("amplitude", fontsize=9)
    if leftmost:
        axis.legend(
            frameon=False,
            fontsize=6.0,
            loc="upper center",
            handlelength=1.8,
            borderpad=0.15,
            labelspacing=0.22,
        )


def draw_coefficients(axis: Any, result: dict[str, Any], leftmost: bool) -> None:
    """Draw the coefficient profile against the Poisson envelope it should follow."""

    degree = result["degree"]
    orders = np.arange(degree + 1)
    floor = 1e-9
    axis.axvline(
        result["lam"],
        color="0.75",
        linewidth=1.2,
        zorder=0,
        label=r"$\lambda=(d/2)^2$",
    )
    axis.plot(
        orders,
        np.maximum(np.abs(result["exact_real"]), floor),
        "s-",
        color=EXACT_COLOUR,
        markersize=3.4,
        linewidth=1.0,
        markerfacecolor="white",
        zorder=2,
        label="exact real",
    )
    axis.plot(
        orders,
        np.maximum(np.abs(result["real"]), floor),
        "o:",
        color=FIT_COLOUR,
        markersize=3.4,
        linewidth=1.2,
        zorder=3,
        label="fitted real",
    )
    axis.plot(
        orders,
        np.maximum(np.abs(result["complex"]), floor),
        "^-",
        color=COMPLEX_COLOUR,
        markersize=3.4,
        linewidth=1.0,
        zorder=4,
        label=r"fitted $|c_n|$",
    )
    axis.set_yscale("log")
    axis.set_ylim(floor * 0.5, 8.0)
    axis.set_xlabel(r"degree $n$", fontsize=9)
    axis.tick_params(labelsize=7)
    if leftmost:
        axis.set_ylabel(r"$|c_n|$", fontsize=9)
    axis.set_title(
        rf"budget $\lambda+5\sqrt{{\lambda}}\approx{result['needed']:.0f}$", fontsize=8
    )
    if leftmost:
        axis.legend(
            frameon=False,
            fontsize=6.0,
            loc="lower left",
            handlelength=1.8,
            borderpad=0.15,
            labelspacing=0.22,
        )


def make_figure(output: Path, *, png: bool = False) -> list[dict[str, Any]]:
    """Draw both blocks: exact coefficients above, a sample below."""

    family = TARGETS["normal"][0]
    rng = np.random.default_rng(SEED)
    figure = plt.figure(figsize=(9.6, 12.4))
    outer = figure.add_gridspec(2, 1, hspace=0.40, top=0.925, bottom=0.045)
    summary = []

    for block, sampled in enumerate((False, True)):
        inner = outer[block].subgridspec(
            3,
            len(SEPARATIONS),
            hspace=0.33,
            wspace=0.22,
            height_ratios=(1.25, 1.0, 1.0),
        )
        for column, d in enumerate(SEPARATIONS):
            if sampled:
                sample = draw(d, SAMPLE_SIZE, rng)
                degree = select_degree_by_likelihood(
                    family, BASELINE, sample, CANDIDATE_DEGREES
                )
            else:
                sample, degree = None, DISPLAY_DEGREE
            result = fit_column(d, degree, sample)
            draw_law(figure.add_subplot(inner[0, column]), result, column == 0)
            draw_amplitude(figure.add_subplot(inner[1, column]), result, column == 0)
            draw_coefficients(figure.add_subplot(inner[2, column]), result, column == 0)
            summary.append({"block": "sample" if sampled else "population", **result})

        box = outer[block].get_position(figure)
        figure.text(
            0.5,
            box.y1 + 0.62 / float(figure.get_size_inches()[1]),
            (
                r"sample, $N=10^5$, degree chosen by held-out likelihood"
                if sampled
                else rf"exact coefficients, $K={DISPLAY_DEGREE}$"
            ),
            ha="center",
            va="bottom",
            fontsize=12,
        )

    path = output / "bimodal-gauss-fits.pdf"
    figure.savefig(path)
    if png:
        figure.savefig(path.with_suffix(".png"), dpi=140)
    plt.close(figure)
    print(f"Figure: {path}")
    return summary


def report(summary: list[dict[str, Any]]) -> None:
    """Print the numbers the panels are annotated from."""

    print()
    print(
        f"{'block':>11} {'d':>4} {'K':>3} {'lambda':>7} {'budget':>7} "
        f"{'TV_real':>10} {'TV_cplx':>10} {'TV_floor':>10} {'cert':>9} {'|Im c|':>8}"
    )
    for row in summary:
        print(
            f"{row['block']:>11} {row['d']:4.0f} {row['degree']:3d} {row['lam']:7.2f} "
            f"{row['needed']:7.1f} {row['tv_real']:10.3e} {row['tv_complex']:10.3e} "
            f"{row['tv_floor']:10.3e} {abs(row['certified']):9.1e} "
            f"{float(np.linalg.norm(np.imag(row['complex']))):8.3f}"
        )


def _parse_args() -> argparse.Namespace:
    """Parse the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=None, help="directory for the figure")
    parser.add_argument(
        "--png",
        action="store_true",
        help="also write a raster copy, for viewing rather than for the note",
    )
    return parser.parse_args()


def main() -> None:
    """Write the figure."""

    args = _parse_args()
    output = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parents[1] / "artifacts" / "bimodal_gauss"
    )
    output.mkdir(parents=True, exist_ok=True)
    report(make_figure(output, png=args.png))


if __name__ == "__main__":
    main()
