"""Choose the reference law from the data before fitting the amplitude.

Every test so far fixed the baseline once and built targets by shifting it, so
the amplitude was left to supply whatever the reference did not. That is the
opposite of the prescription: the reference exists to absorb the low-order
structure -- location, and dispersion where the family has a parameter for it --
so that the coefficients describe only the departure from it.

The difference is not cosmetic. An equal mixture of two gamma members separated
by 0.8 in the natural parameter has mean 38 and variance 2475, while the fixed
baseline has mean 3 and variance 3.6. Asking a degree-K polynomial amplitude to
move the mean by a factor of thirteen and the variance by a factor of seven
hundred tests the reference, not the parametrisation.

Matching is by moments. The mean is set directly; the remaining parameter, where
one exists, is solved for so that the model variance equals the sample variance.
Two limitations are worth stating rather than hiding:

* Poisson has no free dispersion, since ``V(mu) = mu``. A moment-matched Poisson
  baseline can absorb the location of an overdispersed target but not its width,
  and the amplitude is then left to supply the excess.
* Binomial can only reach ``V < mu`` and negative binomial only ``V > mu``, so a
  target on the wrong side of its baseline family cannot be matched at all. That
  is information rather than a failure: it says the reference family is the wrong
  one, which is a choice the user makes before any fitting begins.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from scipy.optimize import brentq

# the second parameter of each family, and whether it increases the variance
DISPERSION_FIELD = {
    "NormalParams": "sigma",
    "GammaParams": "r",
    "BinomialParams": "N",
    "NegativeBinomialParams": "r",
    "GHSParams": "r",
}


def dispersion_range(name: str) -> tuple[float, float]:
    """Return a bracket for the dispersion parameter wide enough to solve in."""

    if name == "BinomialParams":
        return 1.0 + 1e-9, 1e9
    return 1e-9, 1e9


def moment_matched(
    family: Any, template: Any, mean: float, variance: float
) -> tuple[Any, str]:
    """Return the family member with the given mean and, if possible, variance.

    The second return value records what was achieved, so a caller can report an
    unmatched dispersion rather than silently proceeding with a reference that
    does not describe the data.
    """
    name = type(template).__name__
    field = DISPERSION_FIELD.get(name)
    if field is None:
        return dataclasses.replace(template, mean=float(mean)), "mean only"

    def defect(value: float) -> float:
        candidate = dataclasses.replace(
            template, mean=float(mean), **{field: float(value)}
        )
        return float(family.variance(candidate)) - variance

    low, high = dispersion_range(name)
    try:
        # the variance is monotone in the dispersion parameter, so a sign change
        # over the bracket is both necessary and sufficient
        if defect(low) * defect(high) > 0.0:
            return dataclasses.replace(template, mean=float(mean)), "unreachable"
        solution = brentq(defect, low, high, xtol=1e-12, rtol=1e-14)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return dataclasses.replace(template, mean=float(mean)), "unreachable"

    matched = dataclasses.replace(
        template, mean=float(mean), **{field: float(solution)}
    )
    return matched, "mean and variance"


def matched_to_sample(
    family: Any, template: Any, sample: np.ndarray
) -> tuple[Any, str]:
    """Return the baseline matched to a sample's first two moments."""

    values = np.asarray(sample, dtype=float)
    return moment_matched(family, template, float(values.mean()), float(values.var()))


# --------------------------------------------------------------------- figures --
def plot_baseline_choice(
    name: str, *, degree: int = 12, targets: Any = None, output_dir: Any = None
) -> list[dict[str, Any]]:
    """Draw why matching the mean helps and matching the variance does not.

    Three columns, reading left to right as cause and effect: the laws with each
    candidate reference, the density ratio the amplitude actually has to
    represent, and the coefficient decay that results. The middle column is the
    explanatory one -- a reference of the wrong location leaves a ratio that
    sweeps over orders of magnitude, and a reference of the wrong shape leaves a
    ratio with a non-polynomial factor that no truncation absorbs.
    """
    import dataclasses

    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    from applications.amplitude_fit_degree import (
        FAMILY_MAX_DEGREE,
        build_targets,
        integrate,
        support_grid,
    )
    from applications.amplitude_fit_recovery import (
        TARGETS,
        fit_amplitude,
        product_matrices,
    )

    degree = min(degree, FAMILY_MAX_DEGREE.get(name, degree))
    family, template, shift = TARGETS[name]
    if targets is None:
        targets = [
            t
            for t in build_targets(name, shift, (0.4, 0.8), 0.8)
            if "mixture" in t.label
        ]
    lattice = family.is_lattice(template)

    figure, axes = plt.subplots(
        len(targets), 3, figsize=(14.0, 3.4 * len(targets)), squeeze=False
    )
    palette = {
        "fixed": "tab:red",
        "mean only": "tab:blue",
        "mean+variance": "tab:green",
    }
    summary = []

    for row, target in enumerate(targets):
        wide = support_grid(family, template, target.members, 40.0)
        density = target.density(wide)
        mean = integrate(family, template, wide, density * wide)
        variance = integrate(family, template, wide, density * (wide - mean) ** 2)
        candidates = {
            "fixed": template,
            "mean only": dataclasses.replace(template, mean=float(mean)),
            "mean+variance": moment_matched(family, template, mean, variance)[0],
        }

        # a display grid covering the target, not the far integration tail
        spread = float(np.sqrt(variance))
        low, high = mean - 4.0 * spread, mean + 4.0 * spread
        grid = (
            np.arange(max(int(low), 0), int(high) + 1)
            if lattice
            else np.linspace(low, high, 800)
        )
        shown = target.density(grid)

        axes[row][0].plot(
            grid,
            shown,
            color="black",
            linewidth=6.0,
            alpha=0.20,
            zorder=1,
            solid_capstyle="round",
            label="target",
        )
        for label, baseline in candidates.items():
            phi = product_matrices(family, baseline, degree)
            basis = np.asarray(family.basis(wide, 2 * degree, baseline), dtype=float)
            coefficients = np.array(
                [
                    integrate(family, baseline, wide, density * basis[:, k])
                    for k in range(2 * degree + 1)
                ]
            )
            fitted = fit_amplitude(phi, coefficients)["coefficients"]

            reference_wide = np.asarray(family.prob(wide, baseline), dtype=float)
            amplitude_wide = np.asarray(
                family.basis_dot(wide, fitted, baseline), dtype=float
            )
            distance = 0.5 * integrate(
                family,
                baseline,
                wide,
                np.abs(reference_wide * amplitude_wide**2 - density),
            )
            summary.append({"target": target.label, "baseline": label, "tv": distance})

            reference = np.asarray(family.prob(grid, baseline), dtype=float)
            amplitude = np.asarray(
                family.basis_dot(grid, fitted, baseline), dtype=float
            )
            colour = palette[label]

            axes[row][0].plot(
                grid, reference, "--", color=colour, linewidth=0.8, alpha=0.55, zorder=2
            )
            axes[row][0].plot(
                grid,
                reference * amplitude**2,
                "-",
                color=colour,
                linewidth=1.5,
                zorder=4,
                label=f"{label}  TV={distance:.1e}",
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                axes[row][1].plot(
                    grid,
                    np.where(reference > 0, shown / reference, np.nan),
                    color=colour,
                    linewidth=1.4,
                    label=label,
                )
            axes[row][2].plot(
                np.arange(degree + 1),
                np.abs(fitted),
                "o-",
                color=colour,
                markersize=4,
                label=label,
            )

        axes[row][0].set_yscale("log")
        axes[row][0].set_ylim(bottom=max(float(shown[shown > 0].min()) * 0.5, 1e-9))
        axes[row][0].set_ylabel("density")
        axes[row][0].set_title(
            f"{target.label}: halo = target, solid = fit, dashed = reference",
            fontsize=9,
        )
        axes[row][0].legend(frameon=False, fontsize=7)

        axes[row][1].set_yscale("log")
        axes[row][1].set_ylabel("$q = p_{\\rm target}/p_{\\rm ref}$")
        axes[row][1].set_title("what the amplitude must represent", fontsize=9)
        axes[row][1].legend(frameon=False, fontsize=7)

        axes[row][2].set_yscale("log")
        axes[row][2].set_xlabel("degree $n$")
        axes[row][2].set_ylabel("$|c_n|$")
        axes[row][2].set_title("resulting coefficient decay", fontsize=9)
        axes[row][2].legend(frameon=False, fontsize=7)

    for axis in axes[-1][:2]:
        axis.set_xlabel("$m$" if lattice else "$x$")
    figure.suptitle(f"{name}: choosing the reference law, $K$ = {degree}", fontsize=11)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "artifacts" / "baseline_choice"
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}_baseline_choice.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    print(f"Figure: {path}")
    return summary


if __name__ == "__main__":
    from applications.amplitude_fit_recovery import FAMILY_NAMES

    for family_name in FAMILY_NAMES:
        for entry in plot_baseline_choice(family_name):
            print(
                f"  {family_name:18s} {entry['target']:14s} "
                f"{entry['baseline']:14s} TV={entry['tv']:.3e}"
            )


# ------------------------------------------------------------ shift solving --
def _admits(family: Any, baseline: Any, step: float) -> bool:
    """Return whether the shifted member exists."""

    try:
        family.shifted_params(baseline, step)
        return True
    except (ValueError, AssertionError, FloatingPointError):
        return False


def one_sided_limit(
    family: Any, baseline: Any, sign: float, ceiling: float = 200.0
) -> float:
    """Return the furthest admissible shift in one direction.

    The two directions must be searched separately: a baseline placed at the
    mixture mean can usually be pushed a long way down but only a short way up,
    or the reverse, depending on which end of its parameter range it sits.
    """
    if _admits(family, baseline, sign * ceiling):
        return sign * ceiling
    low, high = 0.0, ceiling
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (
            (middle, high)
            if _admits(family, baseline, sign * middle)
            else (low, middle)
        )
    return sign * low


def shift_reaching_mean(family: Any, baseline: Any, mean: float) -> float:
    """Return the natural shift whose member has the requested mean."""

    low = one_sided_limit(family, baseline, -1.0) * (1.0 - 1e-9)
    high = one_sided_limit(family, baseline, +1.0) * (1.0 - 1e-9)

    def defect(step: float) -> float:
        return float(family.mean(family.shifted_params(baseline, step))) - mean

    if defect(low) * defect(high) > 0.0:
        return low if abs(defect(low)) < abs(defect(high)) else high
    return float(brentq(defect, low, high, xtol=1e-14, rtol=1e-15))


def select_degree_by_likelihood(
    family: Any,
    baseline: Any,
    sample: np.ndarray,
    degrees: tuple[int, ...],
    epsilon: float = 1e-2,
    folds: int = 5,
) -> int:
    """Choose the truncation by held-out likelihood.

    The likelihood is evaluated on the baseline-regularised model, the mixing
    serving only to keep the logarithm finite where the amplitude passes through
    zero at an observed point. It is used to score candidate degrees, never to
    optimise: as an objective it carries one logarithmic barrier per distinct
    observed value, whereas the coefficient loss it selects for is a polynomial.
    """
    from applications.amplitude_fit_recovery import fit_amplitude, product_matrices

    band = 2 * max(degrees)
    features = np.asarray(family.basis(sample, band, baseline), dtype=float)
    parts = np.array_split(np.arange(len(sample)), folds)
    sums = features.sum(axis=0)

    scores = {degree: 0.0 for degree in degrees}
    for part in parts:
        held = features[part]
        inside = (sums - held.sum(axis=0)) / (len(sample) - len(part))
        for degree in degrees:
            phi = product_matrices(family, baseline, degree)
            coefficients = fit_amplitude(phi, inside[: 2 * degree + 1])["coefficients"]
            squared = (held[:, : degree + 1] @ coefficients) ** 2
            scores[degree] += -float(
                np.mean(np.log((1.0 - epsilon) * squared + epsilon))
            )
    return min(scores, key=scores.get)


# ------------------------------------------------------- comparable separations --
def largest_valid_shift(family: Any, baseline: Any, ceiling: float = 50.0) -> float:
    """Return the largest ``d`` for which both ``eta +- d`` stay in the family.

    Several families restrict the natural parameter -- the gamma and the negative
    binomial to negative values, the GHS to a strip -- so a symmetric pair of
    shifts is only defined up to a limit that depends on where the baseline sits.
    """

    def valid(step: float) -> bool:
        try:
            family.shifted_params(baseline, step)
            family.shifted_params(baseline, -step)
            return True
        except (ValueError, AssertionError, FloatingPointError):
            return False

    if valid(ceiling):
        return ceiling
    low, high = 0.0, ceiling
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if valid(middle) else (low, middle)
    return low


def separation_for_mean_ratio(
    family: Any, baseline: Any, ratio: float
) -> tuple[float, str]:
    """Return the shift whose two members' means stand in the given ratio.

    A fixed step in the natural parameter is not a comparable difficulty across
    families: for the Poisson it multiplies the mean by ``exp(d)``, while for the
    gamma, where ``eta = -r/mu``, the mean diverges as the parameter approaches
    zero, so the same step of 0.8 gives a mean ratio of 2.2 in one family and 49
    in the other. Fixing the ratio of the component means instead puts the
    mixtures on a common footing.
    """
    limit = largest_valid_shift(family, baseline)

    def achieved(step: float) -> float:
        plus = float(family.mean(family.shifted_params(baseline, step)))
        minus = float(family.mean(family.shifted_params(baseline, -step)))
        return plus / minus

    top = limit * (1.0 - 1e-9)
    if achieved(top) < ratio:
        return top, f"unreachable, best ratio {achieved(top):.2f}"
    solution = brentq(lambda s: achieved(s) - ratio, 1e-9, top, xtol=1e-14, rtol=1e-15)
    return float(solution), "exact"


def separation_for_gap(family: Any, baseline: Any, gap: float) -> tuple[float, str]:
    """Return the shift giving a standardised gap between the two components.

    The gap is ``(mu_plus - mu_minus)`` divided by the mean of the two component
    standard deviations, which is the usual measure of how separated a mixture
    is: below about two the components merge into one mode, above it they
    resolve. Unlike a ratio of means it is defined for the symmetric families,
    whose baseline mean is zero.
    """
    limit = largest_valid_shift(family, baseline)

    def achieved(step: float) -> float:
        plus = family.shifted_params(baseline, step)
        minus = family.shifted_params(baseline, -step)
        spread = 0.5 * (
            float(np.sqrt(family.variance(plus)))
            + float(np.sqrt(family.variance(minus)))
        )
        return (float(family.mean(plus)) - float(family.mean(minus))) / spread

    top = limit * (1.0 - 1e-9)
    if achieved(top) < gap:
        return top, f"unreachable, best gap {achieved(top):.2f}"
    solution = brentq(lambda s: achieved(s) - gap, 1e-9, top, xtol=1e-14, rtol=1e-15)
    return float(solution), "exact"
