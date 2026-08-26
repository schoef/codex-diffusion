"""Baselines, targets, and the metric used to score a fit.

The choices here are the application's, not the package's: which six baselines
to fit against, which laws to try to recover, and how to measure the result.

Three families of target are provided. A ``shifted`` member of the baseline
family has a known amplitude in closed form -- the half-shift amplitude times
the Hellinger affinity -- so it is the only one against which a fit can be
checked coefficient by coefficient. An equal ``mixture`` of two oppositely
shifted members has exact ratio coefficients but no closed-form amplitude, being
the square root of a sum, and is bimodal once the components separate. A
``truncated`` baseline has a jump, which no polynomial ratio represents.

Reference coefficients are obtained for every target by integrating ``phi_k``
against the true density, so all three are handled identically. Agreement is
measured in total variation, which is bounded, is the largest disagreement the
two laws can have about any event, and stays meaningful for a target outside the
model class.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

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

QUADRATURE_POINTS = 20001

# The Krawtchouk basis terminates at the binomial index, and the fit needs the
# product tensor out to 2K, so the binomial baseline with N = 12 caps at K = 6.
FAMILY_MAX_DEGREE = {"binomial": 6}


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


# ----------------------------------------------------------------------- sweep --
def usable_degrees(name: str, degrees: tuple[int, ...]) -> tuple[int, ...]:
    """Return the requested degrees, capped where the OPS basis terminates."""

    cap = FAMILY_MAX_DEGREE.get(name)
    return degrees if cap is None else tuple(d for d in degrees if d <= cap)


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
