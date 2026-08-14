"""Probe the fitting landscape along great circles through the known optimum.

The restart experiments say the basin around the true amplitude is small, but not
why. Two very different pictures fit that observation: a small smooth basin among
many others, or a genuinely oscillatory objective. They call for opposite
remedies, so it is worth settling which one holds.

Along a great circle ``c(a) = cos(a) c* + sin(a) v`` with ``v`` orthonormal to
``c*``, the coefficient map is

    R_k(c(a)) = cos^2(a) (c*' Phi_k c*) + 2 sin(a) cos(a) (c*' Phi_k v)
                                        + sin^2(a) (v' Phi_k v),

which carries only the harmonics zero and two. Squaring it, the least-squares
objective carries only the harmonics zero, two and four. So ``J`` restricted to
any great circle is exactly a degree-two trigonometric polynomial in ``2a``:
pi-periodic -- which is the sign redundancy ``c ~ -c`` -- with at most four
critical points and at most two minima per period. Oscillation is impossible,
and the harmonic content is a prediction this module checks by Fourier transform.

The likelihood is a different object and is probed separately. It has a
logarithmic barrier wherever the amplitude passes through zero, so it does not
see the sphere as one connected domain but as a union of sign chambers, and a
descent method cannot cross between them.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from scipy.linalg import null_space

from applications.amplitude_fit_recovery import (
    FAMILY_NAMES,
    TARGETS,
    exact_amplitude,
    exact_ratio_coefficients,
    product_matrices,
)

DEFAULT_DEGREE = 6
DEFAULT_DIRECTIONS = 200
DEFAULT_RESOLUTION = 4096
DEFAULT_SEED = 11
HARMONIC_TOLERANCE = 1e-9


def great_circle(
    centre: np.ndarray, direction: np.ndarray, angles: np.ndarray
) -> np.ndarray:
    """Return the unit vectors ``cos(a) centre + sin(a) direction``."""

    return np.outer(np.cos(angles), centre) + np.outer(np.sin(angles), direction)


def tangent_directions(centre: np.ndarray, count: int, rng: Any) -> np.ndarray:
    """Return ``count`` random unit vectors orthogonal to ``centre``."""

    basis = null_space(centre[None, :])
    raw = rng.standard_normal((count, basis.shape[1]))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw @ basis.T


def objective_on_circle(
    phi: np.ndarray, target: np.ndarray, points: np.ndarray, k_max: int | None = None
) -> np.ndarray:
    """Return the unweighted least-squares objective along a circle of points."""

    highest = phi.shape[0] - 1 if k_max is None else int(k_max)
    active = np.arange(1, highest + 1)
    residual = np.einsum("an,knm,am->ak", points, phi[active], points) - target[active]
    return 0.5 * np.sum(residual**2, axis=1)


def harmonic_content(values: np.ndarray) -> np.ndarray:
    """Return the normalised magnitude of each Fourier harmonic of a full turn."""

    spectrum = np.abs(np.fft.rfft(values)) / len(values)
    return spectrum / max(spectrum[0], 1e-300)


def count_minima(values: np.ndarray) -> int:
    """Return the number of strict local minima of a periodic sampled function."""

    lower = np.roll(values, 1)
    upper = np.roll(values, -1)
    return int(np.count_nonzero((values < lower) & (values < upper)))


def basin_half_width(values: np.ndarray, angles: np.ndarray) -> float:
    """Return the angular distance from the truth to the first uphill turning point.

    The truth sits at index zero. Walking outwards, the basin ends at the first
    local maximum; beyond it a descent method is pulled somewhere else.
    """
    for index in range(1, len(values) - 1):
        if values[index] > values[index - 1] and values[index] > values[index + 1]:
            return float(angles[index])
    return float(angles[-1])


def run_landscape(
    name: str,
    *,
    degree: int = DEFAULT_DEGREE,
    directions: int = DEFAULT_DIRECTIONS,
    resolution: int = DEFAULT_RESOLUTION,
    k_max: int | None = None,
    shift: float | None = None,
    seed: int = DEFAULT_SEED,
) -> None:
    """Scan great circles through the exact amplitude and report their shape."""

    if name not in TARGETS:
        raise ValueError(
            f"unknown family {name!r}; choose one of: {', '.join(FAMILY_NAMES)}"
        )

    family, baseline, default_shift = TARGETS[name]
    shift = default_shift if shift is None else float(shift)

    phi = product_matrices(family, baseline, degree)
    exact = exact_amplitude(family, baseline, shift, degree)
    unit = exact / np.linalg.norm(exact)
    target = exact_ratio_coefficients(family, baseline, shift, 2 * degree)

    angles = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    rng = np.random.default_rng(seed)
    vectors = tangent_directions(unit, directions, rng)

    print()
    print(
        f"{name}: landscape probe, shift {shift:g}, K = {degree}, "
        f"{directions} great circles, {resolution} points each"
        + ("" if k_max is None else f", k_max = {k_max}")
    )

    beyond, minima, widths, deeper, gaps = [], [], [], 0, []
    for vector in vectors:
        values = objective_on_circle(
            phi, target, great_circle(unit, vector, angles), k_max
        )
        spectrum = harmonic_content(values)
        beyond.append(float(np.max(spectrum[5:])))
        minima.append(count_minima(values))
        widths.append(basin_half_width(values, angles))
        # is the truth the best point on this circle, and by how much
        others = values[1:]
        if float(np.min(others)) < values[0] - 1e-12:
            deeper += 1
            gaps.append(float(values[0] - np.min(others)))

    beyond = np.asarray(beyond)
    print(
        f"  harmonics above the fourth: max {beyond.max():.2e} "
        f"(prediction: zero)  -> {'CONFIRMED' if beyond.max() < HARMONIC_TOLERANCE else 'VIOLATED'}"
    )
    print(
        f"  local minima per full turn: min {min(minima)}, median "
        f"{int(np.median(minima))}, max {max(minima)}  (prediction: at most 4)"
    )
    print(
        f"  basin half-width in radians: median {np.median(widths):.3f}, "
        f"10th percentile {np.percentile(widths, 10):.3f}, min {min(widths):.3f}"
    )
    print(
        f"  circles on which the truth is not the lowest point: {deeper}/{directions}"
        + (f", deepest by {max(gaps):.2e}" if gaps else "")
    )

    # A random perturbation of size sigma per component lands this far away in
    # angle, which is the quantity a restart experiment actually controls.
    print()
    print(f"  {'sigma':>7} {'angle from truth':>17} {'inside median basin':>21}")
    for sigma in (0.05, 0.1, 0.2, 0.3):
        kicked = unit + sigma * rng.standard_normal((512, degree + 1))
        kicked /= np.linalg.norm(kicked, axis=1, keepdims=True)
        angle = np.arccos(np.clip(np.abs(kicked @ unit), -1.0, 1.0))
        print(
            f"  {sigma:7.2f} {float(np.median(angle)):17.3f} "
            f"{float(np.mean(angle < np.median(widths))) * 100:20.0f}%"
        )


def run_likelihood_landscape(
    name: str,
    *,
    degree: int = DEFAULT_DEGREE,
    sample_size: int = 10**4,
    directions: int = 32,
    resolution: int = 2048,
    shift: float | None = None,
    seed: int = DEFAULT_SEED,
) -> None:
    """Scan the same circles for the log-likelihood, which is chambered.

    Where the amplitude has a zero on the support, the likelihood of any sample
    point at that location diverges. The sphere is therefore cut into sign
    chambers that no descent path can cross, which is a different obstruction
    from the one the least-squares objective presents.
    """
    family, baseline, default_shift = TARGETS[name]
    shift = default_shift if shift is None else float(shift)
    shifted = family.shifted_params(baseline, shift)

    exact = exact_amplitude(family, baseline, shift, degree)
    unit = exact / np.linalg.norm(exact)

    rng = np.random.default_rng(seed)
    observations = np.asarray(family.sample(shifted, sample_size, rng=rng))
    features = np.asarray(family.basis(observations, degree, baseline), dtype=float)

    angles = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    vectors = tangent_directions(unit, directions, rng)

    print()
    print(
        f"{name}: likelihood probe, shift {shift:g}, K = {degree}, "
        f"N = {sample_size}, {directions} great circles"
    )

    barriers, minima, widths = [], [], []
    for vector in vectors:
        points = great_circle(unit, vector, angles)
        amplitude = features @ points.T
        # the mean log-likelihood, up to the reference term which is constant
        with np.errstate(divide="ignore"):
            values = -np.mean(np.log(amplitude**2), axis=0)
        barriers.append(int(np.count_nonzero(~np.isfinite(values))))
        finite = np.where(np.isfinite(values), values, np.inf)
        minima.append(count_minima(np.where(np.isfinite(finite), finite, 1e300)))
        widths.append(basin_half_width(finite, angles))

    print(
        f"  circles carrying a divergence: "
        f"{np.count_nonzero(np.asarray(barriers) > 0)}/{directions}"
    )
    print(
        f"  local minima per full turn: median {int(np.median(minima))}, "
        f"max {max(minima)}"
    )
    print(f"  basin half-width in radians: median {np.median(widths):.3f}")


def _parse_args() -> argparse.Namespace:
    """Parse the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="poisson", choices=(*FAMILY_NAMES, "all"))
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--directions", type=int, default=DEFAULT_DIRECTIONS)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--likelihood", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the landscape probe from the command line."""

    args = _parse_args()
    names = FAMILY_NAMES if args.family == "all" else (args.family,)
    for name in names:
        run_landscape(
            name,
            degree=args.degree,
            directions=args.directions,
            resolution=args.resolution,
            k_max=args.k_max,
            shift=args.shift,
            seed=args.seed,
        )
        if args.likelihood:
            run_likelihood_landscape(
                name, degree=args.degree, shift=args.shift, seed=args.seed
            )


if __name__ == "__main__":
    main()
