"""Sampling helpers shared by the family implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator


def resolve_generator(rng: Generator | None) -> Generator:
    """Return the supplied generator, or a fresh default one."""

    return np.random.default_rng() if rng is None else rng


def inverse_cdf_sample(
    family: Any,
    params: Any,
    size: Any,
    rng: Generator,
    *,
    grid: np.ndarray,
    edge_tolerance: float = 1e-10,
    quadrature_tolerance: float = 1e-12,
) -> np.ndarray:
    """Draw continuous variates by inverting a numerically integrated CDF.

    This is the fallback for families with no closed-form variate generator.
    The grid is checked rather than trusted, by two independent tests, so that
    a failure says which property of the grid was inadequate.

    A grid can fail in two unrelated ways, and one number cannot distinguish
    them: it can be too narrow, leaving real mass outside its endpoints, or too
    coarse, so that the trapezoid rule misintegrates a density it does cover.
    The first is measured by the density at the endpoints relative to its peak,
    the second by the integral's distance from one.

    Passing both tests bounds the *mass* error but not the quantile error: the
    inverse is interpolated linearly in the CDF, so accuracy in the returned
    variates is still set by the grid spacing.

    Raises
    ------
    ValueError
        If the endpoints carry more than ``edge_tolerance`` of the peak density
        (too narrow), or the integral differs from one by more than
        ``quadrature_tolerance`` (too coarse).
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("grid must be one-dimensional and strictly increasing")

    density = np.asarray(family.prob(grid, params), dtype=float)
    if not np.all(np.isfinite(density)):
        raise ValueError("the density is not finite everywhere on the grid")

    peak = float(np.max(density))
    if peak <= 0.0:
        raise ValueError("the density vanishes on the whole grid")
    edge = float(max(density[0], density[-1])) / peak
    if edge > edge_tolerance:
        raise ValueError(
            f"inverse-CDF grid is too narrow: its endpoints still carry {edge:.3e} "
            f"of the peak density, above the {edge_tolerance:g} tolerance. Widen "
            "the grid."
        )

    total = float(np.trapezoid(density, grid))
    if not np.isfinite(total) or abs(total - 1.0) > quadrature_tolerance:
        raise ValueError(
            "inverse-CDF grid is too coarse: integrating the density over it by "
            f"the trapezoid rule gives {total:.15f} rather than one, off by "
            f"{abs(total - 1.0):.3e}, above the {quadrature_tolerance:g} "
            "tolerance. Refine the grid."
        )

    cdf = np.concatenate(
        ([0.0], np.cumsum(np.diff(grid) * 0.5 * (density[1:] + density[:-1])))
    )
    cdf /= cdf[-1]
    increasing = np.concatenate(([True], np.diff(cdf) > 0.0))
    return np.interp(rng.random(size), cdf[increasing], grid[increasing])


def symmetric_grid(
    center: float, scale: float, *, width: float, points: int
) -> np.ndarray:
    """Return a uniform grid of ``points`` nodes spanning ``width`` scales."""

    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("center must be finite and scale must be positive")
    half = width * scale
    return np.linspace(center - half, center + half, int(points))
