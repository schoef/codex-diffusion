"""Generalized-hyperbolic-secant/Meixner-Pollaczek NEF-QVF family."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln, loggamma

from ._family import Family, finite, polynomial_degrees, same_fixed
from .jacobi import GHS
from .params import GHSParams
from .sampling import inverse_cdf_sample, symmetric_grid

_LOG_2PI = np.log(2.0 * np.pi)


class GHSFamily(Family):
    """GHS law with shape ``r`` and Meixner-Pollaczek basis.

    The natural parameter is ``eta = 2 * atan(mean / r)`` on ``(-pi, pi)``.
    This module uses the normalization whose variance is
    ``r / 2 + mean**2 / (2 * r)``.
    """

    family_code = GHS
    params_type = GHSParams

    def _validate(self, params: GHSParams) -> None:
        mean = np.asarray(params.mean)
        r = np.asarray(params.r)
        if not finite(mean, r) or np.any(r <= 0):
            raise ValueError("GHS requires finite mean and r > 0")

    def _validate_ops(self, params: GHSParams, n_max: int) -> None:
        self._validate(params)

    def _log_prob(self, x: np.ndarray, params: GHSParams) -> np.ndarray:
        mean = np.asarray(params.mean)
        r = np.asarray(params.r)
        eta = 2.0 * np.arctan(mean / r)
        cumulant = -2.0 * r * np.log(2.0 * np.cos(eta / 2.0))
        base = 2.0 * np.real(loggamma(r + 1j * x)) - _LOG_2PI - gammaln(2.0 * r)
        value = base + eta * x - cumulant
        return np.where(np.isfinite(x), value, -np.inf)

    def _ops_parameters(self, params: GHSParams) -> tuple[Any, Any]:
        return params.mean, params.r

    def _sample(self, params: GHSParams, size, rng) -> np.ndarray:
        """Draw GHS variates by inverting a numerically integrated CDF.

        GHS has a density but no elementary variate generator, so the CDF is
        built on a grid and inverted. The grid spans many standard deviations
        because the tails decay exponentially rather than being bounded, and
        ``inverse_cdf_sample`` refuses to proceed if it fails to carry the mass.

        Note that this samples the law itself. It says nothing about time
        evolution: GHS has no positivity-preserving relaxation kernel, so there
        is no forward Markov process to draw a noised pair from.
        """
        center = float(np.asarray(params.mean))
        scale = float(np.sqrt(np.asarray(self.variance(params))))
        grid = symmetric_grid(center, scale, width=45.0, points=400_001)
        return inverse_cdf_sample(self, params, size, rng, grid=grid)

    def _one_shot_sample(self, x, t, params, rng) -> np.ndarray:
        raise NotImplementedError(
            "GHS has no one-shot kernel: its spectrally defined relaxation "
            "semigroup is not positivity preserving, so no forward Markov "
            "process exists. Marginal members still flow in closed form via "
            "from_shift_coordinate."
        )

    def variance(self, params: GHSParams) -> np.ndarray:
        """Return ``r / 2 + mean**2 / (2 * r)``."""

        self._check_type(params)
        self._validate(params)
        mean, r = np.broadcast_arrays(params.mean, params.r)
        return np.asarray(r / 2.0 + mean**2 / (2.0 * r))

    def natural_parameter(self, params: GHSParams) -> np.ndarray:
        """Return ``eta = 2 * atan(mean / r)``."""

        self._check_type(params)
        self._validate(params)
        mean, r = np.broadcast_arrays(params.mean, params.r)
        return np.asarray(2.0 * np.arctan(mean / r))

    def from_natural(self, eta: Any, r: Any) -> GHSParams:
        """Construct GHS parameters from ``eta`` in ``(-pi, pi)``."""

        eta_array, r_array = np.broadcast_arrays(eta, r)
        if (
            not finite(eta_array, r_array)
            or np.any(np.abs(eta_array) >= np.pi)
            or np.any(r_array <= 0)
        ):
            raise ValueError("GHS requires -pi < eta < pi and r > 0")
        params = GHSParams(r_array * np.tan(eta_array / 2.0), r_array)
        self._validate(params)
        return params

    def shifted_params(self, params: GHSParams, natural_shift: Any) -> GHSParams:
        """Shift ``eta`` while preserving the Meixner-Pollaczek shape."""

        self._check_type(params)
        self._validate(params)
        return self.from_natural(
            self.natural_parameter(params) + np.asarray(natural_shift),
            params.r,
        )

    def shift_coordinate(self, natural_shift: Any, params: GHSParams) -> np.ndarray:
        """Return the trigonometric Meixner-Pollaczek shift coordinate."""

        self.shifted_params(params, natural_shift)
        eta, shift = np.broadcast_arrays(self.natural_parameter(params), natural_shift)
        phi = np.pi / 2.0 + eta / 2.0
        return np.asarray(np.sin(shift / 2.0) / np.sin(phi + shift / 2.0))

    def from_shift_coordinate(self, z: Any, params: GHSParams) -> GHSParams:
        """Invert the trigonometric shift coordinate at a baseline."""

        self._check_type(params)
        self._validate(params)
        eta, r, z_array = np.broadcast_arrays(
            self.natural_parameter(params), params.r, z
        )
        phi = np.pi / 2.0 + eta / 2.0
        shifted_phi = np.arctan2(
            np.sin(phi),
            np.cos(phi) - z_array,
        )
        shifted_eta = 2.0 * shifted_phi - np.pi
        return self.from_natural(shifted_eta, r)

    def shift_coefficients(
        self, natural_shift: Any, n_max: int, params: GHSParams
    ) -> np.ndarray:
        """Return normalized Meixner-Pollaczek shift coefficients."""

        self._validate_ops(params, n_max)
        degree = polynomial_degrees(n_max)
        _, r = np.broadcast_arrays(params.mean, params.r)
        z = self.shift_coordinate(natural_shift, params)
        log_gamma = 0.5 * (
            gammaln(2.0 * r[..., None] + degree)
            - gammaln(2.0 * r[..., None])
            - gammaln(degree + 1.0)
        )
        return np.exp(log_gamma) * z[..., None] ** degree

    def log_affinity(self, params1: GHSParams, params2: GHSParams) -> np.ndarray:
        """Return log affinity for members with equal shape ``r``."""

        self._check_type(params1)
        self._check_type(params2)
        self._validate(params1)
        self._validate(params2)
        r1, r2 = same_fixed("r", params1.r, params2.r)
        mean1, mean2, r = np.broadcast_arrays(
            params1.mean, params2.mean, r1 + np.zeros_like(r2)
        )
        eta1 = 2.0 * np.arctan(mean1 / r)
        eta2 = 2.0 * np.arctan(mean2 / r)
        return np.asarray(
            r
            * (
                np.log(np.cos(eta1 / 2.0))
                + np.log(np.cos(eta2 / 2.0))
                - 2.0 * np.log(np.cos((eta1 + eta2) / 4.0))
            )
        )


GHSFamily6 = GHSFamily()
GHS = GHSFamily6
ghs = GHS


if __name__ == "__main__":
    params = GHSParams(mean=np.array([-1.0, 1.0]), r=1.5)
    x = np.linspace(-20.0, 20.0, 200_001)
    probabilities = GHS.prob_grid(x, params)
    first_basis = GHS.basis(params.mean, 1, params)[..., 1]
    assert np.allclose(np.trapezoid(probabilities, x, axis=-1), 1.0, atol=2e-8)
    assert np.allclose(first_basis, 0.0)
    assert np.all(np.abs(GHS.natural_parameter(params)) < np.pi)
    print("GHS checks passed")
