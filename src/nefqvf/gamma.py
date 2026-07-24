"""Gamma/Laguerre NEF-QVF family."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln

from ._family import Family, finite, polynomial_degrees, same_fixed
from .jacobi import GAMMA
from .params import GammaParams


class GammaFamily(Family):
    """Gamma law with fixed shape ``r`` and positive-gauge Laguerre basis.

    The scale is derived as ``theta = mean / r``. The package rephases the
    standard Laguerre polynomials by ``(-1)**n`` so all Jacobi off-diagonals
    are positive.
    """

    family_code = GAMMA
    params_type = GammaParams

    def _validate(self, params: GammaParams) -> None:
        mean = np.asarray(params.mean)
        r = np.asarray(params.r)
        if not finite(mean, r) or np.any(mean <= 0) or np.any(r <= 0):
            raise ValueError("Gamma requires finite mean > 0 and r > 0")

    def _validate_ops(self, params: GammaParams, n_max: int) -> None:
        self._validate(params)

    def _log_prob(self, x: np.ndarray, params: GammaParams) -> np.ndarray:
        mean = np.asarray(params.mean)
        r = np.asarray(params.r)
        theta = mean / r
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = (r - 1.0) * np.log(x) - x / theta - r * np.log(theta) - gammaln(r)
        return np.where(np.isfinite(x) & (x > 0), value, -np.inf)

    def _ops_parameters(self, params: GammaParams) -> tuple[Any, Any]:
        return params.mean, params.r

    def variance(self, params: GammaParams) -> np.ndarray:
        """Return ``mean**2 / r``."""

        self._check_type(params)
        self._validate(params)
        mean, r = np.broadcast_arrays(params.mean, params.r)
        return np.asarray(mean**2 / r)

    def natural_parameter(self, params: GammaParams) -> np.ndarray:
        """Return ``eta = -r / mean``."""

        self._check_type(params)
        self._validate(params)
        mean, r = np.broadcast_arrays(params.mean, params.r)
        return np.asarray(-r / mean)

    def from_natural(self, eta: Any, r: Any) -> GammaParams:
        """Construct Gamma parameters from negative ``eta`` and shape ``r``."""

        eta_array, r_array = np.broadcast_arrays(eta, r)
        if not finite(eta_array) or np.any(eta_array >= 0):
            raise ValueError("Gamma requires finite eta < 0")
        params = GammaParams(-r_array / eta_array, r_array)
        self._validate(params)
        return params

    def shifted_params(self, params: GammaParams, natural_shift: Any) -> GammaParams:
        """Shift ``eta`` while preserving the Gamma shape."""

        self._check_type(params)
        self._validate(params)
        return self.from_natural(
            self.natural_parameter(params) + np.asarray(natural_shift),
            params.r,
        )

    def shift_coordinate(self, natural_shift: Any, params: GammaParams) -> np.ndarray:
        """Return ``xi = theta * shift / (1 - theta * shift)``."""

        self.shifted_params(params, natural_shift)
        mean, r, shift = np.broadcast_arrays(params.mean, params.r, natural_shift)
        theta = mean / r
        return np.asarray(theta * shift / (1.0 - theta * shift))

    def from_shift_coordinate(self, xi: Any, params: GammaParams) -> GammaParams:
        """Construct the shifted member with mean ``mean * (1 + xi)``."""

        self._check_type(params)
        self._validate(params)
        mean, r, xi_array = np.broadcast_arrays(params.mean, params.r, xi)
        shifted = GammaParams(mean * (1.0 + xi_array), r)
        self._validate(shifted)
        return shifted

    def shift_coefficients(
        self, natural_shift: Any, n_max: int, params: GammaParams
    ) -> np.ndarray:
        """Return normalized positive-gauge Laguerre shift coefficients."""

        self._validate_ops(params, n_max)
        degree = polynomial_degrees(n_max)
        _, r = np.broadcast_arrays(params.mean, params.r)
        xi = self.shift_coordinate(natural_shift, params)
        gamma = np.exp(
            0.5
            * (
                gammaln(r[..., None] + degree)
                - gammaln(r[..., None])
                - gammaln(degree + 1.0)
            )
        )
        return gamma * xi[..., None] ** degree

    def log_affinity(self, params1: GammaParams, params2: GammaParams) -> np.ndarray:
        """Return log affinity for two Gamma members with equal shape."""

        self._check_type(params1)
        self._check_type(params2)
        self._validate(params1)
        self._validate(params2)
        r1, r2 = same_fixed("r", params1.r, params2.r)
        mean1, mean2, r = np.broadcast_arrays(
            params1.mean, params2.mean, r1 + np.zeros_like(r2)
        )
        log_ratio = (
            np.log(2.0)
            + 0.5 * np.log(mean1)
            + 0.5 * np.log(mean2)
            - np.logaddexp(np.log(mean1), np.log(mean2))
        )
        return np.asarray(r * log_ratio)


Gamma = GammaFamily()
gamma = Gamma


if __name__ == "__main__":
    params = GammaParams(mean=np.array([2.0, 5.0]), r=3.0)
    x = np.linspace(1e-5, 30.0, 100_000)
    probabilities = Gamma.prob_grid(x, params)
    centered = Gamma.basis(params.mean, 1, params)[..., 1]
    assert np.allclose(np.trapezoid(probabilities, x, axis=-1), 1.0, atol=2e-5)
    assert np.allclose(centered, 0.0)
    assert np.all(Gamma.jacobi_coefficients(1, params)[0] > 0)
    print("Gamma checks passed")
