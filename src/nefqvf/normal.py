"""Normal/Hermite NEF-QVF family."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln

from ._family import Family, finite, polynomial_degrees, same_fixed
from .jacobi import NORMAL
from .params import NormalParams

_LOG_2PI = np.log(2.0 * np.pi)


class NormalFamily(Family):
    """Normal law with fixed ``sigma`` and orthonormal Hermite basis.

    The public mean is ``mu``, the natural parameter is
    ``eta = mu / sigma**2``, and the positive-gauge shift coordinate is
    ``xi = sigma * natural_shift``.
    """

    family_code = NORMAL
    params_type = NormalParams

    def _validate(self, params: NormalParams) -> None:
        mean = np.asarray(params.mean)
        sigma = np.asarray(params.sigma)
        if not finite(mean, sigma) or np.any(sigma <= 0):
            raise ValueError("Normal requires finite mean and sigma > 0")

    def _validate_ops(self, params: NormalParams, n_max: int) -> None:
        self._validate(params)

    def _log_prob(self, x: np.ndarray, params: NormalParams) -> np.ndarray:
        mean = np.asarray(params.mean)
        sigma = np.asarray(params.sigma)
        with np.errstate(over="ignore", invalid="ignore"):
            value = -0.5 * ((x - mean) / sigma) ** 2 - np.log(sigma) - 0.5 * _LOG_2PI
        return np.where(np.isfinite(x), value, -np.inf)

    def _ops_parameters(self, params: NormalParams) -> tuple[Any, Any]:
        return params.mean, params.sigma

    def variance(self, params: NormalParams) -> np.ndarray:
        """Return ``sigma**2`` with the full parameter batch shape."""

        self._check_type(params)
        self._validate(params)
        mean, sigma = np.broadcast_arrays(params.mean, params.sigma)
        return np.asarray(sigma**2 + np.zeros_like(mean))

    def natural_parameter(self, params: NormalParams) -> np.ndarray:
        """Return ``eta = mean / sigma**2``."""

        self._check_type(params)
        self._validate(params)
        mean, sigma = np.broadcast_arrays(params.mean, params.sigma)
        return np.asarray(mean / sigma**2)

    def from_natural(self, eta: Any, sigma: Any) -> NormalParams:
        """Construct Normal parameters from ``eta`` and fixed ``sigma``."""

        eta_array, sigma_array = np.broadcast_arrays(eta, sigma)
        params = NormalParams(eta_array * sigma_array**2, sigma_array)
        self._validate(params)
        return params

    def shifted_params(self, params: NormalParams, natural_shift: Any) -> NormalParams:
        """Shift ``eta`` while preserving ``sigma``."""

        self._check_type(params)
        self._validate(params)
        return self.from_natural(
            self.natural_parameter(params) + np.asarray(natural_shift),
            params.sigma,
        )

    def shift_coordinate(self, natural_shift: Any, params: NormalParams) -> np.ndarray:
        """Return the Hermite shift coordinate ``xi = sigma * shift``."""

        shifted = self.shifted_params(params, natural_shift)
        _, sigma, shift = np.broadcast_arrays(shifted.mean, params.sigma, natural_shift)
        return np.asarray(sigma * shift)

    def from_shift_coordinate(self, xi: Any, params: NormalParams) -> NormalParams:
        """Construct the shifted member from a Hermite coordinate ``xi``."""

        self._check_type(params)
        self._validate(params)
        mean, sigma, xi_array = np.broadcast_arrays(params.mean, params.sigma, xi)
        shifted = NormalParams(mean + sigma * xi_array, sigma)
        self._validate(shifted)
        return shifted

    def shift_coefficients(
        self, natural_shift: Any, n_max: int, params: NormalParams
    ) -> np.ndarray:
        """Return ``xi**n / sqrt(n!)`` through degree ``n_max``."""

        self._validate_ops(params, n_max)
        degree = polynomial_degrees(n_max)
        xi = self.shift_coordinate(natural_shift, params)
        gamma = np.exp(-0.5 * gammaln(degree + 1.0))
        return gamma * xi[..., None] ** degree

    def log_affinity(self, params1: NormalParams, params2: NormalParams) -> np.ndarray:
        """Return log affinity for two members with equal ``sigma``."""

        self._check_type(params1)
        self._check_type(params2)
        self._validate(params1)
        self._validate(params2)
        sigma1, sigma2 = same_fixed("sigma", params1.sigma, params2.sigma)
        mean1, mean2, sigma = np.broadcast_arrays(
            params1.mean, params2.mean, sigma1 + np.zeros_like(sigma2)
        )
        return np.asarray(-((mean1 - mean2) ** 2) / (8.0 * sigma**2))


Normal = NormalFamily()
normal = Normal


if __name__ == "__main__":
    params = NormalParams(mean=np.array([-1.0, 1.0]), sigma=2.0)
    values = Normal.log_prob(np.array([-1.0, 1.0]), params)
    grid = Normal.prob_grid(np.linspace(-12.0, 12.0, 20_001), params)
    basis_values = Normal.basis(params.mean, 2, params)
    assert values.shape == (2,)
    assert np.allclose(np.trapezoid(grid, dx=24.0 / 20_000, axis=-1), 1.0, atol=1e-7)
    assert np.allclose(basis_values[..., 1], 0.0)
    print("Normal checks passed")
