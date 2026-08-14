"""Poisson/Charlier NEF-QVF family."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln, xlogy

from ._family import Family, finite, integer_support, polynomial_degrees
from .jacobi import POISSON
from .params import PoissonParams


class PoissonFamily(Family):
    """Poisson law with an orthonormal Charlier basis.

    The public mean is ``mu``, the natural parameter is ``log(mu)``, and the
    shift coordinate is ``z = exp(shift) - 1``.
    """

    family_code = POISSON
    params_type = PoissonParams

    def _validate(self, params: PoissonParams) -> None:
        mean = np.asarray(params.mean)
        if not finite(mean) or np.any(mean < 0):
            raise ValueError("Poisson requires finite mean >= 0")

    def _validate_ops(self, params: PoissonParams, n_max: int) -> None:
        self._validate(params)
        if np.any(np.asarray(params.mean) <= 0):
            raise ValueError("The Charlier basis requires mean > 0")

    def _log_prob(self, x: np.ndarray, params: PoissonParams) -> np.ndarray:
        mean = np.asarray(params.mean)
        with np.errstate(invalid="ignore"):
            value = xlogy(x, mean) - mean - gammaln(x + 1.0)
        support = integer_support(x) & (x >= 0)
        return np.where(support, value, -np.inf)

    def _ops_parameters(self, params: PoissonParams) -> tuple[Any, Any]:
        return params.mean, 0.0

    def _sample(self, params: PoissonParams, size, rng) -> np.ndarray:
        """Draw Poisson counts."""

        return rng.poisson(float(params.mean), size=size)

    def variance(self, params: PoissonParams) -> np.ndarray:
        """Return the Poisson variance, equal to its mean."""

        return self.mean(params)

    def natural_parameter(self, params: PoissonParams) -> np.ndarray:
        """Return ``eta = log(mean)``, including ``-inf`` at zero."""

        self._check_type(params)
        self._validate(params)
        with np.errstate(divide="ignore"):
            return np.asarray(np.log(params.mean))

    def from_natural(
        self, eta: Any, fixed_parameters: Any | None = None
    ) -> PoissonParams:
        """Construct Poisson parameters from ``eta``."""

        if fixed_parameters is not None:
            raise ValueError("Poisson has no fixed parameter")
        eta_array = np.asarray(eta)
        if np.any(np.isnan(eta_array)) or np.any(eta_array == np.inf):
            raise ValueError("Poisson requires finite eta or eta = -inf")
        params = PoissonParams(np.exp(eta_array))
        self._validate(params)
        return params

    def shifted_params(
        self, params: PoissonParams, natural_shift: Any
    ) -> PoissonParams:
        """Shift ``eta`` and therefore multiply the mean exponentially."""

        self._check_type(params)
        self._validate(params)
        return self.from_natural(
            self.natural_parameter(params) + np.asarray(natural_shift)
        )

    def shift_coordinate(self, natural_shift: Any, params: PoissonParams) -> np.ndarray:
        """Return the Charlier shift coordinate ``expm1(natural_shift)``."""

        self.shifted_params(params, natural_shift)
        return np.asarray(np.expm1(natural_shift))

    def from_shift_coordinate(self, z: Any, params: PoissonParams) -> PoissonParams:
        """Construct the shifted member with mean ``mean * (1 + z)``."""

        self._check_type(params)
        self._validate(params)
        mean, z_array = np.broadcast_arrays(params.mean, z)
        shifted = PoissonParams(mean * (1.0 + z_array))
        self._validate(shifted)
        return shifted

    def shift_coefficients(
        self, natural_shift: Any, n_max: int, params: PoissonParams
    ) -> np.ndarray:
        """Return normalized Charlier probability-ratio coefficients."""

        self._validate_ops(params, n_max)
        degree = polynomial_degrees(n_max)
        mean = self.mean(params)
        z = self.shift_coordinate(natural_shift, params)
        gamma = np.exp(
            0.5 * degree * np.log(mean[..., None]) - 0.5 * gammaln(degree + 1.0)
        )
        return gamma * z[..., None] ** degree

    def log_affinity(
        self, params1: PoissonParams, params2: PoissonParams
    ) -> np.ndarray:
        """Return the closed-form Poisson log affinity."""

        self._check_type(params1)
        self._check_type(params2)
        self._validate(params1)
        self._validate(params2)
        mean1, mean2 = np.broadcast_arrays(params1.mean, params2.mean)
        return np.asarray(-0.5 * (np.sqrt(mean1) - np.sqrt(mean2)) ** 2)


Poisson = PoissonFamily()
poisson = Poisson


if __name__ == "__main__":
    params = PoissonParams(mean=np.array([2.0, 8.0]))
    counts = np.arange(50)
    probabilities = Poisson.prob_grid(counts, params)
    basis_values = Poisson.basis(counts, 3, PoissonParams(4.0))
    weights = Poisson.prob(counts, PoissonParams(4.0))
    weighted_gram = (basis_values.T * weights) @ basis_values
    assert probabilities.shape == (2, 50)
    assert np.allclose(probabilities.sum(axis=-1), 1.0)
    assert np.allclose(weighted_gram, np.eye(4), atol=1e-10)
    print("Poisson checks passed")
