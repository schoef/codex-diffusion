"""Negative-binomial/Meixner NEF-QVF family."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln, xlogy

from ._family import (
    Family,
    finite,
    integer_support,
    polynomial_degrees,
    same_fixed,
)
from .jacobi import NEGATIVE_BINOMIAL
from .params import NegativeBinomialParams


class NegativeBinomialFamily(Family):
    """Negative-binomial law with shape ``r`` and Meixner basis.

    The count law uses ``c = mean / (r + mean)``. Standard Meixner
    polynomials are rephased by ``(-1)**n`` for positive Jacobi
    off-diagonals.
    """

    family_code = NEGATIVE_BINOMIAL
    params_type = NegativeBinomialParams

    def _validate(self, params: NegativeBinomialParams) -> None:
        mean = np.asarray(params.mean)
        r = np.asarray(params.r)
        if not finite(mean, r) or np.any(mean < 0) or np.any(r <= 0):
            raise ValueError("Negative binomial requires finite mean >= 0 and r > 0")

    def _validate_ops(self, params: NegativeBinomialParams, n_max: int) -> None:
        self._validate(params)
        if np.any(np.asarray(params.mean) <= 0):
            raise ValueError("The Meixner basis requires mean > 0")

    def _log_prob(self, x: np.ndarray, params: NegativeBinomialParams) -> np.ndarray:
        mean = np.asarray(params.mean)
        r = np.asarray(params.r)
        c = mean / (r + mean)
        log_success = np.log(r) - np.log(r + mean)
        with np.errstate(invalid="ignore"):
            value = (
                gammaln(x + r)
                - gammaln(r)
                - gammaln(x + 1.0)
                + r * log_success
                + xlogy(x, c)
            )
        support = integer_support(x) & (x >= 0)
        return np.where(support, value, -np.inf)

    def _ops_parameters(self, params: NegativeBinomialParams) -> tuple[Any, Any]:
        return params.mean, params.r

    def _sample(self, params: NegativeBinomialParams, size, rng) -> np.ndarray:
        """Draw negative-binomial counts.

        NumPy parameterises the law by the success probability, which is
        ``r / (r + mean)`` for the mean used here.
        """
        r = float(params.r)
        return rng.negative_binomial(r, r / (r + float(params.mean)), size=size)

    def _one_shot_sample(self, x, t, params, rng) -> np.ndarray:
        """Birth-death-immigration transition: binomial thinning, then a
        negative-binomial replenishment at the contracted parameter."""

        if np.any(~(integer_support(x) & (x >= 0))):
            raise ValueError("x must be nonnegative integers")
        r = float(params.r)
        mean = float(params.mean)
        c = mean / (mean + r)
        w = np.exp(-t)
        denominator = 1.0 - c * w
        keep = w * (1.0 - c) / denominator
        c_t = c * -np.expm1(-t) / denominator
        survivors = rng.binomial(x.astype(np.int64), keep)
        return survivors + rng.negative_binomial(r + survivors, 1.0 - c_t)

    def variance(self, params: NegativeBinomialParams) -> np.ndarray:
        """Return ``mean + mean**2 / r``."""

        self._check_type(params)
        self._validate(params)
        mean, r = np.broadcast_arrays(params.mean, params.r)
        return np.asarray(mean + mean**2 / r)

    def natural_parameter(self, params: NegativeBinomialParams) -> np.ndarray:
        """Return ``eta = log(c)`` for ``c = mean / (r + mean)``."""

        self._check_type(params)
        self._validate(params)
        mean, r = np.broadcast_arrays(params.mean, params.r)
        with np.errstate(divide="ignore"):
            return np.asarray(np.log(mean) - np.log(r + mean))

    def from_natural(self, eta: Any, r: Any) -> NegativeBinomialParams:
        """Construct parameters from negative ``eta`` and shape ``r``."""

        eta_array, r_array = np.broadcast_arrays(eta, r)
        if np.any(np.isnan(eta_array)) or np.any(eta_array >= 0):
            raise ValueError("Negative-binomial eta must satisfy eta < 0")
        c = np.exp(eta_array)
        params = NegativeBinomialParams(r_array * c / (1.0 - c), r_array)
        self._validate(params)
        return params

    def shifted_params(
        self, params: NegativeBinomialParams, natural_shift: Any
    ) -> NegativeBinomialParams:
        """Shift ``eta = log(c)`` while preserving ``r``."""

        self._check_type(params)
        self._validate(params)
        return self.from_natural(
            self.natural_parameter(params) + np.asarray(natural_shift),
            params.r,
        )

    def shift_coordinate(
        self, natural_shift: Any, params: NegativeBinomialParams
    ) -> np.ndarray:
        """Return the Meixner shift coordinate."""

        shifted = self.shifted_params(params, natural_shift)
        mean, r, shifted_mean = np.broadcast_arrays(params.mean, params.r, shifted.mean)
        c = mean / (r + mean)
        shifted_c = shifted_mean / (r + shifted_mean)
        return np.asarray((shifted_c - c) / (1.0 - shifted_c))

    def from_shift_coordinate(
        self, z: Any, params: NegativeBinomialParams
    ) -> NegativeBinomialParams:
        """Construct the shifted member from Meixner coordinate ``z``."""

        self._check_type(params)
        self._validate(params)
        mean, r, z_array = np.broadcast_arrays(params.mean, params.r, z)
        c = mean / (r + mean)
        shifted_c = (c + z_array) / (1.0 + z_array)
        shifted = NegativeBinomialParams(
            r * shifted_c / (1.0 - shifted_c),
            r,
        )
        self._validate(shifted)
        return shifted

    def shift_coefficients(
        self,
        natural_shift: Any,
        n_max: int,
        params: NegativeBinomialParams,
    ) -> np.ndarray:
        """Return normalized Meixner shift coefficients."""

        self._validate_ops(params, n_max)
        degree = polynomial_degrees(n_max)
        mean, r = np.broadcast_arrays(params.mean, params.r)
        c = mean / (r + mean)
        z = self.shift_coordinate(natural_shift, params)
        log_gamma = 0.5 * (
            gammaln(r[..., None] + degree)
            - gammaln(r[..., None])
            - gammaln(degree + 1.0)
            - degree * np.log(c[..., None])
        )
        return np.exp(log_gamma) * z[..., None] ** degree

    def log_affinity(
        self,
        params1: NegativeBinomialParams,
        params2: NegativeBinomialParams,
    ) -> np.ndarray:
        """Return log affinity for members with equal shape ``r``."""

        self._check_type(params1)
        self._check_type(params2)
        self._validate(params1)
        self._validate(params2)
        r1, r2 = same_fixed("r", params1.r, params2.r)
        mean1, mean2, r = np.broadcast_arrays(
            params1.mean, params2.mean, r1 + np.zeros_like(r2)
        )
        c1 = mean1 / (r + mean1)
        c2 = mean2 / (r + mean2)
        numerator = 0.5 * (np.log1p(-c1) + np.log1p(-c2))
        denominator = np.log1p(-np.sqrt(c1 * c2))
        return np.asarray(r * (numerator - denominator))


NegativeBinomial = NegativeBinomialFamily()
negative_binomial = NegativeBinomial


if __name__ == "__main__":
    params = NegativeBinomialParams(mean=np.array([2.0, 8.0]), r=3.0)
    counts = np.arange(150)
    probabilities = NegativeBinomial.prob_grid(counts, params)
    scalar_params = NegativeBinomialParams(mean=4.0, r=2.5)
    basis_values = NegativeBinomial.basis(counts, 3, scalar_params)
    gram = (
        basis_values.T * NegativeBinomial.prob(counts, scalar_params)
    ) @ basis_values
    assert np.allclose(probabilities.sum(axis=-1), 1.0)
    assert np.allclose(gram, np.eye(4), atol=1e-9)
    print("Negative-binomial checks passed")
