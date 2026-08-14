"""Binomial/Krawtchouk NEF-QVF family."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit, gammaln, xlog1py, xlogy

from ._family import (
    Family,
    finite,
    integer_support,
    polynomial_degrees,
    same_fixed,
)
from .jacobi import BINOMIAL
from .params import BinomialParams


class BinomialFamily(Family):
    """Binomial law with fixed trials ``N`` and Krawtchouk basis.

    The standard Krawtchouk polynomials are rephased by ``(-1)**n`` to obtain
    the package's positive-positive off-diagonal convention. The basis terminates at degree ``N``.
    """

    family_code = BINOMIAL
    params_type = BinomialParams

    def _validate(self, params: BinomialParams) -> None:
        mean = np.asarray(params.mean)
        trials = np.asarray(params.N)
        if (
            not finite(mean, trials)
            or np.any(trials < 0)
            or np.any(trials != np.floor(trials))
            or np.any(mean < 0)
            or np.any(mean > trials)
        ):
            raise ValueError(
                "Binomial requires integer N >= 0 and finite 0 <= mean <= N"
            )

    def _validate_ops(self, params: BinomialParams, n_max: int) -> None:
        self._validate(params)
        mean, trials = np.broadcast_arrays(params.mean, params.N)
        if np.any(trials < 1) or np.any(mean <= 0) or np.any(mean >= trials):
            raise ValueError("The Krawtchouk basis requires N >= 1 and 0 < mean < N")
        if np.any(n_max > trials):
            raise ValueError("The Krawtchouk basis terminates at degree N")

    def _log_prob(self, x: np.ndarray, params: BinomialParams) -> np.ndarray:
        mean = np.asarray(params.mean)
        trials = np.asarray(params.N)
        probability = np.divide(
            mean,
            trials,
            out=np.zeros(np.broadcast_shapes(mean.shape, trials.shape)),
            where=trials != 0,
        )
        with np.errstate(invalid="ignore"):
            value = (
                gammaln(trials + 1.0)
                - gammaln(x + 1.0)
                - gammaln(trials - x + 1.0)
                + xlogy(x, probability)
                + xlog1py(trials - x, -probability)
            )
        support = integer_support(x) & (x >= 0) & (x <= trials)
        return np.where(support, value, -np.inf)

    def _ops_parameters(self, params: BinomialParams) -> tuple[Any, Any]:
        return params.mean, params.N

    def _sample(self, params: BinomialParams, size, rng) -> np.ndarray:
        """Draw binomial counts at ``N`` trials and success rate ``mean / N``."""

        trials = int(params.N)
        return rng.binomial(trials, float(params.mean) / trials, size=size)

    def variance(self, params: BinomialParams) -> np.ndarray:
        """Return ``mean * (1 - mean / N)``."""

        self._check_type(params)
        self._validate(params)
        mean, trials = np.broadcast_arrays(params.mean, params.N)
        probability = np.divide(
            mean, trials, out=np.zeros_like(mean, dtype=float), where=trials != 0
        )
        return np.asarray(mean * (1.0 - probability))

    def natural_parameter(self, params: BinomialParams) -> np.ndarray:
        """Return the Bernoulli log odds ``log(p / (1 - p))``."""

        self._check_type(params)
        self._validate(params)
        mean, trials = np.broadcast_arrays(params.mean, params.N)
        probability = np.divide(
            mean, trials, out=np.zeros_like(mean, dtype=float), where=trials != 0
        )
        with np.errstate(divide="ignore"):
            return np.asarray(np.log(probability) - np.log1p(-probability))

    def from_natural(self, eta: Any, N: Any) -> BinomialParams:
        """Construct Binomial parameters from log odds and trials ``N``."""

        eta_array, trials = np.broadcast_arrays(eta, N)
        if np.any(np.isnan(eta_array)):
            raise ValueError("Binomial eta cannot be NaN")
        params = BinomialParams(trials * expit(eta_array), trials)
        self._validate(params)
        return params

    def shifted_params(
        self, params: BinomialParams, natural_shift: Any
    ) -> BinomialParams:
        """Shift the log odds while preserving ``N``."""

        self._check_type(params)
        self._validate(params)
        return self.from_natural(
            self.natural_parameter(params) + np.asarray(natural_shift),
            params.N,
        )

    def shift_coordinate(
        self, natural_shift: Any, params: BinomialParams
    ) -> np.ndarray:
        """Return the Krawtchouk shift coordinate."""

        self.shifted_params(params, natural_shift)
        mean, trials, shift = np.broadcast_arrays(params.mean, params.N, natural_shift)
        probability = mean / trials
        exponential_shift = np.exp(shift)
        denominator = 1.0 - probability + probability * exponential_shift
        return np.asarray(probability * (exponential_shift - 1.0) / denominator)

    def from_shift_coordinate(self, z: Any, params: BinomialParams) -> BinomialParams:
        """Construct the shifted member from Krawtchouk coordinate ``z``."""

        self._check_type(params)
        self._validate(params)
        mean, trials, z_array = np.broadcast_arrays(params.mean, params.N, z)
        probability = mean / trials
        shifted_probability = probability + (1.0 - probability) * z_array
        shifted = BinomialParams(trials * shifted_probability, trials)
        self._validate(shifted)
        return shifted

    def shift_coefficients(
        self, natural_shift: Any, n_max: int, params: BinomialParams
    ) -> np.ndarray:
        """Return normalized Krawtchouk coefficients through degree ``N``."""

        self._validate_ops(params, n_max)
        degree = polynomial_degrees(n_max)
        mean, trials = np.broadcast_arrays(params.mean, params.N)
        probability = mean / trials
        complement = 1.0 - probability
        z = self.shift_coordinate(natural_shift, params)
        log_gamma = 0.5 * (
            gammaln(trials[..., None] + 1.0)
            - gammaln(degree + 1.0)
            - gammaln(trials[..., None] - degree + 1.0)
            + degree * np.log(complement[..., None] / probability[..., None])
        )
        return np.exp(log_gamma) * z[..., None] ** degree

    def _maximum_ops_degree(self, params: BinomialParams) -> int:
        """Return the finite Krawtchouk basis degree ``N``."""

        trials = np.asarray(params.N)
        if trials.ndim != 0:
            raise ValueError("linearization_tensor currently requires scalar N")
        return int(trials)

    def log_affinity(
        self, params1: BinomialParams, params2: BinomialParams
    ) -> np.ndarray:
        """Return log affinity for members with the same trial count."""

        self._check_type(params1)
        self._check_type(params2)
        self._validate(params1)
        self._validate(params2)
        trials1, trials2 = same_fixed("N", params1.N, params2.N)
        mean1, mean2, trials = np.broadcast_arrays(
            params1.mean, params2.mean, trials1 + np.zeros_like(trials2)
        )
        p1 = np.divide(
            mean1, trials, out=np.zeros_like(mean1, dtype=float), where=trials != 0
        )
        p2 = np.divide(
            mean2, trials, out=np.zeros_like(mean2, dtype=float), where=trials != 0
        )
        base = np.sqrt(p1 * p2) + np.sqrt((1.0 - p1) * (1.0 - p2))
        with np.errstate(divide="ignore", invalid="ignore"):
            value = trials * np.log(base)
        return np.where(trials == 0, 0.0, value)


Binomial = BinomialFamily()
binomial = Binomial


if __name__ == "__main__":
    params = BinomialParams(mean=3.0, N=10)
    counts = np.arange(11)
    probabilities = Binomial.prob(counts, params)
    basis_values = Binomial.basis(counts, 5, params)
    gram = (basis_values.T * probabilities) @ basis_values
    assert np.allclose(probabilities.sum(), 1.0)
    assert np.allclose(gram, np.eye(6), atol=1e-11)
    assert np.all(Binomial.jacobi_coefficients(np.arange(6), params)[0][1:] > 0)
    print("Binomial checks passed")
