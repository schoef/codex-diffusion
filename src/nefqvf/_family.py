"""Common public mechanics for one-dimensional NEF-QVF families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .broadcasting import (
    broadcast_parameter_values,
    replace_parameter_values,
)
from .jacobi import basis, basis_dot, jacobi_coefficients
from .linearization import linearization_tensor_from_jacobi


class Family(ABC):
    """Reference NumPy interface shared by all six family implementations.

    Public methods validate once, preserve NumPy broadcasting, and delegate
    family-specific formulas to compact private methods. Family modules export
    stateless singleton instances of these implementations.
    """

    family_code: int
    params_type: type

    def _check_type(self, params: Any) -> None:
        """Reject parameter records belonging to a different family."""

        if not isinstance(params, self.params_type):
            raise TypeError(
                f"expected {self.params_type.__name__}, got {type(params).__name__}"
            )

    @abstractmethod
    def _validate(self, params: Any) -> None:
        """Validate the probability-law parameter domain."""

    @abstractmethod
    def _validate_ops(self, params: Any, n_max: int) -> None:
        """Validate the stricter domain of the orthogonal basis."""

    @abstractmethod
    def _log_prob(self, x: np.ndarray, params: Any) -> np.ndarray:
        """Evaluate a vectorized log density after public validation."""

    @abstractmethod
    def _ops_parameters(self, params: Any) -> tuple[Any, Any]:
        """Return the mean and family-specific fixed Jacobi parameter."""

    def log_prob(self, x: Any, params: Any) -> np.ndarray:
        """Evaluate with ordinary NumPy broadcasting.

        The result shape is the broadcast shape of ``x`` and every parameter
        field.
        """
        self._check_type(params)
        self._validate(params)
        return np.asarray(self._log_prob(np.asarray(x), params))

    def prob(self, x: Any, params: Any) -> np.ndarray:
        """Evaluate probabilities or densities with paired broadcasting."""

        return np.exp(self.log_prob(x, params))

    def log_prob_grid(
        self, x: Any, params: Any, *, chunk_size: int | None = None
    ) -> np.ndarray:
        """Evaluate the outer product of parameter batches and observations.

        If the parameter fields broadcast to ``batch_shape``, the result shape
        is ``batch_shape + np.shape(x)``. ``chunk_size`` limits the number of
        flattened observations evaluated at once.
        """
        self._check_type(params)
        self._validate(params)
        x_array = np.asarray(x)
        names, values, batch_shape = broadcast_parameter_values(params)
        observation_shape = x_array.shape

        if chunk_size is None or x_array.ndim == 0:
            parameter_views = tuple(
                value.reshape(batch_shape + (1,) * x_array.ndim) for value in values
            )
            grid_params = replace_parameter_values(params, names, parameter_views)
            x_view = x_array.reshape((1,) * len(batch_shape) + observation_shape)
            return np.asarray(self._log_prob(x_view, grid_params))

        if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        flattened_x = x_array.ravel()
        result = np.empty(batch_shape + (flattened_x.size,), dtype=float)
        parameter_views = tuple(value.reshape(batch_shape + (1,)) for value in values)
        grid_params = replace_parameter_values(params, names, parameter_views)
        for start in range(0, flattened_x.size, int(chunk_size)):
            stop = min(start + int(chunk_size), flattened_x.size)
            x_view = flattened_x[start:stop].reshape(
                (1,) * len(batch_shape) + (stop - start,)
            )
            result[..., start:stop] = self._log_prob(x_view, grid_params)
        return result.reshape(batch_shape + observation_shape)

    def prob_grid(
        self, x: Any, params: Any, *, chunk_size: int | None = None
    ) -> np.ndarray:
        """Evaluate an explicit parameter-by-observation probability grid."""

        return np.exp(self.log_prob_grid(x, params, chunk_size=chunk_size))

    def mean(self, params: Any) -> np.ndarray:
        """Return the broadcast public mean parameter."""

        self._check_type(params)
        self._validate(params)
        names, values, _ = broadcast_parameter_values(params)
        return np.asarray(values[names.index("mean")])

    @abstractmethod
    def variance(self, params: Any) -> np.ndarray:
        """Return the analytic variance."""

    @abstractmethod
    def natural_parameter(self, params: Any) -> np.ndarray:
        """Convert public mean parameters to the natural parameter ``eta``."""

    @abstractmethod
    def from_natural(self, eta: Any, fixed_parameters: Any) -> Any:
        """Construct public parameters from ``eta`` and fixed parameters."""

    @abstractmethod
    def shifted_params(self, params: Any, natural_shift: Any) -> Any:
        """Return the family member at ``eta + natural_shift``."""

    @abstractmethod
    def shift_coordinate(self, natural_shift: Any, params: Any) -> np.ndarray:
        """Return the positive-Jacobi-gauge shift coordinate ``xi``."""

    @abstractmethod
    def from_shift_coordinate(self, xi: Any, params: Any) -> Any:
        """Invert ``xi`` at the supplied baseline parameters."""

    @abstractmethod
    def shift_coefficients(
        self, natural_shift: Any, n_max: int, params: Any
    ) -> np.ndarray:
        """Return probability-ratio coefficients ``gamma_n xi**n``."""

    @abstractmethod
    def log_affinity(self, params1: Any, params2: Any) -> np.ndarray:
        """Return the log Hellinger affinity between family members."""

    def affinity(self, params1: Any, params2: Any) -> np.ndarray:
        """Return the Hellinger affinity between family members."""

        return np.exp(self.log_affinity(params1, params2))

    def jacobi_coefficients(self, n: Any, params: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return positive-gauge recurrence coefficients ``(a_n, b_n)``."""

        self._check_type(params)
        self._validate_ops(params, 0)
        mean, fixed = self._ops_parameters(params)
        return jacobi_coefficients(self.family_code, n, mean, fixed)

    def basis(
        self, x: Any, n_max: int, params: Any, *, grid: bool = False
    ) -> np.ndarray:
        """Evaluate the orthonormal basis through degree ``n_max``.

        The final result axis is polynomial degree. Paired evaluation uses
        ordinary broadcasting; ``grid=True`` prepends the parameter batch
        shape to the observation shape.
        """
        self._check_type(params)
        self._validate_ops(params, n_max)
        mean, fixed = self._ops_parameters(params)
        return basis(self.family_code, x, n_max, mean, fixed, grid=grid)

    def basis_dot(
        self, x: Any, coefficients: Any, params: Any, *, grid: bool = False
    ) -> np.ndarray:
        """Evaluate a basis expansion with a Clenshaw recurrence.

        ``coefficients[..., n]`` multiplies ``phi_n``. Under ``grid=True``,
        coefficient batch dimensions align with the parameter batch.
        """
        self._check_type(params)
        coefficient_array = np.asarray(coefficients)
        if coefficient_array.ndim == 0:
            raise ValueError("coefficients must have a final degree axis")
        self._validate_ops(params, coefficient_array.shape[-1] - 1)
        mean, fixed = self._ops_parameters(params)
        return basis_dot(
            self.family_code,
            x,
            coefficient_array,
            mean,
            fixed,
            grid=grid,
        )

    def _maximum_ops_degree(self, params: Any) -> int | None:
        """Return a finite basis cap, or ``None`` for infinite systems."""

        return None

    def linearization_tensor(self, n_max: int, params: Any) -> np.ndarray:
        """Return ``Lambda[m, n, k] = E[phi_m phi_n phi_k]``.

        This recurrence-based implementation currently accepts scalar family
        parameters. The returned tensor has shape
        ``(n_max + 1, n_max + 1, n_max + 1)``.
        """
        self._check_type(params)
        self._validate_ops(params, n_max)
        _, _, batch_shape = broadcast_parameter_values(params)
        if batch_shape:
            raise ValueError("linearization_tensor currently requires scalar params")

        maximum_degree = self._maximum_ops_degree(params)
        workspace_degree = 2 * n_max
        if maximum_degree is not None:
            workspace_degree = min(workspace_degree, maximum_degree)

        degree = np.arange(workspace_degree + 1)
        a, b = self.jacobi_coefficients(degree, params)
        return linearization_tensor_from_jacobi(a, b, n_max)


def finite(*values: Any) -> bool:
    """Return whether every scalar or array value is finite."""

    return all(np.all(np.isfinite(np.asarray(value))) for value in values)


def same_fixed(name: str, value1: Any, value2: Any) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast and require equal fixed parameters for affinity formulas."""

    first, second = np.broadcast_arrays(np.asarray(value1), np.asarray(value2))
    if np.any(first != second):
        raise ValueError(f"affinity requires equal {name}")
    return first, second


def integer_support(x: np.ndarray) -> np.ndarray:
    """Return a mask selecting finite integer-valued observations."""

    return np.isfinite(x) & (x == np.floor(x))


def polynomial_degrees(n_max: int) -> np.ndarray:
    """Validate ``n_max`` and return degrees zero through ``n_max``."""

    if not isinstance(n_max, (int, np.integer)) or n_max < 0:
        raise ValueError("n_max must be a nonnegative integer")
    return np.arange(n_max + 1)
