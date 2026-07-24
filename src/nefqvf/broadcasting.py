"""Broadcasting helpers shared by probability and polynomial evaluators."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import numpy as np


def parameter_values(params: Any) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    """Extract parameter field names and array views in dataclass order."""

    names = tuple(field.name for field in fields(params))
    values = tuple(np.asarray(getattr(params, name)) for name in names)
    return names, values


def broadcast_parameter_values(
    params: Any,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], tuple[int, ...]]:
    """Broadcast every field of a parameter dataclass to one batch shape."""

    names, values = parameter_values(params)
    shape = np.broadcast_shapes(*(value.shape for value in values))
    broadcasted = tuple(np.broadcast_to(value, shape) for value in values)
    return names, broadcasted, shape


def replace_parameter_values(
    params: Any, names: tuple[str, ...], values: tuple[np.ndarray, ...]
) -> Any:
    """Return a parameter record with selected fields replaced by array views."""

    return replace(params, **dict(zip(names, values, strict=True)))


def paired_values(*values: Any) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    """Broadcast values for elementwise, or paired, evaluation."""

    arrays = tuple(np.asarray(value) for value in values)
    shape = np.broadcast_shapes(*(array.shape for array in arrays))
    return tuple(np.broadcast_to(array, shape) for array in arrays), shape


def outer_values(
    x: Any, *parameters: Any
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...], tuple[int, ...]]:
    """Reshape parameters and observations for an explicit outer grid.

    Parameter axes precede all observation axes in the returned views.
    Broadcasting the views therefore produces ``batch_shape + x.shape``.
    """

    x_array = np.asarray(x)
    parameter_arrays = tuple(np.asarray(value) for value in parameters)
    batch_shape = np.broadcast_shapes(*(value.shape for value in parameter_arrays))
    observation_shape = x_array.shape

    x_view = x_array.reshape((1,) * len(batch_shape) + observation_shape)
    parameter_views = tuple(
        np.broadcast_to(value, batch_shape).reshape(
            batch_shape + (1,) * len(observation_shape)
        )
        for value in parameter_arrays
    )
    return x_view, parameter_views, batch_shape, observation_shape
