"""OPS product linearization from Jacobi recurrence coefficients."""

from __future__ import annotations

import numpy as np


def linearization_tensor_from_jacobi(
    a: np.ndarray,
    b: np.ndarray,
    n_max: int,
) -> np.ndarray:
    """Return ``Lambda[m, n, k]`` for degrees through ``n_max``.

    The calculation uses only the Jacobi recurrence. Internally it retains all
    degrees needed by products before returning the requested tensor block.
    ``a`` and ``b`` must therefore extend to at least the largest product
    degree required by the caller, normally ``2 * n_max``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("a and b must be one-dimensional arrays of equal length")
    if not isinstance(n_max, (int, np.integer)) or n_max < 0:
        raise ValueError("n_max must be a nonnegative integer")
    if a.size < n_max + 1:
        raise ValueError("Jacobi coefficient arrays are too short")
    if not np.isclose(a[0], 0.0):
        raise ValueError("the Jacobi recurrence requires a_0 = 0")
    if np.any(a[1:] <= 0.0):
        raise ValueError("off-diagonal Jacobi coefficients must be positive")

    workspace_degree = a.size - 1
    # full[m, n, :] stores the basis coefficients of phi_m * phi_n.
    full = np.zeros((n_max + 1, n_max + 1, a.size), dtype=float)
    for n in range(n_max + 1):
        full[0, n, n] = 1.0
        for m in range(n_max):
            current = full[m, n]
            x_current = b * current
            x_current[1:] += a[1:] * current[:-1]
            x_current[:-1] += a[1:] * current[1:]

            previous = 0.0 if m == 0 else full[m - 1, n]
            following = (x_current - b[m] * current - a[m] * previous) / a[m + 1]

            # Products at this step cannot exceed degree n + m + 1. Removing
            # roundoff beyond that support keeps later recurrence steps clean.
            support_end = min(n + m + 1, workspace_degree)
            following[support_end + 1 :] = 0.0
            full[m + 1, n] = following

    return full[:, :, : n_max + 1]
