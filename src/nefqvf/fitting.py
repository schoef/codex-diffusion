"""Recovering an amplitude from density-ratio coefficients.

The Born parametrisation writes a law as ``p = p_ref h^2`` with
``h = sum_n c_n phi_n``, so the density ratio has coefficients

    R_k(c) = c' Phi_k c,    [Phi_k]_{mn} = Lambda_{mnk},

quadratic in the amplitude. Recovering ``c`` from measured ``R`` is therefore a
nonlinear inverse problem, and normalisation is the sphere ``||c|| = 1`` because
``Lambda_{mn0} = delta_{mn}``.

This module holds the parts of that problem which depend on nothing but the
family: the product matrices, the coefficient map, and a Riemannian
Levenberg-Marquardt solver on the sphere. Which baselines and which targets are
worth fitting is an application question and lives outside the package.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import null_space


def product_matrices(family: Any, baseline: Any, degree: int) -> np.ndarray:
    """Return ``Phi[k] = Lambda[:K+1, :K+1, k]`` for ``k = 0 ... 2K``."""

    tensor = family.linearization_tensor(2 * degree, baseline)
    return np.ascontiguousarray(
        np.transpose(tensor[: degree + 1, : degree + 1, :], (2, 0, 1))
    )


def ratio_coefficients(coefficients: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Return ``R_k(c) = c^T Phi_k c``."""

    return phi.reshape(phi.shape[0], -1) @ np.outer(coefficients, coefficients).ravel()


# ------------------------------------------------------------------- fitting --
def fit_amplitude(
    phi: np.ndarray,
    target: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    tau: float = 0.0,
    penalty: np.ndarray | None = None,
    k_max: int | None = None,
    max_iterations: int = 200,
    tolerance: float = 1e-13,
) -> dict[str, Any]:
    """Minimise the coefficient objective on the unit sphere by Riemannian LM.

    Degree zero is excluded from the residual: ``R_0(c) = ||c||^2`` equals one
    identically on the sphere, so it carries no information and would only add a
    null row to the Jacobian.
    """
    degree = phi.shape[1] - 1
    penalty = (
        np.diag(np.arange(degree + 1, dtype=float))
        if penalty is None
        else np.diag(np.asarray(penalty, dtype=float))
    )
    highest = phi.shape[0] - 1 if k_max is None else int(k_max)
    active = np.arange(1, highest + 1)
    phi_active = phi[active]
    target_active = np.asarray(target, dtype=float)[active]
    weight_matrix = np.eye(active.size) if weight is None else np.asarray(weight)

    c = np.zeros(degree + 1) if initial is None else np.array(initial, dtype=float)
    if initial is None:
        c[0] = 1.0
    c = c / np.linalg.norm(c)

    def residual(vec: np.ndarray) -> np.ndarray:
        rho = np.outer(vec, vec)
        return phi_active.reshape(phi_active.shape[0], -1) @ rho.ravel() - target_active

    def objective(vec: np.ndarray) -> float:
        r = residual(vec)
        return float(0.5 * r @ weight_matrix @ r + 0.5 * tau * vec @ penalty @ vec)

    mu = 1e-3
    value = objective(c)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        r = residual(c)
        rows, n, _ = phi_active.shape
        jacobian = 2.0 * (phi_active.reshape(rows * n, n) @ c).reshape(rows, n)
        tangent = null_space(c[None, :])

        jb = jacobian @ tangent
        hessian = jb.T @ weight_matrix @ jb + tau * tangent.T @ penalty @ tangent
        gradient = jb.T @ weight_matrix @ r + tau * tangent.T @ penalty @ c
        if np.linalg.norm(gradient) < tolerance:
            break

        accepted = False
        for _ in range(40):
            step = np.linalg.solve(hessian + mu * np.eye(hessian.shape[0]), -gradient)
            trial = c + tangent @ step
            trial = trial / np.linalg.norm(trial)
            if trial[0] < 0.0:
                trial = -trial
            trial_value = objective(trial)
            if trial_value < value:
                c, value, accepted = trial, trial_value, True
                mu = max(mu * 0.3, 1e-14)
                break
            mu *= 3.0
        if not accepted:
            break

    # the curvature Gauss-Newton discards, relative to the part it keeps
    r = residual(c)
    rows, n, _ = phi_active.shape
    jacobian = 2.0 * (phi_active.reshape(rows * n, n) @ c).reshape(rows, n)
    kept = jacobian.T @ weight_matrix @ jacobian
    discarded = 2.0 * np.tensordot(weight_matrix @ r, phi_active, axes=1)
    return {
        "coefficients": c,
        "objective": value,
        "iterations": iterations,
        "residual_norm": float(np.linalg.norm(r)),
        "curvature_ratio": float(
            np.linalg.norm(discarded, 2) / max(np.linalg.norm(kept, 2), 1e-300)
        ),
    }
