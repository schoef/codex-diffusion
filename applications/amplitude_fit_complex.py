"""Test 5: does the complex amplitude reach the convex optimum?

The real rank-one fit minimises over the matrices ``cc^T``, which are the
rank-one *boundary* of the feasible set ``rho >= 0``, ``Tr rho = 1``. The convex
relaxation minimises over the whole set and therefore gives a lower bound. The
claim under test is that the bound is attained by an amplitude once the
coefficients are allowed to be complex: a nonnegative polynomial in one variable
is a sum of at most two squares, and one complex amplitude carries exactly two.

Three quantities are computed for the same data and compared.

* ``J_real`` -- the fit of Section "Fitting a law by its amplitude", the
  Levenberg-Marquardt method of ``amplitude_fit_recovery`` on the real sphere.
* ``J_complex`` -- the same method on the complex sphere, in the real
  coordinates ``(a, b)`` with the norm direction ``c`` and the phase direction
  ``ic`` removed from the tangent space.
* ``J_relaxed`` -- the convex problem, by accelerated projected gradient on the
  states. Its accuracy is certified rather than assumed: for any feasible
  ``rho``, convexity bounds the distance to the optimum by

      Tr(M rho) - lambda_min(M),      M = sum_k (W r)_k Phi_k,

  and that is the same matrix and the same eigenvalue that decide optimality of
  a real fit, so the solver and the note's test share their only nontrivial
  step.

The eigenvalue test is the practical content: at a converged real fit, one
eigendecomposition of ``M`` says whether the fit is already the convex optimum
(``mu = lambda_min``) and, when it is not, hands over the direction ``v`` in
which the imaginary part should be grown.

Growing it is not enough on its own. The complex problem is a reparametrisation
of a convex one but is not itself convex, and the local optimiser does get stuck
-- on a well separated mixture it stalls many orders of magnitude above the
optimum. So the correspondence is also used constructively, in the direction the
proposition is proved: the convex optimum is *factorised* into an amplitude
through the roots of its law, which lands on the optimum without searching for
it. That route is what makes the equality of the two optima visible numerically.

Targets matter here in a way they do not for recovery. A shifted member of the
baseline family *is* one square, so the real fit should already be optimal and
the gap should be numerical noise. An equal mixture of two oppositely shifted
members is not, and it is the case where the second square should pay.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import null_space

from applications.amplitude_fit_degree import (
    Target,
    integrate,
    mixture_target,
    reference_coefficients,
    shifted_target,
    target_grid,
    truncated_target,
)
from applications.amplitude_fit_recovery import (
    FAMILY_NAMES,
    TARGETS,
    empirical_coefficients,
    fit_amplitude,
)
from applications.baseline_matching import separation_for_gap

DEFAULT_DEGREE = 6
DEFAULT_SEED = 11
DEFAULT_GAP = 2.0
# the step in the imaginary direction that leaves a stationary real fit; small
# enough to stay in the quadratic regime of the descent expansion, large enough
# not to be undone by the first Levenberg-Marquardt step
DEFAULT_KICK = 1e-2
DEFAULT_RELAXATION_ITERATIONS = 200000
DEFAULT_RELAXATION_TOLERANCE = 1e-14
FIGURE_SUBDIRECTORY = "complex_amplitude"


# ------------------------------------------------------------------- geometry --
def _active(phi: np.ndarray, target: np.ndarray, k_max: int | None):
    """Return the matched block of the problem, degree zero excluded.

    ``R_0`` is the norm and is one identically on the sphere, so it carries no
    information and would only add a null row to the Jacobian.
    """
    highest = phi.shape[0] - 1 if k_max is None else int(k_max)
    active = np.arange(1, highest + 1)
    return phi[active], np.asarray(target, dtype=float)[active]


def _gauge(c: np.ndarray) -> np.ndarray:
    """Return the representative of ``c`` with unit norm and ``c_0`` positive.

    The norm and the global phase are the two continuous redundancies of the
    complex model. Conjugation is the remaining discrete one and is left alone.
    """
    c = c / np.linalg.norm(c)
    lead = c[0]
    if abs(lead) > 1e-12:
        c = c * (np.conj(lead) / abs(lead))
    return c


def ratio_coefficients_complex(c: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Return ``R_k(c) = c^dagger Phi_k c``, real because ``Phi_k`` is real."""

    a, b = np.real(c), np.imag(c)
    return np.einsum("n,knm,m->k", a, phi, a) + np.einsum("n,knm,m->k", b, phi, b)


def _objective(
    phi: np.ndarray,
    target: np.ndarray,
    c: np.ndarray,
    *,
    weight: np.ndarray | None = None,
    k_max: int | None = None,
) -> float:
    """Return the coefficient objective of an amplitude, without fitting."""

    phi_active, target_active = _active(phi, target, k_max)
    weight_matrix = np.eye(target_active.size) if weight is None else np.asarray(weight)
    residual = ratio_coefficients_complex(c, phi_active) - target_active
    return float(0.5 * residual @ weight_matrix @ residual)


def residual_matrix(
    phi_active: np.ndarray, weight: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    """Return ``M = sum_k (W r)_k Phi_k``, the gradient of the convex objective."""

    return np.einsum("k,knm->nm", weight @ residual, phi_active)


def certified_gap(
    phi: np.ndarray,
    target: np.ndarray,
    c: np.ndarray,
    *,
    weight: np.ndarray | None = None,
    k_max: int | None = None,
    relative: bool = False,
) -> float:
    """Return a bound on how far an amplitude is from the convex optimum.

    For any feasible state, convexity gives ``J(rho) - J* <= Tr(M rho) -
    lambda_min(M)``. At a *complex* amplitude the state is ``rho = c c^dagger``,
    so the first term is ``c^dagger M c``, real and equal to
    ``a^T M a + b^T M b``. The distinction matters: taking the real part alone,
    as the real-fit test does, is not this quantity once ``b`` is nonzero.

    The point of computing it here is accuracy. The Levenberg-Marquardt fit
    converges far harder than a first-order method on the spectrahedron, so this
    is a much tighter certificate for the convex optimum than the relaxation's
    own gap -- and it certifies the amplitude, which is what is being claimed.
    """
    phi_active, target_active = _active(phi, target, k_max)
    weight_matrix = np.eye(target_active.size) if weight is None else np.asarray(weight)
    residual = ratio_coefficients_complex(c, phi_active) - target_active
    matrix = residual_matrix(phi_active, weight_matrix, residual)
    a, b = np.real(c), np.imag(c)
    spectrum = np.linalg.eigvalsh(matrix)
    gap = float(a @ matrix @ a + b @ matrix @ b) - float(spectrum[0])
    if not relative:
        return gap
    return gap / max(float(np.max(np.abs(spectrum))), 1e-300)


def optimality_test(
    phi: np.ndarray,
    target: np.ndarray,
    coefficients: np.ndarray,
    *,
    weight: np.ndarray | None = None,
    k_max: int | None = None,
) -> dict[str, Any]:
    """Decide whether a stationary fit is optimal for the convex relaxation.

    Stationarity makes the fitted vector an eigenvector of ``M``; the fit is the
    convex optimum exactly when its eigenvalue is the smallest one. The bottom
    eigenvector is the direction in which the objective decreases when it is
    not, at second order in the step.
    """
    phi_active, target_active = _active(phi, target, k_max)
    weight_matrix = np.eye(target_active.size) if weight is None else np.asarray(weight)

    c = np.asarray(coefficients)
    residual = ratio_coefficients_complex(c, phi_active) - target_active
    matrix = residual_matrix(phi_active, weight_matrix, residual)

    values, vectors = np.linalg.eigh(matrix)
    a = np.real(c)
    mu = (
        float(a @ matrix @ a / (a @ a)) if np.linalg.norm(a) > 0.0 else float(values[0])
    )
    scale = max(float(np.max(np.abs(values))), 1e-300)
    return {
        "matrix": matrix,
        "mu": mu,
        "lambda_min": float(values[0]),
        "gap": float(mu - values[0]),
        "relative_gap": float((mu - values[0]) / scale),
        "direction": vectors[:, 0],
    }


# ------------------------------------------------------------------- complex --
def fit_complex_amplitude(
    phi: np.ndarray,
    target: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    tau: float = 0.0,
    penalty: np.ndarray | None = None,
    k_max: int | None = None,
    max_iterations: int = 400,
    tolerance: float = 1e-13,
) -> dict[str, Any]:
    """Minimise the coefficient objective on the complex unit sphere.

    The method is the real one of ``amplitude_fit_recovery.fit_amplitude`` in the
    doubled coordinates ``u = (a, b)``: the residual is quadratic in ``u`` with
    the block matrices ``diag(Phi_k, Phi_k)``, so the Jacobian is again linear in
    the parameter. The only change is the tangent space, which now has the phase
    direction ``ic = (-b, a)`` removed alongside the norm direction ``u``.
    """
    degree = phi.shape[1] - 1
    penalty = (
        np.diag(np.arange(degree + 1, dtype=float))
        if penalty is None
        else np.diag(np.asarray(penalty, dtype=float))
    )
    phi_active, target_active = _active(phi, target, k_max)
    weight_matrix = np.eye(target_active.size) if weight is None else np.asarray(weight)

    if initial is None:
        c = np.zeros(degree + 1, dtype=complex)
        c[0] = 1.0
    else:
        c = np.asarray(initial, dtype=complex)
    c = _gauge(c)

    def residual(vec: np.ndarray) -> np.ndarray:
        return ratio_coefficients_complex(vec, phi_active) - target_active

    def objective(vec: np.ndarray) -> float:
        r = residual(vec)
        a, b = np.real(vec), np.imag(vec)
        return float(
            0.5 * r @ weight_matrix @ r
            + 0.5 * tau * (a @ penalty @ a + b @ penalty @ b)
        )

    mu = 1e-3
    value = objective(c)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        a, b = np.real(c), np.imag(c)
        r = residual(c)
        jacobian = 2.0 * np.concatenate(
            (
                np.einsum("knm,m->kn", phi_active, a),
                np.einsum("knm,m->kn", phi_active, b),
            ),
            axis=1,
        )
        # the norm direction and the phase direction, both of which leave the law
        gauge = np.vstack((np.concatenate((a, b)), np.concatenate((-b, a))))
        tangent = null_space(gauge)

        jb = jacobian @ tangent
        block = np.block(
            [
                [penalty, np.zeros_like(penalty)],
                [np.zeros_like(penalty), penalty],
            ]
        )
        hessian = jb.T @ weight_matrix @ jb + tau * tangent.T @ block @ tangent
        gradient = jb.T @ weight_matrix @ r + tau * tangent.T @ block @ np.concatenate(
            (a, b)
        )
        if np.linalg.norm(gradient) < tolerance:
            break

        accepted = False
        for _ in range(40):
            step = np.linalg.solve(hessian + mu * np.eye(hessian.shape[0]), -gradient)
            trial_real = np.concatenate((a, b)) + tangent @ step
            trial = _gauge(trial_real[: degree + 1] + 1j * trial_real[degree + 1 :])
            trial_value = objective(trial)
            if trial_value < value:
                c, value, accepted = trial, trial_value, True
                mu = max(mu * 0.3, 1e-14)
                break
            mu *= 3.0
        if not accepted:
            break

    r = residual(c)
    matrix = residual_matrix(phi_active, weight_matrix, r)
    return {
        "coefficients": c,
        "objective": value,
        "iterations": iterations,
        "residual_norm": float(np.linalg.norm(r)),
        "imaginary_weight": float(np.linalg.norm(np.imag(c))),
        "residual_matrix": matrix,
    }


def continued_complex_fit(
    phi: np.ndarray,
    target: np.ndarray,
    *,
    weight: np.ndarray | None = None,
    k_max: int | None = None,
    kick: float = DEFAULT_KICK,
) -> dict[str, Any]:
    """Fit on the real sphere, test, and continue into the imaginary direction.

    A real solution is a stationary point of the complex problem -- the gradient
    in ``b`` vanishes identically at ``b = 0`` -- so an optimiser started at a
    real vector stays there. The eigenvalue test is what decides whether that is
    a solution or only a stationary point, and it supplies the escape.
    """
    real = fit_amplitude(phi, target, weight=weight, k_max=k_max)
    a = real["coefficients"]
    test = optimality_test(phi, target, a.astype(complex), weight=weight, k_max=k_max)

    start = _gauge(a.astype(complex) + 1j * kick * test["direction"])
    complex_fit = fit_complex_amplitude(
        phi, target, initial=start, weight=weight, k_max=k_max
    )
    if complex_fit["objective"] > real["objective"]:
        complex_fit = fit_complex_amplitude(
            phi, target, initial=a.astype(complex), weight=weight, k_max=k_max
        )
    return {"real": real, "test": test, "complex": complex_fit}


# --------------------------------------------------------------------- convex --
def project_to_states(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest ``rho >= 0`` with ``Tr rho = 1`` in Frobenius norm.

    The projection acts on the spectrum: eigenvalues are projected onto the
    probability simplex, which is the shift ``max(w - theta, 0)`` with the single
    ``theta`` that restores the trace.
    """
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    index = np.arange(1, ordered.size + 1)
    admissible = ordered - (cumulative - 1.0) / index > 0.0
    count = int(np.max(index[admissible]))
    theta = (cumulative[count - 1] - 1.0) / count
    return (vectors * np.maximum(values - theta, 0.0)) @ vectors.T


def relaxed_optimum(
    phi: np.ndarray,
    target: np.ndarray,
    *,
    weight: np.ndarray | None = None,
    k_max: int | None = None,
    initial: np.ndarray | None = None,
    max_iterations: int = DEFAULT_RELAXATION_ITERATIONS,
    tolerance: float = DEFAULT_RELAXATION_TOLERANCE,
) -> dict[str, Any]:
    """Solve the convex coefficient-matching problem over the states.

    Minimise ``J(rho) = (1/2)(R(rho) - Rhat)^T W (R(rho) - Rhat)`` over
    ``rho >= 0``, ``Tr rho = 1``. The objective is a convex quadratic in ``rho``
    and the feasible set has a closed-form projection, so this is accelerated
    projected gradient with the exact Lipschitz constant of the gradient.

    Convergence is certified rather than assumed. For any feasible ``rho``,
    convexity gives ``J(rho) - J* <= Tr(M rho) - lambda_min(M)`` with ``M`` the
    gradient, which is the same eigenvalue quantity as the optimality test of a
    real fit; that number is returned as ``gap``. Conditioning is poor near the
    optimum -- the optimal ``rho`` is low rank, so the active constraints are
    degenerate -- which is why the gap is reported alongside the value.
    """
    phi_active, target_active = _active(phi, target, k_max)
    weight_matrix = np.eye(target_active.size) if weight is None else np.asarray(weight)
    dimension = phi.shape[1]

    linear = phi_active.reshape(phi_active.shape[0], -1)
    lipschitz = float(np.linalg.norm(linear.T @ weight_matrix @ linear, 2))
    step = 1.0 / max(lipschitz, 1e-300)

    def gradient_and_residual(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        residual = linear @ state.ravel() - target_active
        return residual_matrix(phi_active, weight_matrix, residual), residual

    rho = (
        np.eye(dimension) / dimension if initial is None else project_to_states(initial)
    )
    look_ahead = rho
    momentum = 1.0
    gap = np.inf
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        matrix, _ = gradient_and_residual(look_ahead)
        proposal = project_to_states(look_ahead - step * matrix)

        # restart the momentum whenever it stops being a descent direction, the
        # standard guard against the oscillation of an accelerated method
        if np.einsum("nm,nm->", look_ahead - proposal, proposal - rho) > 0.0:
            momentum, look_ahead = 1.0, proposal
            rho = proposal
            continue
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        look_ahead = proposal + ((momentum - 1.0) / next_momentum) * (proposal - rho)
        rho, momentum = proposal, next_momentum

        if iterations % 25 == 0:
            matrix, _ = gradient_and_residual(rho)
            gap = float(
                np.einsum("nm,mn->", matrix, rho) - float(np.linalg.eigvalsh(matrix)[0])
            )
            if gap <= tolerance:
                break

    matrix, residual = gradient_and_residual(rho)
    gap = float(
        np.einsum("nm,mn->", matrix, rho) - float(np.linalg.eigvalsh(matrix)[0])
    )
    value = float(0.5 * residual @ weight_matrix @ residual)
    return {
        "matrix": rho,
        "objective": value,
        "gap": gap,
        "lower_bound": value - gap,
        "iterations": iterations,
        "eigenvalues": np.linalg.eigvalsh(rho)[::-1],
    }


# ----------------------------------------------------------------- half line --
# the wall below which a family has no support; ``None`` where there is none
SUPPORT_WALLS: dict[str, tuple[float | None, float | None]] = {
    "normal": (None, None),
    "ghs": (None, None),
    "gamma": (0.0, None),
    "poisson": (0.0, None),
    "negative-binomial": (0.0, None),
    "binomial": (0.0, None),
}


def jacobi_matrix(family: Any, baseline: Any, size: int) -> np.ndarray:
    """Return the Jacobi matrix of multiplication by ``x`` at the given size."""

    a, b = family.jacobi_coefficients(np.arange(size + 1), baseline)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.diag(b[:size]) + np.diag(a[1:size], 1) + np.diag(a[1:size], -1)


def wall_matrix(family: Any, baseline: Any, name: str, size: int) -> np.ndarray:
    """Return ``pi(J)``, multiplication by the wall polynomial on OPS coefficients.

    Only a single wall is handled: the wall polynomial is then of degree one and
    the second block has size ``K``. A bounded interval would make it quadratic
    and the block size ``K - 1``, which is a different bookkeeping.
    """
    lower, upper = SUPPORT_WALLS[name]
    if lower is not None and upper is not None:
        raise NotImplementedError("a two-sided wall polynomial is not implemented")
    if lower is None and upper is None:
        raise ValueError(f"{name} has no wall; the global relaxation is already exact")
    matrix = jacobi_matrix(family, baseline, size)
    identity = np.eye(size)
    return matrix - lower * identity if lower is not None else upper * identity - matrix


def localised_matrices(
    family: Any, baseline: Any, name: str, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the coefficient matrices of the localised model, and the rescaling.

    The model of Eq. "positivity on the support" is
    ``q = sigma_0 + pi sigma_1`` with ``sigma_0`` of degree ``2K`` and
    ``sigma_1`` of degree ``2K - 2``, so the unknown is a pair of states. Its
    coefficient map is ``R = R(rho_0) + pi(J) R(rho_1)``, affine in the pair.

    The normalisation ``R_0 = 1`` couples the two blocks and would spoil the
    sphere, but the coupling matrix is exactly ``A = pi(J)`` restricted, which is
    positive definite because the wall polynomial is positive on the support.
    Substituting ``d_1 = A^{1/2} c_1`` makes ``R_0`` the plain squared norm of
    the stacked vector again. What comes back is therefore the *same* problem as
    on the whole line -- unit sphere, ``R_k = v^dagger Xi_k v`` -- with
    ``Xi_k`` of size ``2K + 1`` in place of ``Phi_k``, so every routine above
    applies to it unchanged.
    """
    phi = fitting_matrices(family, baseline, degree)
    highest = phi.shape[0] - 1
    wall = wall_matrix(family, baseline, name, highest + 1)

    # A = sum_j pi(J)[0, j] Phi_j on the second block, and its inverse square root
    block = phi[:, :degree, :degree]
    coupling = np.einsum("j,jmn->mn", wall[0], block)
    values, vectors = np.linalg.eigh(coupling)
    if values[0] <= 0.0:
        raise ValueError("the wall coupling is not positive definite")
    inverse_root = (vectors / np.sqrt(values)) @ vectors.T

    matrices = np.zeros((highest + 1, 2 * degree + 1, 2 * degree + 1))
    for k in range(highest + 1):
        matrices[k, : degree + 1, : degree + 1] = phi[k]
        second = np.einsum("j,jmn->mn", wall[k], block)
        matrices[k, degree + 1 :, degree + 1 :] = inverse_root @ second @ inverse_root
    return np.ascontiguousarray(matrices), inverse_root


def split_localised(v: np.ndarray, inverse_root: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the two amplitudes ``(c_0, c_1)`` of a stacked localised vector."""

    degree = inverse_root.shape[0]
    return np.asarray(v[: degree + 1]), inverse_root @ np.asarray(v[degree + 1 :])


def law_from_localised(
    target: Any, v: np.ndarray, inverse_root: np.ndarray, name: str, grid: np.ndarray
) -> np.ndarray:
    """Return ``p_ref (|h_0|^2 + pi |h_1|^2)`` on the grid."""

    family, baseline = target.family, target.baseline
    c0, c1 = split_localised(v, inverse_root)
    reference = np.asarray(family.prob(grid, baseline), dtype=float)

    def modulus(c: np.ndarray) -> np.ndarray:
        real = np.asarray(family.basis_dot(grid, np.real(c), baseline), dtype=float)
        imaginary = np.asarray(
            family.basis_dot(grid, np.imag(c), baseline), dtype=float
        )
        return real**2 + imaginary**2

    lower, upper = SUPPORT_WALLS[name]
    wall = grid - lower if lower is not None else upper - grid
    return reference * (modulus(c0) + wall * modulus(c1))


# -------------------------------------------------------------- factorisation --
def factorise_state(
    family: Any,
    baseline: Any,
    rho: np.ndarray,
    degree: int,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Return the complex amplitude of the law of a state, by its roots.

    This is the constructive side of the correspondence. Whatever its rank,
    ``q_rho = phi^T rho phi`` is a sum of squares of real polynomials and is
    therefore nonnegative on the whole line, not merely on the support. A
    nonnegative polynomial of degree ``2K`` in one variable factorises as
    ``P Pbar`` with ``deg P <= K``: its non-real roots come in conjugate pairs
    and its real roots have even multiplicity, so taking one root out of each
    pair and half of each real multiplicity builds ``P``.

    Expanding that ``P`` in the OPS gives the amplitude with ``q_c = q_rho``
    exactly, which is what the local optimiser has to find by search and often
    does not.

    At rank two or less no roots are needed at all. A state
    ``rho = w_1 u_1 u_1^T + w_2 u_2 u_2^T`` has
    ``q_rho = (sqrt(w_1) u_1 . phi)^2 + (sqrt(w_2) u_2 . phi)^2``, and a sum of
    two real squares *is* one complex square, ``f^2 + g^2 = |f + ig|^2``. So the
    amplitude is read straight off the spectrum, exactly and at any degree. That
    path is taken whenever it applies, because the root route loses accuracy as
    the degree grows and is only needed to reduce a rank of three or more.
    """
    weights, vectors = np.linalg.eigh(0.5 * (rho + rho.T))
    weights = np.maximum(weights, 0.0)
    significant = np.flatnonzero(weights > 1e-12 * max(float(weights[-1]), 1e-300))
    if significant.size <= 2:
        parts = [np.sqrt(weights[index]) * vectors[:, index] for index in significant]
        while len(parts) < 2:
            parts.append(np.zeros(rho.shape[0]))
        return _gauge(parts[0] + 1j * parts[1])

    # Everything below stays inside the three-term recurrence. The law's own OPS
    # coefficients give the roots through the comrade matrix, and the reference
    # measure's Gauss rule projects the factor back onto the basis; no power-basis
    # coefficient vector is ever formed, which is the step that made the earlier
    # interpolation route unusable past degree twenty.
    cap = terminating_degree(family, baseline)
    if cap is not None and 2 * degree > cap:
        raise ValueError(
            f"the basis terminates at degree {cap}, so the degree-{2 * degree} "
            "polynomial has no OPS expansion; use the rank-two seed"
        )
    tensor = family.linearization_tensor(2 * degree, baseline)
    coefficients = np.einsum("mnk,mn->k", tensor[: degree + 1, : degree + 1, :], rho)

    roots, leading = _comrade_roots(family, baseline, coefficients)
    scale = max(float(np.max(np.abs(roots))), 1.0)
    imaginary = np.imag(roots)
    upper = roots[imaginary > 1e-7 * scale]
    real_part = np.sort(np.real(roots[np.abs(imaginary) <= 1e-7 * scale]))
    # a real root of a nonnegative polynomial has even multiplicity, so the
    # sorted list pairs up and one of each pair belongs to P
    chosen = np.concatenate((upper, real_part[::2]))
    if 2 * chosen.size != roots.size or leading <= 0.0:
        raise ValueError(
            f"the law is not a square: {chosen.size} roots selected of "
            f"{roots.size // 2}, leading {leading:.3e}"
        )

    # project P onto the OPS with the reference measure's own Gauss rule, which
    # is exact here: P phi_n has degree at most 2K and the rule has K + 1 nodes
    nodes, quadrature = _gauss_rule(family, baseline, degree + 1)
    amplitude = np.sqrt(leading) * np.prod(nodes[:, None] - chosen[None, :], axis=1)
    basis = np.asarray(family.basis(nodes, degree, baseline), dtype=float)
    c = basis.T @ (quadrature * amplitude)

    # the route checks itself: a factorisation that no longer reproduces the
    # coefficients it came from is an error, not a merely poor starting point
    error = float(
        np.max(
            np.abs(
                ratio_coefficients_complex(
                    c, np.transpose(tensor, (2, 0, 1))[:, : degree + 1, : degree + 1]
                )
                - coefficients
            )
        )
    ) / max(float(np.max(np.abs(coefficients))), 1e-300)
    if error > tolerance:
        raise ValueError(
            f"the degree-{2 * degree} root factorisation lost the law: "
            f"relative error {error:.2e}"
        )
    return _gauge(np.asarray(c, dtype=complex))


def _gauss_rule(
    family: Any, baseline: Any, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``count``-point Gauss rule of the reference law.

    Golub-Welsch: the nodes are the eigenvalues of the truncated Jacobi matrix
    and the weights the squared first components of its eigenvectors. Exact for
    polynomials of degree up to ``2 * count - 1``.
    """
    a, b = family.jacobi_coefficients(np.arange(count), baseline)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    matrix = np.diag(b[:count]) + np.diag(a[1:count], 1) + np.diag(a[1:count], -1)
    values, vectors = np.linalg.eigh(matrix)
    return values, vectors[0, :] ** 2


def _comrade_roots(
    family: Any, baseline: Any, coefficients: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the roots of ``sum_k R_k phi_k`` and its leading power coefficient.

    Barnett's comrade matrix: the recurrence ``x phi = J phi + a_d phi_d e`` and
    the vanishing of the polynomial turn the root problem into the eigenvalues of
    the Jacobi matrix under a rank-one correction. It is the companion matrix a
    recurrence-defined basis deserves, and unlike the power-basis companion it
    stays conditioned because ``J`` is symmetric tridiagonal.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    scale = max(float(np.max(np.abs(coefficients))), 1e-300)
    support = np.flatnonzero(np.abs(coefficients) > 1e-13 * scale)
    top = int(support[-1])
    if top < 2:
        raise ValueError("the law is constant; there is nothing to factorise")

    a, b = family.jacobi_coefficients(np.arange(top + 1), baseline)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    matrix = np.diag(b[:top]) + np.diag(a[1:top], 1) + np.diag(a[1:top], -1)
    matrix[top - 1, :] -= (a[top] / coefficients[top]) * coefficients[:top]

    # phi_d has leading power coefficient 1 / (a_1 ... a_d)
    leading = float(coefficients[top] / np.prod(a[1 : top + 1]))
    return np.linalg.eigvals(matrix), leading


def block_rank_two_seed(rho: np.ndarray, split: int) -> np.ndarray:
    """Return the localised amplitude pair from the two blocks of a state.

    The half-line representation needs each of ``sigma_0`` and ``sigma_1`` to be
    a sum of two squares, not the stacked matrix to have rank two overall, so the
    reduction is applied to the blocks separately. Exact when both blocks are of
    rank two, which is what Markov-Lukacs on a half line asserts of the optimum.
    """
    first = rank_two_seed(rho[:split, :split]) * np.sqrt(
        max(float(np.trace(rho[:split, :split])), 0.0)
    )
    second = rank_two_seed(rho[split:, split:]) * np.sqrt(
        max(float(np.trace(rho[split:, split:])), 0.0)
    )
    return _gauge(np.concatenate((first, second)))


def rank_two_seed(rho: np.ndarray) -> np.ndarray:
    """Return the complex amplitude of the two leading eigenvectors of a state.

    Exact when the state has rank two, and a starting point otherwise. Unlike
    the root factorisation it costs one eigendecomposition and is as accurate at
    degree fifty as at degree five, so it is what keeps the constructive route
    usable where the roots are not.
    """
    weights, vectors = np.linalg.eigh(0.5 * (rho + rho.T))
    weights = np.maximum(weights, 0.0)
    leading = vectors[:, -1] * np.sqrt(weights[-1])
    second = vectors[:, -2] * np.sqrt(weights[-2]) if weights.size > 1 else 0.0
    return _gauge(leading + 1j * second)


def terminating_degree(family: Any, baseline: Any) -> int | None:
    """Return the degree at which the OPS terminates, or ``None`` if it does not.

    Only the three lattice families with bounded support have one; for the
    binomial it is the trial count ``N``.
    """
    return family._maximum_ops_degree(baseline)  # noqa: SLF001


def fitting_matrices(family: Any, baseline: Any, degree: int) -> np.ndarray:
    """Return ``Phi_k`` for the coefficients a degree-``K`` fit can match.

    On an unbounded support the product ``phi_m phi_n`` has degree ``m + n``, so
    a degree-``K`` amplitude produces coefficients out to ``2K`` and the fit
    needs all of them. On a support of ``N + 1`` points there are only ``N + 1``
    coefficients in existence: the product folds back into degrees ``<= N``, and
    ``R_0 ... R_N`` already determine the law completely. Asking for ``2K``
    there is not a stronger requirement but an impossible one, and capping it is
    what lets ``K`` run up to ``N``, where the amplitude spans every function on
    the support and the fit becomes exact.
    """
    cap = terminating_degree(family, baseline)
    if cap is not None and degree > cap:
        raise ValueError(f"the basis terminates at degree {cap}, below K = {degree}")
    highest = 2 * degree if cap is None else min(2 * degree, cap)
    tensor = family.linearization_tensor(max(degree, highest), baseline)
    return np.ascontiguousarray(
        np.transpose(tensor[: degree + 1, : degree + 1, : highest + 1], (2, 0, 1))
    )


def _factorisation_nodes(family: Any, baseline: Any, degree: int) -> np.ndarray:
    """Return interpolation nodes spanning the bulk of the reference law.

    Chebyshev nodes on a few standard deviations: the interpolation is of the
    polynomial itself, exactly, so the nodes only have to be distinct and well
    spread. Clustering them where the reference law lives keeps the linear
    system that expands ``P`` in the OPS well conditioned.
    """
    centre = float(family.mean(baseline))
    spread = float(np.sqrt(family.variance(baseline)))
    count = 2 * degree + 1
    grid = np.cos(np.pi * (np.arange(count) + 0.5) / count)
    return centre + 3.0 * spread * grid


# ------------------------------------------------------------------ reporting --
def law_from_amplitude(target: Any, c: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return ``p_ref |h_c|^2`` on the grid for a real or complex amplitude."""

    family, baseline = target.family, target.baseline
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    real = np.asarray(family.basis_dot(grid, np.real(c), baseline), dtype=float)
    imaginary = np.asarray(family.basis_dot(grid, np.imag(c), baseline), dtype=float)
    return reference * (real**2 + imaginary**2)


def law_from_matrix(target: Any, rho: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return ``p_ref phi^T rho phi`` on the grid, through the spectrum of rho."""

    family, baseline = target.family, target.baseline
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    weights, vectors = np.linalg.eigh(rho)
    ratio = np.zeros_like(grid, dtype=float)
    for weight, vector in zip(weights, vectors.T, strict=True):
        if weight <= 0.0:
            continue
        ratio += (
            weight
            * np.asarray(family.basis_dot(grid, vector, baseline), dtype=float) ** 2
        )
    return reference * ratio


def distance(target: Any, law: np.ndarray, grid: np.ndarray) -> float:
    """Return the total-variation distance of a fitted law from the target."""

    difference = np.abs(law - target.density(grid))
    return 0.5 * integrate(target.family, target.baseline, grid, difference)


def shape_target(name: str) -> Any:
    """Return the neighbouring Gamma shape, whose ratio is linear in ``x``.

    The section's own example of a wall that binds: ``dGamma(r+1, theta) /
    dGamma(r, theta) = x / (r theta)``. It is a perfectly good density ratio on
    the support and negative off it, so no globally nonnegative sum of squares
    can reproduce it, while the localised form does so exactly with
    ``sigma_0 = 0`` and ``sigma_1 = 1 / (r theta)``.
    """
    family, baseline, _ = TARGETS[name]
    if name != "gamma":
        raise ValueError("the shape target is specific to the Gamma family")
    scale = float(baseline.mean) / float(baseline.r)
    member = type(baseline)(
        mean=(float(baseline.r) + 1.0) * scale, r=float(baseline.r) + 1.0
    )
    return Target(
        label="shape r+1",
        family=family,
        baseline=baseline,
        sample=lambda size, rng: np.asarray(family.sample(member, size, rng=rng)),
        density=lambda x: np.asarray(family.prob(x, member), dtype=float),
        members=(member,),
    )


def build_target(name: str, kind: str, separation: float | None, gap: float) -> Any:
    """Return one of the three targets of the truncation study.

    A mixture is specified by its standardised gap rather than by a natural
    shift, since the same shift means different things in different families:
    the gap of two is where the components stop merging into one mode, and it is
    reachable in all six.
    """
    if kind == "shifted":
        return shifted_target(name, TARGETS[name][2])
    if kind == "mixture":
        if separation is None:
            family, baseline, _ = TARGETS[name]
            separation, status = separation_for_gap(family, baseline, gap)
            if status != "exact":
                print(f"  [{name}] mixture gap {gap:g} {status}")
        return mixture_target(name, separation)
    if kind == "truncated":
        return truncated_target(name)
    if kind == "shape":
        return shape_target(name)
    raise ValueError(f"unknown target {kind!r}")


def run_comparison(
    name: str,
    kind: str,
    *,
    degree: int = DEFAULT_DEGREE,
    separation: float | None = None,
    gap: float = DEFAULT_GAP,
    sample_size: int | None = None,
    seed: int = DEFAULT_SEED,
    plot: bool = False,
) -> dict[str, Any]:
    """Fit one target three ways and report the three optima."""

    family, baseline, _ = TARGETS[name]
    try:
        target = build_target(name, kind, separation, gap)
    except (ValueError, AssertionError) as error:
        print(f"\n{name}: {kind} target unavailable: {error}")
        return {}
    grid = target_grid(target)
    phi = fitting_matrices(family, baseline, degree)

    matched = phi.shape[0] - 1
    if sample_size is None:
        observed = reference_coefficients(target, matched, grid)
        source = "population"
    else:
        rng = np.random.default_rng(seed)
        observations = target.sample(sample_size, rng)
        observed, _ = empirical_coefficients(family, baseline, observations, matched)
        source = f"sample N = {sample_size}"

    fits = continued_complex_fit(phi, observed)
    relaxed = relaxed_optimum(phi, observed)

    # Two rounds of: factorise the convex optimum into an amplitude, polish it,
    # and restart the convex solver from the amplitude it found. The relaxation
    # is convex, so a warm start changes only the speed, and its own gap
    # certificate is what makes the comparison sound whatever the start was.
    factorised: dict[str, Any] = {"error": "not attempted"}
    for _ in range(2):
        seeds: list[tuple[str, np.ndarray]] = []
        try:
            seeds.append(
                ("roots", factorise_state(family, baseline, relaxed["matrix"], degree))
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            factorised = {"error": str(error)}
        seeds.append(("rank two", rank_two_seed(relaxed["matrix"])))

        for label, seed_vector in seeds:
            candidate = fit_complex_amplitude(phi, observed, initial=seed_vector)
            candidate["seed_objective"] = _objective(phi, observed, seed_vector)
            candidate["seed"] = label
            if candidate["objective"] < factorised.get("objective", np.inf):
                factorised = candidate
            if candidate["objective"] < fits["complex"]["objective"]:
                fits["complex"] = candidate

        best = fits["complex"]["coefficients"]
        warm = relaxed_optimum(
            phi, observed, initial=np.real(np.outer(best, np.conj(best)))
        )
        if warm["objective"] >= relaxed["objective"]:
            break
        relaxed = warm

    real_law = law_from_amplitude(target, fits["real"]["coefficients"], grid)
    complex_law = law_from_amplitude(target, fits["complex"]["coefficients"], grid)
    relaxed_law = law_from_matrix(target, relaxed["matrix"], grid)

    result = {
        "family": name,
        "target": target.label,
        "kind": kind,
        "degree": degree,
        "source": source,
        "real": fits["real"],
        "complex": fits["complex"],
        "test": fits["test"],
        "relaxed": relaxed,
        "factorised": factorised,
        "tv_real": distance(target, real_law, grid),
        "tv_complex": distance(target, complex_law, grid),
        "tv_relaxed": distance(target, relaxed_law, grid),
        "target_object": target,
        "grid": grid,
        "phi": phi,
        "observed": observed,
        "laws": {"real": real_law, "complex": complex_law, "relaxed": relaxed_law},
    }
    report(result)
    if plot:
        plot_comparison(result)
    return result


def run_localised_comparison(
    name: str,
    kind: str,
    *,
    degree: int = DEFAULT_DEGREE,
    separation: float | None = None,
    gap: float = DEFAULT_GAP,
) -> dict[str, Any]:
    """Fit a walled family in the global class and in the one its support allows.

    The global relaxation asks for a sum of squares, which is nonnegativity on
    the whole line. A family with a wall only needs nonnegativity beyond it, and
    the section's localised representation ``q = sigma_0 + pi sigma_1`` is the
    exact class there. Both are convex, both are certified by their own gap, and
    the difference between their optima is what the wall was costing.
    """
    family, baseline, _ = TARGETS[name]
    target = build_target(name, kind, separation, gap)
    grid = target_grid(target)

    phi = fitting_matrices(family, baseline, degree)
    xi, inverse_root = localised_matrices(family, baseline, name, degree)
    observed = reference_coefficients(target, phi.shape[0] - 1, grid)

    def polish(matrices, candidates, relaxed):
        """Fit from every seed, then restart the relaxation from the best fit."""

        best: dict[str, Any] | None = None
        for seed in candidates:
            fit = fit_complex_amplitude(matrices, observed, initial=seed)
            if best is None or fit["objective"] < best["objective"]:
                best = fit
        vector = best["coefficients"]
        warm = relaxed_optimum(
            matrices, observed, initial=np.real(np.outer(vector, np.conj(vector)))
        )
        return best, (warm if warm["objective"] < relaxed["objective"] else relaxed)

    # global sum of squares, as everywhere above, with the same seeds
    global_relaxed = relaxed_optimum(phi, observed)
    seeds = [rank_two_seed(global_relaxed["matrix"])]
    try:
        seeds.append(
            factorise_state(family, baseline, global_relaxed["matrix"], degree)
        )
    except (ValueError, np.linalg.LinAlgError):
        pass
    seeds.append(fit_amplitude(phi, observed)["coefficients"].astype(complex))
    global_fit, global_relaxed = polish(phi, seeds, global_relaxed)

    # the localised class, which is the same problem with Xi in place of Phi. The
    # global solution is feasible for it -- take sigma_1 = 0 -- so it is offered
    # as a seed alongside the blockwise reduction; the localised optimum can
    # never be the worse of the two, and a search that says otherwise is stuck.
    embedded = np.zeros(2 * degree + 1, dtype=complex)
    embedded[: degree + 1] = global_fit["coefficients"]
    localised_relaxed = relaxed_optimum(xi, observed)
    for _ in range(2):
        localised_fit, localised_relaxed = polish(
            xi,
            [
                block_rank_two_seed(localised_relaxed["matrix"], degree + 1),
                _gauge(embedded),
            ],
            localised_relaxed,
        )

    global_law = law_from_amplitude(target, global_fit["coefficients"], grid)
    localised_law = law_from_localised(
        target, localised_fit["coefficients"], inverse_root, name, grid
    )
    c0, c1 = split_localised(localised_fit["coefficients"], inverse_root)

    print()
    print(
        f"{name}: {target.label}, K = {degree}, population, "
        f"wall at {SUPPORT_WALLS[name][0]:g}"
    )
    global_gap = certified_gap(phi, observed, global_fit["coefficients"])
    localised_gap = certified_gap(xi, observed, localised_fit["coefficients"])
    print(
        f"  {'class':>22} {'J_complex':>12} {'certified':>11} {'J_relaxed':>12} {'TV':>11}"
    )
    print(
        f"  {'global sum of squares':>22} {global_fit['objective']:12.4e} "
        f"{abs(global_gap):11.1e} {global_relaxed['objective']:12.4e} "
        f"{distance(target, global_law, grid):11.4e}"
    )
    print(
        f"  {'localised (wall)':>22} {localised_fit['objective']:12.4e} "
        f"{abs(localised_gap):11.1e} {localised_relaxed['objective']:12.4e} "
        f"{distance(target, localised_law, grid):11.4e}"
    )
    difference = global_fit["objective"] - localised_fit["objective"]
    resolvable = abs(difference) > 10.0 * max(abs(global_gap), abs(localised_gap))
    print(
        f"  the wall constraint costs {difference:.4e}"
        + (
            f" ({100.0 * difference / max(global_fit['objective'], 1e-300):.2f}%)"
            if resolvable
            else ", which the certificates cannot resolve"
        )
        + f"; the second block carries {np.linalg.norm(c1) ** 2:.4f} of the weight"
    )
    return {
        "family": name,
        "kind": kind,
        "degree": degree,
        "target_object": target,
        "grid": grid,
        "global": {"relaxed": global_relaxed, "fit": global_fit, "law": global_law},
        "localised": {
            "relaxed": localised_relaxed,
            "fit": localised_fit,
            "law": localised_law,
            "inverse_root": inverse_root,
            "amplitudes": (c0, c1),
        },
        "observed": observed,
    }


def report(result: dict[str, Any]) -> None:
    """Print one comparison."""

    real, complex_fit = result["real"], result["complex"]
    test, relaxed = result["test"], result["relaxed"]
    scale = max(real["objective"], 1e-300)

    print()
    print(
        f"{result['family']}: {result['target']}, K = {result['degree']}, "
        f"{result['source']}"
    )
    print(
        f"  eigenvalue test at the real fit: mu = {test['mu']:.6e}, "
        f"lambda_min = {test['lambda_min']:.6e}, "
        f"relative gap = {test['relative_gap']:.2e}"
    )
    factorised = result["factorised"]
    if "error" in factorised:
        print(f"  factorisation of the convex optimum failed: {factorised['error']}")
    else:
        print(
            f"  factorisation of the convex optimum ({factorised['seed']} seed): "
            f"J = {factorised['seed_objective']:.6e} before polishing, "
            f"{factorised['objective']:.6e} after"
        )
    print(
        f"  {'J_real':>12} {'J_complex':>12} {'J_relaxed':>12} "
        f"{'gap':>10} {'|Im c|':>9} {'rank(rho)':>10}"
    )
    weights = relaxed["eigenvalues"]
    rank = int(np.count_nonzero(weights > 1e-8 * max(float(weights[0]), 1e-300)))
    print(
        f"  {real['objective']:12.5e} {complex_fit['objective']:12.5e} "
        f"{relaxed['objective']:12.5e} {relaxed['gap']:10.2e} "
        f"{complex_fit['imaginary_weight']:9.3e} {rank:10d}"
    )
    print(
        f"  total variation from the target: real {result['tv_real']:.4e}, "
        f"complex {result['tv_complex']:.4e}, relaxed {result['tv_relaxed']:.4e}"
    )

    improvement = (real["objective"] - complex_fit["objective"]) / scale
    attained = abs(complex_fit["objective"] - relaxed["objective"]) <= max(
        10.0 * relaxed["gap"], 1e-9 * scale
    )
    print(
        f"  complex fit improves the objective by {100.0 * improvement:.3f}% and "
        + ("attains" if attained else "does NOT attain")
        + " the convex optimum"
    )


# -------------------------------------------------------------------- figures --
REAL_STYLE = {"color": "tab:blue", "linestyle": ":", "linewidth": 1.6}
COMPLEX_STYLE = {"color": "tab:green", "linestyle": "-", "linewidth": 1.4}
RELAXED_STYLE = {"color": "0.55", "linestyle": "--", "linewidth": 1.2}


def _draw(axis, grid, values, lattice, **style):
    """Draw a curve, or the atoms of one, on a lattice support."""

    if lattice:
        style = {**style, "marker": "o", "markersize": 3.0, "linewidth": 0.8}
    axis.plot(grid, values, **style)


def _window(target: Any, grid: np.ndarray) -> np.ndarray:
    """Return the part of the integration grid worth looking at.

    The grid is sized for the integrals, which is far wider than the support of
    the law; the plots want the bulk.
    """
    density = target.density(grid)
    inside = np.flatnonzero(density > 1e-5 * float(np.max(density)))
    if inside.size == 0:
        return np.ones_like(grid, dtype=bool)
    low, high = int(inside[0]), int(inside[-1])
    pad = max(1, (high - low) // 8)
    mask = np.zeros_like(grid, dtype=bool)
    mask[max(low - pad, 0) : min(high + pad + 1, grid.size)] = True
    return mask


def plot_comparison(result: dict[str, Any], *, output_dir: Any = None) -> str:
    """Draw one comparison: the laws, the amplitudes, the coefficients, the test.

    Six panels, because the three fits differ in different places. The law panel
    shows whether the difference matters at all; the amplitude panel shows the
    node that the real fit is forced into and the complex one is not; the
    coefficient panels show what the objective actually sees; and the last two
    show the eigenvalue test that decides the whole question, together with the
    rank of the state it decides about.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = result["target_object"]
    family, baseline = target.family, target.baseline
    lattice = family.is_lattice(baseline)
    grid = result["grid"]
    mask = _window(target, grid)
    x = grid[mask]

    real_c = result["real"]["coefficients"]
    complex_c = result["complex"]["coefficients"]
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 7.6))

    # (a) the laws
    axis = axes[0, 0]
    _draw(
        axis,
        x,
        np.asarray(family.prob(x, baseline), dtype=float),
        lattice,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=r"baseline $p_{\rm ref}$",
    )
    _draw(
        axis,
        x,
        target.density(x),
        lattice,
        color="crimson",
        linewidth=1.8,
        label="target",
    )
    _draw(
        axis, x, result["laws"]["real"][mask], lattice, label="real fit", **REAL_STYLE
    )
    _draw(
        axis,
        x,
        result["laws"]["complex"][mask],
        lattice,
        label="complex fit",
        **COMPLEX_STYLE,
    )
    _draw(
        axis,
        x,
        result["laws"]["relaxed"][mask],
        lattice,
        label=r"relaxed $\rho$",
        **RELAXED_STYLE,
    )
    axis.set_xlabel("$m$" if lattice else "$x$")
    axis.set_ylabel("density")
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("laws", fontsize=10)

    # (b) the amplitudes: a node of the real fit is a node of its law
    axis = axes[0, 1]
    axis.axhline(0.0, color="0.6", linewidth=0.8)
    real_h = np.asarray(family.basis_dot(x, real_c, baseline), dtype=float)
    part_a = np.asarray(family.basis_dot(x, np.real(complex_c), baseline), dtype=float)
    part_b = np.asarray(family.basis_dot(x, np.imag(complex_c), baseline), dtype=float)
    _draw(axis, x, real_h, lattice, label="real $h$", **REAL_STYLE)
    _draw(
        axis,
        x,
        np.hypot(part_a, part_b),
        lattice,
        label=r"complex $|h_c|$",
        **COMPLEX_STYLE,
    )
    _draw(
        axis,
        x,
        part_a,
        lattice,
        color="tab:green",
        linestyle="-.",
        linewidth=0.9,
        label=r"$\mathrm{Re}\,h_c$",
    )
    _draw(
        axis,
        x,
        part_b,
        lattice,
        color="tab:orange",
        linestyle="-.",
        linewidth=0.9,
        label=r"$\mathrm{Im}\,h_c$",
    )
    # the amplitude diverges in the far tail, where the law has no mass; scale to
    # the region that carries it instead
    density = target.density(x)
    core = density > 1e-3 * float(np.max(density))
    if np.any(core):
        span = np.concatenate(
            [
                curve[core]
                for curve in (real_h, part_a, part_b, np.hypot(part_a, part_b))
            ]
        )
        low, high = float(np.min(span)), float(np.max(span))
        pad = 0.2 * max(high - low, 1e-12)
        axis.set_ylim(low - pad, high + pad)
    axis.set_xlabel("$m$" if lattice else "$x$")
    axis.set_ylabel("$h$")
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("amplitudes (a zero of $h$ is a node of the law)", fontsize=10)

    # (c) the coefficients the objective matches
    axis = axes[0, 2]
    phi, observed = result["phi"], result["observed"]
    degrees = np.arange(observed.size)
    axis.plot(
        degrees,
        observed,
        "o",
        markersize=4.0,
        color="0.25",
        label=r"target $\widehat R_k$",
    )
    axis.plot(
        degrees,
        ratio_coefficients_complex(real_c.astype(complex), phi),
        "x",
        markersize=5.5,
        color="tab:blue",
        label="real fit",
    )
    axis.plot(
        degrees,
        ratio_coefficients_complex(complex_c, phi),
        "+",
        markersize=6.5,
        color="tab:green",
        label="complex fit",
    )
    axis.plot(
        degrees,
        np.einsum("knm,mn->k", phi, result["relaxed"]["matrix"]),
        "s",
        markersize=4.5,
        markerfacecolor="none",
        color="0.55",
        label=r"relaxed $\rho$",
    )
    axis.axhline(0.0, color="0.6", linewidth=0.8)
    axis.set_xlabel("degree $k$")
    axis.set_ylabel("$R_k$")
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("ratio coefficients", fontsize=10)

    # (d) the residuals, which is where the three fits separate
    axis = axes[1, 0]
    floor = 1e-18
    for coefficients, label, style in (
        (
            ratio_coefficients_complex(real_c.astype(complex), phi),
            "real fit",
            REAL_STYLE,
        ),
        (ratio_coefficients_complex(complex_c, phi), "complex fit", COMPLEX_STYLE),
        (
            np.einsum("knm,mn->k", phi, result["relaxed"]["matrix"]),
            r"relaxed $\rho$",
            RELAXED_STYLE,
        ),
    ):
        axis.plot(
            degrees,
            np.maximum(np.abs(coefficients - observed), floor),
            marker="o",
            markersize=3.0,
            label=label,
            **style,
        )
    axis.set_yscale("log")
    axis.set_xlabel("degree $k$")
    axis.set_ylabel(r"$|R_k - \widehat R_k|$")
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("coefficient residuals", fontsize=10)

    # (e) the eigenvalue test at the real fit
    axis = axes[1, 1]
    test = result["test"]
    spectrum = np.linalg.eigvalsh(test["matrix"])
    axis.plot(
        np.arange(spectrum.size),
        spectrum,
        "o",
        markersize=4.5,
        color="0.25",
        label=r"spectrum of $M$",
    )
    axis.axhline(
        test["mu"],
        color="tab:blue",
        linestyle=":",
        linewidth=1.6,
        label=r"$\mu=a^{\top}Ma$",
    )
    axis.axhline(
        test["lambda_min"],
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=r"$\lambda_{\min}(M)$",
    )
    # the test lives at the bottom of the spectrum; the top eigenvalue is orders
    # of magnitude away and would flatten it
    threshold = max(abs(test["lambda_min"]), abs(test["mu"]), 1e-300) / 10.0
    axis.set_yscale("symlog", linthresh=threshold)
    axis.set_xlabel("eigenvalue index")
    axis.set_ylabel("eigenvalue")
    axis.legend(frameon=False, fontsize=8)
    axis.set_title(
        rf"optimality test: relative gap {test['relative_gap']:.2e}", fontsize=10
    )

    # (f) the result: three optima, and the rank of the state that bounds them
    axis = axes[1, 2]
    values = np.array(
        [
            result["real"]["objective"],
            result["complex"]["objective"],
            result["relaxed"]["objective"],
        ]
    )
    positive = values[values > 0.0]
    bottom = 0.1 * float(np.min(positive)) if positive.size else 1e-30
    axis.bar(
        ["real", "complex", "relaxed"],
        np.maximum(values, bottom),
        color=["tab:blue", "tab:green", "0.55"],
        width=0.6,
    )
    axis.set_yscale("log")
    axis.set_ylabel(r"$\mathcal{J}_K$")
    weights = result["relaxed"]["eigenvalues"]
    rank = int(np.count_nonzero(weights > 1e-8 * max(float(weights[0]), 1e-300)))
    axis.set_title(
        rf"optima; $\mathrm{{rank}}\,\rho^\star={rank}$, gap "
        rf"{abs(result['relaxed']['gap']):.1e}",
        fontsize=10,
    )
    for index, value in enumerate(values):
        axis.annotate(
            f"{value:.2e}",
            (index, max(value, bottom)),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )

    figure.suptitle(
        f"{result['family']}: {result['target']}, K = {result['degree']}, "
        f"{result['source']}   |   TV real {result['tv_real']:.2e}, "
        f"complex {result['tv_complex']:.2e}, relaxed {result['tv_relaxed']:.2e}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "artifacts" / FIGURE_SUBDIRECTORY
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{result['family']}_{result['kind']}_K{result['degree']}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"  figure: {path}")
    return str(path)


def plot_localised(result: dict[str, Any], *, output_dir: Any = None) -> str:
    """Draw the two classes side by side: global sum of squares, and localised.

    Four panels. The laws show whether the difference is visible at all; the
    amplitude panel shows the second block, which is the whole point of the wall;
    the residuals show what the objective sees; the last shows the two optima.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = result["target_object"]
    family, baseline = target.family, target.baseline
    name, degree = result["family"], result["degree"]
    lattice = family.is_lattice(baseline)
    grid = result["grid"]
    mask = _window(target, grid)
    x = grid[mask]
    localised, globally = result["localised"], result["global"]

    figure, axes = plt.subplots(1, 4, figsize=(19.0, 4.2))

    axis = axes[0]
    _draw(
        axis,
        x,
        target.density(x),
        lattice,
        color="crimson",
        linewidth=2.0,
        label="target",
    )
    _draw(axis, x, globally["law"][mask], lattice, label="global fit", **REAL_STYLE)
    _draw(
        axis, x, localised["law"][mask], lattice, label="localised fit", **COMPLEX_STYLE
    )
    axis.set_xlabel("$m$" if lattice else "$x$")
    axis.set_ylabel("density")
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("laws", fontsize=10)

    # the two blocks of the localised amplitude
    axis = axes[1]
    c0, c1 = localised["amplitudes"]
    lower = SUPPORT_WALLS[name][0]

    def modulus(c: np.ndarray) -> np.ndarray:
        real = np.asarray(family.basis_dot(x, np.real(c), baseline), dtype=float)
        imag = np.asarray(family.basis_dot(x, np.imag(c), baseline), dtype=float)
        return real**2 + imag**2

    axis.axhline(0.0, color="0.6", linewidth=0.8)
    _draw(axis, x, modulus(c0), lattice, label=r"$\sigma_0=|h_0|^2$", **COMPLEX_STYLE)
    _draw(
        axis,
        x,
        (x - lower) * modulus(c1),
        lattice,
        color="tab:orange",
        linestyle="-.",
        linewidth=1.3,
        label=r"$\pi\,\sigma_1$",
    )
    _draw(
        axis,
        x,
        np.asarray(
            family.basis_dot(x, np.real(globally["fit"]["coefficients"]), baseline)
        )
        ** 2
        + np.asarray(
            family.basis_dot(x, np.imag(globally["fit"]["coefficients"]), baseline)
        )
        ** 2,
        lattice,
        label="global $q$",
        **REAL_STYLE,
    )
    axis.set_xlabel("$m$" if lattice else "$x$")
    axis.set_ylabel("$q$")
    axis.legend(frameon=False, fontsize=8)
    axis.set_title(
        rf"the two blocks; $\pi$ vanishes at the wall {lower:g}", fontsize=10
    )

    # residuals
    axis = axes[2]
    observed = result["observed"]
    degrees = np.arange(observed.size)
    phi = fitting_matrices(family, baseline, degree)
    xi, _ = localised_matrices(family, baseline, name, degree)
    for coefficients, label, style in (
        (
            ratio_coefficients_complex(globally["fit"]["coefficients"], phi),
            "global",
            REAL_STYLE,
        ),
        (
            ratio_coefficients_complex(localised["fit"]["coefficients"], xi),
            "localised",
            COMPLEX_STYLE,
        ),
    ):
        axis.plot(
            degrees,
            np.maximum(np.abs(coefficients - observed), 1e-18),
            marker="o",
            markersize=3.0,
            label=label,
            **style,
        )
    axis.set_yscale("log")
    axis.set_xlabel("degree $k$")
    axis.set_ylabel(r"$|R_k - \widehat R_k|$")
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("coefficient residuals", fontsize=10)

    # optima
    axis = axes[3]
    values = np.array([globally["fit"]["objective"], localised["fit"]["objective"]])
    positive = values[values > 0.0]
    bottom = 0.1 * float(np.min(positive)) if positive.size else 1e-30
    axis.bar(
        ["global", "localised"],
        np.maximum(values, bottom),
        color=["tab:blue", "tab:green"],
        width=0.55,
    )
    axis.set_yscale("log")
    axis.set_ylabel(r"$\mathcal{J}_K$")
    for index, value in enumerate(values):
        axis.annotate(
            f"{value:.2e}",
            (index, max(value, bottom)),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    axis.set_title(
        rf"$\|c_1\|^2 = {np.linalg.norm(c1) ** 2:.4f}$ in the second block",
        fontsize=10,
    )

    figure.suptitle(
        f"{name}: {target.label}, K = {degree}   |   the support wall, "
        f"global sum of squares against the localised class",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    output = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "artifacts" / FIGURE_SUBDIRECTORY
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}_{result['kind']}_K{degree}_localised.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"  figure: {path}")
    return str(path)


def _parse_args() -> argparse.Namespace:
    """Parse the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="normal", choices=(*FAMILY_NAMES, "all"))
    parser.add_argument(
        "--target",
        default="all",
        choices=("shifted", "mixture", "truncated", "shape", "all"),
    )
    parser.add_argument(
        "--localised",
        action="store_true",
        help="also fit the class the support wall allows, for a walled family",
    )
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument(
        "--gap",
        type=float,
        default=DEFAULT_GAP,
        help="standardised separation of the mixture components",
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=None,
        help="natural shift of the mixture, overriding --gap",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="fit empirical coefficients instead of the exact ones",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--plot", action="store_true", help="write a figure for every comparison"
    )
    return parser.parse_args()


def main() -> None:
    """Run the comparison from the command line."""

    args = _parse_args()
    names = FAMILY_NAMES if args.family == "all" else (args.family,)
    kinds = (
        ("shifted", "mixture", "truncated") if args.target == "all" else (args.target,)
    )
    for name in names:
        family, baseline, _ = TARGETS[name]
        cap = terminating_degree(family, baseline)
        if cap is not None and args.degree > cap:
            print(f"{name}: the OPS basis terminates at degree {cap}; skipped")
            continue
        degrees = (args.degree,)
        for kind in kinds:
            if args.localised:
                if SUPPORT_WALLS[name] == (None, None):
                    print(f"\n{name}: no support wall; the global class is exact")
                    continue
                try:
                    result = run_localised_comparison(
                        name,
                        kind,
                        degree=degrees[0],
                        separation=args.separation,
                        gap=args.gap,
                    )
                except (ValueError, AssertionError, NotImplementedError) as error:
                    print(f"\n{name}: {kind} localised fit unavailable: {error}")
                    continue
                if args.plot:
                    plot_localised(result)
                continue
            run_comparison(
                name,
                kind,
                degree=degrees[0],
                separation=args.separation,
                gap=args.gap,
                sample_size=args.samples,
                seed=args.seed,
                plot=args.plot,
            )


if __name__ == "__main__":
    main()
