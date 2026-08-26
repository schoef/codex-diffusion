"""Complex MPS amplitudes on the orthonormal polynomial basis.

The amplitude over ``d`` sites is a matrix-product state on the physical
index ``k``: ``h(x) = prod_i (sum_k A_i[:, k, :] phi_k(x_i))`` with bond
dimension ``chi`` and open boundaries. Because the basis is orthonormal
under the baseline, the standard MPS algebra applies verbatim: canonical
forms make the Born normalisation ``<h|h> = 1`` a local condition, local
moments are contractions with identity environments, and the sequential
sampler is exact. The one-site model is ``d = 1, chi = 1`` and the pair
model of ``two_site_diffusion`` is ``d = 2`` with the bond merged.

Tensors are numpy arrays of shape ``(chi_left, K + 1, chi_right)`` with
``chi = 1`` at both ends of the chain.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ----------------------------------------------------------------- algebra --
def vacuum_state(d: int, degree: int) -> list[np.ndarray]:
    """Return the product vacuum: every site in the degree-zero state."""

    tensors = []
    for _ in range(d):
        tensor = np.zeros((1, degree + 1, 1), dtype=complex)
        tensor[0, 0, 0] = 1.0
        tensors.append(tensor)
    return tensors


def norm_squared(tensors: list[np.ndarray]) -> float:
    """Return ``<h|h>`` by transfer contraction."""

    left = np.ones((1, 1), dtype=complex)
    for tensor in tensors:
        left = np.einsum("ab,akr,bks->rs", left, tensor.conj(), tensor)
    return float(np.real(left[0, 0]))


def right_canonicalise(tensors: list[np.ndarray]) -> list[np.ndarray]:
    """Return an equivalent state with every tensor right-canonical.

    Right-canonical: ``sum_k A[k] A[k]^dagger = I`` on the left bond, so
    contractions to the right of any site are the identity. The overall
    norm is absorbed into the first tensor and then normalised away.
    """
    tensors = [tensor.copy() for tensor in tensors]
    for index in range(len(tensors) - 1, 0, -1):
        chi_l, n, chi_r = tensors[index].shape
        matrix = tensors[index].reshape(chi_l, n * chi_r)
        # LQ decomposition via QR of the conjugate transpose.
        q, r = np.linalg.qr(matrix.conj().T)
        keep = min(chi_l, n * chi_r)
        tensors[index] = q.conj().T[:keep].reshape(keep, n, chi_r)
        tensors[index - 1] = np.einsum(
            "lkr,rm->lkm", tensors[index - 1], r.conj().T[:, :keep]
        )
    tensors[0] = tensors[0] / np.linalg.norm(tensors[0])
    return tensors


def merge_bond(tensors: list[np.ndarray], bond: int) -> np.ndarray:
    """Return the merged two-site tensor ``B`` at ``bond`` (sites bond, bond+1)."""

    return np.einsum("lkm,mjr->lkjr", tensors[bond], tensors[bond + 1])


def split_bond(
    merged: np.ndarray, chi_max: int, *, center_right: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split a merged tensor by SVD, truncating the bond to ``chi_max``.

    With ``center_right`` the left factor is an isometry (left-canonical)
    and the centre of the chain moves right; otherwise the mirror. Returns
    the two site tensors and the discarded squared Schmidt weight.
    """
    chi_l, n1, n2, chi_r = merged.shape
    matrix = merged.reshape(chi_l * n1, n2 * chi_r)
    left, values, right = np.linalg.svd(matrix, full_matrices=False)
    keep = min(chi_max, int(np.count_nonzero(values > 1e-14)) or 1)
    discarded = float(np.sum(values[keep:] ** 2))
    left, values, right = left[:, :keep], values[:keep], right[:keep]
    scale = np.linalg.norm(values)
    values = values / max(scale, 1e-300)
    if center_right:
        first = left.reshape(chi_l, n1, keep)
        second = (values[:, None] * right).reshape(keep, n2, chi_r)
    else:
        first = (left * values[None, :]).reshape(chi_l, n1, keep)
        second = right.reshape(keep, n2, chi_r)
    return first, second, discarded


def move_center_right(tensors: list[np.ndarray], site: int) -> None:
    """Make ``tensors[site]`` left-canonical, pushing the centre to ``site+1``."""

    chi_l, n, chi_r = tensors[site].shape
    matrix = tensors[site].reshape(chi_l * n, chi_r)
    q, r = np.linalg.qr(matrix)
    keep = q.shape[1]
    tensors[site] = q.reshape(chi_l, n, keep)
    tensors[site + 1] = np.einsum("lm,mkr->lkr", r, tensors[site + 1])


def bond_spectrum(tensors: list[np.ndarray], bond: int) -> np.ndarray:
    """Return the Schmidt values across ``bond`` of the normalised state."""

    state = right_canonicalise(tensors)
    for site in range(bond):
        move_center_right(state, site)
    merged = merge_bond(state, bond)
    chi_l, n1, n2, chi_r = merged.shape
    values = np.linalg.svd(merged.reshape(chi_l * n1, n2 * chi_r), compute_uv=False)
    return values / max(np.linalg.norm(values), 1e-300)


# ----------------------------------------------------------------- moments --
def bond_fit_stack(phi: np.ndarray, chi_l: int, chi_r: int) -> np.ndarray:
    """Return the observable stack for a merged bond tensor.

    In mixed canonical form the pair moments at the bond are
    ``R_{jk} = vec(B)^dagger (I x Phi_j x Phi_k x I) vec(B)``: the one-site
    complex fit runs unchanged on this stack, exactly as at two sites.
    """
    k_max = phi.shape[0] - 1
    n = phi.shape[1]
    eye_l = np.eye(chi_l)
    eye_r = np.eye(chi_r)
    stack = np.einsum(
        "ab,jcd,kef,gh->jkacegbdfh", eye_l, phi, phi, eye_r, optimize=True
    )
    dimension = chi_l * n * n * chi_r
    return np.ascontiguousarray(stack.reshape((k_max + 1) ** 2, dimension, dimension))


def pair_moment_matrix(
    tensors: list[np.ndarray], phi: np.ndarray, bond: int
) -> np.ndarray:
    """Return the model pair moments ``E[phi_j(X_bond) phi_k(X_bond+1)]``."""

    state = right_canonicalise(tensors)
    for site in range(bond):
        move_center_right(state, site)
    merged = merge_bond(state, bond)
    moments = np.einsum(
        "labr,jac,kbd,lcdr->jk",
        merged.conj(),
        phi,
        phi,
        merged,
        optimize=True,
    )
    return np.real(moments)


# ---------------------------------------------------------------- sampling --
def evaluate(
    tensors: list[np.ndarray],
    family: Any,
    baseline: Any,
    samples: np.ndarray,
) -> np.ndarray:
    """Return ``h(x)`` for each row of ``samples`` (shape ``(size, d)``)."""

    degree = tensors[0].shape[1] - 1
    size = samples.shape[0]
    state = np.ones((size, 1), dtype=complex)
    for site, tensor in enumerate(tensors):
        basis = np.asarray(
            family.basis(samples[:, site], degree, baseline), dtype=float
        )
        state = np.einsum("sl,sk,lkr->sr", state, basis, tensor, optimize=True)
    return state[:, 0]


def sequential_sample(
    tensors: list[np.ndarray],
    family: Any,
    baseline: Any,
    size: int,
    grid: np.ndarray,
    rng: Any,
) -> np.ndarray:
    """Draw exactly from the Born law of the MPS, site by site.

    With the remainder of the chain right-canonical, the conditional law at
    the current site given the drawn prefix is ``p_ref(x) |v A(x)|^2`` up to
    normalisation, where ``v`` is the per-sample bond state.
    """
    state = right_canonicalise(tensors)
    degree = state[0].shape[1] - 1
    lattice = family.is_lattice(baseline)
    basis = np.asarray(family.basis(grid, degree, baseline), dtype=float)
    reference = np.asarray(family.prob(grid, baseline), dtype=float)

    draws = np.empty((size, len(state)))
    vectors = np.ones((size, 1), dtype=complex)
    for site, tensor in enumerate(state):
        contracted = np.einsum("gk,lkr->glr", basis, tensor, optimize=True)
        conditional = np.einsum("sl,glr->sgr", vectors, contracted, optimize=True)
        densities = reference[None, :] * np.sum(np.abs(conditional) ** 2, axis=2)
        if lattice:
            cumulative = np.cumsum(densities, axis=1)
            cumulative /= cumulative[:, -1:]
            picks = np.sum(cumulative < rng.random((size, 1)), axis=1)
            picks = np.minimum(picks, grid.size - 1)
        else:
            cells = 0.5 * (densities[:, 1:] + densities[:, :-1]) * np.diff(grid)
            cumulative = np.concatenate(
                (np.zeros((size, 1)), np.cumsum(cells, axis=1)), axis=1
            )
            cumulative /= cumulative[:, -1:]
            uniforms = rng.random(size)
            positions = np.array(
                [
                    np.interp(uniforms[index], cumulative[index], grid)
                    for index in range(size)
                ]
            )
            picks = np.searchsorted(grid, positions)
            picks = np.minimum(picks, grid.size - 1)
            draws[:, site] = positions
        if lattice:
            draws[:, site] = grid[picks]
        vectors = conditional[np.arange(size), picks]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-300)
    return draws
