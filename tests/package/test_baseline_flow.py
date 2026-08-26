"""The baseline-change unitary is a one-parameter group in Fisher arclength.

The note's claim: the generator of the baseline flow is the fixed
antisymmetric tridiagonal matrix G0 with weights sqrt((n+1)(1+n a2))/2,
and, in the note's convention U_mn = <m;eta'|n;eta>,
U(eta -> eta') = expm(theta G0) with theta the Fisher--Rao arclength
int dmu/sqrt(V).  Checked against the exact mixed Gram matrix
U_mn = int sqrt(p p') phi_m' phi_n domega for a Weyl family (Poisson,
a2 = 0) and an su(1,1) family (negative binomial, a2 = 1/r).
"""

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

from nefqvf import (
    NegativeBinomial,
    NegativeBinomialParams,
    Poisson,
    PoissonParams,
)

K = 14
GRID = np.arange(0, 400)


def exact_unitary(family, params0, params1):
    basis0 = np.asarray(family.basis(GRID, K, params0), dtype=float)
    basis1 = np.asarray(family.basis(GRID, K, params1), dtype=float)
    weight = np.sqrt(
        np.asarray(family.prob(GRID, params0), dtype=float)
        * np.asarray(family.prob(GRID, params1), dtype=float)
    )
    return basis1.T @ (weight[:, None] * basis0)


def generator(a2):
    matrix = np.zeros((K + 1, K + 1))
    n = np.arange(K)
    weights = 0.5 * np.sqrt((n + 1) * (1 + n * a2))
    matrix[n, n + 1] = weights
    matrix[n + 1, n] = -weights
    return matrix


def test_poisson_flow_is_one_parameter():
    lam0, lam1 = 6.0, 8.5
    unitary = exact_unitary(Poisson, PoissonParams(lam0), PoissonParams(lam1))
    theta = 2.0 * (np.sqrt(lam1) - np.sqrt(lam0))
    predicted = expm(theta * generator(0.0))
    assert np.abs(unitary[:10, :10] - predicted[:10, :10]).max() < 1e-5


def test_negative_binomial_flow_is_one_parameter():
    r, mu0, mu1 = 3.0, 4.0, 6.0
    unitary = exact_unitary(
        NegativeBinomial,
        NegativeBinomialParams(mu0, r),
        NegativeBinomialParams(mu1, r),
    )
    theta, _ = quad(lambda mu: 1.0 / np.sqrt(mu + mu * mu / r), mu0, mu1)
    predicted = expm(theta * generator(1.0 / r))
    assert np.abs(unitary[:10, :10] - predicted[:10, :10]).max() < 1e-4


def test_vacuum_overlap_is_the_affinity():
    """(e^{theta G0})_{00} must equal the Hellinger affinity."""

    lam0, lam1 = 6.0, 8.5
    theta = 2.0 * (np.sqrt(lam1) - np.sqrt(lam0))
    predicted = expm(theta * generator(0.0))[0, 0]
    affinity = np.exp(-0.125 * theta**2)
    assert abs(predicted - affinity) < 1e-8
