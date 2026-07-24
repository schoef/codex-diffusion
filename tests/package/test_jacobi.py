"""Numerical tests for OPS recurrences, orthogonality, and basis contraction."""

import numpy as np
import pytest

from nefqvf import (
    GHS,
    Binomial,
    BinomialParams,
    Gamma,
    GammaParams,
    GHSParams,
    NegativeBinomial,
    NegativeBinomialParams,
    Normal,
    NormalParams,
    Poisson,
    PoissonParams,
)

BASIS_CASES = [
    (Normal, NormalParams(0.7, 1.3), np.linspace(-4.0, 5.0, 31), 5),
    (Poisson, PoissonParams(3.7), np.arange(25), 5),
    (Gamma, GammaParams(4.5, 2.3), np.linspace(0.1, 12.0, 31), 5),
    (Binomial, BinomialParams(4.2, 12), np.arange(13), 5),
    (
        NegativeBinomial,
        NegativeBinomialParams(5.5, 2.7),
        np.arange(31),
        5,
    ),
    (GHS, GHSParams(0.8, 1.7), np.linspace(-5.0, 5.0, 31), 5),
]


@pytest.mark.parametrize(("family", "params", "x", "n_max"), BASIS_CASES)
def test_first_basis_function_is_standardized_observation(family, params, x, n_max):
    basis = family.basis(x, n_max, params)
    expected = (x - family.mean(params)) / np.sqrt(family.variance(params))
    assert np.allclose(basis[..., 0], 1.0)
    assert np.allclose(basis[..., 1], expected)


@pytest.mark.parametrize(("family", "params", "x", "n_max"), BASIS_CASES)
def test_basis_dot_matches_materialized_basis(family, params, x, n_max):
    coefficients = np.linspace(-0.5, 0.8, n_max + 1)
    materialized = np.sum(family.basis(x, n_max, params) * coefficients, axis=-1)
    direct = family.basis_dot(x, coefficients, params)
    assert np.allclose(direct, materialized, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize(("family", "params", "_x", "n_max"), BASIS_CASES)
def test_jacobi_off_diagonals_are_positive(family, params, _x, n_max):
    a_n, _ = family.jacobi_coefficients(np.arange(n_max + 1), params)
    assert a_n[0] == 0.0
    assert np.all(a_n[1:] > 0.0)


def test_discrete_orthonormality():
    cases = [
        (Poisson, PoissonParams(3.7), np.arange(80), 5),
        (Binomial, BinomialParams(4.2, 12), np.arange(13), 12),
        (
            NegativeBinomial,
            NegativeBinomialParams(5.5, 2.7),
            np.arange(220),
            5,
        ),
    ]
    for family, params, x, n_max in cases:
        probability = family.prob(x, params)
        basis = family.basis(x, n_max, params)
        gram = (basis.T * probability) @ basis
        assert np.allclose(gram, np.eye(n_max + 1), atol=2e-9)


def test_continuous_orthonormality():
    cases = [
        (
            Normal,
            NormalParams(0.7, 1.3),
            np.linspace(-10.0, 11.0, 200_001),
            4,
            2e-9,
        ),
        (
            Gamma,
            GammaParams(4.5, 2.3),
            np.linspace(1e-7, 100.0, 250_001),
            4,
            2e-7,
        ),
        (
            GHS,
            GHSParams(0.8, 1.7),
            np.linspace(-30.0, 30.0, 250_001),
            4,
            2e-8,
        ),
    ]
    for family, params, x, n_max, tolerance in cases:
        probability = family.prob(x, params)
        basis = family.basis(x, n_max, params)
        gram = np.empty((n_max + 1, n_max + 1))
        for m in range(n_max + 1):
            for n in range(n_max + 1):
                gram[m, n] = np.trapezoid(probability * basis[:, m] * basis[:, n], x)
        assert np.allclose(gram, np.eye(n_max + 1), atol=tolerance)


def test_grid_basis_and_parameter_batched_coefficients():
    params = NormalParams(mean=np.array([-1.0, 1.0]), sigma=2.0)
    x = np.linspace(-2.0, 2.0, 7)
    coefficients = np.array([[1.0, 2.0], [3.0, 4.0]])
    basis = Normal.basis(x, 1, params, grid=True)
    direct = Normal.basis_dot(x, coefficients, params, grid=True)
    assert basis.shape == (2, 7, 2)
    assert direct.shape == (2, 7)
    assert np.allclose(direct, np.sum(basis * coefficients[:, None, :], axis=-1))


def test_binomial_basis_terminates_at_N():
    params = BinomialParams(mean=2.0, N=5)
    assert Binomial.basis(np.arange(6), 5, params).shape == (6, 6)
    with pytest.raises(ValueError):
        Binomial.basis(np.arange(6), 6, params)


@pytest.mark.parametrize(
    ("family", "params"),
    [
        (Poisson, PoissonParams(0.0)),
        (Binomial, BinomialParams(0.0, 5)),
        (NegativeBinomial, NegativeBinomialParams(0.0, 2.0)),
    ],
)
def test_degenerate_laws_do_not_define_an_ops(family, params):
    with pytest.raises(ValueError):
        family.basis(0.0, 1, params)
