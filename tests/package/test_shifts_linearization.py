"""Shift-kernel and Jacobi product-linearization tensor tests."""

from dataclasses import fields

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

SHIFT_CASES = [
    (Normal, NormalParams(0.0, 1.2), 0.4, 4),
    (Poisson, PoissonParams(4.0), 0.3, 4),
    (Gamma, GammaParams(3.0, 2.5), 0.2, 4),
    (Binomial, BinomialParams(3.6, 12), 0.5, 4),
    (
        NegativeBinomial,
        NegativeBinomialParams(4.0, 3.0),
        0.1,
        4,
    ),
    (GHS, GHSParams(0.0, 1.5), 0.5, 4),
]


def assert_params_allclose(first, second):
    for field in fields(first):
        assert np.allclose(
            getattr(first, field.name),
            getattr(second, field.name),
        )


@pytest.mark.parametrize(
    ("family", "baseline", "natural_shift", "n_max"),
    SHIFT_CASES,
)
def test_shift_coordinate_coefficients_and_inverse(
    family,
    baseline,
    natural_shift,
    n_max,
):
    shifted = family.shifted_params(baseline, natural_shift)
    z = family.shift_coordinate(natural_shift, baseline)
    reconstructed = family.from_shift_coordinate(z, baseline)
    coefficients = family.shift_coefficients(
        natural_shift,
        n_max,
        baseline,
    )

    assert_params_allclose(reconstructed, shifted)
    assert coefficients.shape == (n_max + 1,)
    assert coefficients[0] == 1.0
    standardized_mean_shift = (float(shifted.mean) - float(baseline.mean)) / np.sqrt(
        float(family.variance(baseline))
    )
    assert np.allclose(coefficients[1], standardized_mean_shift)


@pytest.mark.parametrize(
    ("family", "baseline", "_natural_shift", "_n_max"),
    SHIFT_CASES,
)
def test_linearization_tensor_symmetry_and_identity(
    family,
    baseline,
    _natural_shift,
    _n_max,
):
    tensor = family.linearization_tensor(3, baseline)

    assert tensor.shape == (4, 4, 4)
    assert np.allclose(tensor[0], np.eye(4))
    assert np.allclose(tensor, np.transpose(tensor, (1, 0, 2)))
    assert np.allclose(tensor, np.transpose(tensor, (2, 1, 0)))


def test_batched_shift_coefficients():
    params = NormalParams(
        mean=np.array([-1.0, 1.0]),
        sigma=np.array([1.0, 2.0]),
    )
    coefficients = Normal.shift_coefficients(
        np.array([0.2, 0.3]),
        4,
        params,
    )

    assert coefficients.shape == (2, 5)
    assert np.all(coefficients[:, 0] == 1.0)
