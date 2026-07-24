"""Shape-contract tests for paired, outer-grid, and chunked evaluation."""

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

FAMILIES = [
    (Normal, NormalParams(np.array([0.0, 1.0]), 1.5), np.linspace(-2.0, 2.0, 12)),
    (Poisson, PoissonParams(np.array([2.0, 4.0])), np.arange(12)),
    (Gamma, GammaParams(np.array([2.0, 4.0]), 2.5), np.linspace(0.1, 8.0, 12)),
    (Binomial, BinomialParams(np.array([2.0, 4.0]), 8), np.arange(9)),
    (
        NegativeBinomial,
        NegativeBinomialParams(np.array([2.0, 4.0]), 2.5),
        np.arange(12),
    ),
    (GHS, GHSParams(np.array([-1.0, 1.0]), 1.5), np.linspace(-3.0, 3.0, 12)),
]


@pytest.mark.parametrize(("family", "params", "x"), FAMILIES)
def test_paired_broadcasting(family, params, x):
    paired = family.log_prob(x[:2], params)
    assert paired.shape == (2,)

    matrix_params = type(params)(
        **{
            field: np.asarray(getattr(params, field)).reshape(
                (-1, 1) if np.asarray(getattr(params, field)).ndim else ()
            )
            for field in params.__dataclass_fields__
        }
    )
    matrix = family.log_prob(x[None, :], matrix_params)
    assert matrix.shape == (2, x.size)


@pytest.mark.parametrize(("family", "params", "x"), FAMILIES)
def test_outer_grid_and_chunking(family, params, x):
    x = np.asarray(x).reshape(3, 4) if np.asarray(x).size == 12 else np.asarray(x)
    direct = family.log_prob_grid(x, params)
    chunked = family.log_prob_grid(x, params, chunk_size=5)
    assert direct.shape == (2,) + x.shape
    assert np.array_equal(direct, chunked)


def test_grid_does_not_require_manual_singletons():
    params = NormalParams(
        mean=np.arange(6.0).reshape(2, 3),
        sigma=np.array([1.0, 2.0, 3.0]),
    )
    x = np.zeros((4, 5))
    assert Normal.log_prob_grid(x, params).shape == (2, 3, 4, 5)
