"""Analytic moment, natural-parameter, and Hellinger-affinity tests."""

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

ROUND_TRIPS = [
    (Normal, NormalParams(1.2, 0.7), (0.7,)),
    (Poisson, PoissonParams(1.2), ()),
    (Gamma, GammaParams(1.2, 2.4), (2.4,)),
    (Binomial, BinomialParams(1.2, 7), (7,)),
    (NegativeBinomial, NegativeBinomialParams(1.2, 2.4), (2.4,)),
    (GHS, GHSParams(1.2, 2.4), (2.4,)),
]


@pytest.mark.parametrize(("family", "params", "fixed"), ROUND_TRIPS)
def test_natural_parameter_round_trip(family, params, fixed):
    eta = family.natural_parameter(params)
    reconstructed = family.from_natural(eta, *fixed)
    assert np.allclose(reconstructed.mean, params.mean)


@pytest.mark.parametrize(
    ("family", "params"),
    [
        (Normal, NormalParams(np.array([1.0, 2.0]), 0.7)),
        (Poisson, PoissonParams(np.array([1.0, 2.0]))),
        (Gamma, GammaParams(np.array([1.0, 2.0]), 2.4)),
        (Binomial, BinomialParams(np.array([1.0, 2.0]), 7)),
        (
            NegativeBinomial,
            NegativeBinomialParams(np.array([1.0, 2.0]), 2.4),
        ),
        (GHS, GHSParams(np.array([1.0, 2.0]), 2.4)),
    ],
)
def test_self_affinity_is_one(family, params):
    assert np.allclose(family.affinity(params, params), 1.0)


def test_affinities_match_direct_evaluation():
    count_cases = [
        (Poisson, PoissonParams(2.0), PoissonParams(5.0), np.arange(80)),
        (Binomial, BinomialParams(2.0, 8), BinomialParams(5.0, 8), np.arange(9)),
        (
            NegativeBinomial,
            NegativeBinomialParams(2.0, 3.0),
            NegativeBinomialParams(5.0, 3.0),
            np.arange(200),
        ),
    ]
    for family, params1, params2, x in count_cases:
        direct = np.sum(np.sqrt(family.prob(x, params1) * family.prob(x, params2)))
        assert np.allclose(family.affinity(params1, params2), direct, atol=1e-12)

    continuous_cases = [
        (
            Normal,
            NormalParams(-1.0, 1.3),
            NormalParams(2.0, 1.3),
            np.linspace(-12.0, 12.0, 100_001),
        ),
        (
            Gamma,
            GammaParams(2.0, 2.5),
            GammaParams(5.0, 2.5),
            np.linspace(1e-7, 40.0, 150_001),
        ),
        (
            GHS,
            GHSParams(-1.0, 1.5),
            GHSParams(2.0, 1.5),
            np.linspace(-30.0, 30.0, 200_001),
        ),
    ]
    for family, params1, params2, x in continuous_cases:
        direct = np.trapezoid(
            np.sqrt(family.prob(x, params1) * family.prob(x, params2)), x
        )
        assert np.allclose(family.affinity(params1, params2), direct, atol=2e-8)


@pytest.mark.parametrize(
    ("family", "params1", "params2"),
    [
        (Normal, NormalParams(0.0, 1.0), NormalParams(0.0, 2.0)),
        (Gamma, GammaParams(1.0, 2.0), GammaParams(1.0, 3.0)),
        (Binomial, BinomialParams(1.0, 3), BinomialParams(1.0, 4)),
        (
            NegativeBinomial,
            NegativeBinomialParams(1.0, 2.0),
            NegativeBinomialParams(1.0, 3.0),
        ),
        (GHS, GHSParams(1.0, 2.0), GHSParams(1.0, 3.0)),
    ],
)
def test_affinity_rejects_different_fixed_parameters(family, params1, params2):
    with pytest.raises(ValueError):
        family.affinity(params1, params2)
