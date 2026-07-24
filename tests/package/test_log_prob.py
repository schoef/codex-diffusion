"""Probability-kernel comparisons with SciPy and support-boundary checks."""

import numpy as np
import pytest
from scipy import stats

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


def test_normal_matches_scipy():
    x = np.linspace(-4.0, 5.0, 23)
    params = NormalParams(mean=0.7, sigma=1.3)
    assert np.allclose(Normal.log_prob(x, params), stats.norm.logpdf(x, 0.7, 1.3))


def test_poisson_matches_scipy():
    x = np.arange(20)
    params = PoissonParams(mean=3.7)
    assert np.allclose(Poisson.log_prob(x, params), stats.poisson.logpmf(x, 3.7))


def test_gamma_matches_scipy():
    x = np.linspace(0.01, 12.0, 29)
    params = GammaParams(mean=4.5, r=2.3)
    expected = stats.gamma.logpdf(x, a=2.3, scale=4.5 / 2.3)
    assert np.allclose(Gamma.log_prob(x, params), expected)


def test_binomial_matches_scipy():
    x = np.arange(13)
    params = BinomialParams(mean=4.2, N=12)
    assert np.allclose(
        Binomial.log_prob(x, params), stats.binom.logpmf(x, 12, 4.2 / 12)
    )


def test_negative_binomial_matches_scipy():
    x = np.arange(30)
    params = NegativeBinomialParams(mean=5.5, r=2.7)
    probability = 2.7 / (2.7 + 5.5)
    assert np.allclose(
        NegativeBinomial.log_prob(x, params),
        stats.nbinom.logpmf(x, 2.7, probability),
    )


@pytest.mark.parametrize(
    ("family", "params", "outside"),
    [
        (Poisson, PoissonParams(2.0), np.array([-1.0, 0.5])),
        (Gamma, GammaParams(2.0, 3.0), np.array([-1.0, 0.0])),
        (Binomial, BinomialParams(2.0, 5), np.array([-1.0, 1.5, 6.0])),
        (
            NegativeBinomial,
            NegativeBinomialParams(2.0, 3.0),
            np.array([-1.0, 0.5]),
        ),
    ],
)
def test_outside_support_is_negative_infinity(family, params, outside):
    assert np.all(np.isneginf(family.log_prob(outside, params)))


def test_boundary_distributions():
    assert Poisson.prob(0, PoissonParams(0.0)) == 1.0
    assert Poisson.prob(1, PoissonParams(0.0)) == 0.0
    assert Binomial.prob(0, BinomialParams(0.0, 5)) == 1.0
    assert Binomial.prob(5, BinomialParams(5.0, 5)) == 1.0
    assert Binomial.prob(0, BinomialParams(0.0, 0)) == 1.0
    assert NegativeBinomial.prob(0, NegativeBinomialParams(0.0, 2.0)) == 1.0


def test_ghs_normalization_and_moments():
    params = GHSParams(mean=0.8, r=1.7)
    x = np.linspace(-25.0, 25.0, 250_001)
    probability = GHS.prob(x, params)
    normalization = np.trapezoid(probability, x)
    numerical_mean = np.trapezoid(x * probability, x)
    numerical_variance = np.trapezoid((x - params.mean) ** 2 * probability, x)
    assert np.allclose(normalization, 1.0, atol=2e-10)
    assert np.allclose(numerical_mean, GHS.mean(params), atol=2e-9)
    assert np.allclose(numerical_variance, GHS.variance(params), atol=2e-8)
