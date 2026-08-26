"""Exact one-shot diffusion kernels: identity, moments, member flow."""

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

DRAWS = 400_000

# (family, baseline, start state away from the mean, admissible shift)
MARKOV_CASES = [
    (Normal, NormalParams(1.0, 1.5), 4.0, 0.4),
    (Poisson, PoissonParams(6.0), 11, 0.3),
    (Gamma, GammaParams(3.0, 2.5), 7.5, 0.3),
    (Binomial, BinomialParams(8.0, 20), 15, 0.4),
    (NegativeBinomial, NegativeBinomialParams(4.0, 3.0), 10, 0.3),
]

IDS = [family.__class__.__name__ for family, *_ in MARKOV_CASES]


@pytest.mark.parametrize(("family", "params", "x0", "_"), MARKOV_CASES, ids=IDS)
def test_time_zero_is_the_identity(family, params, x0, _):
    x = np.array([x0, x0, x0])
    draws = family.one_shot_sample(x, 0.0, params, rng=np.random.default_rng(0))
    assert np.array_equal(draws, x)


@pytest.mark.parametrize(("family", "params", "x0", "_"), MARKOV_CASES, ids=IDS)
@pytest.mark.parametrize("t", [0.35, 1.2])
def test_conditional_moments(family, params, x0, _, t):
    """Mean relaxes as ``mu + w (x - mu)``; the variance is the affine form
    ``V (1 - w^2) + V' w (1 - w) (x - mu)`` of the note."""

    rng = np.random.default_rng(1234)
    x = np.full(DRAWS, x0)
    draws = np.asarray(family.one_shot_sample(x, t, params, rng=rng), dtype=float)

    w = np.exp(-t)
    mu = float(family.mean(params))
    variance = float(family.variance(params))
    slope = float(family.variance_slope(params))
    predicted_mean = mu + w * (x0 - mu)
    predicted_var = variance * (1.0 - w**2) + slope * w * (1.0 - w) * (x0 - mu)

    mean_error = abs(draws.mean() - predicted_mean)
    assert mean_error < 6.0 * np.sqrt(predicted_var / DRAWS)
    assert abs(draws.var() - predicted_var) < 0.05 * predicted_var


@pytest.mark.parametrize(("family", "params", "_", "shift"), MARKOV_CASES, ids=IDS)
def test_member_flow_invariance(family, params, _, shift):
    """Noising a family member lands on the member at the contracted shift
    coordinate ``z -> w z``, matching the marginal flow used by the toy
    model."""

    t = 0.8
    rng = np.random.default_rng(99)
    z = family.shift_coordinate(shift, params)
    start = family.shifted_params(params, shift)
    target = family.from_shift_coordinate(np.exp(-t) * z, params)

    x0 = family.sample(start, DRAWS, rng=rng)
    draws = np.asarray(family.one_shot_sample(x0, t, params, rng=rng), dtype=float)

    target_mean = float(family.mean(target))
    target_var = float(family.variance(target))
    assert abs(draws.mean() - target_mean) < 6.0 * np.sqrt(target_var / DRAWS)
    assert abs(draws.var() - target_var) < 0.05 * target_var


@pytest.mark.parametrize(("family", "params", "x0", "_"), MARKOV_CASES, ids=IDS)
def test_large_time_forgets_the_start(family, params, x0, _):
    rng = np.random.default_rng(7)
    x = np.full(DRAWS, x0)
    draws = np.asarray(family.one_shot_sample(x, 12.0, params, rng=rng), dtype=float)
    mu = float(family.mean(params))
    variance = float(family.variance(params))
    assert abs(draws.mean() - mu) < 6.0 * np.sqrt(variance / DRAWS)
    assert abs(draws.var() - variance) < 0.05 * variance


def test_broadcasting_shapes():
    params = PoissonParams(6.0)
    rng = np.random.default_rng(3)
    x = np.array([0, 3, 7, 11, 2])
    assert Poisson.one_shot_sample(x, 0.5, params, rng=rng).shape == (5,)
    t = np.array([0.0, 0.1, 0.5, 1.0, 4.0])
    assert Poisson.one_shot_sample(x, t, params, rng=rng).shape == (5,)
    assert Poisson.one_shot_sample(4, t, params, rng=rng).shape == (5,)
    assert Poisson.one_shot_sample(4, 0.5, params, rng=rng).shape == ()


def test_lattice_kernels_stay_on_the_lattice():
    rng = np.random.default_rng(11)
    for family, params, x0 in [
        (Poisson, PoissonParams(6.0), 11),
        (Binomial, BinomialParams(8.0, 20), 15),
        (NegativeBinomial, NegativeBinomialParams(4.0, 3.0), 10),
    ]:
        draws = family.one_shot_sample(np.full(1000, x0), 0.7, params, rng=rng)
        assert np.all(draws == np.floor(draws))
        assert np.all(draws >= 0)
    binomial_draws = Binomial.one_shot_sample(
        np.full(1000, 15), 0.7, BinomialParams(8.0, 20), rng=rng
    )
    assert np.all(binomial_draws <= 20)


def test_ghs_has_no_kernel():
    with pytest.raises(NotImplementedError, match="positivity"):
        GHS.one_shot_sample(0.5, 0.3, GHSParams(0.0, 2.0))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="nonnegative"):
        Poisson.one_shot_sample(3, -0.1, PoissonParams(6.0))
    with pytest.raises(ValueError, match="integers"):
        Poisson.one_shot_sample(2.5, 0.3, PoissonParams(6.0))
    with pytest.raises(ValueError, match="\\[0, N\\]"):
        Binomial.one_shot_sample(25, 0.3, BinomialParams(8.0, 20))
    with pytest.raises(ValueError, match="scalar"):
        Poisson.one_shot_sample(3, 0.3, PoissonParams(np.array([2.0, 6.0])))
