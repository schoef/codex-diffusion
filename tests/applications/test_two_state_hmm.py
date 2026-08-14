"""Checks for the two-state hidden-Markov benchmark."""

import numpy as np
import pytest

from applications.two_state_hmm import (
    BASELINES,
    FAMILY_NAMES,
    emission_parameters,
    latent_correlation,
    log_likelihood,
    log_likelihood_by_enumeration,
    noised_emissions,
    predicted_moments,
    run_benchmark,
    sample_latent,
    sample_observations,
    transition_matrix,
)


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_transfer_matrix_matches_enumeration(family_name):
    """The O(L) likelihood must equal the sum over all 2**L latent paths."""

    family, baseline, separation = BASELINES[family_name]
    rng = np.random.default_rng(0)
    _, _, z = emission_parameters(family, baseline, separation)
    plus, minus = noised_emissions(family, baseline, z, 0.4)
    states = sample_latent(7, 32, 0.2, rng)
    observations = sample_observations(family, plus, minus, states, rng)

    transfer = log_likelihood(family, observations, plus, minus, 0.2)
    enumerated = log_likelihood_by_enumeration(family, observations, plus, minus, 0.2)
    assert np.allclose(transfer, enumerated, atol=1e-10)


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_independent_sites_factorise(family_name):
    """At epsilon = 1/2 the law must be a product of one-site mixtures."""

    family, baseline, separation = BASELINES[family_name]
    rng = np.random.default_rng(1)
    _, _, z = emission_parameters(family, baseline, separation)
    plus, minus = noised_emissions(family, baseline, z, 0.6)
    states = sample_latent(6, 16, 0.5, rng)
    observations = sample_observations(family, plus, minus, states, rng)

    joint = log_likelihood(family, observations, plus, minus, 0.5)
    factorised = np.sum(
        np.logaddexp(
            np.asarray(family.log_prob(observations, plus)) + np.log(0.5),
            np.asarray(family.log_prob(observations, minus)) + np.log(0.5),
        ),
        axis=1,
    )
    assert np.allclose(joint, factorised, atol=1e-10)


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_flow_contracts_the_mean_coordinate(family_name):
    """Inverting the contracted shift must contract the mean at rate exp(-t)."""

    family, baseline, separation = BASELINES[family_name]
    plus, minus, z = emission_parameters(family, baseline, separation)
    baseline_mean = float(family.mean(baseline))
    for t in (0.0, 0.25, 1.5, 5.0):
        damping = float(np.exp(-t))
        noised = noised_emissions(family, baseline, z, t)
        for member, raw in zip(noised, (plus, minus)):
            expected = baseline_mean + damping * (
                float(family.mean(raw)) - baseline_mean
            )
            assert float(family.mean(member)) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_variance_forms_agree(family_name):
    """The two one-site variance forms agree only if V is quadratic."""

    family, baseline, separation = BASELINES[family_name]
    _, _, z = emission_parameters(family, baseline, separation)
    for t in (0.0, 0.5, 2.0):
        plus, minus = noised_emissions(family, baseline, z, t)
        moments = predicted_moments(family, baseline, plus, minus)
        assert moments["variance_two_state"] == pytest.approx(
            moments["variance_expanded"], rel=1e-12
        )


def test_late_time_law_is_the_product_baseline():
    """As t grows the emissions collapse and the law factorises."""

    family, baseline, separation = BASELINES["gamma"]
    rng = np.random.default_rng(2)
    _, _, z = emission_parameters(family, baseline, separation)
    plus, minus = noised_emissions(family, baseline, z, 40.0)
    states = sample_latent(5, 8, 0.3, rng)
    observations = sample_observations(family, plus, minus, states, rng)

    late = log_likelihood(family, observations, plus, minus, 0.3)
    product = np.sum(np.asarray(family.log_prob(observations, baseline)), axis=1)
    assert np.allclose(late, product, atol=1e-10)


def test_latent_two_point_function():
    """The sampled latent chain must reproduce rho**lag."""

    rng = np.random.default_rng(3)
    epsilon = 0.2
    states = sample_latent(32, 200_000, epsilon, rng)
    for lag in (1, 3, 7):
        empirical = float(np.mean(states[:, :-lag] * states[:, lag:]))
        assert empirical == pytest.approx(
            float(latent_correlation(epsilon, lag)), abs=5e-3
        )


def test_transition_matrix_is_stochastic_and_symmetric():
    """Rows sum to one, and the matrix is symmetric under the (+1,-1) order."""

    transition = transition_matrix(0.3)
    assert np.allclose(transition.sum(axis=1), 1.0)
    assert np.allclose(transition, transition.T)
    with pytest.raises(ValueError, match="epsilon"):
        transition_matrix(1.5)


def test_unknown_family_is_rejected():
    """The entry point should report the accepted family names."""

    with pytest.raises(ValueError, match="unknown family"):
        run_benchmark("not-a-family")
