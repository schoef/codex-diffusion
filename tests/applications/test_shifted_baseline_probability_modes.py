"""End-to-end checks for all shifted-baseline demonstration configurations."""

import numpy as np
import pytest

from applications.shifted_baseline_probability_modes import (
    CONFIGURATION_FACTORIES,
    check_linearization,
    check_projection,
    compute_linearization,
    compute_projection,
    damped_coefficients,
    exact_damped_params,
    integrate,
    reconstructed_density,
)


@pytest.mark.parametrize("family_name", CONFIGURATION_FACTORIES)
def test_shifted_baseline_probability_modes(family_name):
    config = CONFIGURATION_FACTORIES[family_name]()
    result = compute_projection(config)
    check_projection(config, result)

    assert result.analytic[0] == 1.0
    assert np.allclose(
        result.quadrature,
        result.analytic,
        atol=config.projection_atol,
        rtol=config.projection_atol,
    )
    standardized_mean_shift = (
        float(config.shifted.mean) - float(config.baseline.mean)
    ) / np.sqrt(float(config.family.variance(config.baseline)))
    assert np.allclose(result.analytic[1], standardized_mean_shift)

    tau = max(config.taus)
    degree = np.arange(config.n_max + 1)
    damped = damped_coefficients(result.analytic, tau)
    assert np.allclose(
        damped,
        result.analytic * np.exp(-degree * tau),
    )

    reconstructed = reconstructed_density(config, result.quadrature, tau)
    exact = config.family.prob(config.grid, exact_damped_params(config, tau))
    assert np.max(np.abs(reconstructed - exact)) < config.exact_density_atol
    assert np.isclose(
        integrate(config, reconstructed),
        1.0,
        atol=max(config.projection_atol, 2e-10),
    )


@pytest.mark.parametrize("family_name", CONFIGURATION_FACTORIES)
def test_lower_modes_do_not_depend_on_truncation_order(family_name):
    config = CONFIGURATION_FACTORIES[family_name]()
    rng = np.random.default_rng(101)
    observations = config.sample(rng, config.shifted, 2_000)
    lower = config.family.basis(observations, 3, config.baseline)
    higher = config.family.basis(
        observations,
        config.n_max,
        config.baseline,
    )

    assert np.array_equal(lower, higher[:, :4])


@pytest.mark.parametrize("family_name", CONFIGURATION_FACTORIES)
def test_analytic_and_sampled_linearization_tensors(family_name):
    config = CONFIGURATION_FACTORIES[family_name]()
    result = compute_linearization(config)
    check_linearization(result)

    assert result.analytic.shape == (4, 4, 4)
    assert np.allclose(result.analytic[0], np.eye(4))


@pytest.mark.parametrize("family_name", CONFIGURATION_FACTORIES)
def test_jacobi_linearization_matches_direct_expectation(family_name):
    config = CONFIGURATION_FACTORIES[family_name]()
    n_max = config.linearization_n_max
    analytic = config.family.linearization_tensor(n_max, config.baseline)
    basis = config.family.basis(config.grid, n_max, config.baseline)
    probability = config.family.prob(config.grid, config.baseline)
    direct = integrate(
        config,
        probability[:, None, None, None]
        * basis[:, :, None, None]
        * basis[:, None, :, None]
        * basis[:, None, None, :],
    )

    assert np.allclose(analytic, direct, atol=1e-8, rtol=1e-10)
