"""Two-site diffusion: Kronecker machinery, branch structure, resolution."""

import numpy as np
import pytest

from applications.amplitude_fit_complex import (
    continued_complex_fit,
    fitting_matrices,
    ratio_coefficients_complex,
)
from applications.two_site_diffusion import (
    amplitude_fidelity,
    branch_expansion,
    build_pair_target,
    exact_pair_amplitude,
    exact_pair_coefficients,
    factorise_pair_moments,
    marginal_weight,
    pair_density,
    pair_fitting_matrices,
    pair_total_variation,
    run_pair_study,
    site_member_amplitude,
)


def test_kronecker_stack_matches_products():
    """A product amplitude must give the outer product of one-site ratios."""

    target = build_pair_target("poisson", 0.57, 0.2)
    phi = fitting_matrices(target.family, target.baseline, 4)
    stack, degrees = pair_fitting_matrices(phi)
    rng = np.random.default_rng(0)
    c1 = rng.normal(size=5) + 1j * rng.normal(size=5)
    c2 = rng.normal(size=5) + 1j * rng.normal(size=5)
    c1, c2 = c1 / np.linalg.norm(c1), c2 / np.linalg.norm(c2)
    pair = ratio_coefficients_complex(np.kron(c1, c2), stack)
    single1 = ratio_coefficients_complex(c1, phi)
    single2 = ratio_coefficients_complex(c2, phi)
    assert np.allclose(pair, np.outer(single1, single2).reshape(-1), atol=1e-12)
    k_max = phi.shape[0] - 1
    j, k = np.divmod(np.arange((k_max + 1) ** 2), k_max + 1)
    assert np.array_equal(degrees, (j + k).astype(float))


def test_exact_pair_coefficients_match_quadrature():
    target = build_pair_target("poisson", 0.57, 0.3)
    family, baseline = target.family, target.baseline
    grid = np.arange(0, 200)
    basis = np.asarray(family.basis(grid, 6, baseline), dtype=float)
    for t in (0.0, 0.5):
        truth = pair_density(target, t, grid)
        quadrature = basis.T @ truth @ basis
        exact = exact_pair_coefficients(target, t, 6)
        assert np.allclose(exact, quadrature, atol=1e-8)


def test_branch_amplitude_is_exact_at_zero_epsilon():
    """At perfect persistence the pure amplitude represents the pair law,
    and the branch expansion reads off the two members at weight 1/2."""

    target = build_pair_target("poisson", 0.57, 0.0)
    matrix = exact_pair_amplitude(target, 0.0, 6)
    members = [
        site_member_amplitude(target, member, 0.0, 6)
        for member in (target.plus, target.minus)
    ]
    weights, residual = branch_expansion(matrix, members)
    assert np.allclose(weights, 0.5, atol=1e-10)
    assert residual < 1e-10
    exact = exact_pair_coefficients(target, 0.0, 12)
    phi = fitting_matrices(target.family, target.baseline, 6)
    stack, _ = pair_fitting_matrices(phi)
    fitted = ratio_coefficients_complex(matrix.reshape(-1), stack)
    assert np.linalg.norm(fitted - exact.reshape(-1)) < 1e-3


def test_cross_moments_resolve_the_latent():
    """The Phase A prediction, on exact moments: with the cross moments in
    the objective the fitted law's moment-curve factorisation recovers the
    members and weights; with marginals only it cannot. The amplitude itself
    is NOT pinned even by the full pair law — the ratio depends on the
    sufficient statistic ``x_1 + x_2`` alone, so a phase gauge in the sum
    variable survives at any number of sites (fidelity < 1 below); the
    latent lives in the law, not in the amplitude gauge."""

    target = build_pair_target("poisson", 0.57, 0.0)
    family, baseline = target.family, target.baseline
    phi = fitting_matrices(family, baseline, 6)
    k_max = phi.shape[0] - 1
    stack, _ = pair_fitting_matrices(phi)
    exact = exact_pair_coefficients(target, 0.0, k_max).reshape(-1)
    reference = exact_pair_amplitude(target, 0.0, 6)

    full = continued_complex_fit(stack, exact)["complex"]
    marginal = continued_complex_fit(stack, exact, weight=marginal_weight(k_max))[
        "complex"
    ]
    full_matrix = full["coefficients"].reshape(7, 7)
    marginal_matrix = marginal["coefficients"].reshape(7, 7)

    grid = np.arange(0, 60)
    assert pair_total_variation(target, 0.0, full_matrix, grid) < 1e-3
    assert pair_total_variation(target, 0.0, marginal_matrix, grid) > 0.03

    fitted = ratio_coefficients_complex(full_matrix.reshape(-1), stack)
    factor = factorise_pair_moments(
        fitted.reshape(k_max + 1, k_max + 1), family, baseline, 1.2
    )
    assert abs(factor["shifts"][0] - 0.285) < 0.02
    assert abs(factor["shifts"][1] + 0.285) < 0.02
    assert factor["epsilon"] < 0.05
    assert max(factor["curve_residuals"]) < 0.05

    junk = ratio_coefficients_complex(marginal_matrix.reshape(-1), stack)
    broken = factorise_pair_moments(
        junk.reshape(k_max + 1, k_max + 1), family, baseline, 1.2
    )
    recovered = np.sort(broken["shifts"])
    missed = (
        abs(recovered[1] - 0.285) > 0.1
        or abs(recovered[0] + 0.285) > 0.1
        or max(broken["curve_residuals"]) > 0.15
    )
    assert missed
    assert amplitude_fidelity(full_matrix, reference) < 0.999


@pytest.fixture(scope="module")
def pair_study():
    return run_pair_study(
        "poisson",
        separation=0.57,
        epsilon=0.05,
        n_slices=3,
        degree=4,
        draws=12_000,
        tau=0.5,
        seed=11,
    )


def test_pair_slices_are_accurate(pair_study):
    for entry in pair_study["slices"]:
        assert entry["total_variation"] < 0.08
        assert np.isfinite(entry["held_out_nll"])


def test_pair_experiment_orders_the_fits(pair_study):
    experiment = pair_study["experiment"]
    assert (
        experiment["full cold"]["total_variation"]
        < experiment["marginal only"]["total_variation"]
    )
    for label in ("warm chain", "full cold", "data"):
        factor = experiment[label]["factorisation"]
        assert abs(factor["shifts"][0] - 0.285) < 0.08
        assert abs(factor["shifts"][1] + 0.285) < 0.08
    broken = experiment["marginal only"]["factorisation"]
    recovered = np.sort(broken["shifts"])
    missed = (
        abs(recovered[1] - 0.285) > 0.1
        or abs(recovered[0] + 0.285) > 0.1
        or max(broken["curve_residuals"]) > 0.15
    )
    assert missed


def test_pair_generation_reaches_the_target(pair_study):
    assert pair_study["generation"]["sample_tv"] < 0.2
