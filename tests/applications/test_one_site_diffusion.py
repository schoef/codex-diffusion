"""End-to-end checks of the one-site diffusion model."""

import numpy as np
import pytest

from applications.amplitude_fit_complex import build_target
from applications.one_site_diffusion import (
    blended_target,
    contracted_member,
    direct_sample,
    fit_schedule,
    mode_variances,
    run_study,
    slice_ratio_coefficients,
    slice_target,
)
from applications.targets import reference_coefficients, target_grid
from nefqvf import GHS, GHSParams, Normal, NormalParams


@pytest.fixture(scope="module")
def poisson_study():
    return run_study(
        "poisson",
        "mixture",
        n_slices=4,
        degree=6,
        draws=20_000,
        gap=1.2,
        tau=0.5,
        particles=1_200,
        proposals=16,
        seed=11,
    )


def test_schedule_is_descending_and_ends_at_zero(poisson_study):
    schedule = poisson_study["schedule"]
    assert np.all(np.diff(schedule) < 0.0)
    assert schedule[-1] == 0.0
    assert schedule[0] > 1.0


def test_every_slice_is_certified_and_accurate(poisson_study):
    for entry in poisson_study["slices"]:
        assert entry["relative_gap"] < 1e-3
        assert entry["total_variation"] < 0.05
        assert np.isfinite(entry["held_out_nll"])


def test_warm_start_matches_the_cold_start_on_an_easy_target(poisson_study):
    warm = poisson_study["slices"][-1]
    cold = poisson_study["cold_start"]
    assert warm["total_variation"] < 0.05
    assert cold["total_variation"] < 0.05


def test_generation_reaches_the_target(poisson_study):
    generation = poisson_study["generation"]
    assert generation["direct_tv"] < 0.06
    assert generation["reverse_tv"] < 0.15
    assert np.all(generation["ess"] > 8.0)


def test_slice_truths_match_quadrature():
    target = build_target("normal", "mixture", None, 1.2)
    grid = target_grid(target)
    for t in (0.0, 0.4, 1.5):
        truth = slice_target(target, t)
        exact = slice_ratio_coefficients(target, t, 8)
        quadrature = reference_coefficients(truth, 8, grid)
        assert np.allclose(exact, quadrature, atol=1e-8)


def test_mode_variance_formula_is_exact():
    """``Var_{p_t}[phi_k] = sum_j Lambda_kkj R_j(t) - R_k(t)^2`` against a
    direct lattice sum: the machine check of the note's SNR schedule."""

    target = build_target("poisson", "mixture", None, 1.4)
    family, baseline = target.family, target.baseline
    k_max = 8
    lam = family.linearization_tensor(2 * k_max, baseline)
    x = np.arange(0, 300)
    basis = np.asarray(family.basis(x, k_max, baseline), dtype=float)
    for t in (0.0, 0.7):
        variances, _ = mode_variances(target, t, k_max, lam)
        p = slice_target(target, t).density(x)
        first = (p[:, None] * basis).sum(axis=0)
        second = (p[:, None] * basis**2).sum(axis=0)
        assert np.allclose(variances, second - first**2, rtol=1e-8, atol=1e-12)


def test_predicted_floor_tracks_the_observed_error(poisson_study):
    for entry in poisson_study["slices"]:
        floor = entry["predicted_floor"]
        assert np.isfinite(floor) and floor > 0.0
        assert entry["total_variation"] < 10.0 * floor
        # deep slices may carry no mode above noise; the data slice must.
        assert 0 <= entry["supported_degree"] <= 6
    assert poisson_study["slices"][-1]["supported_degree"] >= 1


def test_steered_split_recovers_the_exact_chamber():
    """On the amplitude ``(h_+ + i h_-)/sqrt(2)`` the latent machinery must
    return weights (1/2, 1/2), a vanishing residual, and the member states."""

    from applications.one_site_diffusion import (
        latent_weights,
        member_amplitude,
        steered_components,
    )

    target = build_target("poisson", "mixture", None, 1.4)
    members = [member_amplitude(target, m, 0.0, 8) for m in target.members]
    exact = (members[0] + 1j * members[1]) / np.sqrt(2.0)
    weights, residual = latent_weights(exact, members)
    assert np.allclose(weights, 0.5, atol=1e-10)
    assert residual < 1e-10
    components = steered_components(exact, members)
    for member, (weight, vector) in zip(members, components, strict=True):
        assert abs(weight - 0.5) < 1e-3
        assert abs(abs(member @ vector) - 1.0) < 1e-3


def test_contracted_member_limits():
    baseline = NormalParams(0.0, 1.0)
    member = NormalParams(2.0, 1.0)
    frozen = contracted_member(Normal, baseline, member, 0.0)
    assert np.isclose(float(Normal.mean(frozen)), 2.0)
    forgotten = contracted_member(Normal, baseline, member, 30.0)
    assert abs(float(Normal.mean(forgotten))) < 1e-10


def test_blended_target_reproduces_the_stacked_objective():
    rng = np.random.default_rng(3)
    data = rng.normal(size=7)
    previous = rng.normal(size=7)
    delta, tau = 0.4, 0.7
    blended, weights = blended_target(data, previous, delta, tau)
    decay = np.exp(-np.arange(7) * delta)

    def stacked(r):
        return 0.5 * np.sum((r - data) ** 2 + tau * (decay * r - previous) ** 2)

    def folded(r):
        return 0.5 * np.sum(weights * (r - blended) ** 2)

    r1, r2 = rng.normal(size=7), rng.normal(size=7)
    difference = (stacked(r1) - folded(r1)) - (stacked(r2) - folded(r2))
    assert abs(difference) < 1e-10


def test_direct_sampler_matches_target_moments():
    rng = np.random.default_rng(4)
    target = build_target("normal", "shifted", None, 1.2)
    grid = target_grid(target)
    exact = slice_ratio_coefficients(target, 0.0, 12)
    coefficients = np.zeros(13, dtype=complex)
    coefficients[0] = 1.0
    # a rank-one truth: the exact half-shift amplitude of the member.
    from applications.targets import exact_amplitude

    family, baseline = target.family, target.baseline
    shift = float(family.natural_parameter(target.members[0])) - float(
        family.natural_parameter(baseline)
    )
    coefficients = exact_amplitude(family, baseline, shift, 12).astype(complex)
    del exact

    sample = direct_sample(target, coefficients, 60_000, rng, grid)
    truth_mean = float(family.mean(target.members[0]))
    truth_var = float(family.variance(target.members[0]))
    assert abs(sample.mean() - truth_mean) < 6.0 * np.sqrt(truth_var / len(sample))
    assert abs(sample.var() - truth_var) < 0.05 * truth_var


def test_ghs_has_no_forward_process():
    baseline = GHSParams(0.0, 2.0)
    target_like = build_target("normal", "shifted", None, 1.2)
    ghs_target = type(target_like)(
        label="ghs shifted",
        family=GHS,
        baseline=baseline,
        sample=lambda size, rng: np.asarray(GHS.sample(baseline, size, rng=rng)),
        density=lambda x: np.asarray(GHS.prob(x, baseline), dtype=float),
        members=(baseline,),
    )
    with pytest.raises(NotImplementedError, match="positivity"):
        fit_schedule(
            ghs_target,
            np.array([1.0, 0.0]),
            4,
            2_000,
            0.0,
            np.random.default_rng(0),
        )
