"""Checks for the one-channel amplitude machinery."""

import numpy as np
import pytest

from applications.amplitude_fit_degree import (
    build_targets,
    integrate,
    mixture_target,
    shifted_target,
    target_grid,
    total_variation,
)
from applications.amplitude_fit_recovery import (
    FAMILY_NAMES,
    TARGETS,
    exact_amplitude,
    exact_ratio_coefficients,
    fit_amplitude,
    product_matrices,
    ratio_coefficients,
)
from applications.baseline_matching import (
    moment_matched,
    select_degree_by_likelihood,
    separation_for_gap,
    shift_reaching_mean,
)
from applications.paper_one_channel import continued_fit, separation_for, width_ratio

DEGREE = 5
# the Krawtchouk basis terminates at N = 12, so 2K must stay within it
LOW_DEGREE, HIGH_DEGREE = 4, 6


def _band_discrepancy(family, baseline, shift, degree):
    """Return ``max_k |R_k(c_exact) - gamma_k z^k|`` over the band ``k <= K``."""

    phi = product_matrices(family, baseline, degree)
    exact = exact_amplitude(family, baseline, shift, degree)
    unit = exact / np.linalg.norm(exact)
    expected = exact_ratio_coefficients(family, baseline, shift, 2 * degree)
    return float(np.max(np.abs(ratio_coefficients(unit, phi) - expected)[: degree + 1]))


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_product_matrices_reproduce_the_shift_coefficients(family_name):
    """``R_k(c_exact)`` must approach ``gamma_k z^k`` as the truncation is raised.

    The two sides come from different code paths -- the linearisation tensor and
    the shift machinery -- so agreement cross-checks the conventions. It cannot
    be exact at fixed ``K``, because the exact amplitude has been truncated; what
    identifies the residue as truncation rather than a convention error is that
    it falls sharply with ``K``.
    """
    family, baseline, shift = TARGETS[family_name]
    coarse = _band_discrepancy(family, baseline, shift, LOW_DEGREE)
    fine = _band_discrepancy(family, baseline, shift, HIGH_DEGREE)
    assert fine < coarse / 5.0
    assert fine < 5e-3


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_zeroth_ratio_coefficient_is_the_norm(family_name):
    """``Phi_0`` is the identity, so ``R_0`` is ``||c||^2`` and carries no shape."""

    family, baseline, _ = TARGETS[family_name]
    phi = product_matrices(family, baseline, DEGREE)
    assert np.allclose(phi[0], np.eye(DEGREE + 1), atol=1e-10)


def _population_error(family, baseline, shift, degree):
    """Return the population fit's distance from the exact truncated amplitude."""

    phi = product_matrices(family, baseline, degree)
    exact = exact_amplitude(family, baseline, shift, degree)
    unit = exact / np.linalg.norm(exact)
    target = exact_ratio_coefficients(family, baseline, shift, 2 * degree)
    return float(np.linalg.norm(fit_amplitude(phi, target)["coefficients"] - unit))


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_population_fit_recovers_the_exact_amplitude(family_name):
    """With exact coefficients the fit must approach the truth as ``K`` rises."""

    family, baseline, shift = TARGETS[family_name]
    coarse = _population_error(family, baseline, shift, LOW_DEGREE)
    fine = _population_error(family, baseline, shift, HIGH_DEGREE)
    assert fine < coarse / 5.0
    assert fine < 5e-3


def test_fit_stays_on_the_sphere_with_a_positive_leading_sign():
    """Normalisation is the constraint, and the sign is fixed to break the fibre."""

    family, baseline, shift = TARGETS["gamma"]
    phi = product_matrices(family, baseline, DEGREE)
    target = exact_ratio_coefficients(family, baseline, shift, 2 * DEGREE)
    c = fit_amplitude(phi, target)["coefficients"]
    assert float(np.linalg.norm(c)) == pytest.approx(1.0, abs=1e-12)
    assert c[0] > 0.0


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_continuation_is_never_worse_than_a_cold_start(family_name):
    """Continuing in ``K`` must not lose to starting from the baseline vector.

    The continuation keeps the better of the two at every degree, so this is a
    guard against the padding or the normalisation being wrong rather than a
    statement about optimisation.
    """
    family, baseline, shift = TARGETS[family_name]
    degree = 6 if family_name == "binomial" else 10
    target = exact_ratio_coefficients(family, baseline, shift, 2 * degree)
    phi = product_matrices(family, baseline, degree)

    def objective(c):
        residual = ratio_coefficients(c, phi)[1:] - target[1 : 2 * degree + 1]
        return float(np.sum(residual**2))

    continued = continued_fit(family, baseline, target, degree)
    cold = fit_amplitude(phi, target)["coefficients"]
    assert objective(continued) <= objective(cold) * (1.0 + 1e-9)
    assert float(np.linalg.norm(continued)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_moment_matching_hits_the_moments_it_claims(family_name):
    """Matching must reach both moments, or report that it could not."""

    family, template, _ = TARGETS[family_name]
    mean, variance = (
        1.3 * float(family.mean(template)) + 0.4,
        2.0 * float(family.variance(template)),
    )
    matched, status = moment_matched(family, template, mean, variance)
    assert float(family.mean(matched)) == pytest.approx(mean, rel=1e-9)
    if status == "mean and variance":
        assert float(family.variance(matched)) == pytest.approx(variance, rel=1e-6)
    else:
        # Poisson has no dispersion parameter and must say so rather than pretend
        assert status in {"mean only", "unreachable"}


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_shift_reaching_mean_inverts_the_mean_map(family_name):
    """Solving for the shift that reaches a mean must reproduce that mean."""

    family, template, _ = TARGETS[family_name]
    wanted = 1.2 * float(family.mean(template)) + 0.3
    step = shift_reaching_mean(family, template, wanted)
    reached = float(family.mean(family.shifted_params(template, step)))
    assert reached == pytest.approx(wanted, rel=1e-8)


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_separation_for_gap_is_achieved_and_capped(family_name):
    """The gap must be met, and the capped separation must respect the cap."""

    family, template, _ = TARGETS[family_name]
    step, status = separation_for_gap(family, template, 2.0)
    if status == "exact":
        plus = family.shifted_params(template, step)
        minus = family.shifted_params(template, -step)
        spread = 0.5 * (
            float(np.sqrt(family.variance(plus)))
            + float(np.sqrt(family.variance(minus)))
        )
        gap = (float(family.mean(plus)) - float(family.mean(minus))) / spread
        assert gap == pytest.approx(2.0, rel=1e-6)

    capped = separation_for(family_name, 2.0)
    assert capped > 0.0
    assert width_ratio(family, template, capped) <= 4.5 + 1e-6


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_total_variation_is_zero_for_the_target_itself(family_name):
    """The metric must vanish on the truth and be bounded by one."""

    family, baseline, shift = TARGETS[family_name]
    target = shifted_target(family_name, shift)
    grid = target_grid(target)
    mass = integrate(family, baseline, grid, target.density(grid))
    assert mass == pytest.approx(1.0, abs=1e-6)

    # the baseline itself, as an amplitude, is the unit vector
    unit = np.zeros(DEGREE + 1)
    unit[0] = 1.0
    baseline_target = shifted_target(family_name, 0.0)
    distance = total_variation(baseline_target, unit, target_grid(baseline_target))
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_degree_selection_prefers_a_small_degree_for_a_simple_target():
    """A single shift needs few degrees, and the rule should find that."""

    family, baseline, shift = TARGETS["poisson"]
    target = shifted_target("poisson", shift)
    rng = np.random.default_rng(0)
    sample = target.sample(20000, rng)
    chosen = select_degree_by_likelihood(family, baseline, sample, (2, 3, 4, 6, 8, 12))
    assert chosen <= 6


def test_build_targets_skips_shifts_the_family_cannot_represent(capsys):
    """A separation outside the parameter range is dropped with a note."""

    targets = build_targets("negative-binomial", 0.1, (0.4, 0.8), 0.8)
    labels = [t.label for t in targets]
    assert not any("d=0.8" in label for label in labels)
    assert "skipping" in capsys.readouterr().out


def test_mixture_density_integrates_to_one():
    """The mixture sampler and its density must describe the same law."""

    target = mixture_target("poisson", 0.4)
    family, baseline = target.family, target.baseline
    grid = target_grid(target)
    density = target.density(grid)
    assert integrate(family, baseline, grid, density) == pytest.approx(1.0, abs=1e-9)

    rng = np.random.default_rng(1)
    sample = target.sample(200000, rng)
    expected = integrate(family, baseline, grid, density * grid)
    assert float(sample.mean()) == pytest.approx(expected, rel=2e-2)
