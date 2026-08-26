"""The joint (theta, c) fit removes the displacement wall."""

import numpy as np

from applications.amplitude_fit_complex import (
    continued_complex_fit,
    fitting_matrices,
    law_from_amplitude,
)
from applications.amplitude_fit_recentred import (
    _distance,
    _true_arclength,
    fit_recentred,
    mean_at_arclength,
    with_mean,
)
from applications.targets import (
    TARGETS,
    Target,
    empirical_coefficients,
    shifted_target,
    support_grid,
)


def test_arclength_integration_roundtrips():
    family, baseline, _ = TARGETS["poisson"]
    theta = 2.0 * (np.sqrt(10.0) - np.sqrt(6.0))
    mu = mean_at_arclength(family, baseline, theta)
    assert abs(mu - 10.0) < 1e-3


def test_recentred_fit_beats_the_wall():
    """A displaced Poisson member at K = 4: the plain fit fails, the joint
    fit reaches the sampling floor and recovers the Fisher location."""

    name, offset, degree = "poisson", 1.0, 4
    family, baseline, _ = TARGETS[name]
    target = shifted_target(name, offset)
    rng = np.random.default_rng(7)
    sample = np.asarray(target.sample(30_000, rng))
    grid = support_grid(family, baseline, target.members)

    phi = fitting_matrices(family, baseline, degree)
    empirical, _ = empirical_coefficients(family, baseline, sample, phi.shape[0] - 1)
    plain = continued_complex_fit(phi, empirical)["complex"]
    plain_tv = _distance(
        target, law_from_amplitude(target, plain["coefficients"], grid), grid
    )

    fit = fit_recentred(family, baseline, sample, degree)
    moved = Target(
        label=target.label,
        family=family,
        baseline=fit["member"],
        sample=target.sample,
        density=target.density,
        members=target.members,
    )
    tv = _distance(target, law_from_amplitude(moved, fit["coefficients"], grid), grid)

    assert plain_tv > 0.1
    assert tv < 0.02
    true_theta = _true_arclength(family, baseline, target.members[0])
    assert abs(fit["theta"] - true_theta) < 0.05


def test_moment_gauge_is_reached():
    """After the orbit polish the fitted law's first moment sits on theta."""

    family, baseline, _ = TARGETS["normal"]
    target = shifted_target("normal", 2.0)
    rng = np.random.default_rng(3)
    sample = np.asarray(target.sample(20_000, rng))
    fit = fit_recentred(family, baseline, sample, 4)
    fitted_mean = float(family.mean(fit["member"]))
    assert abs(fitted_mean - sample.mean()) < 0.05
    # residual shape is mean-free relative to the moved baseline
    grid = support_grid(family, baseline, target.members)
    moved = Target(
        label="",
        family=family,
        baseline=fit["member"],
        sample=target.sample,
        density=target.density,
        members=target.members,
    )
    law = law_from_amplitude(moved, fit["coefficients"], grid)
    law_mean = float(np.trapezoid(grid * law, grid))
    assert abs(law_mean - fitted_mean) < 0.05


def test_with_mean_keeps_shape():
    family, baseline, _ = TARGETS["gamma"]
    member = with_mean(family, baseline, 5.0)
    assert float(family.mean(member)) == 5.0
    assert member.r == baseline.r
