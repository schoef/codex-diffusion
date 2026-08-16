"""Checks for the complex amplitude and the convex relaxation."""

from pathlib import Path

import numpy as np
import pytest

from applications.amplitude_fit_complex import (
    _gauge,
    _gauss_rule,
    _objective,
    block_rank_two_seed,
    build_target,
    certified_gap,
    distance,
    factorise_state,
    fit_complex_amplitude,
    fitting_matrices,
    law_from_amplitude,
    law_from_localised,
    localised_matrices,
    optimality_test,
    plot_comparison,
    project_to_states,
    ratio_coefficients_complex,
    relaxed_optimum,
    run_comparison,
    split_localised,
    terminating_degree,
)
from applications.amplitude_fit_degree import reference_coefficients, target_grid
from applications.amplitude_fit_recovery import (
    FAMILY_NAMES,
    TARGETS,
    fit_amplitude,
    product_matrices,
)

DEGREE = 5
GAP = 2.0
# families whose mixture at this separation leaves the reachable set at this
# truncation: the density ratio grows too fast for a degree-2K expansion, and
# every method agrees on a fit that is useless. They are exercised on the other
# two targets instead.
UNREACHABLE_MIXTURES = ("gamma", "negative-binomial")


def _problem(family_name, kind="truncated", degree=DEGREE):
    """Return the product matrices and the exact coefficients of one target."""

    family, baseline, _ = TARGETS[family_name]
    target = build_target(family_name, kind, None, GAP)
    grid = target_grid(target)
    return (
        product_matrices(family, baseline, degree),
        reference_coefficients(target, 2 * degree, grid),
        family,
        baseline,
    )


def test_the_gauge_puts_the_leading_coefficient_on_the_positive_real_axis():
    """Norm and global phase are the two continuous redundancies; both are fixed."""

    c = _gauge(np.array([1.0 + 1.0j, 2.0 - 0.5j, -3.0j]))
    assert float(np.linalg.norm(c)) == pytest.approx(1.0, abs=1e-14)
    assert c[0].real > 0.0
    assert c[0].imag == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_the_law_is_blind_to_the_phase_and_to_conjugation(family_name):
    """The two redundancies of Proposition "the relaxation is the complex Born
    model" must be exact, not approximate: the coefficient map is what the fit
    sees, and it may not distinguish them."""

    phi, _, _, _ = _problem(family_name)
    rng = np.random.default_rng(3)
    c = _gauge(rng.normal(size=DEGREE + 1) + 1j * rng.normal(size=DEGREE + 1))

    reference = ratio_coefficients_complex(c, phi)
    rotated = ratio_coefficients_complex(np.exp(0.7j) * c, phi)
    conjugated = ratio_coefficients_complex(np.conj(c), phi)
    assert np.allclose(reference, rotated, atol=1e-13)
    assert np.allclose(reference, conjugated, atol=1e-13)
    # Phi_0 is the identity, so R_0 is the norm
    assert reference[0] == pytest.approx(1.0, abs=1e-13)


def test_the_projection_returns_a_state():
    """The relaxation stays feasible by projection, so the projection must land
    exactly on the set and leave a state alone."""

    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(6, 6))
    state = project_to_states(matrix + matrix.T)
    assert np.trace(state) == pytest.approx(1.0, abs=1e-12)
    assert float(np.linalg.eigvalsh(state)[0]) > -1e-12
    assert np.allclose(state, project_to_states(state), atol=1e-12)


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_factorisation_reproduces_the_law_of_a_state(family_name):
    """A state's law is a sum of squares of real polynomials, hence nonnegative
    on the whole line, hence ``P Pbar`` with ``deg P <= K``. The amplitude built
    from the roots must return the coefficients of the state it came from."""

    phi, _, family, baseline = _problem(family_name)
    rng = np.random.default_rng(17)
    factor = rng.normal(size=(DEGREE + 1, 3))
    rho = factor @ factor.T
    rho = rho / np.trace(rho)

    c = factorise_state(family, baseline, rho, DEGREE)
    expected = np.einsum("knm,mn->k", phi, rho)
    assert np.allclose(ratio_coefficients_complex(c, phi), expected, atol=1e-8)


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_the_complex_fit_never_loses_to_the_real_fit(family_name):
    """The real fit is the complex one restricted to ``b = 0``, so it cannot win.

    The complex fit is started from the factorised convex optimum, which is the
    route that does not depend on the local optimiser finding its way.
    """
    phi, observed, family, baseline = _problem(family_name)
    real = fit_amplitude(phi, observed)
    relaxed = relaxed_optimum(phi, observed)
    seed = factorise_state(family, baseline, relaxed["matrix"], DEGREE)
    complex_fit = fit_complex_amplitude(phi, observed, initial=seed)

    assert complex_fit["objective"] <= real["objective"] * (1.0 + 1e-9)


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_the_complex_fit_attains_the_convex_optimum(family_name):
    """The claim of the section: the relaxation's lower bound is reached by an
    amplitude. The convex value is certified by its own duality gap, so the
    comparison does not rest on the convex solver having converged."""

    phi, observed, family, baseline = _problem(family_name)
    relaxed = relaxed_optimum(phi, observed)
    seed = factorise_state(family, baseline, relaxed["matrix"], DEGREE)
    complex_fit = fit_complex_amplitude(phi, observed, initial=seed)

    scale = max(relaxed["objective"], 1e-30)
    assert (
        complex_fit["objective"] >= relaxed["objective"] - abs(relaxed["gap"]) - 1e-14
    )
    assert (
        complex_fit["objective"]
        <= relaxed["objective"] + 10.0 * abs(relaxed["gap"]) + 1e-9 * scale
    )


def test_the_eigenvalue_test_predicts_the_descent_it_offers():
    """Proposition "when a real fit is optimal for the relaxation" also states
    the size of the improvement, ``eps^2 (v^T M v - mu)``. Checking the constant
    and not merely its sign is what tests the derivation.
    """
    phi, observed, _, _ = _problem("normal", kind="truncated")
    real = fit_amplitude(phi, observed)
    a = real["coefficients"]
    test = optimality_test(phi, observed, a.astype(complex))
    v = test["direction"]

    predicted = float(v @ test["matrix"] @ v) - test["mu"]
    assert predicted < 0.0

    errors = []
    for epsilon in (4e-3, 2e-3, 1e-3):
        c = (a + 1j * epsilon * v) / np.sqrt(1.0 + epsilon**2)
        observed_change = _objective(phi, observed, c) - real["objective"]
        errors.append(abs(observed_change - epsilon**2 * predicted))
    # the neglected term is O(eps^4), so halving eps must cut the error by ~16
    assert errors[0] > 8.0 * errors[1]
    assert errors[1] > 8.0 * errors[2]


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_a_real_fit_is_a_stationary_point_of_the_complex_problem(family_name):
    """The gradient in ``b`` vanishes identically at ``b = 0``, so an optimiser
    started at a real vector stays real. This is why the eigenvalue test, and
    not a restart, is what finds the second square."""

    phi, observed, _, _ = _problem(family_name)
    real = fit_amplitude(phi, observed)
    stayed = fit_complex_amplitude(
        phi, observed, initial=real["coefficients"].astype(complex)
    )
    assert float(np.linalg.norm(np.imag(stayed["coefficients"]))) < 1e-10
    assert stayed["objective"] == pytest.approx(real["objective"], rel=1e-6)


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_the_relaxation_lower_bounds_both_amplitude_fits(family_name):
    """The rank-one matrices are feasible for the relaxation, so its optimum is
    a lower bound. A violation beyond the certified gap would mean the convex
    solver had left the feasible set."""

    for kind in ("shifted", "mixture", "truncated"):
        if kind == "mixture" and family_name in UNREACHABLE_MIXTURES:
            continue
        phi, observed, _, _ = _problem(family_name, kind=kind)
        relaxed = relaxed_optimum(phi, observed)
        real = fit_amplitude(phi, observed)
        assert relaxed["objective"] <= real["objective"] + abs(relaxed["gap"]) + 1e-12


def test_the_comparison_figure_is_written(tmp_path):
    """A smoke test for the figure: every panel must survive real data."""

    result = run_comparison("normal", "truncated", degree=3)
    path = plot_comparison(result, output_dir=tmp_path)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_the_matched_coefficients_stop_where_the_basis_does():
    """Off a lattice the product reaches degree ``2K``; on one it folds back into
    the ``N + 1`` coefficients that exist, and asking for more is impossible
    rather than merely demanding."""

    family, baseline, _ = TARGETS["normal"]
    assert fitting_matrices(family, baseline, 5).shape == (11, 6, 6)

    family, baseline, _ = TARGETS["binomial"]
    trials = terminating_degree(family, baseline)
    assert trials == 12
    assert fitting_matrices(family, baseline, 4).shape == (9, 5, 5)
    assert fitting_matrices(family, baseline, 8).shape == (13, 9, 9)
    with pytest.raises(ValueError, match="terminates"):
        fitting_matrices(family, baseline, trials + 1)


def test_a_lattice_amplitude_is_exact_at_the_terminating_degree():
    """On ``N + 1`` points a degree-``N`` amplitude spans every function, so its
    square spans every nonnegative one. The hard-edge target that no truncation
    below ``N`` can reach must therefore be fitted exactly, and by a real
    amplitude: there is nothing left for a second square to do."""

    family, baseline, _ = TARGETS["binomial"]
    trials = terminating_degree(family, baseline)
    target = build_target("binomial", "truncated", None, GAP)
    grid = target_grid(target)

    phi = fitting_matrices(family, baseline, trials)
    observed = reference_coefficients(target, phi.shape[0] - 1, grid)
    real = fit_amplitude(phi, observed)
    law = law_from_amplitude(target, real["coefficients"], grid)

    assert real["objective"] < 1e-16
    assert distance(target, law, grid) < 1e-8


def test_the_localised_model_puts_the_normalisation_back_on_the_sphere():
    """The wall coupling is positive definite, so rescaling by its square root
    turns ``R_0 = 1`` into the plain norm and ``Xi_0`` into the identity. That is
    what lets every routine written for the whole line apply unchanged."""

    family, baseline, _ = TARGETS["gamma"]
    xi, _ = localised_matrices(family, baseline, "gamma", DEGREE)
    assert xi.shape == (2 * DEGREE + 1, 2 * DEGREE + 1, 2 * DEGREE + 1)
    assert np.allclose(xi[0], np.eye(2 * DEGREE + 1), atol=1e-9)
    for matrix in xi:
        assert np.allclose(matrix, matrix.T, atol=1e-9)


def test_the_localised_coefficients_are_the_law_they_claim():
    """``R_k`` from the tensor must equal the quadrature of
    ``q = |h_0|^2 + pi |h_1|^2``, which is what makes the second block mean what
    the section says it means."""

    family, baseline, _ = TARGETS["gamma"]
    xi, inverse_root = localised_matrices(family, baseline, "gamma", DEGREE)
    rng = np.random.default_rng(5)
    v = rng.normal(size=2 * DEGREE + 1) + 1j * rng.normal(size=2 * DEGREE + 1)
    v = v / np.linalg.norm(v)

    c0, c1 = split_localised(v, inverse_root)
    nodes, weights = _gauss_rule(family, baseline, 2 * DEGREE + 1)

    def modulus(c):
        real = np.asarray(family.basis_dot(nodes, np.real(c), baseline), dtype=float)
        imag = np.asarray(family.basis_dot(nodes, np.imag(c), baseline), dtype=float)
        return real**2 + imag**2

    law = modulus(c0) + nodes * modulus(c1)
    basis = np.asarray(family.basis(nodes, xi.shape[0] - 1, baseline), dtype=float)
    assert np.allclose(
        ratio_coefficients_complex(v, xi), basis.T @ (weights * law), atol=1e-9
    )


def test_the_wall_is_what_makes_the_gamma_shape_ratio_reachable():
    """The section's own example. ``dGamma(r+1)/dGamma(r) = x / (r theta)`` is a
    valid ratio on the support and negative off it, so a globally nonnegative sum
    of squares cannot represent it at any truncation, while the localised class
    represents it exactly. Both optima are certified, so the comparison is not a
    statement about either optimiser."""

    family, baseline, _ = TARGETS["gamma"]
    degree = 3
    target = build_target("gamma", "shape", None, GAP)
    grid = target_grid(target)

    phi = fitting_matrices(family, baseline, degree)
    xi, inverse_root = localised_matrices(family, baseline, "gamma", degree)
    observed = reference_coefficients(target, phi.shape[0] - 1, grid)

    globally = relaxed_optimum(phi, observed)
    locally = relaxed_optimum(xi, observed)

    # the localised class contains the global one -- take sigma_1 = 0
    assert locally["objective"] <= globally["objective"] + abs(globally["gap"])
    # and it is strictly better here, by far more than either certificate
    assert globally["objective"] - locally["objective"] > 100.0 * max(
        abs(globally["gap"]), abs(locally["gap"])
    )
    assert locally["objective"] < 1e-9
    assert globally["objective"] > 1e-5


def test_the_localised_amplitude_pair_attains_the_localised_optimum():
    """The half-line Markov-Lukacs statement is that each block is a sum of two
    squares, so one complex amplitude per block suffices."""

    family, baseline, _ = TARGETS["gamma"]
    degree = 3
    target = build_target("gamma", "shape", None, GAP)
    grid = target_grid(target)
    xi, inverse_root = localised_matrices(family, baseline, "gamma", degree)
    observed = reference_coefficients(target, xi.shape[0] - 1, grid)

    relaxed = relaxed_optimum(xi, observed)
    seed = block_rank_two_seed(relaxed["matrix"], degree + 1)
    fit = fit_complex_amplitude(xi, observed, initial=seed)

    assert fit["objective"] < 1e-8
    assert certified_gap(xi, observed, fit["coefficients"]) < 1e-6
    # the law it defines is the target's, to the accuracy of the fit
    law = law_from_localised(target, fit["coefficients"], inverse_root, "gamma", grid)
    assert distance(target, law, grid) < 1e-4


def test_a_family_without_a_wall_is_refused():
    """On R the global class is already exact, so a localised model there would
    be a silently redundant second block."""

    family, baseline, _ = TARGETS["normal"]
    with pytest.raises(ValueError, match="no wall"):
        localised_matrices(family, baseline, "normal", DEGREE)
