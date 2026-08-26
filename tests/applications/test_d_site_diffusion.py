"""d-site diffusion: MPS algebra, bond fits, latent readout on the chain."""

import numpy as np
import pytest

from applications.amplitude_fit_complex import (
    fitting_matrices,
    ratio_coefficients_complex,
)
from applications.d_site_diffusion import (
    bond_pair_law,
    model_nll,
    run_chain_study,
)
from applications.mps_amplitude import (
    bond_spectrum,
    evaluate,
    norm_squared,
    pair_moment_matrix,
    right_canonicalise,
    sequential_sample,
    split_bond,
    vacuum_state,
)
from applications.two_site_diffusion import (
    build_pair_target,
    exact_pair_amplitude,
    exact_pair_coefficients,
    pair_density,
)


def _random_state(rng, d, degree, chi):
    shapes = [(1, degree + 1, chi)]
    shapes += [(chi, degree + 1, chi)] * (d - 2)
    shapes += [(chi, degree + 1, 1)]
    return [rng.normal(size=shape) + 1j * rng.normal(size=shape) for shape in shapes]


def test_canonicalisation_preserves_the_state():
    rng = np.random.default_rng(0)
    target = build_pair_target("poisson", 0.57, 0.0)
    family, baseline = target.family, target.baseline
    tensors = _random_state(rng, 4, 4, 3)
    canon = right_canonicalise(tensors)
    assert abs(norm_squared(canon) - 1.0) < 1e-10
    x = np.column_stack(
        [np.asarray(family.sample(baseline, 6, rng=rng)) for _ in range(4)]
    ).astype(float)
    ratio = evaluate(tensors, family, baseline, x) / evaluate(
        canon, family, baseline, x
    )
    assert np.std(np.abs(ratio)) / np.mean(np.abs(ratio)) < 1e-10


def test_product_state_pair_moments():
    rng = np.random.default_rng(1)
    target = build_pair_target("poisson", 0.57, 0.0)
    phi = fitting_matrices(target.family, target.baseline, 4)
    c1 = rng.normal(size=5) + 1j * rng.normal(size=5)
    c2 = rng.normal(size=5) + 1j * rng.normal(size=5)
    c1, c2 = c1 / np.linalg.norm(c1), c2 / np.linalg.norm(c2)
    product = [c1.reshape(1, 5, 1), c2.reshape(1, 5, 1)]
    moments = pair_moment_matrix(product, phi, 0)
    outer = np.outer(
        ratio_coefficients_complex(c1, phi), ratio_coefficients_complex(c2, phi)
    )
    assert np.allclose(moments, outer, atol=1e-12)


def test_exact_pair_amplitude_is_bond_two():
    """The eps = 0 branch amplitude embeds exactly at chi = 2, and its MPS
    pair moments reproduce the exact mixture law."""

    target = build_pair_target("poisson", 0.57, 0.0)
    phi = fitting_matrices(target.family, target.baseline, 5)
    matrix = exact_pair_amplitude(target, 0.0, 5)
    first, second, discarded = split_bond(
        matrix.reshape(1, 6, 6, 1), 2, center_right=False
    )
    assert discarded < 1e-20
    mps = [first, second]
    exact = exact_pair_coefficients(target, 0.0, phi.shape[0] - 1)
    assert np.linalg.norm(pair_moment_matrix(mps, phi, 0) - exact) < 5e-3
    spectrum = bond_spectrum(mps, 0)
    assert spectrum[1] > 0.1
    assert spectrum[2] < 1e-12


def test_bond_pair_law_normalises():
    target = build_pair_target("poisson", 0.57, 0.0)
    family, baseline = target.family, target.baseline
    matrix = exact_pair_amplitude(target, 0.0, 5)
    grid = np.arange(0, 60)
    law = bond_pair_law(matrix.reshape(1, 6, 6, 1), family, baseline, grid)
    assert abs(law.sum() - 1.0) < 1e-6
    truth = pair_density(target, 0.0, grid)
    assert 0.5 * np.abs(law - truth).sum() < 5e-3


def test_sequential_sampler_matches_the_baseline():
    rng = np.random.default_rng(3)
    target = build_pair_target("poisson", 0.57, 0.0)
    family, baseline = target.family, target.baseline
    grid = np.arange(0, 60)
    draws = sequential_sample(vacuum_state(3, 4), family, baseline, 40_000, grid, rng)
    mean = float(family.mean(baseline))
    variance = float(family.variance(baseline))
    for site in range(3):
        assert abs(draws[:, site].mean() - mean) < 6.0 * np.sqrt(variance / 40_000)


@pytest.fixture(scope="module")
def chain_study():
    return run_chain_study(
        "poisson",
        d=4,
        n_slices=3,
        degree=4,
        chi=2,
        sweeps=2,
        draws=16_000,
        seed=7,
    )


def test_chain_slices_are_accurate(chain_study):
    for entry in chain_study["slices"]:
        assert entry["pair_tv"] < 0.06
        assert np.isfinite(entry["nll_gap"])
    assert chain_study["slices"][-1]["nll_gap"] < 0.25


def test_chain_recovers_the_latent_at_the_data_slice(chain_study):
    final = chain_study["slices"][-1]
    assert abs(final["shifts"][0] - 0.285) < 0.08
    assert abs(final["shifts"][1] + 0.285) < 0.08
    # the weight estimate is noisier than the shifts at fixture scale
    assert abs(final["epsilon_hat"] - 0.05) < 0.15
    assert final["bond_spectrum"][1] > 0.05


def test_chain_generation_reaches_the_target(chain_study):
    generation = chain_study["generation"]
    assert np.mean(generation["site_tv"]) < 0.08
    assert generation["pair_sample_tv"] < 0.2


def test_chain_model_nll_is_finite(chain_study):
    final = chain_study["slices"][-1]
    target = chain_study["target"]
    sample = np.asarray(target.sample(500, np.random.default_rng(0)))
    assert np.isfinite(model_nll(final["tensors"], target, sample))
