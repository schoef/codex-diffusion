"""The two-site diffusion model: the smallest case with a bond.

Two sites carry a shared latent: the emissions of the two-state hidden Markov
toy at lag one. The model is one complex amplitude on the product basis,
``h(x_1, x_2) = sum C[k_1, k_2] phi_{k_1}(x_1) phi_{k_2}(x_2)``, so the whole
one-site fitting machinery applies verbatim on the Kronecker stack
``Phi_j (x) Phi_k`` with ``c = vec(C)`` — and the bond dimension of the model
is the Schmidt rank of ``C``. What one site cannot see becomes measurable
here: the cross moments ``E[phi_j(X_1) phi_k(X_2)]`` form a low-rank matrix
whose factors are the latent components, so the factorisation chambers that
are likelihood-degenerate at one site split apart.

One class fact is exercised deliberately: a pure amplitude gives a pair
density matrix of rank at most two, while the hidden-Markov law at flip
probability ``0 < epsilon`` is a rank-four mixture. The pure model is exact
at ``epsilon = 0`` (perfect persistence) and at ``epsilon = 1/2`` (independent
sites, bond one), and carries a class floor in between.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.amplitude_fit_complex import (
    certified_gap,
    continued_complex_fit,
    fit_complex_amplitude,
    fitting_matrices,
    ratio_coefficients_complex,
    terminating_degree,
)
from applications.one_site_diffusion import direct_sample
from applications.targets import FAMILY_NAMES, exact_amplitude, support_grid
from applications.two_state_hmm import (
    BASELINES,
    emission_parameters,
    noised_emissions,
    sample_latent,
    sample_observations,
    transition_matrix,
)

DEFAULT_SLICES = 8
DEFAULT_DEGREE = 6
DEFAULT_DRAWS = 60_000
DEFAULT_SEPARATION = 0.57
DEFAULT_EPSILON = 0.05
DEFAULT_TAU = 0.5
DEFAULT_SEED = 5
TERMINAL_FRACTION = 1e-4
FIGURE_SUBDIRECTORY = "two_site_diffusion"


# ------------------------------------------------------------------ target --
@dataclass
class PairTarget:
    """The lag-one pair law of the two-state hidden Markov toy."""

    label: str
    family: Any
    baseline: Any
    epsilon: float
    plus: Any
    minus: Any
    z: np.ndarray

    @property
    def weights(self) -> np.ndarray:
        """Joint latent weights: stationary 1/2 times the transition row."""

        return 0.5 * transition_matrix(self.epsilon)

    def members_at(self, t: float) -> tuple[Any, Any]:
        return noised_emissions(self.family, self.baseline, self.z, t)

    def sample(self, size: int, rng: Any) -> np.ndarray:
        states = sample_latent(2, size, self.epsilon, rng)
        return sample_observations(self.family, self.plus, self.minus, states, rng)


def build_pair_target(name: str, separation: float, epsilon: float) -> PairTarget:
    family, baseline, _ = BASELINES[name]
    plus, minus, z = emission_parameters(family, baseline, separation)
    return PairTarget(
        label=f"hmm pair sep={separation:g} eps={epsilon:g}",
        family=family,
        baseline=baseline,
        epsilon=epsilon,
        plus=plus,
        minus=minus,
        z=z,
    )


def pair_density(target: PairTarget, t: float, grid: np.ndarray) -> np.ndarray:
    """Return the exact noised joint density on ``grid x grid``."""

    family = target.family
    members = target.members_at(t)
    columns = [np.asarray(family.prob(grid, member), dtype=float) for member in members]
    weights = target.weights
    law = np.zeros((grid.size, grid.size))
    for i, first in enumerate(columns):
        for j, second in enumerate(columns):
            law += weights[i, j] * np.outer(first, second)
    return law


def exact_pair_coefficients(target: PairTarget, t: float, k_max: int) -> np.ndarray:
    """Return the exact ``R_{jk}(t) = E_{p_t}[phi_j(X_1) phi_k(X_2)]`` matrix."""

    family, baseline = target.family, target.baseline
    rows = []
    for member in target.members_at(t):
        shift = float(family.natural_parameter(member)) - float(
            family.natural_parameter(baseline)
        )
        rows.append(
            np.asarray(family.shift_coefficients(shift, k_max, baseline), dtype=float)
        )
    weights = target.weights
    matrix = np.zeros((k_max + 1, k_max + 1))
    for i, first in enumerate(rows):
        for j, second in enumerate(rows):
            matrix += weights[i, j] * np.outer(first, second)
    return matrix


def empirical_pair_coefficients(
    family: Any, baseline: Any, sample: np.ndarray, k_max: int
) -> np.ndarray:
    """Return ``mean phi_j(x_1) phi_k(x_2)`` over a sample of pairs."""

    first = np.asarray(family.basis(sample[:, 0], k_max, baseline), dtype=float)
    second = np.asarray(family.basis(sample[:, 1], k_max, baseline), dtype=float)
    return first.T @ second / len(sample)


# ----------------------------------------------------------- pair machinery --
def pair_fitting_matrices(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Kronecker stack ``Phi_j (x) Phi_k`` and the pair degrees.

    ``R_{jk}(C) = vec(C)^dagger (Phi_j (x) Phi_k) vec(C)``, so the one-site
    complex fit, its continuation and its certificate all run unchanged on
    this stack. The degrees ``j + k`` set the decay rates of the pair modes.
    """
    k_max = phi.shape[0] - 1
    n = phi.shape[1]
    stack = np.einsum("jab,kcd->jkacbd", phi, phi).reshape(
        (k_max + 1) ** 2, n * n, n * n
    )
    j, k = np.divmod(np.arange((k_max + 1) ** 2), k_max + 1)
    return stack, (j + k).astype(float)


def pair_chi_squared(coefficients: np.ndarray, degrees: np.ndarray, t: Any) -> Any:
    """Return ``chi^2(t) = sum_p R_p^2 exp(-2 d_p t)`` over pair modes."""

    flat = np.asarray(coefficients, dtype=float).reshape(-1)[1:]
    rates = degrees[1:]
    t_array = np.asarray(t, dtype=float)
    return np.sum(flat**2 * np.exp(-2.0 * rates * t_array[..., None]), axis=-1)


def build_pair_schedule(
    coefficients: np.ndarray,
    degrees: np.ndarray,
    n_slices: int,
    terminal_fraction: float = TERMINAL_FRACTION,
) -> np.ndarray:
    """Return slice times log-uniform in the pair chi-squared."""

    if n_slices < 1:
        raise ValueError("the schedule needs at least one slice")
    chi_zero = float(pair_chi_squared(coefficients, degrees, 0.0))
    if not np.isfinite(chi_zero) or chi_zero <= 0.0:
        raise ValueError("the empirical pair coefficients carry no signal")

    def solve(level: float) -> float:
        low, high = 0.0, 80.0
        for _ in range(200):
            middle = 0.5 * (low + high)
            if float(pair_chi_squared(coefficients, degrees, middle)) > level:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    fractions = terminal_fraction ** (np.arange(n_slices, 0, -1) / n_slices)
    times = [solve(chi_zero * fraction) for fraction in fractions]
    return np.array(times + [0.0])


def pair_blended_target(
    data_target: np.ndarray,
    previous: np.ndarray,
    delta: float,
    tau: float,
    degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold the consistency tie into the pair target, decay ``exp(-d_p delta)``."""

    decay = np.exp(-degrees * delta)
    tie_weight = tau * decay**2
    weights = 1.0 + tie_weight
    blended = (data_target + tau * decay * previous) / weights
    return blended, weights


def pair_law(target: PairTarget, matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return the model law ``p_ref x p_ref |h_C|^2`` on ``grid x grid``."""

    family, baseline = target.family, target.baseline
    k_max = matrix.shape[0] - 1
    basis = np.asarray(family.basis(grid, k_max, baseline), dtype=float)
    amplitude = basis @ matrix @ basis.T
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    return np.outer(reference, reference) * np.abs(amplitude) ** 2


def _cell_weights(family: Any, baseline: Any, grid: np.ndarray) -> np.ndarray:
    return np.ones_like(grid) if family.is_lattice(baseline) else np.gradient(grid)


def pair_total_variation(
    target: PairTarget, t: float, matrix: np.ndarray, grid: np.ndarray
) -> float:
    """Return the exact TV between the model and the noised pair law."""

    truth = pair_density(target, t, grid)
    law = pair_law(target, matrix, grid)
    cell = _cell_weights(target.family, target.baseline, grid)
    return 0.5 * float(np.sum(np.abs(truth - law) * np.outer(cell, cell)))


def held_out_pair_nll(
    target: PairTarget, matrix: np.ndarray, sample: np.ndarray
) -> float:
    """Return the exact held-out negative log-likelihood of a fitted slice."""

    family, baseline = target.family, target.baseline
    k_max = matrix.shape[0] - 1
    first = np.asarray(family.basis(sample[:, 0], k_max, baseline), dtype=float)
    second = np.asarray(family.basis(sample[:, 1], k_max, baseline), dtype=float)
    amplitude = np.einsum("nj,jk,nk->n", first, matrix, second)
    log_reference = np.asarray(
        family.log_prob(sample[:, 0], baseline), dtype=float
    ) + np.asarray(family.log_prob(sample[:, 1], baseline), dtype=float)
    return -float(np.mean(log_reference + np.log(np.abs(amplitude) ** 2 + 1e-300)))


def pair_floor(
    target: PairTarget,
    t: float,
    k_max: int,
    n_train: int,
    grid: np.ndarray,
) -> tuple[float, int]:
    """Return the typical-sample floor and supported pair count at slice ``t``.

    Same recipe as one site: the variance of every pair mode is truncated to
    the reachable region ``N p_t(x_1, x_2) dx >= 1`` of the joint law.
    """
    family, baseline = target.family, target.baseline
    # the reachability criterion needs a statistically meaningful cell, not
    # the quadrature step: on a fine continuous grid every 2-d cell carries
    # an expected count far below one. Coarsen to ~160 points per axis.
    if not family.is_lattice(baseline):
        step = max(1, grid.size // 160)
        grid = grid[::step]
    truth = pair_density(target, t, grid)
    cell = _cell_weights(family, baseline, grid)
    mass = truth * np.outer(cell, cell)
    mask = n_train * mass >= 1.0
    if not np.any(mask):
        return float("nan"), 0
    rows, columns = np.nonzero(mask)
    basis = np.asarray(family.basis(grid, k_max, baseline), dtype=float)
    features = basis[rows][:, :, None] * basis[columns][:, None, :]
    features = features.reshape(len(rows), (k_max + 1) ** 2)
    weighted = mass[mask][:, None]
    first = np.sum(weighted * features, axis=0)
    second = np.sum(weighted * features**2, axis=0)
    variances = np.maximum(second - first**2, 0.0)
    exact = exact_pair_coefficients(target, t, k_max).reshape(-1)
    noise = float(np.sum(variances[1:]) / n_train)
    floor = 0.5 * np.sqrt(noise)
    signal = np.abs(exact[1:])
    ratio = signal / np.sqrt(np.maximum(variances[1:], 1e-300) / n_train)
    supported = int(np.count_nonzero(ratio >= 1.0))
    return floor, supported


# ---------------------------------------------------------- latent structure --
def site_member_amplitude(
    target: PairTarget, member: Any, t: float, degree: int
) -> np.ndarray:
    """Return the unit-norm one-site amplitude of a contracted member."""

    family, baseline = target.family, target.baseline
    shift = float(family.natural_parameter(member)) - float(
        family.natural_parameter(baseline)
    )
    z = family.shift_coordinate(shift, baseline)
    moved = family.from_shift_coordinate(np.exp(-t) * z, baseline)
    moved_shift = float(family.natural_parameter(moved)) - float(
        family.natural_parameter(baseline)
    )
    vector = exact_amplitude(family, baseline, moved_shift, degree)
    return vector / np.linalg.norm(vector)


def site_density_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the site-1 reduced density matrix ``rho_1 = C C^dagger``."""

    return matrix @ matrix.conj().T


def hermitian_latent_weights(
    rho: np.ndarray, members: list[np.ndarray]
) -> tuple[np.ndarray, float]:
    """Gram-corrected member weights of a Hermitian density matrix."""

    projectors = [np.outer(v, v) for v in members]
    gram_squared = np.array(
        [[float(np.sum(p * q)) for q in projectors] for p in projectors]
    )
    overlaps = np.array([float(np.real(v @ rho @ v)) for v in members])
    weights = np.linalg.solve(gram_squared, overlaps)
    model = sum(w * p for w, p in zip(weights, projectors, strict=True))
    residual = float(np.linalg.norm(rho - model) / np.linalg.norm(rho))
    return weights, residual


def hermitian_steered_components(
    rho: np.ndarray, members: list[np.ndarray]
) -> list[tuple[float, np.ndarray]]:
    """Split a Hermitian rank-2 ``rho`` into two pure states, one per member.

    Same construction as the one-site split, but the decomposition freedom is
    now a 2 x 2 unitary, so a relative phase is scanned alongside the angle.
    """
    values, vectors = np.linalg.eigh(rho)
    root = vectors[:, -2:] * np.sqrt(np.maximum(np.real(values[-2:]), 0.0))

    def columns(theta: float, phase: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array(
            [[c, -s * np.exp(1j * phase)], [s * np.exp(-1j * phase), c]]
        )
        return root @ rotation

    def score(u: np.ndarray, order: list[int]) -> float:
        total = 0.0
        for column, member_index in enumerate(order):
            norm = float(np.real(u[:, column].conj() @ u[:, column]))
            if norm > 1e-300:
                overlap = abs(members[member_index] @ u[:, column]) ** 2
                total += overlap / norm
        return total

    best: tuple[float, float, float, list[int]] | None = None
    for order in ([0, 1], [1, 0]):
        for theta in np.linspace(0.0, np.pi, 61, endpoint=False):
            for phase in np.linspace(0.0, np.pi, 31, endpoint=False):
                u = columns(theta, phase)
                value = score(u, order)
                if best is None or value > best[0]:
                    best = (value, theta, phase, order)
    _, theta, phase, order = best
    step_theta, step_phase = np.pi / 61.0, np.pi / 31.0
    for theta_fine in np.linspace(theta - step_theta, theta + step_theta, 41):
        for phase_fine in np.linspace(phase - step_phase, phase + step_phase, 41):
            u = columns(theta_fine, phase_fine)
            value = score(u, order)
            if value > best[0]:
                best = (value, theta_fine, phase_fine, order)
    _, theta, phase, order = best
    u = columns(theta, phase)
    components: dict[int, tuple[float, np.ndarray]] = {}
    for column, member_index in enumerate(order):
        weight = float(np.real(u[:, column].conj() @ u[:, column]))
        vector = u[:, column] / np.sqrt(max(weight, 1e-300))
        components[member_index] = (weight, vector)
    return [components[index] for index in range(len(members))]


def shared_latent_site_means(
    target: PairTarget,
    matrix: np.ndarray,
    members: list[np.ndarray],
    grid: np.ndarray,
    rng: Any,
    n_toys: int,
    toy_size: int,
) -> np.ndarray:
    """Return toy means from the member-aligned split of the site-1 ``rho``."""

    rho = site_density_matrix(matrix)
    rho = rho / np.trace(rho).real
    components = hermitian_steered_components(rho, members)
    weights = np.array([max(w, 0.0) for w, _ in components])
    weights = weights / weights.sum()
    counts = rng.multinomial(n_toys, weights)
    means = []
    for count, (_, vector) in zip(counts, components, strict=True):
        if count == 0:
            continue
        draws = direct_sample(target, vector, count * toy_size, rng, grid)
        means.append(draws.reshape(count, toy_size).mean(axis=1))
    return np.concatenate(means)


def branch_expansion(
    matrix: np.ndarray, members: list[np.ndarray]
) -> tuple[np.ndarray, float]:
    """Expand the fitted amplitude in the member branches ``B_s = u_s u_s^T``.

    The branches are not orthogonal, so the complex least squares runs
    through their Gram matrix. Returns the normalised branch weights
    ``|beta_s|^2`` and the relative residual outside the branch span — the
    amplitude-level latent readout, which unlike the site-reduced density
    matrix carries no decoherence penalty from overlapping records.
    """
    branches = [np.outer(v, v).reshape(-1) for v in members]
    vector = matrix.reshape(-1)
    gram = np.array([[p.conj() @ q for q in branches] for p in branches])
    overlaps = np.array([p.conj() @ vector for p in branches])
    beta = np.linalg.solve(gram, overlaps)
    model = sum(b * p for b, p in zip(beta, branches, strict=True))
    residual = float(np.linalg.norm(vector - model) / np.linalg.norm(vector))
    weights = np.abs(beta) ** 2
    return weights / weights.sum(), residual


def amplitude_fidelity(matrix: np.ndarray, reference: np.ndarray) -> float:
    """Return ``|<h|h_ref>|^2``, maximised over the conjugation chamber."""

    direct = abs(np.vdot(reference.reshape(-1), matrix.reshape(-1))) ** 2
    conjugated = abs(np.vdot(reference.reshape(-1), matrix.conj().reshape(-1))) ** 2
    return float(max(direct, conjugated))


def steered_branch_split(
    matrix: np.ndarray, members: list[np.ndarray]
) -> list[tuple[float, np.ndarray]]:
    """Split the rank-2 amplitude into two product branches, one per member.

    Rank-one splits ``C = sum_s x_s y_s^T`` form a GL(2) family; scanning the
    unitary part aligns each left factor with its member. Returns the branch
    weights and the unit left factors — the fitted one-site branch shapes.
    """
    left, values, right = np.linalg.svd(matrix)
    root = np.sqrt(np.maximum(values[:2], 0.0))
    x0 = left[:, :2] * root
    y0 = right[:2, :].conj().T * root

    def factors(theta: float, phase: float) -> tuple[np.ndarray, np.ndarray]:
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array(
            [[c, -s * np.exp(1j * phase)], [s * np.exp(-1j * phase), c]]
        )
        return x0 @ rotation, y0 @ rotation.conj()

    def score(x: np.ndarray, order: list[int]) -> float:
        total = 0.0
        for column, member_index in enumerate(order):
            norm = float(np.real(x[:, column].conj() @ x[:, column]))
            if norm > 1e-300:
                total += abs(members[member_index] @ x[:, column]) ** 2 / norm
        return total

    best: tuple[float, float, float, list[int]] | None = None
    for order in ([0, 1], [1, 0]):
        for theta in np.linspace(0.0, np.pi, 61, endpoint=False):
            for phase in np.linspace(0.0, np.pi, 31, endpoint=False):
                x, _ = factors(theta, phase)
                value = score(x, order)
                if best is None or value > best[0]:
                    best = (value, theta, phase, order)
    _, theta, phase, order = best
    step_theta, step_phase = np.pi / 61.0, np.pi / 31.0
    for theta_fine in np.linspace(theta - step_theta, theta + step_theta, 41):
        for phase_fine in np.linspace(phase - step_phase, phase + step_phase, 41):
            x, _ = factors(theta_fine, phase_fine)
            value = score(x, order)
            if value > best[0]:
                best = (value, theta_fine, phase_fine, order)
    _, theta, phase, order = best
    x, y = factors(theta, phase)
    components: dict[int, tuple[float, np.ndarray]] = {}
    for column, member_index in enumerate(order):
        weight = float(
            np.real(x[:, column].conj() @ x[:, column])
            * np.real(y[:, column].conj() @ y[:, column])
        )
        norm = np.linalg.norm(x[:, column])
        components[member_index] = (weight, x[:, column] / max(norm, 1e-300))
    total = sum(w for w, _ in components.values())
    return [
        (components[index][0] / total, components[index][1])
        for index in range(len(members))
    ]


def branch_latent_means(
    target: PairTarget,
    matrix: np.ndarray,
    members: list[np.ndarray],
    grid: np.ndarray,
    rng: Any,
    n_toys: int,
    toy_size: int,
) -> np.ndarray:
    """Return toy means: one branch of the split amplitude per toy."""

    components = steered_branch_split(matrix, members)
    weights = np.array([max(w, 0.0) for w, _ in components])
    weights = weights / weights.sum()
    counts = rng.multinomial(n_toys, weights)
    means = []
    for count, (_, vector) in zip(counts, components, strict=True):
        if count == 0:
            continue
        draws = direct_sample(target, vector, count * toy_size, rng, grid)
        means.append(draws.reshape(count, toy_size).mean(axis=1))
    return np.concatenate(means)


def moment_curve_residual(
    shift: float, span: np.ndarray, family: Any, baseline: Any, k_max: int
) -> float:
    """Distance of the coherent moment-curve point from the given span."""

    r = np.asarray(family.shift_coefficients(shift, k_max, baseline), dtype=float)
    projected = span @ (span.T @ r)
    return float(np.linalg.norm(r - projected) / np.linalg.norm(r))


def factorise_pair_moments(
    matrix: np.ndarray,
    family: Any,
    baseline: Any,
    width: float,
    points: int = 301,
) -> dict[str, Any]:
    """Recover members and latent weights from a pair moment matrix.

    ``M = R W R^T`` with the columns of ``R`` on the family's coherent moment
    curve ``r(j)_k = gamma_k z(j)^k``. Two sites alone leave a one-parameter
    family of two-component product decompositions; the family constraint —
    the components must lie on the moment curve — removes it: the curve is
    scanned for its two intersections with the top-two column span of ``M``.
    The recovered ``W`` estimates the joint latent weights, hence the flip
    probability. This readout uses only the law's moments, so it is immune to
    the amplitude phase gauge in the sum variable.
    """
    k_max = matrix.shape[0] - 1
    symmetric = 0.5 * (matrix + matrix.T)
    _, vectors = np.linalg.eigh(symmetric)
    span = vectors[:, -2:]

    def residual(shift: float) -> float:
        return moment_curve_residual(shift, span, family, baseline, k_max)

    scan = np.linspace(-width, width, points)
    values = np.array([residual(shift) for shift in scan])
    interior = (
        np.nonzero((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))[0] + 1
    )
    candidates = list(interior[np.argsort(values[interior])])
    # noisy moments can wash out a local minimum: fall back to the best
    # well-separated scan points so the factorisation always returns two.
    spacing = max(points // 10, 2)
    for index in np.argsort(values):
        if len(candidates) >= 2:
            break
        if all(abs(index - kept) >= spacing for kept in candidates):
            candidates.append(int(index))
    shifts = []
    for index in candidates[:2]:
        step = scan[1] - scan[0]
        fine = np.linspace(scan[index] - step, scan[index] + step, 81)
        fine_values = np.array([residual(shift) for shift in fine])
        shifts.append(float(fine[np.argmin(fine_values)]))
    shifts = sorted(shifts, reverse=True)
    columns = np.column_stack(
        [
            np.asarray(family.shift_coefficients(s, k_max, baseline), dtype=float)
            for s in shifts
        ]
    )
    pseudo = np.linalg.pinv(columns)
    weights = pseudo @ symmetric @ pseudo.T
    total = float(np.sum(weights))
    epsilon = float((weights[0, 1] + weights[1, 0]) / max(total, 1e-300))
    curve_residuals = [residual(s) for s in shifts]
    return {
        "shifts": shifts,
        "weights": weights / max(total, 1e-300),
        "epsilon": epsilon,
        "curve_residuals": curve_residuals,
    }


def exact_pair_amplitude(target: PairTarget, t: float, degree: int) -> np.ndarray:
    """Return the exact coefficient matrix at ``epsilon = 0``.

    With perfect persistence the pair law is a two-component mixture of
    products, and ``h = (h_+ h_+ + i h_- h_-)/sqrt(2)`` realises it exactly.
    """
    plus = site_member_amplitude(target, target.plus, t, degree)
    minus = site_member_amplitude(target, target.minus, t, degree)
    matrix = (np.outer(plus, plus) + 1j * np.outer(minus, minus)) / np.sqrt(2.0)
    return matrix / np.linalg.norm(matrix)


# ------------------------------------------------------------ the slice loop --
def marginal_weight(k_max: int) -> np.ndarray:
    """Return the diagonal weight that keeps only the marginal pair modes.

    Cross modes (both degrees positive) get weight zero: fitting with this
    weight is the one-site information twice, and leaves the factorisation
    chamber of the joint amplitude undetermined — the control experiment for
    the claim that the cross moments are what resolve it.
    """
    j, k = np.divmod(np.arange(1, (k_max + 1) ** 2), k_max + 1)
    return np.diag(np.where((j > 0) & (k > 0), 0.0, 1.0))


def diagnostic_grid(family: Any, baseline: Any, members: tuple) -> np.ndarray:
    """Return the support grid, decimated to diagnostic resolution.

    The full quadrature grid has ~2e4 points; pair laws, samplers and
    two-dimensional total variations square it, so continuous families are
    evaluated on ~500 points — ample for smooth densities.
    """
    grid = support_grid(family, baseline, members)
    if family.is_lattice(baseline):
        return grid
    step = max(1, grid.size // 500)
    return grid[::step]


def fit_pair_schedule(
    target: PairTarget,
    schedule: np.ndarray,
    degree: int,
    draws: int,
    tau: float,
    rng: Any,
) -> dict[str, Any]:
    """Fit one pair amplitude per slice down the schedule, warm and tied."""

    family, baseline = target.family, target.baseline
    phi = fitting_matrices(family, baseline, degree)
    k_max = phi.shape[0] - 1
    stack, degrees = pair_fitting_matrices(phi)
    grid = diagnostic_grid(family, baseline, (target.plus, target.minus))

    pool = np.asarray(target.sample(draws, rng))
    held = pool[: draws // 5]
    train = pool[draws // 5 :]

    slices = []
    previous: np.ndarray | None = None
    for index in range(1, len(schedule)):
        t = float(schedule[index])
        delta = float(schedule[index - 1] - schedule[index])
        noised = np.column_stack(
            [
                family.one_shot_sample(train[:, site], t, params=baseline, rng=rng)
                for site in (0, 1)
            ]
        )
        empirical = empirical_pair_coefficients(family, baseline, noised, k_max)
        empirical_vec = empirical.reshape(-1)
        if previous is None or tau == 0.0:
            fit_target, weight = empirical_vec, None
        else:
            previous_vec = ratio_coefficients_complex(previous.reshape(-1), stack)
            blended, weights = pair_blended_target(
                empirical_vec, previous_vec, delta, tau, degrees
            )
            fit_target, weight = blended, np.diag(weights[1:])

        # both starts, better kept — same policy as one site.
        result = continued_complex_fit(stack, fit_target, weight=weight)["complex"]
        if previous is not None:
            warm = fit_complex_amplitude(
                stack, fit_target, initial=previous.reshape(-1), weight=weight
            )
            if warm["objective"] < result["objective"]:
                result = warm
        vector = result["coefficients"]
        matrix = vector.reshape(phi.shape[1], phi.shape[1])
        relative_gap = certified_gap(
            stack, fit_target, vector, weight=weight, relative=True
        )

        exact = exact_pair_coefficients(target, t, k_max)
        fitted = ratio_coefficients_complex(vector, stack).reshape(k_max + 1, k_max + 1)
        held_noised = np.column_stack(
            [
                family.one_shot_sample(held[:, site], t, params=baseline, rng=rng)
                for site in (0, 1)
            ]
        )
        floor, supported = pair_floor(target, t, k_max, len(train), grid)
        cross_singulars = np.linalg.svd(empirical[1:, 1:], compute_uv=False)
        bond_singulars = np.linalg.svd(matrix, compute_uv=False)
        slices.append(
            {
                "t": t,
                "matrix": matrix,
                "objective": float(result["objective"]),
                "relative_gap": float(relative_gap),
                "total_variation": pair_total_variation(target, t, matrix, grid),
                "predicted_floor": floor,
                "supported_pairs": supported,
                "coefficient_error": float(np.linalg.norm(fitted - exact)),
                "held_out_nll": held_out_pair_nll(target, matrix, held_noised),
                "empirical": empirical,
                "fitted": fitted,
                "exact": exact,
                "cross_singulars": cross_singulars[:4],
                "bond_singulars": bond_singulars[:4],
            }
        )
        previous = matrix

    return {
        "target": target,
        "schedule": schedule,
        "phi": phi,
        "stack": stack,
        "degrees": degrees,
        "grid": grid,
        "slices": slices,
        "train": train,
        "held": held,
    }


def pair_sample(
    target: PairTarget, matrix: np.ndarray, size: int, rng: Any, grid: np.ndarray
) -> np.ndarray:
    """Draw exactly from the pair law: site 1 from ``rho_1``, site 2 pure.

    This is the sequential MPS sampler at two sites: the site-1 marginal is
    ``phi^T rho_1 phi p_ref``, and conditioning on the drawn ``x_1`` collapses
    site 2 to the pure state ``C^T phi(x_1)``.
    """
    family, baseline = target.family, target.baseline
    k_max = matrix.shape[0] - 1
    rho = site_density_matrix(matrix)
    values, vectors = np.linalg.eigh(rho)
    keep = np.real(values) > 1e-12
    weights = np.real(values[keep])
    weights = weights / weights.sum()
    counts = rng.multinomial(size, weights)
    first = np.concatenate(
        [
            direct_sample(target, vectors[:, keep][:, index], count, rng, grid)
            for index, count in enumerate(counts)
            if count > 0
        ]
    )
    first = rng.permutation(first)
    basis = np.asarray(family.basis(first, k_max, baseline), dtype=float)
    conditional = basis @ matrix
    grid_basis = np.asarray(family.basis(grid, k_max, baseline), dtype=float)
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    amplitudes = conditional @ grid_basis.T
    densities = reference[None, :] * np.abs(amplitudes) ** 2
    if family.is_lattice(baseline):
        cumulative = np.cumsum(densities, axis=1)
        cumulative /= cumulative[:, -1:]
        picks = np.sum(cumulative < rng.random((size, 1)), axis=1)
        second = grid[np.minimum(picks, grid.size - 1)]
    else:
        cells = 0.5 * (densities[:, 1:] + densities[:, :-1]) * np.diff(grid)
        cumulative = np.concatenate(
            (np.zeros((size, 1)), np.cumsum(cells, axis=1)), axis=1
        )
        cumulative /= cumulative[:, -1:]
        uniforms = rng.random(size)
        second = np.array(
            [
                np.interp(uniforms[index], cumulative[index], grid)
                for index in range(size)
            ]
        )
    return np.column_stack([first, second])


# ------------------------------------------------------------------ studies --
def run_pair_study(
    name: str,
    *,
    separation: float = DEFAULT_SEPARATION,
    epsilon: float = DEFAULT_EPSILON,
    n_slices: int = DEFAULT_SLICES,
    degree: int = DEFAULT_DEGREE,
    draws: int = DEFAULT_DRAWS,
    tau: float = DEFAULT_TAU,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Fit the pair schedule and run the latent-resolution experiment."""

    rng = np.random.default_rng(seed)
    target = build_pair_target(name, separation, epsilon)
    family, baseline = target.family, target.baseline

    cap = terminating_degree(family, baseline)
    if cap is not None and 2 * degree > cap:
        raise ValueError("per-site degree exceeds the terminating cap")

    probe = np.asarray(target.sample(min(draws, 20_000), rng))
    phi = fitting_matrices(family, baseline, degree)
    k_max = phi.shape[0] - 1
    probe_matrix = empirical_pair_coefficients(family, baseline, probe, k_max)
    _, degrees = pair_fitting_matrices(phi)
    schedule = build_pair_schedule(probe_matrix.reshape(-1), degrees, n_slices)

    result = fit_pair_schedule(target, schedule, degree, draws, tau, rng)
    stack = result["stack"]
    grid = result["grid"]

    # the latent-resolution experiment at t = 0: the same data, fitted with
    # and without the cross moments in the objective.
    final = result["slices"][-1]
    empirical_vec = final["empirical"].reshape(-1)
    full_cold = continued_complex_fit(stack, empirical_vec)["complex"]
    marginal_only = continued_complex_fit(
        stack, empirical_vec, weight=marginal_weight(k_max)
    )["complex"]
    members = [
        site_member_amplitude(target, member, 0.0, degree)
        for member in (target.plus, target.minus)
    ]
    reference = exact_pair_amplitude(target, 0.0, degree)
    experiment = {}
    for label, vector in [
        ("warm chain", final["matrix"].reshape(-1)),
        ("full cold", full_cold["coefficients"]),
        ("marginal only", marginal_only["coefficients"]),
    ]:
        matrix = vector.reshape(degree + 1, degree + 1)
        weights, residual = branch_expansion(matrix, members)
        fitted_moments = ratio_coefficients_complex(matrix.reshape(-1), stack).reshape(
            k_max + 1, k_max + 1
        )
        experiment[label] = {
            "matrix": matrix,
            "weights": weights,
            "residual": residual,
            "fidelity": amplitude_fidelity(matrix, reference),
            "total_variation": pair_total_variation(target, 0.0, matrix, grid),
            "factorisation": factorise_pair_moments(
                fitted_moments, family, baseline, 2.0 * separation
            ),
        }
    experiment["data"] = {
        "factorisation": factorise_pair_moments(
            final["empirical"], family, baseline, 2.0 * separation
        )
    }
    result["experiment"] = experiment
    result["members"] = members

    generated = pair_sample(target, final["matrix"], 4_000, rng, grid)
    truth = pair_density(target, 0.0, grid)
    if family.is_lattice(baseline):
        edges = np.concatenate((grid - 0.5, [grid[-1] + 0.5]))
        counts, _, _ = np.histogram2d(
            generated[:, 0], generated[:, 1], bins=(edges, edges)
        )
        sample_tv = 0.5 * float(np.sum(np.abs(counts / len(generated) - truth)))
    else:
        sample_tv = float("nan")
    result["generation"] = {"sample": generated, "sample_tv": sample_tv}
    result["name"] = name
    result["separation"] = separation
    result["epsilon"] = epsilon
    result["degree"] = degree
    result["seed"] = seed
    return result


def report(result: dict[str, Any]) -> None:
    """Print the per-slice table and the latent-resolution experiment."""

    target = result["target"]
    print(f"\n{result['name']} — {target.label}")
    print(
        f"  schedule: {len(result['slices'])} slices, T = {result['schedule'][0]:.3f}"
    )
    print(
        "      t     objective   rel.gap        TV     floor  P*"
        "   sigma2(emp)  sigma2(bond)     NLL"
    )
    for entry in result["slices"]:
        print(
            f"  {entry['t']:6.3f}  {entry['objective']:10.3e}"
            f"  {entry['relative_gap']:8.1e}"
            f"  {entry['total_variation']:8.2e}"
            f"  {entry['predicted_floor']:8.2e}"
            f"  {entry['supported_pairs']:3d}"
            f"  {entry['cross_singulars'][1]:11.3e}"
            f"  {entry['bond_singulars'][1]:11.3e}"
            f"  {entry['held_out_nll']:7.4f}"
        )
    print("  latent-resolution experiment at t = 0:")
    half = 0.5 * result["separation"]
    print(
        f"    true member shifts ({half:+.3f}, {-half:+.3f}), eps {result['epsilon']:.3f}"
    )
    for label, entry in result["experiment"].items():
        factor = entry["factorisation"]
        line = (
            f"    {label:13s}  shifts ({factor['shifts'][0]:+.3f},"
            f" {factor['shifts'][1]:+.3f})  eps^ {factor['epsilon']:.3f}"
            f"  curve res ({factor['curve_residuals'][0]:.3f},"
            f" {factor['curve_residuals'][1]:.3f})"
        )
        if "total_variation" in entry:
            line += (
                f"  TV {entry['total_variation']:.3e}  fidelity {entry['fidelity']:.4f}"
            )
        print(line)
    print(
        f"  generation TV (sequential sampler): {result['generation']['sample_tv']:.3e}"
    )


def plot_pair_study(result: dict[str, Any], *, output_dir: Any = None) -> str:
    """Write the six-panel two-site diagnostic figure."""

    target = result["target"]
    family = target.family
    grid = result["grid"]
    slices = result["slices"]
    times = [entry["t"] for entry in slices]

    figure, axes = plt.subplots(2, 3, figsize=(16.0, 8.5))

    axis = axes[0, 0]
    truth = pair_density(target, 0.0, grid)
    law = pair_law(target, slices[-1]["matrix"], grid)
    window = np.nonzero(truth.sum(axis=1) > 1e-8)[0]
    lo, hi = grid[window[0]], grid[window[-1]]
    axis.contour(grid, grid, truth.T, levels=6, colors="k", linewidths=1.2)
    axis.contour(grid, grid, law.T, levels=6, colors="C0", linewidths=0.9)
    axis.set_xlim(lo, hi)
    axis.set_ylim(lo, hi)
    axis.set_xlabel("site 1")
    axis.set_ylabel("site 2")
    axis.set_title("joint law at t = 0: truth (black), model (blue)")

    axis = axes[0, 1]
    axis.semilogy(times, [entry["total_variation"] for entry in slices], "o-")
    axis.semilogy(times, [max(entry["relative_gap"], 1e-18) for entry in slices], "s--")
    floors = [entry["predicted_floor"] for entry in slices]
    if np.all(np.isfinite(floors)):
        axis.semilogy(times, floors, "k:", lw=2.0)
        labels = ["total variation", "certified relative gap", "predicted floor"]
    else:
        labels = ["total variation", "certified relative gap"]
    axis.set_xlabel("t")
    axis.legend(labels, fontsize=8)
    axis.set_title("per-slice error against the predicted floor")

    axis = axes[0, 2]
    for index in range(3):
        axis.semilogy(
            times,
            [entry["cross_singulars"][index] for entry in slices],
            "o-",
            label=f"sigma_{index + 1}",
        )
    axis.set_xlabel("t")
    axis.legend(fontsize=8)
    axis.set_title("cross-moment singular values (rank-2 prediction)")

    axis = axes[1, 0]
    for index in range(3):
        axis.semilogy(
            times,
            [max(entry["bond_singulars"][index], 1e-12) for entry in slices],
            "o-",
            label=f"bond sigma_{index + 1}",
        )
    axis.set_xlabel("t")
    axis.legend(fontsize=8)
    axis.set_title("bond spectrum of the fitted amplitude")

    axis = axes[1, 1]
    experiment = result["experiment"]
    labels = [label for label in experiment if "total_variation" in experiment[label]]
    curve = [
        max(experiment[label]["factorisation"]["curve_residuals"]) for label in labels
    ]
    tvs = [experiment[label]["total_variation"] for label in labels]
    positions = np.arange(len(labels))
    axis.bar(positions - 0.18, curve, width=0.36, label="moment-curve residual")
    axis.bar(positions + 0.18, tvs, width=0.36, label="joint TV")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=8)
    axis.legend(fontsize=8)
    axis.set_title("latent resolution at t = 0")

    axis = axes[1, 2]
    baseline = target.baseline
    style = {"drawstyle": "steps-mid"} if family.is_lattice(baseline) else {}
    for member in (target.plus, target.minus):
        axis.plot(
            grid,
            np.asarray(family.prob(grid, member), dtype=float),
            color="0.6",
            lw=2.5,
            **style,
        )
    for position, label in enumerate(["full cold", "marginal only"]):
        factor = result["experiment"][label]["factorisation"]
        for column, shift in enumerate(factor["shifts"]):
            member = family.shifted_params(baseline, shift)
            axis.plot(
                grid,
                np.asarray(family.prob(grid, member), dtype=float),
                color=f"C{position}",
                lw=1.0,
                label=label if column == 0 else None,
                **style,
            )
    window = np.nonzero(pair_density(target, 0.0, grid).sum(axis=1) > 1e-8)[0]
    axis.set_xlim(grid[window[0]], grid[window[-1]])
    axis.legend(fontsize=8)
    axis.set_title("members recovered from fitted moments (true in grey)")

    figure.suptitle(f"{result['name']} — {target.label} — two sites")
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result['name']}-eps{result['epsilon']:g}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


# ---------------------------------------------------------------------- CLI --
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="poisson")
    parser.add_argument("--separation", type=float, default=DEFAULT_SEPARATION)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--slices", type=int, default=DEFAULT_SLICES)
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    names = FAMILY_NAMES if args.family == "all" else [args.family]
    for name in names:
        if name == "ghs":
            print("ghs — skipped: no positivity-preserving one-shot kernel")
            continue
        result = run_pair_study(
            name,
            separation=args.separation,
            epsilon=args.epsilon,
            n_slices=args.slices,
            degree=args.degree,
            draws=args.draws,
            tau=args.tau,
            seed=args.seed,
        )
        report(result)
        if args.plot:
            print(f"  figure: {plot_pair_study(result, output_dir=args.output)}")


if __name__ == "__main__":
    main()
