"""The one-site diffusion model: the note's construction collapsed to d = 1.

One amplitude is fitted per slice of a noise schedule, warm-started from the
previous slice, tied to it by the consistency penalty, and certified against
the convex relaxation. Because family invariance keeps the slice truths in
closed form, every slice is scored against the exact noised law rather than
against a longer run of the model itself. Generation runs both ways: exact
direct sampling from the final slice, and the reverse Doob kernel down the
schedule. Everything tensor-network is absent by construction; everything
else in the algorithm is exercised.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from applications.amplitude_fit_complex import (
    build_target,
    certified_gap,
    continued_complex_fit,
    distance,
    fit_complex_amplitude,
    fitting_matrices,
    law_from_amplitude,
    ratio_coefficients_complex,
    terminating_degree,
)
from applications.targets import (
    FAMILY_NAMES,
    Target,
    empirical_coefficients,
    exact_amplitude,
    target_grid,
)

DEFAULT_SLICES = 8
DEFAULT_DEGREE = 8
DEFAULT_DRAWS = 60_000
DEFAULT_GAP = 1.4
DEFAULT_TAU = 0.5
DEFAULT_PARTICLES = 4_000
DEFAULT_PROPOSALS = 24
DEFAULT_SEED = 5
TERMINAL_FRACTION = 1e-4
FIGURE_SUBDIRECTORY = "one_site_diffusion"


# ------------------------------------------------------------- slice truths --
def contracted_member(family: Any, baseline: Any, member: Any, t: float) -> Any:
    """Return the family member reached from ``member`` after noising time ``t``.

    Family invariance: the reference process contracts the shift coordinate,
    ``z -> exp(-t) z``, so the noised member is a rescale and an inversion.
    """
    shift = float(family.natural_parameter(member)) - float(
        family.natural_parameter(baseline)
    )
    z = family.shift_coordinate(shift, baseline)
    return family.from_shift_coordinate(np.exp(-t) * z, baseline)


def slice_target(target: Target, t: float) -> Target:
    """Return the exact noised law at slice ``t`` as a target of its own.

    Requires a target built from family members (shifted or mixture): the
    noised law is then the equal mixture of the contracted members.
    """
    if not target.members:
        raise ValueError("slice truths need a member-built target")
    family, baseline = target.family, target.baseline
    noised = tuple(
        contracted_member(family, baseline, member, t) for member in target.members
    )

    def density(x: np.ndarray) -> np.ndarray:
        values = [np.asarray(family.prob(x, member), dtype=float) for member in noised]
        return np.mean(values, axis=0)

    def sample(size: int, rng: Any) -> np.ndarray:
        labels = rng.integers(0, len(noised), size)
        draws = [np.asarray(family.sample(member, size, rng=rng)) for member in noised]
        return np.choose(labels, draws)

    return Target(
        label=f"{target.label} at t={t:g}",
        family=family,
        baseline=baseline,
        sample=sample,
        density=density,
        members=noised,
    )


def slice_ratio_coefficients(target: Target, t: float, k_max: int) -> np.ndarray:
    """Return the exact ratio coefficients of the noised law at slice ``t``."""

    family, baseline = target.family, target.baseline
    rows = []
    for member in target.members:
        contracted = contracted_member(family, baseline, member, t)
        shift = float(family.natural_parameter(contracted)) - float(
            family.natural_parameter(baseline)
        )
        rows.append(
            np.asarray(family.shift_coefficients(shift, k_max, baseline), dtype=float)
        )
    return np.mean(np.reshape(rows, (len(rows), k_max + 1)), axis=0)


def mode_variances(
    target: Target, t: float, k_max: int, lam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``Var_{p_t}[phi_k]`` for ``k <= k_max`` and the exact band.

    This is the note's signal-to-noise formula: the second moment of
    ``phi_k`` under the noised law is ``sum_j Lambda[k,k,j] R_j(t)``, exact
    because the slice coefficients are exact, so the sampling variance of
    every empirical coefficient at every slice costs no re-noising.
    """
    band = lam.shape[2] - 1
    exact = slice_ratio_coefficients(target, t, band)
    diagonal = lam[np.arange(k_max + 1), np.arange(k_max + 1), :]
    variances = diagonal @ exact - exact[: k_max + 1] ** 2
    return np.maximum(variances, 0.0), exact


def effective_variances(
    target: Target, t: float, k_max: int, n_train: int, grid: np.ndarray
) -> np.ndarray | None:
    """Return ``Var_{p_t}[phi_k]`` truncated to the sample-reachable region."""

    family, baseline = target.family, target.baseline
    truth = slice_target(target, t)
    density = np.asarray(truth.density(grid), dtype=float)
    cell = np.ones_like(grid) if family.is_lattice(baseline) else np.gradient(grid)
    mass = density * cell
    mask = n_train * mass >= 1.0
    if not np.any(mask):
        return None
    basis = np.asarray(family.basis(grid[mask], k_max, baseline), dtype=float)
    weighted = mass[mask][:, None]
    first = np.sum(weighted * basis, axis=0)
    second = np.sum(weighted * basis**2, axis=0)
    return np.maximum(second - first**2, 0.0)


def snr_prediction(
    target: Target,
    t: float,
    k_max: int,
    band: int,
    n_train: int,
    grid: np.ndarray,
) -> tuple[float, int]:
    """Return the typical-sample error floor and the data-supported degree.

    The exact variance of ``mode_variances`` is the wrong yardstick for a
    finite sample: at high degree it is dominated by tail states whose
    expected count in ``n_train`` draws is far below one, so the realised
    coefficient noise is much smaller than the asymptotic one. The floor
    therefore truncates the variance to the reachable region
    ``n_train * p_t(x) dx >= 1`` and reads
    ``(1/2) sqrt(sum_k Var_k / N + tail)``: coefficient noise at the matched
    degrees plus the squared exact coefficients beyond them. The supported
    degree is the last mode whose signal-to-noise exceeds one.
    """
    variances = effective_variances(target, t, k_max, n_train, grid)
    if variances is None:
        return float("nan"), 0
    exact = slice_ratio_coefficients(target, t, band)
    noise = float(np.sum(variances[1:]) / n_train)
    tail = float(np.sum(exact[k_max + 1 :] ** 2))
    floor = 0.5 * np.sqrt(noise + tail)
    signal = np.abs(exact[1 : k_max + 1])
    ratio = signal / np.sqrt(np.maximum(variances[1:], 1e-300) / n_train)
    above = np.nonzero(ratio >= 1.0)[0]
    supported = int(above.max() + 1) if above.size else 0
    return floor, supported


# ---------------------------------------------------------------- schedule --
def chi_squared(coefficients: np.ndarray, t: Any) -> np.ndarray:
    """Return ``chi^2(t) = sum_k>0 R_k^2 exp(-2 k t)`` for the given profile."""

    degrees = np.arange(1, len(coefficients))
    weights = np.asarray(coefficients, dtype=float)[1:] ** 2
    t_array = np.asarray(t, dtype=float)
    return np.sum(weights * np.exp(-2.0 * degrees * t_array[..., None]), axis=-1)


def build_schedule(
    coefficients: np.ndarray,
    n_slices: int,
    terminal_fraction: float = TERMINAL_FRACTION,
) -> np.ndarray:
    """Return slice times ``T = t_0 > ... > t_L = 0``, log-uniform in chi^2.

    The bond entropy of the multichannel schedule vanishes at one site, so the
    slices are placed at equal increments of ``log chi^2(t)`` instead, with the
    horizon ``T`` set by the terminal fraction of the initial chi-squared.
    """
    if n_slices < 1:
        raise ValueError("the schedule needs at least one slice")
    chi_zero = float(chi_squared(coefficients, 0.0))
    if not np.isfinite(chi_zero) or chi_zero <= 0.0:
        raise ValueError("the empirical coefficients carry no signal")

    def solve(level: float) -> float:
        low, high = 0.0, 80.0
        for _ in range(200):
            middle = 0.5 * (low + high)
            if float(chi_squared(coefficients, middle)) > level:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    fractions = terminal_fraction ** (np.arange(n_slices, 0, -1) / n_slices)
    times = [solve(chi_zero * fraction) for fraction in fractions]
    return np.array(times + [0.0])


# ------------------------------------------------------------ the objective --
def blended_target(
    data_target: np.ndarray,
    previous: np.ndarray,
    delta: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold the consistency tie into the coefficient target and weight.

    The tie term ``tau [R_k(prev) - exp(-k delta) R_k(c)]^2`` is quadratic in
    the same ``R_k(c)`` as the data term, so their sum is one weighted least
    squares with a blended target: no new optimiser is needed. Returns the
    blended target vector and the per-degree weights.
    """
    degrees = np.arange(len(data_target))
    decay = np.exp(-degrees * delta)
    tie_weight = tau * decay**2
    weights = 1.0 + tie_weight
    blended = (data_target + tau * decay * previous) / weights
    return blended, weights


def slice_objective_pieces(
    phi: np.ndarray,
    empirical: np.ndarray,
    previous_fit: np.ndarray | None,
    delta: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the fitting target and diagonal weight for one slice."""

    if previous_fit is None or tau == 0.0:
        return empirical, None
    previous_ratio = np.real(ratio_coefficients_complex(previous_fit, phi))
    blended, weights = blended_target(empirical, previous_ratio, delta, tau)
    return blended, np.diag(weights[1:])


# ------------------------------------------------------------ the slice loop --
def fit_schedule(
    target: Target,
    schedule: np.ndarray,
    degree: int,
    draws: int,
    tau: float,
    rng: Any,
) -> dict[str, Any]:
    """Fit one amplitude per slice down the schedule, warm-started and tied."""

    family, baseline = target.family, target.baseline
    phi = fitting_matrices(family, baseline, degree)
    k_max = phi.shape[0] - 1
    grid = target_grid(target)

    pool = np.asarray(target.sample(draws, rng))
    held = pool[: draws // 5]
    train = pool[draws // 5 :]

    cap = terminating_degree(family, baseline)
    band = 2 * k_max if cap is None else min(2 * k_max, cap)

    slices = []
    previous: np.ndarray | None = None
    for index in range(1, len(schedule)):
        t = float(schedule[index])
        delta = float(schedule[index - 1] - schedule[index])
        noised = family.one_shot_sample(train, t, params=baseline, rng=rng)
        empirical, _ = empirical_coefficients(family, baseline, noised, k_max)
        fit_target, weight = slice_objective_pieces(
            phi, empirical, previous, delta, tau
        )

        # fits are cheap at one site, so both starts are always run and the
        # better kept: the certificate alone cannot arbitrate, because the
        # eigenvalue difference it computes drowns in floating-point noise of
        # size eps * ||M|| once the high-degree product matrices are large.
        result = continued_complex_fit(phi, fit_target, weight=weight)["complex"]
        if previous is not None:
            warm = fit_complex_amplitude(
                phi, fit_target, initial=previous, weight=weight
            )
            if warm["objective"] < result["objective"]:
                result = warm
        coefficients = result["coefficients"]
        relative_gap = certified_gap(
            phi, fit_target, coefficients, weight=weight, relative=True
        )

        truth = slice_target(target, t)
        law = law_from_amplitude(target, coefficients, grid)
        exact = slice_ratio_coefficients(target, t, k_max)
        fitted = np.real(ratio_coefficients_complex(coefficients, phi))
        held_noised = family.one_shot_sample(held, t, params=baseline, rng=rng)
        floor, supported = snr_prediction(target, t, k_max, band, len(train), grid)
        variances = effective_variances(target, t, k_max, len(train), grid)
        noise_band = None if variances is None else np.sqrt(variances / len(train))
        slices.append(
            {
                "t": t,
                "predicted_floor": floor,
                "supported_degree": supported,
                "empirical": empirical,
                "fitted": fitted,
                "exact": exact,
                "noise_band": noise_band,
                "coefficients": coefficients,
                "objective": float(result["objective"]),
                "relative_gap": float(relative_gap),
                "total_variation": distance(truth, law, grid),
                "coefficient_error": float(np.linalg.norm(fitted - exact)),
                "held_out_nll": held_out_nll(target, coefficients, held_noised),
            }
        )
        previous = coefficients

    return {
        "target": target,
        "schedule": schedule,
        "phi": phi,
        "grid": grid,
        "slices": slices,
        "train": train,
        "held": held,
    }


def held_out_nll(target: Target, coefficients: np.ndarray, sample: np.ndarray) -> float:
    """Return the exact held-out negative log-likelihood of a fitted slice."""

    family, baseline = target.family, target.baseline
    real = np.asarray(
        family.basis_dot(sample, np.real(coefficients), baseline), dtype=float
    )
    imaginary = np.asarray(
        family.basis_dot(sample, np.imag(coefficients), baseline), dtype=float
    )
    log_reference = np.asarray(family.log_prob(sample, baseline), dtype=float)
    return -float(np.mean(log_reference + np.log(real**2 + imaginary**2 + 1e-300)))


# ---------------------------------------------------------------- generation --
def amplitude_components(
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weights and vectors of ``rho = a a^T + b b^T``, rank <= 2."""

    a, b = np.real(coefficients), np.imag(coefficients)
    rho = np.outer(a, a) + np.outer(b, b)
    values, vectors = np.linalg.eigh(rho)
    keep = values > 1e-12
    weights = values[keep]
    return weights / weights.sum(), vectors[:, keep]


def direct_sample(
    target: Target,
    coefficients: np.ndarray,
    size: int,
    rng: Any,
    grid: np.ndarray,
) -> np.ndarray:
    """Draw exactly from ``p_ref |h_c|^2`` by component then a 1-D draw."""

    family, baseline = target.family, target.baseline
    weights, vectors = amplitude_components(coefficients)
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    lattice = family.is_lattice(baseline)
    counts = rng.multinomial(size, weights)

    pieces = []
    for count, vector in zip(counts, vectors.T, strict=True):
        if count == 0:
            continue
        component = np.asarray(family.basis_dot(grid, vector, baseline), dtype=float)
        density = reference * component**2
        if lattice:
            pieces.append(rng.choice(grid, size=count, p=density / density.sum()))
        else:
            cdf = np.concatenate(
                (
                    [0.0],
                    np.cumsum(np.diff(grid) * 0.5 * (density[1:] + density[:-1])),
                )
            )
            cdf /= cdf[-1]
            increasing = np.concatenate(([True], np.diff(cdf) > 0.0))
            pieces.append(
                np.interp(rng.random(count), cdf[increasing], grid[increasing])
            )
    return rng.permutation(np.concatenate(pieces))


def reverse_sample(
    result: dict[str, Any],
    particles: int,
    proposals: int,
    rng: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the reverse Doob sampler down the schedule.

    Each step proposes from the exact forward kernel and tilts by the fitted
    ratio at the target slice; the returned effective-sample-size history is
    the diagnostic for the step-size against proposal-count trade.
    """
    target = result["target"]
    family, baseline = target.family, target.baseline
    schedule = result["schedule"]
    slices = result["slices"]

    y = np.asarray(family.sample(baseline, particles, rng=rng))
    ess_history = []
    for index in range(len(schedule) - 1):
        delta = float(schedule[index] - schedule[index + 1])
        coefficients = slices[index]["coefficients"]
        drawn = family.one_shot_sample(
            np.repeat(y, proposals), delta, params=baseline, rng=rng
        ).reshape(particles, proposals)
        flat = drawn.reshape(-1)
        real = np.asarray(
            family.basis_dot(flat, np.real(coefficients), baseline), dtype=float
        )
        imaginary = np.asarray(
            family.basis_dot(flat, np.imag(coefficients), baseline), dtype=float
        )
        weights = (real**2 + imaginary**2).reshape(particles, proposals) + 1e-300
        totals = weights.sum(axis=1, keepdims=True)
        weights = weights / totals
        ess_history.append(float(np.mean(1.0 / np.sum(weights**2, axis=1))))
        cumulative = np.cumsum(weights, axis=1)
        uniforms = rng.random((particles, 1))
        chosen = np.sum(cumulative < uniforms, axis=1)
        y = drawn[np.arange(particles), np.minimum(chosen, proposals - 1)]
    return y, np.array(ess_history)


def sample_total_variation(
    target: Target, sample: np.ndarray, grid: np.ndarray
) -> float:
    """Return a binned total-variation distance of a sample from the target."""

    family, baseline = target.family, target.baseline
    if family.is_lattice(baseline):
        edges = np.concatenate((grid - 0.5, [grid[-1] + 0.5]))
        counts, _ = np.histogram(sample, bins=edges)
        return 0.5 * float(np.sum(np.abs(counts / len(sample) - target.density(grid))))
    low = np.quantile(grid, 0.001)
    high = np.quantile(grid, 0.999)
    edges = np.linspace(low, high, 81)
    counts, _ = np.histogram(sample, bins=edges)
    midpoints = 0.5 * (edges[1:] + edges[:-1])
    masses = target.density(midpoints) * np.diff(edges)
    inside = counts.sum() / len(sample)
    return 0.5 * float(np.sum(np.abs(counts / len(sample) - masses)) + (1.0 - inside))


# ---------------------------------------------------------- latent structure --
def member_amplitude(target: Target, member: Any, t: float, k_max: int) -> np.ndarray:
    """Return the unit-norm amplitude of the contracted member at slice ``t``."""

    family, baseline = target.family, target.baseline
    moved = contracted_member(family, baseline, member, t)
    shift = float(family.natural_parameter(moved)) - float(
        family.natural_parameter(baseline)
    )
    vector = exact_amplitude(family, baseline, shift, k_max)
    return vector / np.linalg.norm(vector)


def latent_weights(
    coefficients: np.ndarray, members: list[np.ndarray]
) -> tuple[np.ndarray, float]:
    """Return the member weights carried by the fitted density matrix.

    The fitted ``rho = a a^T + b b^T`` is compared to ``sum_s w_s |a_s><a_s|``
    in the Frobenius norm; the coherent states are not orthogonal, so the
    least squares runs through their Gram matrix (``Tr[A_i A_j] = G_ij^2``).
    Returns the weights and the relative residual outside the member span.
    """
    a, b = np.real(coefficients), np.imag(coefficients)
    rho = np.outer(a, a) + np.outer(b, b)
    projectors = [np.outer(v, v) for v in members]
    gram_squared = np.array(
        [[float(np.sum(p * q)) for q in projectors] for p in projectors]
    )
    overlaps = np.array([float(v @ rho @ v) for v in members])
    weights = np.linalg.solve(gram_squared, overlaps)
    model = sum(w * p for w, p in zip(weights, projectors, strict=True))
    residual = float(np.linalg.norm(rho - model) / np.linalg.norm(rho))
    return weights, residual


def steered_components(
    coefficients: np.ndarray, members: list[np.ndarray]
) -> list[tuple[float, np.ndarray]]:
    """Split the fitted rank-2 ``rho`` into two pure states, one per member.

    Two-state decompositions of ``rho`` form a one-parameter family
    ``U = [e_1 sqrt(mu_1), e_2 sqrt(mu_2)] O`` with ``U U^T = rho`` and ``O``
    a rotation: the eigenvectors themselves are the even/odd combinations of
    the members, not the members. The rotation angle is chosen to align each
    column with its member, and the returned weights are the column norms.
    """
    a, b = np.real(coefficients), np.imag(coefficients)
    rho = np.outer(a, a) + np.outer(b, b)
    values, vectors = np.linalg.eigh(rho)
    root = vectors[:, -2:] * np.sqrt(np.maximum(values[-2:], 0.0))

    def score(theta: float, order: list[int]) -> float:
        c, s = np.cos(theta), np.sin(theta)
        u = root @ np.array([[c, -s], [s, c]])
        total = 0.0
        for column, member_index in enumerate(order):
            norm = float(u[:, column] @ u[:, column])
            if norm > 1e-300:
                total += float(members[member_index] @ u[:, column]) ** 2 / norm
        return total

    best: tuple[float, float, list[int]] | None = None
    for order in ([0, 1], [1, 0]):
        for theta in np.linspace(0.0, np.pi, 181, endpoint=False):
            value = score(theta, order)
            if best is None or value > best[0]:
                best = (value, theta, order)
    _, theta, order = best
    step = np.pi / 181.0
    for theta_fine in np.linspace(theta - step, theta + step, 201):
        value = score(theta_fine, order)
        if value > best[0]:
            best = (value, theta_fine, order)
    _, theta, order = best
    c, s = np.cos(theta), np.sin(theta)
    u = root @ np.array([[c, -s], [s, c]])
    components: dict[int, tuple[float, np.ndarray]] = {}
    for column, member_index in enumerate(order):
        weight = float(u[:, column] @ u[:, column])
        vector = u[:, column] / np.sqrt(max(weight, 1e-300))
        components[member_index] = (weight, vector)
    return [components[index] for index in range(len(members))]


def shared_latent_means(
    target: Target,
    coefficients: np.ndarray,
    members: list[np.ndarray],
    grid: np.ndarray,
    rng: Any,
    n_toys: int,
    toy_size: int,
) -> np.ndarray:
    """Return one sample mean per toy, the latent drawn once per toy.

    This is the exchangeable version of the model: a pure state from the
    steered split of ``rho`` per toy, then ``toy_size`` i.i.d. draws from
    its law. If the fitted density matrix carries the two member lumps, the
    means histogram is sharply bimodal at the contracted member means.
    """
    components = steered_components(coefficients, members)
    weights = np.array([max(w, 0.0) for w, _ in components])
    weights = weights / weights.sum()
    counts = rng.multinomial(n_toys, weights)
    means = []
    for count, (_, vector) in zip(counts, components, strict=True):
        if count == 0:
            continue
        draws = direct_sample(
            target, vector.astype(complex), count * toy_size, rng, grid
        )
        means.append(draws.reshape(count, toy_size).mean(axis=1))
    return np.concatenate(means)


def ensemble_laws(result: dict[str, Any], runs: int) -> list[np.ndarray]:
    """Refit the whole schedule on fresh data ``runs`` times; return t=0 laws."""

    target = result["target"]
    laws = []
    for offset in range(1, runs + 1):
        rng = np.random.default_rng(result["seed"] + 1000 * offset)
        refit = fit_schedule(
            target,
            result["schedule"],
            result["degree"],
            result["draws"],
            result["tau"],
            rng,
        )
        laws.append(
            law_from_amplitude(
                target, refit["slices"][-1]["coefficients"], result["grid"]
            )
        )
    return laws


# ------------------------------------------------------------------ studies --
def run_study(
    name: str,
    kind: str,
    *,
    n_slices: int = DEFAULT_SLICES,
    degree: int = DEFAULT_DEGREE,
    draws: int = DEFAULT_DRAWS,
    gap: float = DEFAULT_GAP,
    tau: float = DEFAULT_TAU,
    particles: int = DEFAULT_PARTICLES,
    proposals: int = DEFAULT_PROPOSALS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Fit the schedule, compare against a cold start, and sample both ways."""

    rng = np.random.default_rng(seed)
    target = build_target(name, kind, None, gap)
    family, baseline = target.family, target.baseline

    probe = np.asarray(target.sample(min(draws, 20_000), rng))
    band = 2 * degree
    cap = terminating_degree(family, baseline)
    if cap is not None:
        band = min(band, cap)
    probe_coefficients, _ = empirical_coefficients(family, baseline, probe, band)
    schedule = build_schedule(probe_coefficients, n_slices)

    result = fit_schedule(target, schedule, degree, draws, tau, rng)
    grid = result["grid"]

    # cold start: the t = 0 slice alone, from the vacuum.
    final_noised = result["train"]
    k_max = result["phi"].shape[0] - 1
    empirical, _ = empirical_coefficients(family, baseline, final_noised, k_max)
    cold = continued_complex_fit(result["phi"], empirical)["complex"]
    cold_law = law_from_amplitude(target, cold["coefficients"], grid)
    result["cold_start"] = {
        "objective": float(cold["objective"]),
        "total_variation": distance(target, cold_law, grid),
    }
    # the warm slice was fitted against the blended tie target; score it on the
    # same raw empirical objective as the cold start for a fair comparison.
    warm_ratio = np.real(
        ratio_coefficients_complex(result["slices"][-1]["coefficients"], result["phi"])
    )
    residual = (warm_ratio - empirical)[1:]
    result["slices"][-1]["raw_objective"] = float(0.5 * residual @ residual)

    warm_final = result["slices"][-1]["coefficients"]
    direct = direct_sample(target, warm_final, particles * proposals, rng, grid)
    reverse, ess = reverse_sample(result, particles, proposals, rng)
    result["generation"] = {
        "direct_tv": sample_total_variation(target, direct, grid),
        "reverse_tv": sample_total_variation(target, reverse, grid),
        "direct_sample": direct,
        "reverse_sample": reverse,
        "ess": ess,
    }
    result["name"] = name
    result["kind"] = kind
    result["degree"] = degree
    result["draws"] = draws
    result["tau"] = tau
    result["seed"] = seed
    return result


def report(result: dict[str, Any]) -> None:
    """Print the per-slice table and the generation summary."""

    target = result["target"]
    print(f"\n{result['name']} — {target.label}")
    print(
        f"  schedule: {len(result['slices'])} slices, T = {result['schedule'][0]:.3f}"
    )
    print("      t     objective   rel.gap        TV     floor  K*    |dR|      NLL")
    for entry in result["slices"]:
        print(
            f"  {entry['t']:6.3f}  {entry['objective']:10.3e}"
            f"  {entry['relative_gap']:8.1e}"
            f"  {entry['total_variation']:8.2e}"
            f"  {entry['predicted_floor']:8.2e}"
            f"  {entry['supported_degree']:3d}"
            f"  {entry['coefficient_error']:6.1e}"
            f"  {entry['held_out_nll']:7.4f}"
        )
    warm = result["slices"][-1]
    cold = result["cold_start"]
    print(
        f"  t=0 warm vs cold: objective {warm['raw_objective']:.3e} / "
        f"{cold['objective']:.3e}, TV {warm['total_variation']:.2e} / "
        f"{cold['total_variation']:.2e}"
    )
    generation = result["generation"]
    print(
        f"  generation TV: direct {generation['direct_tv']:.3e}, "
        f"reverse {generation['reverse_tv']:.3e}, "
        f"mean ESS {generation['ess'].mean():.1f}"
    )


def plot_study(result: dict[str, Any], *, output_dir: Any = None) -> str:
    """Write a four-panel diagnostic figure to the artifacts directory."""

    target = result["target"]
    family, baseline = target.family, target.baseline
    grid = result["grid"]
    schedule = result["schedule"]
    slices = result["slices"]
    lattice = family.is_lattice(baseline)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    axis = axes[0, 0]
    shown = [0, len(slices) // 2, len(slices) - 1]
    for index in shown:
        entry = slices[index]
        truth = slice_target(target, entry["t"])
        law = law_from_amplitude(target, entry["coefficients"], grid)
        style = {"drawstyle": "steps-mid"} if lattice else {}
        axis.plot(grid, truth.density(grid), color="0.6", lw=2.5, **style)
        axis.plot(grid, law, lw=1.0, label=f"t={entry['t']:.2f}", **style)
    window = truth.density(grid) > 1e-9
    axis.set_xlim(grid[window].min(), grid[window].max())
    axis.legend(fontsize=8)
    axis.set_title("fitted slices against exact slice laws")

    axis = axes[0, 1]
    times = [entry["t"] for entry in slices]
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

    axis = axes[1, 0]
    generation = result["generation"]
    for sample, label in [
        (generation["direct_sample"], "direct"),
        (generation["reverse_sample"], "reverse Doob"),
    ]:
        if lattice:
            edges = np.concatenate((grid - 0.5, [grid[-1] + 0.5]))
        else:
            edges = np.linspace(grid.min(), grid.max(), 81)
        axis.hist(sample, bins=edges, density=True, histtype="step", label=label)
    axis.plot(grid, target.density(grid), color="k", lw=1.5, label="target")
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    axis.plot(grid, reference, color="0.5", ls="--", lw=1.2, label="baseline")
    axis.set_xlim(grid[window].min(), grid[window].max())
    axis.legend(fontsize=8)
    axis.set_title("generation at t = 0")

    axis = axes[1, 1]
    axis.plot(schedule[:-1], generation["ess"], "o-")
    axis.set_xlabel("slice time")
    axis.set_ylabel("effective sample size per particle")
    axis.set_title("reverse-sampler proposal quality")

    figure.suptitle(f"{result['name']} — {target.label}")
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result['name']}-{result['kind']}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def plot_latent(
    result: dict[str, Any], *, runs: int = 10, output_dir: Any = None
) -> str:
    """Write the latent-structure figure: ensemble, mode decay, weights, toys.

    The two member panels need the two-component mixture target; on a
    single-member target they carry a note instead.
    """
    target = result["target"]
    family, baseline = target.family, target.baseline
    grid = result["grid"]
    slices = result["slices"]
    k_max = result["phi"].shape[0] - 1
    amplitude_degree = len(slices[-1]["coefficients"]) - 1
    times = np.array([entry["t"] for entry in slices])
    rng = np.random.default_rng(result["seed"] + 77)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    lattice = family.is_lattice(baseline)
    style = {"drawstyle": "steps-mid"} if lattice else {}

    # -- panel A: ten independently trained t = 0 laws against the truth.
    axis = axes[0, 0]
    for law in ensemble_laws(result, runs):
        axis.plot(grid, law, color="C0", lw=0.7, alpha=0.6, **style)
    for member in target.members:
        member_density = np.asarray(family.prob(grid, member), dtype=float)
        axis.plot(grid, member_density, color="0.7", lw=1.0, **style)
    axis.plot(grid, target.density(grid), color="k", lw=1.8, label="target", **style)
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    axis.plot(grid, reference, color="0.4", ls="--", lw=1.2, label="baseline")
    window = target.density(grid) > 1e-9
    axis.set_xlim(grid[window].min(), grid[window].max())
    axis.legend(fontsize=8)
    axis.set_title(f"{runs} independently trained t = 0 laws (members in grey)")

    # -- panel B: mode decay along the schedule, the ILSE regression lines.
    axis = axes[0, 1]
    exact_zero = slice_ratio_coefficients(target, 0.0, k_max)
    dense = np.linspace(0.0, float(result["schedule"][0]), 200)
    shown_modes = [k for k in (1, 2, 3, 4, 6, 8) if k <= k_max]
    for position, k in enumerate(shown_modes):
        color = f"C{position}"
        axis.semilogy(
            dense, np.abs(exact_zero[k]) * np.exp(-k * dense), color=color, lw=1.0
        )
        axis.semilogy(
            times,
            [abs(entry["empirical"][k]) for entry in slices],
            "o",
            color=color,
            ms=4,
            label=f"k={k}",
        )
        axis.semilogy(
            times,
            [abs(entry["fitted"][k]) for entry in slices],
            "x",
            color=color,
            ms=5,
        )
        bands = [
            entry["noise_band"][k]
            for entry in slices
            if entry["noise_band"] is not None
        ]
        if len(bands) == len(slices):
            axis.semilogy(times, bands, ls=":", color=color, lw=0.8)
    axis.set_ylim(bottom=1e-6)
    axis.set_xlabel("t")
    axis.legend(fontsize=7, ncol=2)
    axis.set_title("mode decay: exact lines, empirical o, fitted x, noise band :")

    two_members = len(target.members) == 2

    # -- panel C: member weights of the fitted density matrix per slice.
    axis = axes[1, 0]
    if two_members:
        weight_rows, residuals = [], []
        for entry in slices:
            members = [
                member_amplitude(target, member, entry["t"], amplitude_degree)
                for member in target.members
            ]
            weights, residual = latent_weights(entry["coefficients"], members)
            weight_rows.append(weights)
            residuals.append(residual)
        weight_rows = np.array(weight_rows)
        axis.plot(times, weight_rows[:, 0], "o-", label="weight on member +")
        axis.plot(times, weight_rows[:, 1], "s-", label="weight on member -")
        axis.plot(times, residuals, "k:", label="residual outside span")
        axis.axhline(0.5, color="0.7", lw=0.8)
        axis.set_xlabel("t")
        axis.set_ylim(-0.1, 1.1)
        axis.legend(fontsize=8)
        axis.set_title("latent weights of the fitted density matrix")
    else:
        axis.axis("off")
        axis.text(0.5, 0.5, "single-member target", ha="center")

    # -- panel D: shared-latent toys, one latent draw per toy of size N.
    axis = axes[1, 1]
    if two_members:
        picks = [len(slices) - 1, len(slices) // 2, 0]
        for position, index in enumerate(picks):
            entry = slices[index]
            members = [
                member_amplitude(target, member, entry["t"], amplitude_degree)
                for member in target.members
            ]
            means = shared_latent_means(
                target, entry["coefficients"], members, grid, rng, 400, 2000
            )
            color = f"C{position}"
            axis.hist(
                means,
                bins=60,
                density=True,
                histtype="step",
                color=color,
                label=f"t={entry['t']:.2f}",
            )
            for member in target.members:
                moved = contracted_member(family, baseline, member, entry["t"])
                axis.axvline(float(family.mean(moved)), color=color, ls=":", lw=0.8)
        axis.set_xlabel("toy mean (N = 2000 per toy)")
        axis.legend(fontsize=8)
        axis.set_title("shared-latent toys: one latent per toy of the split rho")
    else:
        axis.axis("off")
        axis.text(0.5, 0.5, "single-member target", ha="center")

    figure.suptitle(f"{result['name']} — {target.label} — latent structure")
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result['name']}-{result['kind']}-latent.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


# ---------------------------------------------------------------------- CLI --
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="poisson")
    parser.add_argument("--target", default="mixture", choices=["shifted", "mixture"])
    parser.add_argument("--slices", type=int, default=DEFAULT_SLICES)
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--gap", type=float, default=DEFAULT_GAP)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--proposals", type=int, default=DEFAULT_PROPOSALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--latent", action="store_true")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    names = FAMILY_NAMES if args.family == "all" else [args.family]
    for name in names:
        if name == "ghs":
            print("ghs — skipped: no positivity-preserving one-shot kernel")
            continue
        result = run_study(
            name,
            args.target,
            n_slices=args.slices,
            degree=args.degree,
            draws=args.draws,
            gap=args.gap,
            tau=args.tau,
            particles=args.particles,
            proposals=args.proposals,
            seed=args.seed,
        )
        report(result)
        if args.plot:
            print(f"  figure: {plot_study(result, output_dir=args.output)}")
        if args.latent:
            print(
                "  figure: "
                f"{plot_latent(result, runs=args.runs, output_dir=args.output)}"
            )


if __name__ == "__main__":
    main()
