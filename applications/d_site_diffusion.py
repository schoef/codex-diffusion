"""The d-site diffusion model: DMRG-style bond sweeps at fixed bond dimension.

The model is a complex MPS amplitude over ``d`` sites; the data are the
emissions of the two-state hidden Markov toy, noised sitewise by the exact
one-shot kernels. Training is a sweep of local bond fits: in mixed canonical
form the pair moments at a bond are ``vec(B)^dagger (I x Phi_j x Phi_k x I)
vec(B)`` of the merged tensor, so every bond update is the one-site complex
fit on a Kronecker stack — warm-started, tied to the previous slice by the
blended pair target, and truncated back to bond dimension ``chi`` by SVD.

Everything is scored against closed forms: the exact lag-one pair law of the
noised chain (family invariance), the exact transfer-matrix likelihood of the
noised data, and the moment-curve factorisation reading the latent members
and the flip probability off the fitted bond moments.
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
    terminating_degree,
)
from applications.mps_amplitude import (
    bond_fit_stack,
    bond_spectrum,
    evaluate,
    merge_bond,
    move_center_right,
    pair_moment_matrix,
    right_canonicalise,
    sequential_sample,
    split_bond,
    vacuum_state,
)
from applications.two_site_diffusion import (
    build_pair_schedule,
    build_pair_target,
    diagnostic_grid,
    empirical_pair_coefficients,
    factorise_pair_moments,
    pair_blended_target,
    pair_density,
)
from applications.two_state_hmm import (
    BASELINES,
    emission_parameters,
    log_likelihood,
    noised_emissions,
    sample_latent,
    sample_observations,
)

DEFAULT_SITES = 8
DEFAULT_SLICES = 6
DEFAULT_DEGREE = 5
DEFAULT_CHI = 2
DEFAULT_SWEEPS = 2
DEFAULT_DRAWS = 60_000
DEFAULT_SEPARATION = 0.57
DEFAULT_EPSILON = 0.05
DEFAULT_TAU = 0.5
DEFAULT_SEED = 5
FIGURE_SUBDIRECTORY = "d_site_diffusion"


# ------------------------------------------------------------------ target --
@dataclass
class ChainTarget:
    """The d-site law of the two-state hidden Markov toy."""

    label: str
    family: Any
    baseline: Any
    epsilon: float
    plus: Any
    minus: Any
    z: np.ndarray
    d: int

    def sample(self, size: int, rng: Any) -> np.ndarray:
        states = sample_latent(self.d, size, self.epsilon, rng)
        return sample_observations(self.family, self.plus, self.minus, states, rng)

    def noised_members(self, t: float) -> tuple[Any, Any]:
        return noised_emissions(self.family, self.baseline, self.z, t)

    def truth_nll(self, noised_sample: np.ndarray, t: float) -> float:
        """Exact negative log-likelihood of the noised law, per sample."""

        plus_t, minus_t = self.noised_members(t)
        loglik = log_likelihood(
            self.family, noised_sample, plus_t, minus_t, self.epsilon
        )
        return -float(np.mean(loglik))


def build_chain_target(
    name: str, d: int, separation: float, epsilon: float
) -> ChainTarget:
    family, baseline, _ = BASELINES[name]
    plus, minus, z = emission_parameters(family, baseline, separation)
    return ChainTarget(
        label=f"hmm chain d={d} sep={separation:g} eps={epsilon:g}",
        family=family,
        baseline=baseline,
        epsilon=epsilon,
        plus=plus,
        minus=minus,
        z=z,
        d=d,
    )


def model_nll(
    tensors: list[np.ndarray], target: ChainTarget, sample: np.ndarray
) -> float:
    """Exact negative log-likelihood of the MPS Born law, per sample."""

    family, baseline = target.family, target.baseline
    amplitude = evaluate(tensors, family, baseline, sample)
    log_reference = np.zeros(len(sample))
    for site in range(sample.shape[1]):
        log_reference += np.asarray(
            family.log_prob(sample[:, site], baseline), dtype=float
        )
    return -float(np.mean(log_reference + np.log(np.abs(amplitude) ** 2 + 1e-300)))


def bond_pair_law(merged: np.ndarray, family: Any, baseline: Any, grid: np.ndarray):
    """Return the model's pair marginal at a bond, on ``grid x grid``.

    In mixed canonical form the environments are identities, so the marginal
    is the bond-traced Born law of the merged tensor.
    """
    degree = merged.shape[1] - 1
    basis = np.asarray(family.basis(grid, degree, baseline), dtype=float)
    fields = np.einsum("ga,labr,hb->lghr", basis, merged, basis, optimize=True)
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    density = np.sum(np.abs(fields) ** 2, axis=(0, 3))
    return np.outer(reference, reference) * density


def pair_tv(pair_target: Any, t: float, law: np.ndarray, grid: np.ndarray) -> float:
    truth = pair_density(pair_target, t, grid)
    family, baseline = pair_target.family, pair_target.baseline
    cell = np.ones_like(grid) if family.is_lattice(baseline) else np.gradient(grid)
    return 0.5 * float(np.sum(np.abs(truth - law) * np.outer(cell, cell)))


# ------------------------------------------------------------ the sweep loop --
def fit_chain_schedule(
    target: ChainTarget,
    schedule: np.ndarray,
    degree: int,
    chi: int,
    draws: int,
    tau: float,
    sweeps: int,
    rng: Any,
) -> dict[str, Any]:
    """Fit one MPS per slice down the schedule, warm-started and tied."""

    family, baseline = target.family, target.baseline
    d = target.d
    phi = fitting_matrices(family, baseline, degree)
    k_max = phi.shape[0] - 1
    j_index, k_index = np.divmod(np.arange((k_max + 1) ** 2), k_max + 1)
    degrees = (j_index + k_index).astype(float)
    grid = diagnostic_grid(family, baseline, (target.plus, target.minus))
    pair_target = build_pair_target(
        _family_name(family), _separation_of(target), target.epsilon
    )
    middle = (d - 2) // 2
    stacks: dict[tuple[int, int, int, int], np.ndarray] = {}

    pool = np.asarray(target.sample(draws, rng))
    held = pool[: draws // 5]
    train = pool[draws // 5 :]

    slices = []
    previous: list[np.ndarray] | None = None
    previous_moments: list[np.ndarray] | None = None
    for index in range(1, len(schedule)):
        t = float(schedule[index])
        delta = float(schedule[index - 1] - schedule[index])
        noised = np.column_stack(
            [
                family.one_shot_sample(train[:, site], t, params=baseline, rng=rng)
                for site in range(d)
            ]
        )
        held_noised = np.column_stack(
            [
                family.one_shot_sample(held[:, site], t, params=baseline, rng=rng)
                for site in range(d)
            ]
        )
        empirical = [
            empirical_pair_coefficients(
                family, baseline, noised[:, bond : bond + 2], k_max
            ).reshape(-1)
            for bond in range(d - 1)
        ]
        if previous_moments is None or tau == 0.0:
            targets = empirical
            weights: list[np.ndarray | None] = [None] * (d - 1)
        else:
            targets, weights = [], []
            for bond in range(d - 1):
                blended, weight_vec = pair_blended_target(
                    empirical[bond], previous_moments[bond], delta, tau, degrees
                )
                targets.append(blended)
                weights.append(np.diag(weight_vec[1:]))

        tensors = (
            right_canonicalise([tensor.copy() for tensor in previous])
            if previous is not None
            else vacuum_state(d, degree)
        )
        objective = 0.0
        for sweep in range(sweeps):
            tensors = right_canonicalise(tensors)
            objective = _one_round(
                tensors,
                targets,
                weights,
                phi,
                chi,
                stacks,
                both_starts=(sweep == 0),
            )

        moments = [pair_moment_matrix(tensors, phi, bond) for bond in range(d - 1)]
        middle_merged = _merged_at(tensors, middle)
        law = bond_pair_law(middle_merged, family, baseline, grid)
        middle_gap = certified_gap(
            stacks[middle_merged.shape],
            targets[middle],
            middle_merged.reshape(-1),
            weight=weights[middle],
            relative=True,
        )
        factor = factorise_pair_moments(
            moments[middle], family, baseline, 2.0 * _separation_of(target)
        )
        slices.append(
            {
                "t": t,
                "objective": objective,
                "relative_gap": float(middle_gap),
                "pair_tv": pair_tv(pair_target, t, law, grid),
                "nll_gap": model_nll(tensors, target, held_noised)
                - target.truth_nll(held_noised, t),
                "bond_spectrum": bond_spectrum(tensors, middle)[:4],
                "shifts": factor["shifts"],
                "epsilon_hat": factor["epsilon"],
                "curve_residuals": factor["curve_residuals"],
                "tensors": [tensor.copy() for tensor in tensors],
            }
        )
        previous = tensors
        previous_moments = [matrix.reshape(-1) for matrix in moments]

    return {
        "target": target,
        "pair_target": pair_target,
        "schedule": schedule,
        "phi": phi,
        "grid": grid,
        "middle": middle,
        "slices": slices,
        "train": train,
        "held": held,
    }


def _fit_bond(
    tensors: list[np.ndarray],
    bond: int,
    targets: list[np.ndarray],
    weights: list[np.ndarray | None],
    phi: np.ndarray,
    chi: int,
    stacks: dict,
    *,
    center_right: bool,
    both_starts: bool,
) -> float:
    """Fit one merged bond tensor in place; returns its objective."""

    merged = merge_bond(tensors, bond)
    shape = merged.shape
    if shape not in stacks:
        stacks[shape] = bond_fit_stack(phi, shape[0], shape[3])
    initial = merged.reshape(-1)
    initial = initial / np.linalg.norm(initial)
    fit = fit_complex_amplitude(
        stacks[shape], targets[bond], initial=initial, weight=weights[bond]
    )
    if both_starts:
        cold = continued_complex_fit(
            stacks[shape], targets[bond], weight=weights[bond]
        )["complex"]
        if cold["objective"] < fit["objective"]:
            fit = cold
    refitted = fit["coefficients"].reshape(shape)
    first, second, _ = split_bond(refitted, chi, center_right=center_right)
    tensors[bond], tensors[bond + 1] = first, second
    return float(fit["objective"])


def _one_round(
    tensors: list[np.ndarray],
    targets: list[np.ndarray],
    weights: list[np.ndarray | None],
    phi: np.ndarray,
    chi: int,
    stacks: dict,
    *,
    both_starts: bool,
) -> float:
    """One right-then-left sweep; returns the summed final bond objectives.

    ``tensors`` must be right-canonical on entry (centre at site 0); the
    centre rides with the sweep, so every merged tensor sees identity
    environments.
    """
    d = len(tensors)
    for bond in range(d - 1):
        _fit_bond(
            tensors,
            bond,
            targets,
            weights,
            phi,
            chi,
            stacks,
            center_right=True,
            both_starts=both_starts,
        )
    objectives = [0.0] * (d - 1)
    for bond in range(d - 2, -1, -1):
        objectives[bond] = _fit_bond(
            tensors,
            bond,
            targets,
            weights,
            phi,
            chi,
            stacks,
            center_right=False,
            both_starts=False,
        )
    return float(np.sum(objectives))


def _merged_at(tensors: list[np.ndarray], bond: int) -> np.ndarray:
    state = right_canonicalise(tensors)
    for site in range(bond):
        move_center_right(state, site)
    return merge_bond(state, bond)


def _separation_of(target: ChainTarget) -> float:
    family, baseline = target.family, target.baseline
    return 2.0 * (
        float(family.natural_parameter(target.plus))
        - float(family.natural_parameter(baseline))
    )


def _family_name(family: Any) -> str:
    for name, (candidate, _, _) in BASELINES.items():
        if candidate is family:
            return name
    raise ValueError("unknown family")


# ------------------------------------------------------------------ studies --
def run_chain_study(
    name: str,
    *,
    d: int = DEFAULT_SITES,
    separation: float = DEFAULT_SEPARATION,
    epsilon: float = DEFAULT_EPSILON,
    n_slices: int = DEFAULT_SLICES,
    degree: int = DEFAULT_DEGREE,
    chi: int = DEFAULT_CHI,
    sweeps: int = DEFAULT_SWEEPS,
    draws: int = DEFAULT_DRAWS,
    tau: float = DEFAULT_TAU,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Fit the chain down the schedule and sample the trained model."""

    rng = np.random.default_rng(seed)
    target = build_chain_target(name, d, separation, epsilon)
    family, baseline = target.family, target.baseline

    cap = terminating_degree(family, baseline)
    if cap is not None and 2 * degree > cap:
        raise ValueError("per-site degree exceeds the terminating cap")

    probe = np.asarray(target.sample(min(draws, 20_000), rng))
    phi = fitting_matrices(family, baseline, degree)
    k_max = phi.shape[0] - 1
    j_index, k_index = np.divmod(np.arange((k_max + 1) ** 2), k_max + 1)
    degrees = (j_index + k_index).astype(float)
    middle = (d - 2) // 2
    probe_matrix = empirical_pair_coefficients(
        family, baseline, probe[:, middle : middle + 2], k_max
    )
    schedule = build_pair_schedule(probe_matrix.reshape(-1), degrees, n_slices)

    result = fit_chain_schedule(target, schedule, degree, chi, draws, tau, sweeps, rng)

    final = result["slices"][-1]
    generated = sequential_sample(
        final["tensors"], family, baseline, 4_000, result["grid"], rng
    )
    grid = result["grid"]
    site_tvs = []
    truth_site = pair_density(result["pair_target"], 0.0, grid).sum(axis=1)
    if family.is_lattice(baseline):
        edges = np.concatenate((grid - 0.5, [grid[-1] + 0.5]))
        for site in range(d):
            counts, _ = np.histogram(generated[:, site], bins=edges)
            site_tvs.append(
                0.5 * float(np.sum(np.abs(counts / len(generated) - truth_site)))
            )
        counts, _, _ = np.histogram2d(
            generated[:, middle], generated[:, middle + 1], bins=(edges, edges)
        )
        pair_sample_tv = 0.5 * float(
            np.sum(
                np.abs(
                    counts / len(generated)
                    - pair_density(result["pair_target"], 0.0, grid)
                )
            )
        )
    else:
        pair_sample_tv = float("nan")
    result["generation"] = {
        "sample": generated,
        "site_tv": site_tvs,
        "pair_sample_tv": pair_sample_tv,
    }
    result["name"] = name
    result["d"] = d
    result["chi"] = chi
    result["separation"] = separation
    result["epsilon"] = epsilon
    result["seed"] = seed
    return result


def report(result: dict[str, Any]) -> None:
    """Print the per-slice table and the latent readout."""

    target = result["target"]
    print(f"\n{result['name']} — {target.label} — chi = {result['chi']}")
    print(
        f"  schedule: {len(result['slices'])} slices, T = {result['schedule'][0]:.3f}"
    )
    print(
        "      t     objective   rel.gap   pair TV   NLL gap"
        "   bond sigma2   shifts recovered      eps^"
    )
    for entry in result["slices"]:
        spectrum = entry["bond_spectrum"]
        sigma2 = spectrum[1] if len(spectrum) > 1 else 0.0
        print(
            f"  {entry['t']:6.3f}  {entry['objective']:10.3e}"
            f"  {entry['relative_gap']:8.1e}"
            f"  {entry['pair_tv']:8.2e}"
            f"  {entry['nll_gap']:8.4f}"
            f"  {sigma2:11.3e}"
            f"   ({entry['shifts'][0]:+.3f}, {entry['shifts'][1]:+.3f})"
            f"  {entry['epsilon_hat']:+.3f}"
        )
    half = 0.5 * result["separation"]
    print(
        f"  true member shifts ({half:+.3f}, {-half:+.3f}), eps {result['epsilon']:.3f}"
    )
    generation = result["generation"]
    if generation["site_tv"]:
        site_tv = np.mean(generation["site_tv"])
        print(
            f"  generation: mean per-site TV {site_tv:.3e},"
            f" middle-pair TV {generation['pair_sample_tv']:.3e}"
        )


def plot_chain_study(result: dict[str, Any], *, output_dir: Any = None) -> str:
    """Write the four-panel d-site diagnostic figure."""

    target = result["target"]
    family, baseline = target.family, target.baseline
    slices = result["slices"]
    times = [entry["t"] for entry in slices]
    grid = result["grid"]

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    axis = axes[0, 0]
    axis.semilogy(times, [entry["pair_tv"] for entry in slices], "o-")
    axis.semilogy(times, [max(entry["nll_gap"], 1e-6) for entry in slices], "s--")
    axis.set_xlabel("t")
    axis.legend(["middle-bond pair TV", "held-out NLL gap"], fontsize=8)
    axis.set_title("per-slice error against the exact chain")

    axis = axes[0, 1]
    for index in range(3):
        axis.semilogy(
            times,
            [
                max(entry["bond_spectrum"][index], 1e-12)
                if len(entry["bond_spectrum"]) > index
                else 1e-12
                for entry in slices
            ],
            "o-",
            label=rf"$\sigma_{index + 1}$",
        )
    axis.set_xlabel("t")
    axis.legend(fontsize=8)
    axis.set_title("middle-bond Schmidt spectrum of the trained MPS")

    axis = axes[1, 0]
    axis.plot(times, [entry["shifts"][0] for entry in slices], "o-")
    axis.plot(times, [entry["shifts"][1] for entry in slices], "s-")
    half = 0.5 * result["separation"]
    axis.axhline(half, color="0.6", lw=0.8)
    axis.axhline(-half, color="0.6", lw=0.8)
    axis.plot(times, [entry["epsilon_hat"] for entry in slices], "^--")
    axis.axhline(result["epsilon"], color="0.6", lw=0.8, ls=":")
    axis.set_xlabel("t")
    axis.legend(["shift +", "shift -", "eps^"], fontsize=8)
    axis.set_title("latent readout from the fitted bond moments")

    axis = axes[1, 1]
    style = {"drawstyle": "steps-mid"} if family.is_lattice(baseline) else {}
    truth_site = pair_density(result["pair_target"], 0.0, grid).sum(axis=1)
    axis.plot(grid, truth_site, color="k", lw=1.5, label="site marginal truth", **style)
    generated = result["generation"]["sample"]
    if family.is_lattice(baseline):
        edges = np.concatenate((grid - 0.5, [grid[-1] + 0.5]))
        axis.hist(
            generated.reshape(-1),
            bins=edges,
            density=True,
            histtype="step",
            label="generated (all sites)",
        )
    reference = np.asarray(family.prob(grid, baseline), dtype=float)
    axis.plot(grid, reference, color="0.5", ls="--", lw=1.2, label="baseline")
    window = truth_site > 1e-8
    axis.set_xlim(grid[window].min(), grid[window].max())
    axis.legend(fontsize=8)
    axis.set_title("generation at t = 0")

    figure.suptitle(f"{result['name']} — {target.label} — chi = {result['chi']}")
    figure.tight_layout()
    directory = (
        Path("artifacts") / FIGURE_SUBDIRECTORY
        if output_dir is None
        else Path(output_dir)
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        f"{result['name']}-d{result['d']}-chi{result['chi']}"
        f"-eps{result['epsilon']:g}.png"
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


# ---------------------------------------------------------------------- CLI --
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="poisson")
    parser.add_argument("--sites", type=int, default=DEFAULT_SITES)
    parser.add_argument("--separation", type=float, default=DEFAULT_SEPARATION)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--slices", type=int, default=DEFAULT_SLICES)
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--chi", type=int, default=DEFAULT_CHI)
    parser.add_argument("--sweeps", type=int, default=DEFAULT_SWEEPS)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.family == "ghs":
        print("ghs — skipped: no positivity-preserving one-shot kernel")
        return
    result = run_chain_study(
        args.family,
        d=args.sites,
        separation=args.separation,
        epsilon=args.epsilon,
        n_slices=args.slices,
        degree=args.degree,
        chi=args.chi,
        sweeps=args.sweeps,
        draws=args.draws,
        tau=args.tau,
        seed=args.seed,
    )
    report(result)
    if args.plot:
        print(f"  figure: {plot_chain_study(result, output_dir=args.output)}")


if __name__ == "__main__":
    main()
