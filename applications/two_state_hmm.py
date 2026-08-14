"""Two-state hidden-Markov benchmark with NEF-QVF emissions.

The benchmark separates spatial dependence from the local observation family.
A symmetric two-state Markov chain supplies the dependence; the emissions are
two members of one NEF-QVF family, and the family can be exchanged without
touching the latent process.

Its point is that everything stays in closed form at every diffusion time, so a
fitted model can be scored against truth rather than against a longer run of
itself. Three facts do the work.

Family invariance. The reference process maps a family member to a family
member, contracting the shift coordinate as ``z -> exp(-t) z``. So the noised
emission is obtained by rescaling two scalars, with no kernel integral, and the
law at time ``t`` is again a two-state hidden Markov model with the *same*
transition matrix.

Exact likelihood. With the latent state space of dimension two, the likelihood
is a transfer-matrix product, evaluable in ``O(L)`` per sample.

Closed-form moments. Noising leaves the correlation length alone and damps the
mean-chart contrast as ``exp(-t)``, and the one-site variance is the variance
function read at the contracted means.

Nothing here is family-specific except the choice of baseline: the flow, the
likelihood, the sampler and every check are generic.

A note on GHS. Its marginal flow is legitimate, since ``z -> exp(-t) z`` is an
identity in the shift coordinate and the contracted member is an ordinary
probability law. What GHS lacks is a positivity-preserving kernel, hence any
*joint* draw of a state and its noised counterpart. This module only ever needs
marginals, so GHS runs; a forward Markov process would not.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator
from scipy.special import logsumexp

from nefqvf import (
    GHS,
    Binomial,
    BinomialParams,
    Gamma,
    GammaParams,
    GHSParams,
    NegativeBinomial,
    NegativeBinomialParams,
    Normal,
    NormalParams,
    Poisson,
    PoissonParams,
)

FAMILY_NAMES = (
    "normal",
    "poisson",
    "gamma",
    "binomial",
    "negative-binomial",
    "ghs",
)

# The only family-specific data in the module: a baseline to relax towards and
# a natural-parameter separation for the two emissions. Both are chosen so that
# the emissions are clearly distinct while staying inside the natural domain.
BASELINES: dict[str, tuple[Any, Any, float]] = {
    "normal": (Normal, NormalParams(mean=0.0, sigma=1.0), 3.0),
    "poisson": (Poisson, PoissonParams(mean=6.0), 1.4),
    "gamma": (Gamma, GammaParams(mean=3.0, r=2.5), 1.0),
    "binomial": (Binomial, BinomialParams(mean=3.6, N=12), 2.0),
    "negative-binomial": (
        NegativeBinomial,
        NegativeBinomialParams(mean=4.0, r=3.0),
        0.6,
    ),
    "ghs": (GHS, GHSParams(mean=0.0, r=1.5), 3.0),
}

DISPLAY_NAMES = {
    "normal": "Normal",
    "poisson": "Poisson",
    "gamma": "Gamma",
    "binomial": "Binomial",
    "negative-binomial": "Negative binomial",
    "ghs": "GHS",
}

# Index 0 is the state +1 and index 1 the state -1, throughout.
STATIONARY = np.array([0.5, 0.5])

# Family-independent defaults, shared by the functions below and the CLI.
DEFAULT_EPSILON = 0.15
DEFAULT_SITES = 24
DEFAULT_SAMPLES = 200_000
DEFAULT_TIMES = (0.0, 0.3, 1.0, 2.0, 3.0)
DEFAULT_LAGS = (1, 2, 4, 8)
# Chosen so that a 48-site path at the default epsilon shows several domains of
# comparable length, rather than one long stretch in which nothing happens.
DEFAULT_SEED = 33
DEFAULT_DRAWS = 16
# A sweep spanning both extremes: domains far longer than the lattice, domains
# far shorter than one site, the uncorrelated case, and the alternating regime
# beyond epsilon = 1/2.
DEFAULT_EPSILONS = (0.002, 0.05, 0.1, 0.15, 0.35, 0.5, 0.9)
# These figures get their own directory, since a sweep produces many of them.
FIGURE_SUBDIRECTORY = "two_state_hmm"


def transition_matrix(epsilon: float) -> np.ndarray:
    """Return the symmetric two-state transition matrix."""

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    return np.array([[1.0 - epsilon, epsilon], [epsilon, 1.0 - epsilon]])


def latent_correlation(epsilon: float, lag: Any) -> np.ndarray:
    """Return ``E[S_i S_j] = rho**lag`` with ``rho = 1 - 2 epsilon``."""

    return np.asarray((1.0 - 2.0 * epsilon) ** np.asarray(lag, dtype=float))


def correlation_length(epsilon: float) -> float:
    """Return ``-1 / log|rho|``, infinite when the chain never flips."""

    rho = abs(1.0 - 2.0 * epsilon)
    if rho == 0.0:
        return 0.0
    if rho == 1.0:
        return np.inf
    return float(-1.0 / np.log(rho))


def emission_parameters(
    family: Any, baseline: Any, separation: float
) -> tuple[Any, Any, np.ndarray]:
    """Return the two emission members and their shift coordinates.

    The emissions sit at ``eta +- separation / 2``. Their shift coordinates are
    what the forward flow acts on, so they are returned alongside.
    """
    half = 0.5 * float(separation)
    plus = family.shifted_params(baseline, +half)
    minus = family.shifted_params(baseline, -half)
    z = np.array(
        [
            float(family.shift_coordinate(+half, baseline)),
            float(family.shift_coordinate(-half, baseline)),
        ]
    )
    return plus, minus, z


def noised_emissions(
    family: Any, baseline: Any, z: np.ndarray, t: float
) -> tuple[Any, Any]:
    """Return the two emissions at diffusion time ``t``.

    This is the whole of the forward process: family invariance contracts the
    shift coordinate, so the noised member is recovered by inverting the
    contracted coordinate at the baseline. No kernel is applied.
    """
    damping = float(np.exp(-t))
    return (
        family.from_shift_coordinate(damping * z[0], baseline),
        family.from_shift_coordinate(damping * z[1], baseline),
    )


def sample_latent(
    sites: int, samples: int, epsilon: float, rng: Generator
) -> np.ndarray:
    """Draw ``samples`` latent paths of length ``sites`` taking values +-1."""

    if sites < 1 or samples < 1:
        raise ValueError("sites and samples must be positive")
    start = 2 * rng.integers(0, 2, size=(samples, 1)) - 1
    if sites == 1:
        return start.astype(float)
    flips = np.where(rng.random((samples, sites - 1)) < epsilon, -1.0, 1.0)
    return np.cumprod(np.concatenate([start, flips], axis=1), axis=1)


def sample_observations(
    family: Any, plus: Any, minus: Any, states: np.ndarray, rng: Generator
) -> np.ndarray:
    """Draw one observation per site, conditionally on the latent path.

    Draws are grouped by state rather than made site by site, which is why the
    library sampler only needs scalar parameters.
    """
    up = states > 0.0
    observations = np.empty(states.shape, dtype=float)
    count_up = int(np.count_nonzero(up))
    count_down = int(up.size - count_up)
    if count_up:
        observations[up] = family.sample(plus, count_up, rng=rng)
    if count_down:
        observations[~up] = family.sample(minus, count_down, rng=rng)
    return observations


def log_likelihood(
    family: Any, observations: np.ndarray, plus: Any, minus: Any, epsilon: float
) -> np.ndarray:
    """Return the exact log likelihood of each sample path.

    This is the transfer-matrix product ``pi^T D(x_1) T ... D(x_L) 1``, run as a
    rescaled forward recursion: the emission vector is factored by its larger
    entry and the running vector renormalised at every site, so neither the
    product of ``L`` densities nor the product of ``L`` transition factors can
    underflow.
    """
    observations = np.atleast_2d(np.asarray(observations, dtype=float))
    transition = transition_matrix(epsilon)
    log_emission = np.stack(
        [
            np.asarray(family.log_prob(observations, plus), dtype=float),
            np.asarray(family.log_prob(observations, minus), dtype=float),
        ],
        axis=-1,
    )

    offset = np.max(log_emission[:, 0, :], axis=-1)
    alpha = STATIONARY * np.exp(log_emission[:, 0, :] - offset[:, None])
    scale = alpha.sum(axis=-1)
    total = np.log(scale) + offset
    alpha = alpha / scale[:, None]

    for site in range(1, observations.shape[1]):
        offset = np.max(log_emission[:, site, :], axis=-1)
        alpha = (alpha @ transition) * np.exp(
            log_emission[:, site, :] - offset[:, None]
        )
        scale = alpha.sum(axis=-1)
        total = total + np.log(scale) + offset
        alpha = alpha / scale[:, None]
    return total


def log_likelihood_by_enumeration(
    family: Any, observations: np.ndarray, plus: Any, minus: Any, epsilon: float
) -> np.ndarray:
    """Return the same log likelihood by summing over all latent paths.

    Exponential in the number of sites, so this is a check on
    :func:`log_likelihood` rather than a usable evaluator.
    """
    observations = np.atleast_2d(np.asarray(observations, dtype=float))
    samples, sites = observations.shape
    transition = transition_matrix(epsilon)
    log_transition = np.log(transition)
    log_emission = np.stack(
        [
            np.asarray(family.log_prob(observations, plus), dtype=float),
            np.asarray(family.log_prob(observations, minus), dtype=float),
        ],
        axis=-1,
    )

    terms = np.empty((samples, 2**sites), dtype=float)
    for index in range(2**sites):
        path = [(index >> shift) & 1 for shift in range(sites)]
        value = np.log(0.5) + sum(
            log_transition[path[site], path[site + 1]] for site in range(sites - 1)
        )
        value = value + sum(log_emission[:, site, path[site]] for site in range(sites))
        terms[:, index] = value
    return logsumexp(terms, axis=1)


def predicted_moments(
    family: Any, baseline: Any, plus: Any, minus: Any
) -> dict[str, float]:
    """Return the closed-form one-site moments for a pair of emissions.

    ``variance_two_state`` averages the variance function over the two members;
    ``variance_expanded`` instead expands about the midpoint, which is equal only
    because the variance function is quadratic. Comparing them therefore tests
    the quadratic-variance property itself.
    """
    mu_plus = float(family.mean(plus))
    mu_minus = float(family.mean(minus))
    centre = 0.5 * (mu_plus + mu_minus)
    contrast = 0.5 * (mu_plus - mu_minus)

    v_plus = float(family.variance(plus))
    v_minus = float(family.variance(minus))
    variance_two_state = 0.5 * (v_plus + v_minus) + contrast**2

    # Every parameter record carries ``mean`` alongside its fixed fields, so the
    # member at the midpoint mean is generic to build.
    midpoint = replace(baseline, mean=centre)
    v_centre = float(family.variance(midpoint))
    if mu_plus != mu_minus:
        second_derivative = (
            float(family.variance_slope(plus)) - float(family.variance_slope(minus))
        ) / (mu_plus - mu_minus)
    else:
        second_derivative = 0.0
    variance_expanded = v_centre + (1.0 + 0.5 * second_derivative) * contrast**2

    return {
        "mean": centre,
        "contrast": contrast,
        "variance_two_state": variance_two_state,
        "variance_expanded": variance_expanded,
        "variance_second_derivative": second_derivative,
    }


def _lag_correlation(values: np.ndarray, lag: int, centre: float) -> float:
    """Return the empirical two-point function at a fixed separation."""

    left = values[:, :-lag] - centre
    right = values[:, lag:] - centre
    return float(np.mean(left * right))


def run_benchmark(
    name: str,
    *,
    epsilon: float = DEFAULT_EPSILON,
    sites: int = DEFAULT_SITES,
    samples: int = DEFAULT_SAMPLES,
    times: tuple[float, ...] = DEFAULT_TIMES,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    seed: int = DEFAULT_SEED,
) -> None:
    """Run the sampler, the exact likelihood and the moment checks."""

    if name not in BASELINES:
        choices = ", ".join(FAMILY_NAMES)
        raise ValueError(f"unknown family {name!r}; choose one of: {choices}")

    family, baseline, separation = BASELINES[name]
    rng = np.random.default_rng(seed)
    plus, minus, z = emission_parameters(family, baseline, separation)
    baseline_mean = float(family.mean(baseline))
    rho = 1.0 - 2.0 * epsilon

    print()
    print(f"{name}: baseline mean {baseline_mean:g}, eta separation {separation:g}")
    print(
        f"epsilon {epsilon:g}, rho {rho:.4f}, correlation length "
        f"{correlation_length(epsilon):.3f} sites; {sites} sites, {samples} samples"
    )

    # --- the latent chain on its own -------------------------------------
    states = sample_latent(sites, samples, epsilon, rng)
    print()
    print("latent two-point function")
    print(f"  {'lag':>4} {'empirical':>12} {'rho**lag':>12} {'difference':>12}")
    for lag in lags:
        if lag >= sites:
            continue
        empirical = _lag_correlation(states, lag, 0.0)
        exact = float(latent_correlation(epsilon, lag))
        print(f"  {lag:4d} {empirical:12.6f} {exact:12.6f} {empirical - exact:12.2e}")

    # --- the emission flow and one-site moments --------------------------
    print()
    print("emission flow: means, and one-site moments of the noised law")
    header = (
        f"  {'t':>5} {'mu+(t)':>10} {'mu-(t)':>10} {'flow err':>10} "
        f"{'mean emp':>10} {'mean pred':>10} {'var emp':>10} {'var pred':>10} "
        f"{'var alt':>10}"
    )
    print(header)
    for t in times:
        noised_plus, noised_minus = noised_emissions(family, baseline, z, t)
        moments = predicted_moments(family, baseline, noised_plus, noised_minus)

        # the flow must agree with the contraction of the mean coordinate
        damping = float(np.exp(-t))
        flow_error = max(
            abs(
                float(family.mean(noised))
                - (baseline_mean + damping * (float(family.mean(raw)) - baseline_mean))
            )
            for noised, raw in ((noised_plus, plus), (noised_minus, minus))
        )

        observations = sample_observations(
            family, noised_plus, noised_minus, states, rng
        )
        print(
            f"  {t:5.2f} {float(family.mean(noised_plus)):10.5f} "
            f"{float(family.mean(noised_minus)):10.5f} {flow_error:10.2e} "
            f"{observations.mean():10.5f} {moments['mean']:10.5f} "
            f"{observations.var():10.5f} {moments['variance_two_state']:10.5f} "
            f"{moments['variance_expanded']:10.5f}"
        )

    # --- the noised two-point function -----------------------------------
    print()
    print("covariance of the noised law, against exp(-2t) Delta_mu**2 rho**lag")
    print(
        f"  {'t':>5} {'lag':>4} {'empirical':>12} {'predicted':>12} {'difference':>12}"
    )
    for t in times:
        noised_plus, noised_minus = noised_emissions(family, baseline, z, t)
        moments = predicted_moments(family, baseline, noised_plus, noised_minus)
        observations = sample_observations(
            family, noised_plus, noised_minus, states, rng
        )
        for lag in lags:
            if lag >= sites:
                continue
            empirical = _lag_correlation(observations, lag, moments["mean"])
            exact = moments["contrast"] ** 2 * float(latent_correlation(epsilon, lag))
            print(
                f"  {t:5.2f} {lag:4d} {empirical:12.6f} {exact:12.6f} "
                f"{empirical - exact:12.2e}"
            )

    # --- the exact likelihood --------------------------------------------
    print()
    print("exact likelihood")
    small_sites, small_samples = 8, 256
    small_states = sample_latent(small_sites, small_samples, epsilon, rng)
    noised_plus, noised_minus = noised_emissions(family, baseline, z, 0.5)
    small = sample_observations(family, noised_plus, noised_minus, small_states, rng)

    transfer = log_likelihood(family, small, noised_plus, noised_minus, epsilon)
    enumerated = log_likelihood_by_enumeration(
        family, small, noised_plus, noised_minus, epsilon
    )
    print(
        f"  transfer matrix vs sum over all 2**{small_sites} latent paths: "
        f"max |difference| = {np.max(np.abs(transfer - enumerated)):.2e}"
    )

    # at epsilon = 1/2 the latent states are independent, so the law is a
    # product of two-component mixtures
    independent = log_likelihood(family, small, noised_plus, noised_minus, 0.5)
    mixture = np.sum(
        logsumexp(
            np.stack(
                [
                    np.asarray(family.log_prob(small, noised_plus)),
                    np.asarray(family.log_prob(small, noised_minus)),
                ],
                axis=-1,
            )
            + np.log(0.5),
            axis=-1,
        ),
        axis=1,
    )
    print(
        "  epsilon = 1/2 vs a product of two-component mixtures: "
        f"max |difference| = {np.max(np.abs(independent - mixture)):.2e}"
    )

    # as t grows both emissions collapse onto the baseline, so the law becomes
    # the product baseline whatever the latent chain does
    late_plus, late_minus = noised_emissions(family, baseline, z, 40.0)
    late = log_likelihood(family, small, late_plus, late_minus, epsilon)
    product = np.sum(np.asarray(family.log_prob(small, baseline)), axis=1)
    print(
        "  t = 40 vs the product baseline: "
        f"max |difference| = {np.max(np.abs(late - product)):.2e}"
    )


def _parse_args() -> argparse.Namespace:
    """Parse the command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="normal", choices=(*FAMILY_NAMES, "all"))
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--sites", type=int, default=DEFAULT_SITES)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="run numerical checks without writing figures",
    )
    parser.add_argument(
        "--variants",
        action="store_true",
        help="sweep the correlation length instead of drawing one toy",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark from the command line."""

    args = _parse_args()
    names = FAMILY_NAMES if args.family == "all" else (args.family,)
    for name in names:
        run_benchmark(
            name,
            epsilon=args.epsilon,
            sites=args.sites,
            samples=args.samples,
            seed=args.seed,
        )
        if args.no_plot:
            continue
        if args.variants:
            plot_correlation_variants(name, seed=args.seed)
        else:
            plot_toy(name, epsilon=args.epsilon, seed=args.seed)


if __name__ == "__main__":
    main()


def plot_toy(
    name: str,
    *,
    epsilon: float = DEFAULT_EPSILON,
    sites: int = 48,
    draws: int = DEFAULT_DRAWS,
    times: tuple[float, ...] = DEFAULT_TIMES,
    seed: int = DEFAULT_SEED,
    output_dir: Any = None,
    tag: str = "",
    extension: str = "png",
    width: float = 12.0,
    row_height: float = 1.5,
) -> None:
    """Draw one latent path and show what noising does to it.

    Each panel puts the lattice of sites on the horizontal axis and the
    observation coordinate on the vertical one. The background is the exact
    emission density at that site, conditional on the sampled latent state, so
    the two latent regimes are visible as two bands. Over it sit the emission
    means as a step line, the baseline mean as a dashed line, and a population
    of ``draws`` observations per site,
    jittered horizontally so that the spread within a site is visible rather
    than overplotted.

    Reading down the panels, the latent path never changes and the contrast
    between the two regimes decays as ``exp(-t)`` while the bands broaden onto
    the common baseline. Only the latent path is shared between panels; the
    observations are drawn afresh at every time, so a given point carries no
    identity from one row to the next and only the population should be read.

    The panels are deliberately independent, not coupled: a data law is not
    represented by an amplitude here, so no point has an identity to carry from
    one time to the next, and a coupling would only invite reading one in.
    """
    if name not in BASELINES:
        choices = ", ".join(FAMILY_NAMES)
        raise ValueError(f"unknown family {name!r}; choose one of: {choices}")

    family, baseline, separation = BASELINES[name]
    # One frozen seed for reproducibility, but an independent stream for the
    # latent path, the jitter and each diffusion time. Re-seeding identically
    # per panel would instead couple them, and only for some families: a
    # location-scale sampler reuses the same standard variate and an inverse-CDF
    # sampler the same uniform, whereas a rejection-based lattice sampler
    # consumes the stream at a parameter-dependent rate and desynchronises. That
    # asymmetry is an artefact of the generator, not of the model, and it would
    # invite reading a point as evolving when nothing here evolves pointwise.
    latent_seed, jitter_seed, *time_seeds = np.random.SeedSequence(seed).spawn(
        2 + len(times)
    )
    rng = np.random.default_rng(latent_seed)
    plus, minus, z = emission_parameters(family, baseline, separation)
    lattice = family.is_lattice(baseline)

    # one latent path, reused at every time and shared by the whole population
    path = sample_latent(sites, 1, epsilon, rng)
    states = np.repeat(path, draws, axis=0)
    up = path[0] > 0.0
    jitter = np.random.default_rng(jitter_seed).uniform(-0.3, 0.3, size=states.shape)

    # Everything is sampled before anything is drawn, so that the vertical range
    # can cover the laws *and* the draws. Sizing it from the laws alone lets an
    # outlying draw stretch the axis past the shading and leave white space.
    panels = []
    for t, child_seed in zip(times, time_seeds):
        noised_plus, noised_minus = noised_emissions(family, baseline, z, t)
        observations = sample_observations(
            family, noised_plus, noised_minus, states, np.random.default_rng(child_seed)
        )
        panels.append((t, noised_plus, noised_minus, observations))

    spread = max(
        float(np.sqrt(family.variance(member)))
        for _, noised_plus, noised_minus, _ in panels
        for member in (noised_plus, noised_minus)
    )
    centre = float(family.mean(baseline))
    drawn_low = min(float(observations.min()) for *_, observations in panels)
    drawn_high = max(float(observations.max()) for *_, observations in panels)
    low = min(centre - 4.5 * spread, drawn_low)
    high = max(centre + 4.5 * spread, drawn_high)

    # cell edges for the mesh, with the density evaluated at the cell centres
    if lattice:
        low = max(low, 0.0)
        coordinate = np.arange(int(np.floor(low)), int(np.ceil(high)) + 1)
        edges = np.append(coordinate - 0.5, coordinate[-1] + 0.5)
    else:
        margin = 0.02 * (high - low)
        edges = np.linspace(low - margin, high + margin, 401)
        coordinate = 0.5 * (edges[:-1] + edges[1:])

    figure, axes = plt.subplots(
        len(times),
        1,
        figsize=(width, row_height * len(times)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for axis, (t, noised_plus, noised_minus, observations) in zip(axes, panels):
        density = np.stack(
            [
                np.asarray(family.prob(coordinate, noised_plus), dtype=float),
                np.asarray(family.prob(coordinate, noised_minus), dtype=float),
            ]
        )
        # column j is the emission density at site j given its latent state
        panel = np.where(up[None, :], density[0][:, None], density[1][:, None])

        axis.pcolormesh(
            np.arange(sites + 1) - 0.5,
            edges,
            panel,
            cmap="Blues",
            shading="flat",
            rasterized=True,
        )

        # the baseline mean, which every emission relaxes onto
        axis.axhline(
            centre,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=r"baseline mean $\mu$",
        )

        levels = np.where(
            up, float(family.mean(noised_plus)), float(family.mean(noised_minus))
        )
        axis.step(
            np.arange(sites),
            levels,
            where="mid",
            color="crimson",
            linewidth=1.4,
            label=r"emission mean $\mu_{S_j,t}$",
        )
        axis.plot(
            (np.arange(sites)[None, :] + jitter).ravel(),
            observations.ravel(),
            "o",
            markersize=1.8,
            alpha=0.45,
            color="black",
            label=rf"{draws} draws per site",
        )
        axis.set_ylabel("$m$" if lattice else "$x$")
        contrast = 0.5 * (
            float(family.mean(noised_plus)) - float(family.mean(noised_minus))
        )
        axis.set_title(
            rf"$t = {t:g}$        $\Delta\mu_t = {contrast:.3f}$",
            fontsize=9,
            pad=3.0,
        )

    axes[-1].set_xlabel("site $j$")
    axes[-1].set_xlim(-0.5, sites - 0.5)
    axes[-1].set_ylim(edges[0], edges[-1])
    rho = 1.0 - 2.0 * epsilon
    length = correlation_length(epsilon)
    figure.suptitle(
        rf"{DISPLAY_NAMES[name]}        $\epsilon = {epsilon:g}$,   "
        rf"$\rho = {rho:+.3f}$,   $\xi = {length:.2f}$ sites",
        fontsize=11,
        y=0.995,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.978),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.968), h_pad=0.8)

    output = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parents[1] / "artifacts" / FIGURE_SUBDIRECTORY
    )
    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / f"{name}_two_state_hmm{tag}.{extension}"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    print(f"Figure: {figure_path}")


def plot_correlation_variants(
    name: str,
    *,
    epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
    sites: int = 48,
    draws: int = DEFAULT_DRAWS,
    times: tuple[float, ...] = DEFAULT_TIMES,
    seed: int = DEFAULT_SEED,
    output_dir: Any = None,
) -> None:
    """Draw the same toy at several correlation lengths.

    The sweep is meant to build intuition about what ``epsilon`` controls, and
    it deliberately includes both extremes. At ``epsilon`` far below one half the
    domains outrun the lattice and the field looks like a single regime; at one
    half the sites are independent and the latent structure disappears, which is
    the product-law null case; above one half ``rho`` turns negative and the
    regimes alternate site by site, so the correlation length describes the
    decay of an oscillating two-point function rather than a domain size.

    The lattice length is held fixed across the sweep so the panels can be read
    against each other.
    """
    for epsilon in epsilons:
        plot_toy(
            name,
            epsilon=epsilon,
            sites=sites,
            draws=draws,
            times=times,
            seed=seed,
            output_dir=output_dir,
            tag=f"_epsilon{epsilon:g}",
        )
