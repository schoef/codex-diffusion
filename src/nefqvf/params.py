"""Immutable parameter records for the six NEF-QVF families."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class NormalParams:
    """Normal parameters with public mean and fixed standard deviation."""

    mean: ArrayLike
    sigma: ArrayLike


@dataclass(frozen=True, slots=True)
class PoissonParams:
    """Poisson parameters, represented directly by the nonnegative mean."""

    mean: ArrayLike


@dataclass(frozen=True, slots=True)
class GammaParams:
    """Gamma parameters with mean and positive shape ``r``."""

    mean: ArrayLike
    r: ArrayLike


@dataclass(frozen=True, slots=True)
class BinomialParams:
    """Binomial parameters with mean and integer number of trials ``N``."""

    mean: ArrayLike
    N: ArrayLike


@dataclass(frozen=True, slots=True)
class NegativeBinomialParams:
    """Negative-binomial parameters with mean and positive shape ``r``."""

    mean: ArrayLike
    r: ArrayLike


@dataclass(frozen=True, slots=True)
class GHSParams:
    """GHS parameters with mean and positive Meixner-Pollaczek shape ``r``."""

    mean: ArrayLike
    r: ArrayLike
