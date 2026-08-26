"""Fast NumPy/SciPy evaluation for the six one-dimensional NEF-QVF families."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Binomial": (".binomial", "Binomial"),
    "BinomialFamily": (".binomial", "BinomialFamily"),
    "BinomialParams": (".params", "BinomialParams"),
    "Gamma": (".gamma", "Gamma"),
    "GammaFamily": (".gamma", "GammaFamily"),
    "GammaParams": (".params", "GammaParams"),
    "GHS": (".ghs", "GHS"),
    "GHSFamily": (".ghs", "GHSFamily"),
    "GHSParams": (".params", "GHSParams"),
    "NegativeBinomial": (".negative_binomial", "NegativeBinomial"),
    "NegativeBinomialFamily": (
        ".negative_binomial",
        "NegativeBinomialFamily",
    ),
    "NegativeBinomialParams": (".params", "NegativeBinomialParams"),
    "Normal": (".normal", "Normal"),
    "NormalFamily": (".normal", "NormalFamily"),
    "NormalParams": (".params", "NormalParams"),
    "Poisson": (".poisson", "Poisson"),
    "PoissonFamily": (".poisson", "PoissonFamily"),
    "PoissonParams": (".params", "PoissonParams"),
    "fit_amplitude": (".fitting", "fit_amplitude"),
    "product_matrices": (".fitting", "product_matrices"),
    "ratio_coefficients": (".fitting", "ratio_coefficients"),
    "binomial": (".binomial", "binomial"),
    "gamma": (".gamma", "gamma"),
    "ghs": (".ghs", "ghs"),
    "negative_binomial": (".negative_binomial", "negative_binomial"),
    "normal": (".normal", "normal"),
    "poisson": (".poisson", "poisson"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load public family objects lazily to keep package import inexpensive."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported names in interactive completion."""

    return sorted((*globals(), *__all__))
