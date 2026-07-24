"""End-to-end checks for the shifted-baseline demonstration."""

import pytest

from applications.shifted_baseline_probability_modes import FAMILY_NAMES, run_demo


@pytest.mark.parametrize("family_name", FAMILY_NAMES)
def test_shifted_baseline_probability_modes(family_name):
    """Every family must pass projection, damping, and Lambda checks."""

    run_demo(family_name, make_plot=False)


def test_unknown_family_is_rejected():
    """The Python entry point should report the accepted family names."""

    with pytest.raises(ValueError, match="unknown family"):
        run_demo("not-a-family", make_plot=False)
