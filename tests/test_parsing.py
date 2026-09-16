"""The parser has to cope with however a paper wrote its numbers."""

import numpy as np
import pytest

from vaccine_data import normalise_category, normalise_columns, parse_quantity, to_proportion


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1e6", 1e6),
        ("1E6", 1e6),
        ("1x10^6", 1e6),
        ("1×10⁶", 1e6),
        ("1.5x10^6", 1.5e6),
        ("2 x 10^6", 2e6),
        # Bare powers: the original parser stripped the caret and read these as 106.
        ("10^6", 1e6),
        ("10⁶", 1e6),
        ("1 000 000", 1e6),
        # Guards against a power-of-ten rule that is too eager: these are
        # ordinary numbers and must not be read as ten to some power.
        ("1000000", 1e6),
        ("106", 106.0),
        ("10", 10.0),
        ("~1e6", 1e6),
        ("1e6 CFU", 1e6),
        ("<10", 10.0),
        ("3.0", 3.0),
        ("3 µg", 3.0),
        ("5ug", 5.0),
        ("2 mg", 2000.0),
    ],
)
def test_quantities_parse_to_the_right_number(text, expected):
    assert parse_quantity(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "-", "NA", "n/a", "not reported", "1e6-1e7", "several", None])
def test_unreadable_cells_become_nan_rather_than_a_wrong_number(text):
    assert np.isnan(parse_quantity(text))


def test_unknown_units_are_refused_instead_of_guessed():
    assert np.isnan(parse_quantity("5 furlongs"))


def test_numbers_pass_through_unchanged():
    assert parse_quantity(1e6) == 1e6
    assert parse_quantity(3) == 3.0


def test_category_and_column_normalisation():
    assert normalise_category("  BALB/c  ") == "balb/c"
    assert normalise_columns(["plague_ type", "CFU", "Post-challenge (1 dose)"]) == [
        "plague_type",
        "cfu",
        "post_challenge_1_dose",
    ]


def test_percentages_and_proportions_are_detected():
    import pandas as pd

    percent, scale = to_proportion(pd.Series([10.0, 50.0, 100.0]))
    assert scale == "percent" and percent.max() == pytest.approx(1.0)
    proportion, scale = to_proportion(pd.Series([0.1, 0.5, 1.0]))
    assert scale == "proportion" and proportion.max() == pytest.approx(1.0)
