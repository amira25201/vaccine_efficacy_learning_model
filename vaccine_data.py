"""Reading and cleaning the vaccine spreadsheet.

The source data is transcribed from published animal studies, so numbers
arrive in whatever form the paper used: 1e6, 1x10^6, 1×10⁶, 10⁶, "3 µg",
"2 mg". `parse_quantity` turns all of those into a float, and returns NaN
rather than a wrong number when it cannot.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

NBSP = "\xa0"
_SUPERSCRIPTS = str.maketrans(
    {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+"}
)
_SUPERSCRIPT_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+")
_MISSING = {"", "-", "--", "na", "n/a", "nan", "none", "not reported", "nr"}

# Multipliers for mass units, normalised to micrograms.
_MASS_UNITS = {"ug": 1.0, "µg": 1.0, "μg": 1.0, "mcg": 1.0, "mg": 1000.0, "g": 1_000_000.0}

_PLAIN = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*([a-zµμ/]*)$")
# Scientific notation written out. The coefficient is optional so that "10^6"
# parses as a million rather than as 106, which is what the original parser
# returned. The caret is required: without it "1000000" would match as ten to
# the power zero.
_POWER = re.compile(r"^(?:([+-]?\d+(?:\.\d+)?)\s*x\s*)?10\s*\^\s*([+-]?\d+)$")


def _lift_superscripts(text: str) -> str:
    """Turn "10⁶" into "10^6" before anything else touches it.

    This has to happen first: Unicode NFKC normalisation rewrites superscript
    digits as ordinary ones, which would make "10⁶" indistinguishable from
    the number 106.
    """
    return _SUPERSCRIPT_RUN.sub(lambda m: "^" + m.group(0).translate(_SUPERSCRIPTS), text)


def _strip_thousand_separators(text: str) -> str:
    """Remove spaces used as digit-group separators, as in "1 000 000".

    Only spaces between digit groups go, so "2 x 10^6" and "3 µg" survive.
    """
    while True:
        collapsed = re.sub(r"(?<=\d) (?=\d{3}(?:\D|$))", "", text)
        if collapsed == text:
            return text
        text = collapsed


def _normalise_text(value) -> str:
    text = _lift_superscripts(str(value).strip())
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(",", "").replace(NBSP, " ")
    text = text.replace("·", "x").replace("×", "x").replace("∗", "x").replace("*", "x")
    return _strip_thousand_separators(text.lower().strip())


def parse_quantity(cell) -> float:
    """Parse one spreadsheet cell into a float, or NaN if it cannot be read."""
    if cell is None:
        return np.nan
    if isinstance(cell, (int, float, np.number)):
        return np.nan if pd.isna(cell) else float(cell)

    text = _normalise_text(cell)
    if text in _MISSING:
        return np.nan

    # Leading qualifiers such as "~1e6" or "<10" carry no numeric information.
    text = text.lstrip("~<>≈≤≥ ").strip()
    # Trailing labels such as "1e6 cfu".
    text = re.sub(r"\s*(cfu|colony[- ]forming units?|per mouse|/mouse)\s*$", "", text).strip()

    match = _POWER.match(text)
    if match:
        coefficient = float(match.group(1)) if match.group(1) else 1.0
        return coefficient * (10.0 ** int(match.group(2)))

    match = _PLAIN.match(text)
    if match:
        value, unit = float(match.group(1)), match.group(2)
        if unit in _MASS_UNITS:
            return value * _MASS_UNITS[unit]
        if unit == "":
            return value
        return np.nan  # a unit we do not understand: refuse rather than guess

    try:
        return float(text)  # plain exponent form, e.g. 1e6
    except ValueError:
        return np.nan


def normalise_category(value):
    if pd.isna(value):
        return value
    return " ".join(unicodedata.normalize("NFKC", str(value)).split()).lower()


def normalise_columns(columns) -> list[str]:
    return (
        pd.Index(columns)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
        .tolist()
    )


def to_proportion(series: pd.Series) -> tuple[pd.Series, str]:
    """
    Put a survival column on the 0-1 scale.

    Returns the series and the scale that was detected, so the same decision
    can be recorded with the model instead of being re-guessed per column.
    """
    values = pd.to_numeric(series, errors="coerce")
    observed = values.dropna()
    if observed.empty:
        return values, "unknown"
    if observed.max() > 1.0:
        return values / 100.0, "percent"
    return values, "proportion"


def load_dataset(path, sheet=0) -> pd.DataFrame:
    """Read the spreadsheet and apply every cleaning step, in order."""
    df = pd.read_excel(path, sheet_name=sheet)
    return clean_dataset(df)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    import config

    df = df.copy()
    df.columns = normalise_columns(df.columns)

    for column in config.CATEGORICAL_FEATURES:
        if column in df.columns:
            df[column] = df[column].map(normalise_category)

    for column in config.NUMERIC_FEATURES:
        if column in df.columns:
            df[column] = df[column].map(parse_quantity).astype(float)

    # Both targets get the same treatment, and the scale is recorded rather
    # than decided separately for each column.
    scales = {}
    for target in config.TARGETS.values():
        if target in df.columns:
            df[target], scales[target] = to_proportion(df[target])
    df.attrs["target_scales"] = scales
    return df


def check_groups(df: pd.DataFrame, group_column: str) -> str:
    """
    Validate the grouping column and describe it.

    Grouping is what stops the same animal appearing in both the training and
    test sets. If the column is missing, or gives one group per row, the
    caller needs to know before trusting any score.
    """
    if group_column not in df.columns:
        raise KeyError(
            f"Grouping column {group_column!r} is not in the spreadsheet. "
            f"Set config.GROUP_COLUMN to the column identifying an experiment "
            f"or animal. Columns found: {sorted(df.columns)}"
        )
    n_groups = df[group_column].nunique()
    if n_groups < 2:
        raise ValueError(f"Column {group_column!r} has {n_groups} distinct value(s); grouped splitting needs at least 2.")
    if n_groups == len(df):
        return (
            f"WARNING: {group_column!r} has one distinct value per row ({n_groups}), so grouping does nothing "
            f"and the scores below are optimistic. Point config.GROUP_COLUMN at the study or animal identifier."
        )
    return f"Grouping by {group_column!r}: {n_groups} groups across {len(df)} rows."
