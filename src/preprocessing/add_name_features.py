"""
Utilities for extracting simple text-based features from project names.
"""

import pandas as pd
from langdetect import detect, LangDetectException


def _detect_language_safe(text: str) -> str:
    """
    Safely detect the language of a text string.

    Returns a two-letter language code (e.g. 'en') or 'unknown'
    if language detection fails.
    """
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def add_name_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate simple numeric features from the 'name' text column,
    using the same logic as for the 'blurb' field.

    Invalid or missing names are excluded from length calculation
    and language detection.

    Generated features:
        - name_missing:     1 if name is missing/invalid, else 0
        - name_len:         number of whitespace-separated tokens
        - name_lang:        detected language code (or 'unknown')
        - name_is_english:  binary language flag (1 if 'en')

    Returns:
        pd.DataFrame: Input DataFrame with additional name-related features.
    """

    df = df.copy()

    # Convert to string
    name_raw = df["name"].astype(str)
    name_strip = name_raw.str.strip()

    invalid_values = {"#NAME", "#NAME?", "??", "?", "N/A", "NA", "nan"}
    invalid_upper = {val.upper() for val in invalid_values}

    is_nan = df["name"].isna()
    is_empty = name_strip == ""
    is_invalid_token = name_strip.str.upper().isin(invalid_upper)
    is_only_digits = name_strip.str.fullmatch(r"\d+").fillna(False)

    contains_letter_or_digit = name_strip.str.contains(r"\d|[^\W\d_]", regex=True)
    is_only_special = ~contains_letter_or_digit & (name_strip != "")

    df["name_missing"] = (
        is_nan |
        is_empty |
        is_invalid_token |
        is_only_digits |
        is_only_special
    ).astype(int)

    # Clean valid names
    name_clean = name_strip.mask(df["name_missing"] == 1, "")

    # Word count
    df["name_len"] = name_clean.str.split().str.len()

    # Language detection
    df["name_lang"] = "unknown"
    mask = (df["name_missing"] == 0) & (name_clean.str.len() > 0)

    if mask.any():
        langs = name_clean[mask].apply(_detect_language_safe)
        df.loc[mask, "name_lang"] = langs

    df["name_is_english"] = (df["name_lang"] == "en").astype(int)

    return df
