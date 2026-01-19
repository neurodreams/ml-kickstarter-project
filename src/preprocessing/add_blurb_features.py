"""
Utilities for extracting simple text-based features from project blurbs.
"""

import pandas as pd

from langdetect import detect, LangDetectException


def _detect_language_safe(text: str) -> str:
    """
    Detects language and returns a two-letter language code (e.g. 'en') 
    or 'unknown' on failure.
    """
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def add_blurb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate simple numeric features from the 'blurb' text column and
    estimate the language of non-missing/valid blurbs.

    The function identifies missing or invalid blurbs and derives several
    lightweight text-based features that can be used in exploratory analysis
    or as model inputs.

    A blurb is considered invalid if:
        - it is NaN
        - it is an empty string after stripping whitespace
        - it contains Excel artefacts (#NAME, ?? etc.)
        - it contains only special characters (i.e., no letters or digits in any language)
        - it contains only digits

    Generated features:
        - blurb_missing:       1 if blurb is missing/invalid, else 0
        - blurb_len:           number of whitespace-separated tokens
        - blurb_lang:          detected language code (if not detected >> 'unknown')
        - blurb_is_english:    1 if detected language is English ('en'), else 0

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe that must contain a 'blurb' column.

    Returns
    -------
    df : pd.DataFrame
        A dataframe with additional blurb-related feature columns.
    """

    df = df.copy()

    # Convert to string to avoid errors when processing mixed types
    blurb_raw = df["blurb"].astype(str)

    # Excel artefacts or meaningless tokens
    invalid_values = {
        "#NAME", "#NAME?", "??", "?", "N/A", "NA", "nan"
    }
    invalid_set_upper = {val.upper() for val in invalid_values}

    blurb_strip = blurb_raw.str.strip()

    is_nan = df["blurb"].isna()
    is_empty = blurb_strip == ""
    is_invalid_token = blurb_strip.str.upper().isin(invalid_set_upper)

    is_only_digits = blurb_strip.str.fullmatch(r"\d+").fillna(False)

    # Check if blurb contains at least one letter (any language) or digit.
    # If not and is not empty it's only special characters.
    contains_letter_or_digit = blurb_strip.str.contains(r"\d|[^\W\d_]", regex=True)
    is_only_special = ~contains_letter_or_digit & (blurb_strip != "")

    df["blurb_missing"] = (
        is_nan |
        is_empty |
        is_invalid_token |
        is_only_digits |
        is_only_special
    ).astype(int)

    blurb_clean = blurb_strip.mask(df["blurb_missing"] == 1, "")

    df["blurb_len"] = blurb_clean.str.split().str.len()

    df["blurb_lang"] = "unknown"
    mask_lang = (df["blurb_missing"] == 0) & (blurb_clean.str.len() > 0)

    if mask_lang.any():
        detected_langs = blurb_clean[mask_lang].apply(_detect_language_safe)
        df.loc[mask_lang, "blurb_lang"] = detected_langs

    # English / other languages
    df["blurb_is_english"] = (df["blurb_lang"] == "en").astype(int)

    return df
