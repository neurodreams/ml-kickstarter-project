"""
Utilities for converting project goal amounts to USD using daily FX rates.
"""

import pandas as pd
from utils.add_fx_mean import add_fx_mean


def calculate_usd_goal(df) -> pd.DataFrame:
    """
    Add USD-adjusted goal values to the dataset.

    This function applies `add_fx_mean` to resolve the daily FX rate for
    each row and computes a new column `usd_goal_fx` as `goal * fx_daily_mean`.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing at least `goal`, `currency`, and `launched_at`.

    Returns
    -------
    pd.DataFrame
        DataFrame with added `fx_daily_mean` and `usd_goal_fx` columns.
    """
    df = df.copy()
    df = add_fx_mean(df)
    df["usd_goal_fx"] = (df["goal"] * df["fx_daily_mean"])

    return df
