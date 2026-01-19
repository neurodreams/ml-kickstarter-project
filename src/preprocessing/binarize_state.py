"""
Utilities for cleaning and binarizing the project state label.
"""

import pandas as pd


def clean_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and binarize the 'state' column:

    - Treat 'canceled' projects as 'failed' by replacing value 'canceled' with 'failed'
    - Keep only rows where state is 'failed' or 'successful' and drop all others

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'state' column.

    Returns
    -------
    df : pd.DataFrame
        A new DataFrame with:
        - 'canceled' merged into 'failed'
        - only 'failed' and 'successful' projects retained
    """
    df = df.copy()

    # Merge 'canceled' with 'failed'
    df["state"] = df["state"].replace({"canceled": "failed"})

    # Keep only failed and successful
    df = df[df["state"].isin(["failed", "successful"])]

    return df
