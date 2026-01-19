"""
Utilities for normalizing project category labels.
"""

import pandas as pd


def merge_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize category parent names by converting 'Journalism' and 'Dance' to 'Others'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'category_parent_name' column.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the corrected 'category_parent_name' values.
    """
    df = df.copy()
    df["category_parent_name"] = df["category_parent_name"].replace(
        {"Journalism": "Others", "Dance": "Others"}
    )
    return df
