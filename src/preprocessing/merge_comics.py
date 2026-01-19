"""
Utilities for standardizing category parent name capitalization.
"""

import pandas as pd


def merge_comics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize category parent names by converting 'comics' to 'Comics'.

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
    df['category_parent_name'] = df['category_parent_name'].replace({
        'comics': 'Comics'
    })
    return df
