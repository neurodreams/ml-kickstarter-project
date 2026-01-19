"""
Utilities for removing projects with zero funding goals from the dataset.
"""

import pandas as pd


def drop_zero_goal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where *goal* value is 0

    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe containing a 'goal' column

    Returns
    -------
    df : pd.DataFrame
        A new DataFrame where all rows with goal == 0 have been dropped.
    """
    df = df.copy()
    df = df[df["goal"] != 0]
    return df
