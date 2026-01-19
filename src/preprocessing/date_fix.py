"""
Utilities for cleaning and imputing project launch and deadline dates.
"""

import pandas as pd


def fix_dates(df: pd.DataFrame, avg_duration_days: int = 34) -> pd.DataFrame:
    """
    Fix and standardize 'launched_at' and 'deadline' values.

    - Drop rows where deadline == 0 (these also have launched_at == 0).
    - For rows where launched_at == 0 but deadline is valid,
      impute launched_at as deadline minus an average project duration.
    - Convert both columns from Unix timestamps to daily datetime values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'launched_at' and 'deadline' columns
        as Unix timestamps (seconds).
    avg_duration_days : int, optional
        Average project duration in days used for imputing missing
        launched_at values. Default is 34.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with fixed and standardized date columns.
    """
    df = df.copy()

    # Remove lines with 'deadline' == 0 (these also have 'launched_at' == 0)
    df = df[df["deadline"] != 0]

    # Count the seconds from avg_duration_days for Unix impution
    seconds_per_day = 24 * 60 * 60
    offset_seconds = avg_duration_days * seconds_per_day

    # Imputate 'launched_at' where 'launched_at' == 0 but 'deadline' != 0
    mask_launched_missing = (df["launched_at"] == 0) & (df["deadline"] != 0)
    df.loc[mask_launched_missing, "launched_at"] = (
        df.loc[mask_launched_missing, "deadline"] - offset_seconds
    )

    # Convert to datetime
    df["launched_at"] = pd.to_datetime(df["launched_at"], unit="s", errors="coerce").dt.floor("D")
    df["deadline"] = pd.to_datetime(df["deadline"], unit="s", errors="coerce").dt.floor("D")

    # Return fixed dataframe
    return df
