"""
Utilities for removing unused or leakage-prone columns from the dataset.
"""

import pandas as pd


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary columns from the dataset:

    - backers_count
    - percent_funded
    - all '*pledged*' columns
    - spotlight
    - staff_pick
    - all *profile* columns
    - all *creator* columns (except 'creator_id')
    - all *URL* columns
    - category_name
    - category_id
    - collected_at
    - state_changed_at
    - usd_type
    - all *location* columns
    - all currency exchange rates 

    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe with original dataset columns.

    Returns
    -------
    df : pd.DataFrame
        An updated Dataframe with the unnecessary columns removed.
    """
    df = df.copy()

    cols_to_drop = [
        "backers_count", "percent_funded",
        "converted_pledged_amount", "pledged", "usd_pledged",
        "spotlight", "staff_pick",
        "profile_blurb", "profile_id", "profile_name",
        "profile_project_id", "profile_state", "profile_state_changed_at",
        "creator_name", "creator_url",
        "category_url", "project_url",
        "category_id",
        "collected_at", "state_changed_at",
        "usd_type", "location_country", "location_id", 
        "location_name", "location_state", "location_type",
        "static_usd_rate", "usd_exchange_rate", "fx_rate", "launched_at",
        "created_at"
    ]

    # Drop only the columns found in the dataframe
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)

    return df
