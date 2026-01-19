"""
Feature engineering utilities for preparing model input data.

This module adds derived numeric and categorical features used
in downstream machine learning models.
"""

import pandas as pd
import numpy as np

from src.preprocessing.add_blurb_features import add_blurb_features
from src.preprocessing.add_name_features import add_name_features

def add_creator_history_features(
    df: pd.DataFrame,
    creator_col: str = "creator_id",
    launch_col: str = "launched_date",
    state_col: str = "state",
    success_label: str = "successful",
) -> pd.DataFrame:
    """
    Add creator-level historical features based on past projects.

    Creates cumulative counts of previous projects and previously
    successful projects for each creator, ordered by launch date.
    """

    df = df.copy()

    # Ensure datetime format
    df["_launch_dt"] = pd.to_datetime(df[launch_col], errors="coerce")

    # Sort by creator and time
    df_sorted = df.sort_values([creator_col, "_launch_dt"])

    # Binary success indicator
    df_sorted["_is_success"] = (df_sorted[state_col] == success_label).astype(int)

    # Cumulative counters per creator
    df_sorted["creator_prev_projects_successful"] = (
        df_sorted.groupby(creator_col)["_is_success"].cumsum()
        - df_sorted["_is_success"]
    )

    df_sorted["creator_prev_projects"] = (
        df_sorted.groupby(creator_col).cumcount()
    )

    # Cleanup
    df_sorted = df_sorted.drop(columns=["_launch_dt", "_is_success"])
    df_final = df_sorted.sort_index()

    return df_final


def add_project_duration(
    df: pd.DataFrame,
    launch_col: str = "launched_date",
    deadline_col: str = "deadline",
    new_col: str = "project_duration_days",
) -> pd.DataFrame:
    """
    Add a column describing project duration in days.

    The duration is calculated as the difference between the
    deadline and launch dates.
    """


    df = df.copy()

    launch_dt = pd.to_datetime(df[launch_col], errors="coerce")
    deadline_dt = pd.to_datetime(df[deadline_col], errors="coerce")

    df[new_col] = (deadline_dt - launch_dt).dt.days

    return df


def add_log_goal_feature(
    df: pd.DataFrame,
    goal_col: str = "usd_goal_fx",
    new_col: str = "usd_goal_fx_log",
) -> pd.DataFrame:
    """
    Add a log-transformed version of the goal column.

    Uses log(goal + 1) to avoid issues with zero values.
    """


    df = df.copy()

    df[goal_col] = pd.to_numeric(df[goal_col], errors="coerce")
    df[new_col] = np.log(df[goal_col] + 1)

    return df


def reduce_subcategories(
    df: pd.DataFrame,
    main_col: str = "category_parent_name",
    sub_col: str = "category_name",
    min_n: int = 2000,
    new_col: str = "category_name_reduced",
) -> pd.DataFrame:
    """
    Reduce rare subcategories into a single 'other' category.

    Subcategories with fewer than `min_n` occurrences are grouped
    into 'other', while ensuring that each main category retains
    at least one subcategory.
    """


    df = df.copy()

    counts = (
        df.groupby([main_col, sub_col])
        .size()
        .reset_index(name="n")
    )

    frequent = counts[counts["n"] >= min_n].copy()

    for main in counts[main_col].unique():
        if main not in frequent[main_col].values:
            top_row = (
                counts[counts[main_col] == main]
                .sort_values("n", ascending=False)
                .iloc[0]
            )
            frequent = pd.concat(
                [frequent, top_row.to_frame().T],
                ignore_index=True,
            )

    frequent["keep"] = True

    df = df.merge(
        frequent[[main_col, sub_col, "keep"]],
        how="left",
        on=[main_col, sub_col],
    )

    df[new_col] = df[sub_col].where(df["keep"].eq(True), "other")
    df = df.drop(columns=["keep"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a preprocessed dataset.

    This function combines text-based, historical, temporal, and
    categorical feature transformations used for modeling.
    """

    df = df.copy()

    df = add_blurb_features(df)
    df = add_name_features(df)

    df = add_creator_history_features(
        df,
        creator_col="creator_id",
        launch_col="launched_date",
        state_col="state",
        success_label="successful"
    )

    df = add_project_duration(
        df,
        launch_col="launched_date",
        deadline_col="deadline",
        new_col="project_duration_days",
    )

    df = add_log_goal_feature(df)

    df = reduce_subcategories(df)

    return df


if __name__ == "__main__":
    df = pd.read_csv("./data/processed_data.csv")
    output_csv = "./data/feature_processed_data.csv"
    df_out = engineer_features(df)
    df_out.to_csv(output_csv, index=False)
