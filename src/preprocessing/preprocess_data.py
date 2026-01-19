"""
Preprocessing pipeline for Kickstarter project raw data.
"""

import pandas as pd

from .date_fix import fix_dates
from .calculate_usd_goal import calculate_usd_goal
from .merge_categories import merge_categories
from .merge_comics import merge_comics
from .binarize_state import clean_state
from .drop_cols import drop_unused_columns
from .drop_zero_goal_rows import drop_zero_goal


def preprocess_data(data: str = "./data/raw_data.csv", return_report: bool = True):
    """
    Run the full domain-level preprocessing pipeline on Kickstarter project data.

    This function loads raw CSV data and applies a sequence of cleaning and
    transformation steps needed to turn the dataset into a consistent,
    analysis-ready format. The steps performed are:

    1. **Fix invalid or missing date fields**
       - Cleans `launched_at` and `deadline` timestamps.
       - Removes rows with completely invalid date fields.

    2. **Drop rows with zero funding goals**
       - Ensures that all remaining projects have a positive goal value.

    3. **Compute `usd_goal_fx`**
       - Adds FX-rate based conversion (`fx_daily_mean`).
       - Computes the project's goal in USD using daily exchange rates.

    4. **Normalize category information**
       - Merges categories into broader parent groups.
       - Applies additional normalisation for comic-related categories.

    5. **Clean and binarize `state`**
       - Keeps only `successful` and `failed` projects.
       - Converts `canceled` → `failed`.

    6. **Drop unused and irrelevant columns**
       - Removes Kickstarter metadata fields not relevant for ML analysis.

    7. **Drop rows containing any remaining NaN values**
       - Ensures final dataset has full feature completeness.

    8. **Export the processed dataset**
       - Saves final CSV (and optionally parquet).

    Parameters
    ----------
    data : str, optional
        Path to the raw CSV file. Default is `"./data/raw_data.csv"`.

    return_report : bool, optional
        If True, prints a summary of row counts after each major cleaning
        step. Default is True.

    Returns
    -------
    pandas.DataFrame
        The fully cleaned and preprocessed dataset.

    Notes
    -----
    - The function writes two output files:
        * `./data/processed_data.csv`
        * `./data/processed_data.parquet`
    - Dropped row counts and final shape are printed when `return_report=True`.
    """

    df = pd.read_csv(data, low_memory=False)
    report = {}

    report["n_raw"] = len(df)

    # 1. Fix date fields
    df = fix_dates(df)
    report["n_after_fix_dates"] = len(df)

    # 2. Drop rows with zero goal
    df = drop_zero_goal(df)
    report["n_after_drop_zero_goal_columns"] = len(df)

    # 3. FX conversion
    df = calculate_usd_goal(df)

    # 4. Category merges
    df = merge_categories(df)
    df = merge_comics(df)

    # 5. Clean/binarize state-column
    df = clean_state(df)
    report["n_after_clean_state"] = len(df)

    # 6. Drop unused columns
    df = drop_unused_columns(df)

    # 7. Drop rows with NaN values
    df = df.dropna()
    df = df.reset_index(drop=True)

    report["n_finalized"] = len(df)

    # Reporting
    if return_report:
        print("\nPreprocessing report")
        for key, value in report.items():
            print(f"{key}: {value}")
        print(f"Dropped rows: {report['n_raw'] - report['n_finalized']}")
        print(f"Final shape: {df.shape}")

    # 8. Save processed data
    df.to_csv("./data/processed_data.csv", index=False)
    df.to_parquet("./data/processed_data.parquet", index=False)


def main():
    """
    Entrypoint for CLI execution via:

        uv run -m code.preprocessing.preprocess_data

    Runs preprocessing using default file paths.
    """
    preprocess_data()


if __name__ == "__main__":
    main()
