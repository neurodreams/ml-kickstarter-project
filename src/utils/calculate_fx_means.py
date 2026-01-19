"""Utils for computing daily FX mean rates from OHLC CSV files."""

import pandas as pd
from pathlib import Path


def calculate_fx_means(data_dir: str | Path) -> None:
    """
    Compute and store daily mean FX rates for CSV files in a directory.

    The function reads all CSV files in the given directory, calculates a
    daily mean FX rate for each row based on OHLC columns, and writes the
    result to new CSV files containing an added 'fx_daily_mean' column.

    The daily mean is calculated as:
        (open + high + low + close) / 4

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Path to the directory that contains the FX CSV files.
    """


    data_dir = Path(data_dir)

    for csv_path in data_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)
        df["fx_daily_mean"] = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4).round(5)
        df.to_csv(f"data/fx_rates/{csv_path.name}")


if __name__ == "__main__":
    data_dir = Path("data/fx_rates/")
    calculate_fx_means(data_dir)