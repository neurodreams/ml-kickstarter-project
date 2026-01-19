"""
Utilities for computing daily FX rates and converting project goals to USD.
"""

import pandas as pd
from datetime import timedelta

def get_fx_date(launched_date):
    """Return the date to use for FX rate lookup.

    Mon–Fri: use the same day.
    Sat:     use Friday (one day earlier).
    Sun:     use Friday (two days earlier).
    """
    date = pd.to_datetime(launched_date)

    weekday = date.weekday() # 0=Mon ... 4=Fri, 5=Sat, 6=Sun

    if weekday == 5: # Saturday
        date = date - timedelta(days=1)
    elif weekday == 6: # Sunday
        date = date - timedelta(days=2)

    return date.normalize() 

def add_fx_mean(df: pd.DataFrame, fx_folder: str = "data/fx_rates") -> pd.DataFrame:
    """Add fx_daily_mean for each row using daily FX CSVs and weekend logic.

    - launched_at is assumed to be epoch seconds.
    - For non-USD rows, FX rate is looked up from <currency>-USD_daily.csv
      using an FX date where Sat/Sun are mapped back to Friday.
    - For USD rows, fx_daily_mean is set to 1.0.

    Parameters
    ----------
    df : pd.DataFrame
        Project-level dataset with at least columns: 'currency', 'launched_at'.
    fx_folder : str, optional
        Folder path where the FX CSV files are stored.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with new columns:
        - launched_date
        - fx_date
        - fx_daily_mean
    """
    
    df = df.copy()
    df["launched_date"] = pd.to_datetime(df["launched_at"], unit="s").dt.floor("D")
    
    # Create fx_date: map weekend dates back to Friday
    df["fx_date"] = df["launched_date"].apply(get_fx_date)
    
    # Separate currencies to USD & others
    df_usd = df[df["currency"] == "USD"].copy()
    df_non_usd = df[df["currency"] != "USD"].copy()

    # USD - USD uses static 1.0 fx-rate
    df_usd["fx_daily_mean"] = 1.0
    
    # Get list of non-USD currencies
    currency_list = set(df_non_usd["currency"])

    merged_parts = []

    for currency in currency_list:
        df_sub = df_non_usd[df_non_usd["currency"] == currency].copy()

        # Get fx-rates from csv
        fx_path = f"{fx_folder}/{currency}-USD_daily.csv"
        fx = pd.read_csv(fx_path)

        fx["fx_date"] = pd.to_datetime(fx["timestamp"]).dt.floor("D")

        df_sub = df_sub.merge(
            fx[["fx_date", "fx_daily_mean"]],
            on = "fx_date",
            how = "left",
        )
        
        merged_parts.append(df_sub)

    df_final = pd.concat([df_usd] + merged_parts, ignore_index=True)

    return df_final 
    

