"""
Utilities for preparing feature matrices and targets for model training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(
    df: pd.DataFrame,
    num_features: list,
    cat_features: list,
    text_blurb,
    text_name,
    rand,
):
    """
    Prepare the dataset for model training and evaluation.

    This function:
    - creates a binary target column ``state_binary`` where
      successful projects are labeled as 1 and others as 0
    - extracts selected numeric, categorical, and text features
    - derives month features from date columns
    - splits the data into training and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the required feature columns.
    num_features : list
        List of numeric feature column names.
    cat_features : list
        List of categorical feature column names.
    text_blurb : str
        Name of the text column containing project blurbs.
    text_name : str
        Name of the text column containing project names.
    rand : int or None
        Random seed for train/test split.

    Returns
    -------
    X_train : pandas.DataFrame
        Training features.
    X_test : pandas.DataFrame
        Test features.
    y_train : pandas.Series
        Training target values.
    y_test : pandas.Series
        Test target values.
    """

    df = df.copy()

    df["state_binary"] = (df["state"] == "successful").astype(int)

    date_features = ["launched_date", "deadline"]
    for col in date_features:
        df[col] = pd.to_datetime(df[col])
        df[col + "_month"] = df[col].dt.month

    X = df[num_features + cat_features + [text_blurb, text_name]]
    y = df["state_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=rand
    )

    return X_train, X_test, y_train, y_test