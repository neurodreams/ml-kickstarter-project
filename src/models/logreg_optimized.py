"""
Train, evaluate, and persist the final Logistic Regression model.

This module defines the optimized Logistic Regression pipeline
used as the final model in the project.
"""

import pandas as pd
from pathlib import Path
import joblib

from utils.data_preparation import prepare_data
from models.evaluate_models import build_preprocessor

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    f1_score,
)


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "feature_processed_data.csv"
MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "objects" / "logreg_final.joblib"
)


NUMERIC_FEATURES = [
    "usd_goal_fx_log",
    "creator_prev_projects_successful",
    "creator_prev_projects",
    "project_duration_days",
    "blurb_len",
    "name_len",
]

CATEGORICAL_FEATURES = [
    "category_name_reduced",
    "category_parent_name",
    "country",
    "currency",
    "launched_date_month",
    "deadline_month",
    "blurb_missing",
    "blurb_is_english",
    "name_missing",
    "name_is_english",
]

TEXT_BLURB = "blurb"
TEXT_NAME = "name"


def load_training_data_and_preprocessor(
    data_path: Path = DATA_PATH,
):
    """
    Load feature data, split it into train and test sets, and build
    the preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """

    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = prepare_data(
        df,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        TEXT_BLURB,
        TEXT_NAME,
        rand=42,
    )

    preprocessor = build_preprocessor(
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        TEXT_BLURB,
        TEXT_NAME,
    )

    return X_train, X_test, y_train, y_test, preprocessor


def build_model(preprocessor: Pipeline) -> Pipeline:
    """
    Build the final Logistic Regression pipeline using fixed hyperparameters.
    """

    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        fit_intercept=True,
        max_iter=200,
        penalty="l2",
        solver="saga",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )


def save_model(model: Pipeline, model_path: Path) -> None:
    """
    Save a fitted model pipeline to disk.
    """

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def fit_and_save_model(
    model_path: Path = MODEL_PATH,
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """
    Fit the final model on the training data and save it to disk.

    Returns:
        model, X_test, y_test
    """

    X_train, X_test, y_train, y_test, preprocessor = (
        load_training_data_and_preprocessor()
    )

    model = build_model(preprocessor)
    model.fit(X_train, y_train)

    save_model(model, model_path)

    return model, X_test, y_test


def evaluate_model(model: Pipeline, X_test, y_test) -> None:
    """
    Print key evaluation metrics and a classification report for the model.
    """

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_scores = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_scores)

    print("=== Testisetin metriikat lopulliselle Logistic Regression -mallille ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    model, X_test, y_test = fit_and_save_model()
    print(f"Malli tallennettu polkuun: {MODEL_PATH}")
    evaluate_model(model, X_test, y_test)
