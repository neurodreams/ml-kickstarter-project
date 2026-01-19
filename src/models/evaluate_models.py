"""
Utilities for training and comparing multiple classification models.
"""

import pandas as pd
import time
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier,
    PassiveAggressiveClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.compose import ColumnTransformer

from utils.data_preparation import prepare_data


def build_preprocessor(numeric_features, categorical_features, text_blurb, text_name):
    """
    Build a preprocessing pipeline for numeric, categorical, and text features.
    """

    LEAKAGE_STOPWORDS = {"canceled", "cancelled", "stretch", "unlocked"}
    STOPWORDS = list(ENGLISH_STOP_WORDS.union(LEAKAGE_STOPWORDS))

    numerical_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    text_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    stop_words=STOPWORDS,
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
            ("blurb", text_pipeline, text_blurb),
            ("name", text_pipeline, text_name),
        ],
        remainder="drop",
    )
    return preprocessor


def get_model_dict(rand):
    """
    Return a dictionary of candidate models for evaluation.
    """

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=rand,
            max_features="sqrt",
        ),
        "logreg_lbfgs": LogisticRegression(
            C=1.0, n_jobs=None, random_state=rand, solver="lbfgs", max_iter=2000
        ),
        "logreg_saga": LogisticRegression(
            random_state=rand,
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=2000,
            n_jobs=-1,
        ),
        "linear_svc": LinearSVC(
            random_state=rand, C=1.0, max_iter=5000, class_weight=None
        ),
        "sgd_hinge": SGDClassifier(
            random_state=rand,
            loss="hinge",
            alpha=1e-4,
            penalty="l2",
            max_iter=2000,
            tol=1e-3,
        ),
        "sgd_logloss": SGDClassifier(
            random_state=rand,
            loss="log_loss",
            alpha=1e-4,
            penalty="l2",
            max_iter=2000,
            tol=1e-3,
        ),
        "ridge_clf": RidgeClassifier(random_state=rand, alpha=1.0),
        "passive_aggressive": PassiveAggressiveClassifier(
            random_state=rand, C=0.5, loss="hinge", max_iter=2000, tol=1e-3
        ),
    }


def evaluate_single_model(
    name, estimator, preprocessor, X_train, X_test, y_train, y_test
):
    """
    Fit a single model pipeline and compute evaluation metrics on the test set.
    """

    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    elapsed = time.perf_counter() - start

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1 score": f1_score(y_test, y_pred),
        "elapsed_seconds": elapsed,
    }


def save_results(results, path="./data/model_results.csv"):
    """
    Save model evaluation results to a CSV file.
    """

    results_df = pd.DataFrame(results)
    results_df.to_csv(path)
    print("Mallien suorituskykyraportti tallennettu polkuun", path)


def evaluate_models(
    df, numeric_features, categorical_features, text_blurb, text_name, rand=None
):
    """
    Train and evaluate multiple models and save a comparison report.
    """

    X_train, X_test, y_train, y_test = prepare_data(
        df, numeric_features, categorical_features, text_blurb, text_name, rand
    )

    preprocessor = build_preprocessor(
        numeric_features, categorical_features, text_blurb, text_name
    )

    models = get_model_dict(rand)
    results = []

    print("Mallien kilpailuraportti:")
    for name, estimator in models.items():
        result = evaluate_single_model(
            name, estimator, preprocessor, X_train, X_test, y_train, y_test
        )
        results.append(result)

        print(
            f"{name} valmis. Accuracy: {result['accuracy']:.4f}, "
            f"Precision: {result['precision']:.4f}, "
            f"Recall: {result['recall']:.4f}, "
            f"F1: {result['f1 score']:.4f}, "
            f"Time spent (s): {result['elapsed_seconds']:.4f}"
        )

    save_results(results)


if __name__ == "__main__":
    df = pd.read_csv("data/feature_processed_data.csv")

    numeric_features = [
        "usd_goal_fx_log",
        "creator_prev_projects_successful",
        "creator_prev_projects",
        "project_duration_days",
        "blurb_len",
        "name_len",
    ]

    categorical_features = [
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

    text_blurb = "blurb"
    text_name = "name"

    evaluate_models(
        df, numeric_features, categorical_features, text_blurb, text_name, rand=None
    )
