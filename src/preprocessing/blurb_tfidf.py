"""
Utilities for TF-IDF vectorization of the 'blurb' text field.

Provides helpers for creating a reusable sklearn pipeline and
for directly fitting and transforming blurb text into TF-IDF matrices.
"""

from typing import Optional, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix


def make_blurb_tfidf_pipeline(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 1),
    stop_words: Optional[str] = "english",
    min_df: int | float = 1,
    max_df: int | float = 1.0,
) -> Pipeline:

    """
    Creates an sklearn pipeline that vectorizes blurb texts using TF-IDF.

    The returned pipeline exposes .fit() and .transform() and accepts
    either a pandas Series or a list of strings.

    Parameters
    ----------
    max_features : int
        Maximum size of the vocabulary.
    ngram_range : tuple(int, int)
        The lower and upper boundaries for n-grams.
    stop_words : str or None
        Stop word setting, e.g. "english" or None.
    min_df : int or float
        Ignore terms that appear in fewer than this number (or proportion) of documents.
    max_df : int or float
        Ignore terms that appear in more than this proportion of documents.

    Returns
    -------
    An sklearn pipeline containing a single TF-IDF vectorizer step.
    """

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        min_df=min_df,
        max_df=max_df,
    )

    # Expandable pipeline
    pipe = Pipeline([
        ("tfidf", tfidf),
    ])
    return pipe


def fit_transform_blurbs(
    train_blurbs: pd.Series,
    test_blurbs: Optional[pd.Series] = None,
    **tfidf_kwargs,
) -> Tuple[csr_matrix, Optional[csr_matrix], TfidfVectorizer]:
    """
    Fit a TF-IDF vectorizer on training blurbs and transform training
    and (optionally) test blurbs.

    Useful when experiments only require the TF-IDF matrices and
    the fitted vectorizer.

    Parameters
    ----------
    train_blurbs : pd.Series
        Training text data.
    test_blurbs : pd.Series or None
        Optional test text data. If None, only X_train is returned.
    **tfidf_kwargs :
        Additional arguments passed to TfidfVectorizer.

    Returns
    -------
    X_train : csr_matrix
        TF-IDF matrix for training data.
    X_test : csr_matrix or None
        TF-IDF matrix for test data.
    vectorizer : TfidfVectorizer
        The fitted vectorizer.
    """

    vectorizer = TfidfVectorizer(**tfidf_kwargs)

    X_train = vectorizer.fit_transform(train_blurbs.fillna(""))
    X_test = vectorizer.transform(test_blurbs.fillna("")) if test_blurbs is not None else None

    return X_train, X_test, vectorizer
