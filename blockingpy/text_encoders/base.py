"""base class for text-to-matrix transformers."""
from abc import ABC, abstractmethod

import pandas as pd


class TextEncoder(ABC):

    """Abstract base class for text-to-matrix transformers."""

    @abstractmethod
    def fit(self, X: pd.Series, y: pd.Series | None = None) -> "TextEncoder":
        """
        Learn any stateful parameters from the data (e.g., vocabulary).
        Default implementation is a no-op, returning self.

        Parameters
        ----------
        X : pandas.Series
            Series of text strings to learn from.
        y : ignored
            Present for API consistency.

        Returns
        -------
        self

        """
        return self

    def fit_transform(self, X: pd.Series, y: pd.Series | None = None) -> pd.DataFrame:
        """
        Equivalent to calling fit(X) then transform(X).

        Parameters
        ----------
        X : pandas.Series
            Series of text strings to fit and transform.
        y : ignored
            Present for API consistency.

        Returns
        -------
        pandas.DataFrame
            Transformed feature matrix.

        """
        return self.fit(X, y).transform(X)

    @abstractmethod
    def transform(self, X: pd.Series) -> pd.DataFrame:
        """
        Transform a pandas Series of strings into a document-term
        DataFrame (sparse or dense), where rows correspond to records
        and columns correspond to features (e.g., n-grams or embeddings).

        Parameters
        ----------
        X : pandas.Series
            Series of text strings to transform.

        Returns
        -------
        pandas.DataFrame
            Transformed feature matrix. May use a SparseDtype internally.

        """
        ...
