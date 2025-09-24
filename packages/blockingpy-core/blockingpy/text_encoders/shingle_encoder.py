"""Module containing the NgramEncoder class."""

from __future__ import annotations

import re

from nltk.util import ngrams
from pandas import Series
from sklearn.feature_extraction.text import CountVectorizer

from ..data_handler import DataHandler
from .base import TextEncoder


class NgramEncoder(TextEncoder):
    """
    Encoder that converts text strings into a sparse document-term matrix
    of character *n*-gram counts, packaged in a :class:`DataHandler`.
    """

    def __init__(
        self,
        n_shingles: int = 2,
        lowercase: bool = True,
        strip_non_alphanum: bool = True,
        max_features: int = 5000,
    ) -> None:
        """
        Create a character *n*-gram encoder.

        Parameters
        ----------
        n_shingles
            Number of characters per shingle.
        lowercase
            If *True*, convert text to lowercase before tokenisation.
        strip_non_alphanum
            If *True*, remove non-alphanumeric characters before shingling.
        max_features
            Maximum number of unique shingles to keep.

        """
        self.n_shingles = n_shingles
        self.lowercase = lowercase
        self.strip_non_alphanum = strip_non_alphanum
        self.max_features = max_features

    def fit(self, X: Series, y: Series | None = None) -> NgramEncoder:
        """Stateless encoder; fitting is a no-op, returned for API parity."""
        return self

    def transform(self, X: Series) -> DataHandler:
        """
        Transform a series of strings into a sparse shingle count matrix.

        Parameters
        ----------
        X
            Series of text strings.

        Returns
        -------
        DataHandler
            ``data``: ``csr_matrix`` of shape *(n_samples, n_features)*;
            ``cols``: list of shingle strings.

        """
        vectorizer = CountVectorizer(
            tokenizer=lambda t: self._tokenize(t),
            max_features=self.max_features,
            lowercase=False,
            token_pattern=None,
        )
        mat = vectorizer.fit_transform(X.tolist())
        return DataHandler(data=mat, cols=vectorizer.get_feature_names_out().tolist())

    def _tokenize(self, text: str) -> list[str]:
        """Split *text* into character shingles of length ``self.n_shingles``."""
        if self.lowercase:
            text = text.lower()
        if self.strip_non_alphanum:
            text = re.sub(r"[^a-z0-9]" if self.lowercase else r"[^A-Za-z0-9]", "", text)
        return ["".join(g) for g in ngrams(text, self.n_shingles)]
