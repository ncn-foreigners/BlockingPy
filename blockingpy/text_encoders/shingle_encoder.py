"""Module containing the NgramEncoder class."""
import re

import pandas as pd
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer

from .base import TextEncoder


class NgramEncoder(TextEncoder):

    """
    Encoder that converts a pandas Series of text into a sparse
    document-term matrix of character n-gram (shingle) counts.
    """

    def __init__(
        self,
        n_shingles: int = 2,
        lowercase: bool = True,
        strip_non_alphanum: bool = True,
        max_features: int = 5000,
    ) -> None:
        """
        Initialize the n-gram encoder.

        Parameters
        ----------
        n_shingles : int, optional
            Number of characters per shingle (default is 2).
        lowercase : bool, optional
            If True, convert text to lowercase before shingling.
        strip_non_alphanum : bool, optional
            If True, remove non-alphanumeric characters.
        max_features : int, optional
            Maximum number of unique shingles to include (default is 5000).

        """
        self.n_shingles = n_shingles
        self.lowercase = lowercase
        self.strip_non_alphanum = strip_non_alphanum
        self.max_features = max_features

    def fit(self, X: pd.Series, y: pd.Series | None = None) -> "NgramEncoder":
        """
        Placeholder for fit method. This encoder does not learn any state
        from the data, but this method is included for API consistency.
        """
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        """
        Transform input texts into a sparse DataFrame of shingle counts.

        Parameters
        ----------
        X : pandas.Series
            Series of text strings to transform.

        Returns
        -------
        pandas.DataFrame
            Sparse DataFrame where columns are shingles and values
            are counts within each string.

        """
        dtm = self._create_sparse_dtm(X)
        return dtm

    def _create_sparse_dtm(self, x: pd.Series) -> pd.DataFrame:
        """
        Build a sparse document-term matrix using CountVectorizer.

        Parameters
        ----------
        x : pandas.Series
            Series of text strings to vectorize.

        Returns
        -------
        pandas.DataFrame
            Sparse DataFrame of shingle counts.

        """
        x = x.tolist() if isinstance(x, pd.Series) else x

        vectorizer = CountVectorizer(
            tokenizer=lambda x: self._tokenize_character_shingles(x),
            max_features=self.max_features,
            token_pattern=None,
        )
        x_dtm_sparse = vectorizer.fit_transform(x)

        x_sparse_df = pd.DataFrame.sparse.from_spmatrix(
            x_dtm_sparse, columns=vectorizer.get_feature_names_out()
        )

        return x_sparse_df

    def _tokenize_character_shingles(self, text: str) -> list[str]:
        """
        Tokenize a single string into a list of character shingles.

        Parameters
        ----------
        text : str
            Input string to tokenize.

        Returns
        -------
        list[str]
            A list of character n-grams.

        """
        if self.lowercase:
            text = text.lower()
        if self.strip_non_alphanum:
            if self.lowercase:
                text = re.sub(r"[^a-z0-9]", "", text)
            else:
                text = re.sub(r"[^A-Za-z0-9]", "", text)
        shingles = ["".join(gram) for gram in ngrams(text, self.n_shingles)]
        return shingles
