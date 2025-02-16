"""
Contains helper functions for blocking operations such as input validation, metrics validation,
algorithm validation and Document Term Matrix (DTM) creation.
"""

import re

import numpy as np
import pandas as pd
from nltk.util import ngrams
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class DistanceMetricValidator:

    """Centralized validation for distance metrics across different algorithms."""

    SUPPORTED_METRICS: dict[str, set[str]] = {
        "hnsw": {
            "l2",
            "euclidean",
            "cosine",
            "ip",
        },
        "annoy": {"euclidean", "manhattan", "hamming", "angular", "dot"},
        "voyager": {"euclidean", "cosine", "inner_product"},
        "faiss": {
            "euclidean",
            "l2",
            "inner_product",
            "cosine",
            "l1",
            "manhattan",
            "linf",
            "canberra",
            "bray_curtis",
            "jensen_shannon",
        },
    }

    NO_METRIC_ALGORITHMS = {"lsh", "kd", "nnd"}  # too many options for nnd to validate

    @classmethod
    def validate_metric(cls, algorithm: str, metric: str) -> None:
        """
        Validate if a metric is supported for given algorithm.

        Parameters
        ----------
        algorithm : str
            Name of the algorithm
        metric : str
            Name of the distance metric

        Raises
        ------
        ValueError
            If metric is not supported for the algorithm

        Notes
        -----
        Supported distance metrics per algorithm:
        - HNSW: l2, euclidean, cosine, ip
        - Annoy: euclidean, manhattan, hamming, angular, dot
        - Voyager: euclidean, cosine, inner_product
        - FAISS: euclidean, l2, inner_product, cosine, l1, manhattan, linf,
                canberra, bray_curtis, jensen_shannon
        - NND: look: https://pynndescent.readthedocs.io/en/latest/api.html

        """
        if algorithm in cls.NO_METRIC_ALGORITHMS:
            return

        if algorithm not in cls.SUPPORTED_METRICS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        if metric not in cls.SUPPORTED_METRICS[algorithm]:
            valid_metrics = ", ".join(sorted(cls.SUPPORTED_METRICS[algorithm]))
            raise ValueError(
                f"Distance metric '{metric}' not supported for {algorithm}. "
                f"Supported metrics are: {valid_metrics}"
            )

    @classmethod
    def get_supported_metrics(cls, algorithm: str) -> set[str]:
        """
        Get set of supported metrics for an algorithm.

        Parameters
        ----------
        algorithm : str
            The name of the algorithm to get supported metrics for.

        Returns
        -------
        set[str]
            A set containing the names of all metrics supported by the specified algorithm.
            Returns an empty set if the algorithm is not recognized.

        """
        return cls.SUPPORTED_METRICS.get(algorithm, set())


class InputValidator:

    """Validates input data and parameters for blocking operations."""

    @staticmethod
    def validate_data(x: pd.Series | sparse.csr_matrix | np.ndarray) -> None:
        """
        Validate input data type.

        Parameters
        ----------
        x : Union[pd.Series, sparse.csr_matrix, np.ndarray]
            Input data to validate

        Raises
        ------
        ValueError
            If input type is not supported

        """
        if not (isinstance(x, np.ndarray) or sparse.issparse(x) or isinstance(x, pd.Series)):
            raise ValueError(
                "Only dense (np.ndarray) or sparse (csr_matrix) matrix "
                "or pd.Series with str data is supported"
            )

    @staticmethod
    def validate_true_blocks(true_blocks: pd.DataFrame | None, deduplication: bool) -> None:
        """
        Validate true blocks data structure.

        Parameters
        ----------
        true_blocks : Optional[pd.DataFrame]
            True blocks information for evaluation
        deduplication : bool
            Whether deduplication is being performed

        Raises
        ------
        ValueError
            If true_blocks structure is invalid

        """
        COLUMN_LEN_RECLIN_TRUE_BLOCKS = 3
        COLUMN_LEN_DEDUP_TRUE_BLOCKS = 2
        if true_blocks is not None:
            if not deduplication:
                if len(true_blocks.columns) != COLUMN_LEN_RECLIN_TRUE_BLOCKS or not all(
                    col in true_blocks.columns for col in ["x", "y", "block"]
                ):
                    raise ValueError(
                        "`true blocks` should be a DataFrame with columns: " "x, y, block"
                    )
            elif len(true_blocks.columns) != COLUMN_LEN_DEDUP_TRUE_BLOCKS or not all(
                col in true_blocks.columns for col in ["x", "block"]
            ):
                raise ValueError("`true blocks` should be a DataFrame with columns: " "x, block")


def tokenize_character_shingles(
    text: str, n: int = 2, lowercase: bool = True, strip_non_alphanum: bool = True
) -> list[str]:
    """
    Generate character n-grams (shingles) from input text.

    Parameters
    ----------
    text : str
        Input text to tokenize
    n : int, optional
        Size of character n-grams (default 2)
    lowercase : bool, optional
        Whether to convert text to lowercase (default True)
    strip_non_alphanum : bool, optional
        Whether to remove non-alphanumeric characters (default True)

    Returns
    -------
    list of str
        List of character n-grams

    Examples
    --------
    >>> tokenize_character_shingles("Hello", n=2)
    ['he', 'el', 'll', 'lo']

    Notes
    -----
    The function processes text in the following order:
    1. Converts to lowercase (if requested)
    2. Removes non-alphanumeric characters (if requested)
    3. Generates n-character shingles

    """
    if lowercase:
        text = text.lower()
    if strip_non_alphanum:
        text = re.sub(r"[^a-z0-9]+", "", text)
    shingles = ["".join(gram) for gram in ngrams(text, n)]
    return shingles


def create_sparse_dtm(x: pd.Series, control_txt: dict, verbose: bool = False) -> pd.DataFrame:
    """
    Create a sparse document-term matrix from input texts.

    Parameters
    ----------
    x : pandas.Series
        Input texts to process
    control_txt : dict
        Configuration dictionary with keys:
        - n_shingles : int
            Size of character n-grams
        - lowercase : bool
            Whether to convert text to lowercase
        - strip_non_alphanum : bool
            Whether to remove non-alphanumeric characters
        - max_features : int
            Maximum number of features to keep
    verbose : bool, optional
        Whether to print additional information (default False)

    Returns
    -------
    pandas.DataFrame
        Sparse dataframe containing the document-term matrix

    Notes
    -----
    The function uses CountVectorizer from scikit-learn with custom
    tokenization based on character n-grams. The resulting matrix
    is stored as a sparse pandas DataFrame.

    Examples
    --------
    >>> texts = pd.Series(['hello world', 'hello there'])
    >>> controls = {'n_shingles': 2, 'lowercase': True,
    ...            'strip_non_alphanum': True, 'max_features': 100}
    >>> dtm = create_sparse_dtm(texts, controls)

    """
    x = x.tolist() if isinstance(x, pd.Series) else x

    vectorizer = CountVectorizer(
        tokenizer=lambda x: tokenize_character_shingles(
            x,
            n=control_txt.get("n_shingles", 2),
            lowercase=control_txt.get("lowercase", True),
            strip_non_alphanum=control_txt.get("strip_non_alphanum", True),
        ),
        max_features=control_txt.get("max_features", 5000),
        token_pattern=None,
    )
    x_dtm_sparse = vectorizer.fit_transform(x)
    x_voc = vectorizer.vocabulary_

    x_sparse_df = pd.DataFrame.sparse.from_spmatrix(
        x_dtm_sparse, columns=vectorizer.get_feature_names_out()
    )

    if verbose:
        print("Vocabulary:", x_voc)
        print("Sparse DataFrame shape:", x_sparse_df.shape)
        print("Sparse DataFrame:\n", x_sparse_df)

    return x_sparse_df


def rearrange_array(indices: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rearrange the array of indices to match the correct order.
    If the algoritm returns the record "itself" for a given row (in deduplication), but not
    as the first nearest neighbor, rearrange the array to fix this issue.
    If the algoritm does not return the record "itself" for a given row (in deduplication),
    insert a dummy value (-1) at the start and shift other indices and distances values.

    Parameters
    ----------
    indices : array-like
        indices returned by the algorithm
    distances : array-like
        distances returned by the algorithm

    Notes
    -----
    This method is necessary because if two records are exactly the same,
    the algorithm will not return itself as the first nearest neighbor in
    deduplication. This method rearranges the array to fix this issue.
    Due to the fact that it is an "approximate" algorithm, it may not return
    the record itself at all.

    """
    n_rows = indices.shape[0]
    result = indices.copy()
    result_dist = distances.copy()

    for i in range(n_rows):
        if result[i][0] != i:
            matches = np.where(result[i] == i)[0]

            if len(matches) == 0:
                result[i][1:] = result[i][:-1]
                result[i][0] = -1
                result_dist[i][1:] = result_dist[i][:-1]
                result_dist[i][0] = -1
            else:
                position = matches[0]
                value_to_move = result[i][position]
                result[i][1 : position + 1] = result[i][0:position]
                result[i][0] = value_to_move

    return result, result_dist
