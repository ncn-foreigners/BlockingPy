"""
Contains helper functions for blocking operations such as input validation, metrics validation,
algorithm validation and Document Term Matrix (DTM) creation.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


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

    NO_METRIC_ALGORITHMS = {"lsh", "kd", "nnd", "gpu_faiss"}  # too many options for nnd to validate

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

    @staticmethod
    def validate_controls_txt(control_txt: dict[str, Any] | None) -> None:
        """
        Validate a user-supplied `control_txt` dict before it is merged with
        package defaults.
        """
        if control_txt is None:
            return

        if not isinstance(control_txt, dict):
            raise ValueError("`control_txt` must be a dictionary.")

        encoder = control_txt.get("encoder", "shingle")
        allowed_encoders = {"shingle", "embedding"}
        if encoder not in allowed_encoders:
            raise ValueError(
                f"Unknown encoder '{encoder}'. " f"Supported encoders: {sorted(allowed_encoders)}"
            )

        allowed_top = {"encoder", "shingle", "embedding"}
        unknown_top = set(control_txt) - allowed_top
        if unknown_top:
            raise ValueError(f"Unknown keys at top level of `control_txt`: {unknown_top}")

        known_specs = {
            "shingle": {"n_shingles", "lowercase", "strip_non_alphanum", "max_features"},
            "embedding": {
                "model",
                "normalize",
                "max_length",
                "emb_batch_size",
                "show_progress_bar",
                "use_multiprocessing",
                "multiprocessing_threshold",
            },
        }
        for section, allowed_keys in known_specs.items():
            if section in control_txt:
                unknown = set(control_txt[section]) - allowed_keys
                if unknown:
                    raise ValueError(f"Unknown keys in control_txt['{section}']: {unknown}")


def rearrange_array(indices: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rearrange the array of indices to match the correct order.
    If the algorithm returns the record "itself" for a given row (in deduplication), but not
    as the first nearest neighbor, rearrange the array to fix this issue.
    If the algorithm does not return the record "itself" for a given row (in deduplication),
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
