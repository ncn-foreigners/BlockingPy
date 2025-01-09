"""Contains the MLPackBlocker class for performing blocking using MLPack algorithms."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from mlpack import knn, lsh

from .base import BlockingMethod

logger = logging.getLogger(__name__)


class MLPackBlocker(BlockingMethod):

    """
    A class for performing blocking using MLPack algorithms (LSH or k-d tree).

    This class implements blocking functionality using either Locality-Sensitive
    Hashing (LSH) or k-d tree algorithms from the MLPack library for efficient
    similarity search and nearest neighbor queries.

    Parameters
    ----------
    None

    Attributes
    ----------
    algo : str or None
        The selected algorithm ('lsh' or 'kd')
    ALGO_MAP : dict
        Mapping of algorithm names to their MLPack implementations

    See Also
    --------
    BlockingMethod : Abstract base class defining the blocking interface

    Notes
    -----
    For more details about the MLPack library and its algorithms, see:
    https://github.com/mlpack

    """

    def __init__(self) -> None:
        """
        Initialize the MLPackBlocker instance.

        Creates a new MLPackBlocker with no algorithm selected.
        """
        self.algo: str
        self.ALGO_MAP: dict[str, str] = {"lsh": "lsh", "kd": "knn"}

    def block(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        k: int,
        verbose: bool | None,
        controls: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Perform blocking using MLPack algorithm (LSH or k-d tree).

        Parameters
        ----------
        x : pandas.DataFrame
            Reference dataset containing features for indexing
        y : pandas.DataFrame
            Query dataset to find nearest neighbors for
        k : int
            Number of nearest neighbors to find
        verbose : bool, optional
            If True, print detailed progress information
        controls : dict
            Algorithm control parameters with the following structure:
            {
                'algo': str,
                'lsh': {  # if using LSH
                    'seed': int,
                    'k_search': int,
                    'bucket_size': int,
                    'hash_width': float,
                    'num_probes': int,
                    'projections': int,
                    'tables': int
                },
                'kd': {   # if using k-d tree
                    'seed': int,
                    'k_search': int,
                    'algorithm': str,
                    'leaf_size': int,
                    'tree_type': str,
                    'epsilon': float,
                    'rho': float,
                    'tau': float,
                    'random_basis': bool
                }
            }

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the blocking results with columns:
            - 'y': indices from query dataset
            - 'x': indices of matched items from reference dataset
            - 'dist': distances to matched items

        Notes
        -----
        The function supports two different algorithms:
        - LSH (Locality-Sensitive Hashing): Better for high-dimensional data
        - k-d tree: Better for low-dimensional data

        """
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self.algo = controls.get("algo", "lsh")
        self._check_algo(self.algo)
        if self.algo == "lsh":
            seed = controls["lsh"].get("seed")
            k_search = controls["lsh"].get("k_search")
        else:
            seed = controls["kd"].get("seed")
            k_search = controls["kd"].get("k_search")

        if k_search > x.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, x.shape[0])
            logger.warning(
                f"k_search ({original_k_search}) is larger than the number of reference points "
                f"({x.shape[0]}). Adjusted k_search to {k_search}."
            )

        logger.info(f"Initializing MLPack {self.algo.upper()} index...")

        if self.algo == "lsh":
            query_result = lsh(
                k=k_search,
                query=y,
                reference=x,
                verbose=verbose,
                seed=seed,
                bucket_size=controls["lsh"].get("bucket_size"),
                hash_width=controls["lsh"].get("hash_width"),
                num_probes=controls["lsh"].get("num_probes"),
                projections=controls["lsh"].get("projections"),
                tables=controls["lsh"].get("tables"),
            )
        else:
            query_result = knn(
                k=k_search,
                query=y,
                reference=x,
                verbose=verbose,
                seed=seed,
                algorithm=controls["kd"].get("algorithm"),
                leaf_size=controls["kd"].get("leaf_size"),
                tree_type=controls["kd"].get("tree_type"),
                epsilon=controls["kd"].get("epsilon"),
                rho=controls["kd"].get("rho"),
                tau=controls["kd"].get("tau"),
                random_basis=controls["kd"].get("random_basis"),
            )

        logger.info("MLPack index query completed.")

        indices = query_result["neighbors"]
        distances = query_result["distances"]

        if k == 2:
            indices, distances = self.rearrange_array(indices, distances)

        result = pd.DataFrame(
            {
                "y": range(y.shape[0]),
                "x": indices[:, k - 1],
                "dist": distances[:, k - 1],
            }
        )

        logger.info("Blocking process completed successfully.")

        return result

    def _check_algo(self, algo: str) -> None:
        """
        Validate the provided algorithm.

        Parameters
        ----------
        algo : str
            The algorithm to validate

        Raises
        ------
        ValueError
            If the provided algorithm is not in the ALGO_MAP

        Notes
        -----
        Valid algorithms are defined in the ALGO_MAP class attribute.
        Currently supports 'lsh' for Locality-Sensitive Hashing and
        'kd' for k-d tree based search.

        """
        if algo not in self.ALGO_MAP:
            valid_algos = ", ".join(self.ALGO_MAP.keys())
            raise ValueError(f"Invalid algorithm '{algo}'. Accepted values are: {valid_algos}.")

    def rearrange_array(self,
                        indices : np.ndarray,
                        distances : np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
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
