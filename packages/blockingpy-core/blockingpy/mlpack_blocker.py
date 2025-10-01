"""Contains the MLPackBlocker class for performing blocking using MLPack algorithms."""

import logging
import warnings
from typing import Any

import mlpack
import pandas as pd

from .base import BlockingMethod
from .data_handler import DataHandler
from .helper_functions import rearrange_array

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
        x: DataHandler,
        y: DataHandler,
        k: int,
        verbose: bool | None,
        controls: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Perform blocking using MLPack algorithm (LSH or k-d tree).

        Parameters
        ----------
        x : DataHandler
            Reference dataset containing features for indexing
        y : DataHandler
            Query dataset to find nearest neighbors for
        k : int
            Number of nearest neighbors to find
        verbose : bool, optional
            If True, print detailed progress information
        controls : dict
            Algorithm control parameters with the following structure:
            {
                'random_seed': int,
                'algo': str # 'lsh' or 'kd',
                'lsh': {  # if using LSH
                    'k_search': int,
                    'bucket_size': int,
                    'hash_width': float,
                    'num_probes': int,
                    'projections': int,
                    'tables': int
                },
                'kd': {   # if using k-d tree
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

        self.x_columns = list(x.cols)
        X = x.to_dense()
        Y = y.to_dense()

        self.algo = controls.get("algo", "lsh")
        self._check_algo(self.algo)
        seed = controls.get("random_seed")
        if self.algo == "lsh":
            k_search = controls["lsh"].get("k_search")
        else:
            k_search = controls["kd"].get("k_search")

        if k_search > X.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, X.shape[0])
            warnings.warn(
                f"k_search ({original_k_search}) is larger than the number of reference points "
                f"({X.shape[0]}). Adjusted k_search to {k_search}.",
                category=UserWarning,
                stacklevel=2,
            )

        logger.info(f"Initializing MLPack {self.algo.upper()} index...")

        if self.algo == "lsh":
            query_result = mlpack.lsh(
                k=k_search,
                query=Y,
                reference=X,
                verbose=verbose,
                seed=seed,
                bucket_size=controls["lsh"].get("bucket_size"),
                hash_width=controls["lsh"].get("hash_width"),
                num_probes=controls["lsh"].get("num_probes"),
                projections=controls["lsh"].get("projections"),
                tables=controls["lsh"].get("tables"),
            )
        else:
            query_result = mlpack.knn(
                k=k_search,
                query=Y,
                reference=X,
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

        K_VAL = 2
        if k == K_VAL:
            indices, distances = rearrange_array(indices, distances)

        result = pd.DataFrame(
            {
                "y": range(Y.shape[0]),
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
