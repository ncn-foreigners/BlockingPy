"""
Contains the VoyagerBlocker class for performing blocking using the
Voyager algorithm from Spotify.
"""

import logging
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from voyager import Index, Space

from .base import BlockingMethod
from .data_handler import DataHandler
from .helper_functions import rearrange_array

logger = logging.getLogger(__name__)


class VoyagerBlocker(BlockingMethod):
    """
    A class for performing blocking using the Voyager algorithm from Spotify.

    This class implements blocking functionality using Spotify's Voyager algorithm
    for efficient approximate nearest neighbor search. It supports multiple distance
    metrics and is designed for high-dimensional data.

    Parameters
    ----------
    None

    Attributes
    ----------
    index : voyager.Index or None
        The Voyager index used for nearest neighbor search
    x_columns : array-like or None
        Column names of the reference dataset
    METRIC_MAP : dict
        Mapping of distance metric names to Voyager Space types

    See Also
    --------
    BlockingMethod : Abstract base class defining the blocking interface
    voyager.Index : The underlying Voyager index implementation

    Raises
    ------
    ValueError
        If path is provided but incorrect

    Notes
    -----
    For more details about the Voyager algorithm and implementation, see:
    https://github.com/spotify/voyager

    """

    METRIC_MAP: dict[str, Space] = {
        "euclidean": Space.Euclidean,
        "cosine": Space.Cosine,
        "inner_product": Space.InnerProduct,
    }

    def __init__(self) -> None:
        """
        Initialize the VoyagerBlocker instance.

        Creates a new VoyagerBlocker with empty index.
        """
        self.index: Index
        self.x_columns: list[str]

    def block(
        self,
        x: DataHandler,
        y: DataHandler,
        k: int,
        verbose: bool | None,
        controls: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Perform blocking using the Voyager algorithm.

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
                'voyager': {
                    'distance': str,
                    'k_search': int,
                    'path': str,
                    'M': int,
                    'ef_construction': int,
                    'max_elements': int,
                    'num_threads': int,
                    'query_ef': int
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
        The algorithm uses a graph-based approach for approximate
        nearest neighbor search. The quality of approximation can be controlled
        through parameters like ef_construction and query_ef.

        """
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self.x_columns = list(x.cols)

        X = x.to_dense()
        Y = y.to_dense()

        distance = controls["voyager"].get("distance")
        space = self.METRIC_MAP[distance]
        k_search = controls["voyager"].get("k_search")
        path = controls["voyager"].get("path")
        seed = controls.get("random_seed")
        if seed is None:
            seed = 1

        if X.shape[0] == 0:
            raise ValueError("Reference dataset `x` must not be empty.")
        if Y.shape[0] == 0:
            raise ValueError("Query dataset `y` must not be empty.")

        self.index = Index(
            space=space,
            num_dimensions=X.shape[1],
            M=controls["voyager"].get("M"),
            ef_construction=controls["voyager"].get("ef_construction"),
            random_seed=seed,
            max_elements=controls["voyager"].get("max_elements"),
        )

        logger.info("Building index...")

        self.index.add_items(
            X,
            num_threads=controls["voyager"].get("num_threads"),
        )

        logger.info("Querying index...")

        l_ind_nns = np.zeros(Y.shape[0], dtype=int)
        l_ind_dist = np.zeros(Y.shape[0])

        if k_search > X.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, X.shape[0])
            warnings.warn(
                f"k_search ({original_k_search}) is larger than the number of reference points "
                f"({X.shape[0]}). Adjusted k_search to {k_search}.",
                category=UserWarning,
                stacklevel=2,
            )

        all_neighbor_ids, all_distances = self.index.query(
            vectors=Y,
            k=k_search,
            num_threads=controls["voyager"].get("num_threads"),
            query_ef=controls["voyager"].get("query_ef"),
        )
        K_VAL = 2
        if k == K_VAL:
            all_neighbor_ids, all_distances = rearrange_array(all_neighbor_ids, all_distances)

        l_ind_nns = all_neighbor_ids[:, k - 1]
        l_ind_dist = all_distances[:, k - 1]

        if path:
            self._save_index(path)

        result = pd.DataFrame(
            {
                "y": np.arange(Y.shape[0]),
                "x": l_ind_nns,
                "dist": l_ind_dist,
            }
        )

        logger.info("Process completed successfully.")

        return result

    def _save_index(self, path: str) -> None:
        """
        Save the Voyager index and column names to files.

        Parameters
        ----------
        path : str
            Directory path where the files will be saved

        Raises
        ------
        ValueError
            If the provided path is incorrect

        Notes
        -----
        Creates two files:
            - 'index.voyager': The Voyager index file
            - 'index-colnames.txt': A text file with column names

        """
        if not os.path.exists(os.path.dirname(path)):
            raise ValueError("Provided path is incorrect")

        path_voy = os.path.join(path, "index.voyager")
        path_voy_cols = os.path.join(path, "index-colnames.txt")

        logger.info(f"Writing an index to {path_voy}")

        self.index.save(path_voy)

        with open(path_voy_cols, "w", encoding="utf-8") as f:
            f.write("\n".join(self.x_columns))
