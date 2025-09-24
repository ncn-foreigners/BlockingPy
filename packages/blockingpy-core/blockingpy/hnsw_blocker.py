"""Contains the HNSWBlocker class for performing blocking using the HNSW algorithm."""

import logging
import os
import warnings
from typing import Any

import hnswlib
import pandas as pd

from .base import BlockingMethod
from .data_handler import DataHandler
from .helper_functions import rearrange_array

logger = logging.getLogger(__name__)


class HNSWBlocker(BlockingMethod):
    """
    A class for performing blocking using the Hierarchical Navigable Small World (HNSW) algorithm.

    This class implements blocking functionality using the HNSW algorithm for efficient
    similarity search and nearest neighbor queries.

    Parameters
    ----------
    None

    Attributes
    ----------
    index : hnswlib.Index or None
        The HNSW index used for nearest neighbor search
    x_columns : array-like or None
        Column names of the reference dataset
    SPACE_MAP : dict
        Mapping of distance metric names to their HNSW implementations

    See Also
    --------
    BlockingMethod : Abstract base class defining the blocking interface

    Notes
    -----
    For more details about the HNSW algorithm, see:
    https://github.com/nmslib/hnswlib

    """

    SPACE_MAP: dict[str, str] = {
        "l2": "l2",
        "euclidean": "l2",
        "cosine": "cosine",
        "ip": "ip",
    }

    def __init__(self) -> None:
        """
        Initialize the HNSWBlocker instance.

        Creates a new HNSWBlocker with empty index.
        """
        self.index: hnswlib.Index
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
        Perform blocking using the HNSW algorithm.

        Parameters
        ----------
        x : pandas.DataFrame
            Reference dataset containing features for indexing
        y : pandas.DataFrame
            Query dataset to find nearest neighbors for
        k : int
            Number of nearest neighbors to find. If k is larger than the number
            of reference points, it will be automatically adjusted
        verbose : bool, optional
            If True, print detailed progress information
        controls : dict
            Algorithm control parameters with the following structure:
            {
                'random_seed': int,
                'hnsw': {
                    'k_search': int,
                    'distance': str,
                    'n_threads': int,
                    'path': str,
                    'ef_c': int,
                    'ef_s': int,
                    'M': int,
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
        The function builds an HNSW index from the reference dataset and finds
        the k-nearest neighbors for each point in the query dataset. The index
        parameters ef_c (construction) and ef_s (search) control the trade-off
        between search accuracy and speed.

        """
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self.x_columns = list(x.cols)

        distance = controls["hnsw"].get("distance")
        n_threads = controls["hnsw"].get("n_threads")
        path = controls["hnsw"].get("path")
        k_search = controls["hnsw"].get("k_search")
        space = self.SPACE_MAP[distance]
        seed = controls.get("random_seed")
        if seed is None:
            seed = 100

        logger.info("Initializing HNSW index...")

        X = x.to_dense()
        Y = y.to_dense()

        self.index = hnswlib.Index(space=space, dim=X.shape[1])
        self.index.init_index(
            max_elements=X.shape[0],
            ef_construction=controls["hnsw"].get("ef_c"),
            M=controls["hnsw"].get("M"),
            random_seed=seed,
        )
        self.index.set_num_threads(n_threads)

        logger.info("Adding items to index...")

        self.index.add_items(X)
        self.index.set_ef(controls["hnsw"].get("ef_s"))

        logger.info("Querying index...")

        if k_search > X.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, X.shape[0])
            warnings.warn(
                f"k_search ({original_k_search}) is larger than the number of reference points "
                f"({X.shape[0]}). Adjusted k_search to {k_search}.",
                category=UserWarning,
                stacklevel=2,
            )

        l_1nn = self.index.knn_query(Y, k=k_search, num_threads=n_threads)
        indices = l_1nn[0]
        distances = l_1nn[1]

        K_VAL = 2
        if k == K_VAL:
            indices, distances = rearrange_array(indices, distances)

        if path:
            self._save_index(path)

        result = pd.DataFrame(
            {
                "y": range(y.shape[0]),
                "x": indices[:, k - 1],
                "dist": distances[:, k - 1],
            }
        )

        logger.info("Process completed successfully.")

        return result

    def _save_index(self, path: str) -> None:
        """
        Save the HNSW index and column names to files.

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
            - 'index.hnsw': The HNSW index file
            - 'index-colnames.txt': A text file with column names

        """
        if not os.path.exists(os.path.dirname(path)):
            raise ValueError("Provided path is incorrect")

        path_ann = os.path.join(path, "index.hnsw")
        path_ann_cols = os.path.join(path, "index-colnames.txt")

        logger.info(f"Writing an index to {path_ann}")

        self.index.save_index(path_ann)
        with open(path_ann_cols, "w", encoding="utf-8") as f:
            f.write("\n".join(self.x_columns))
