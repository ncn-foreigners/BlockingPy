"""Contains the FaissBlocker class for performing blocking using one of the FAISS algorithms."""

import logging
import os
import warnings
from typing import Any

import faiss
import numpy as np
import pandas as pd

from .base import BlockingMethod
from .data_handler import DataHandler
from .helper_functions import rearrange_array

logger = logging.getLogger(__name__)


class FaissBlocker(BlockingMethod):
    """
    A class for performing blocking using the FAISS (Facebook AI Similarity Search) algorithm.

    This class implements blocking functionality using Facebook's FAISS library for
    efficient similarity search and nearest neighbor queries. It supports multiple
    distance metrics and is optimized for high-performance computing.

    Parameters
    ----------
    None

    Attributes
    ----------
    index : faiss.Index
        The FAISS index used for nearest neighbor search
    x_columns : array-like or None
        Column names of the reference dataset
    METRIC_MAP : dict
        Mapping of distance metric names to FAISS metric types

    See Also
    --------
    BlockingMethod : Abstract base class defining the blocking interface
    faiss.Index : The underlying FAISS index implementation

    Notes
    -----
    The available Index types from FAISS are: 'flat', 'hnsw', and 'lsh'.
    - 'flat' is a brute-force exact search (most accurate but slowest)
    - 'hnsw' is a Hierarchical Navigable Small World graph algorithm
        (good balance of speed and accuracy)
    - 'lsh' is a Locality Sensitive Hashing algorithm
        (fastest but approximate results)

    For more details about the FAISS library and implementation, see:
    https://github.com/facebookresearch/faiss

    Some distance metrics require special handling:
    - Cosine similarity is implemented through L2 normalization
    - Jensen-Shannon and Canberra metrics require smoothing to handle zero values
    - Selected distance metrics does not affect the algorithm if 'lsh' was selected

    Faiss does not support `random_seed` parameter. Instead, it handles reproducibility
    inside the algorithm. For more details, see:
    https://gist.github.com/mdouze/1892178b5663b80e85ab076966c59c28

    """

    METRIC_MAP: dict[str, Any] = {
        "euclidean": faiss.METRIC_L2,
        "l2": faiss.METRIC_L2,
        "inner_product": faiss.METRIC_INNER_PRODUCT,
        "cosine": faiss.METRIC_INNER_PRODUCT,
        # note: later handled by vector normalisation
        # see:(https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
        "l1": faiss.METRIC_L1,
        "manhattan": faiss.METRIC_L1,
        "linf": faiss.METRIC_Linf,
        # "lp" : faiss.METRIC_Lp,
        "canberra": faiss.METRIC_Canberra,
        # note: requires smoothing since 0/0 is undefined
        "bray_curtis": faiss.METRIC_BrayCurtis,
        "jensen_shannon": faiss.METRIC_JensenShannon,
        # note: requires smoothing since log(0) is undefined
    }

    def __init__(self) -> None:
        """
        Initialize the FaissBlocker instance.

        Creates a new FaissBlocker with empty index.
        """
        self.index: faiss.Index
        self.x_columns: list[str]

    def block(  # noqa: PLR0915, PLR0912
        self,
        x: DataHandler,
        y: DataHandler,
        k: int,
        verbose: bool | None,
        controls: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Perform blocking using the FAISS algorithm.

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
                'faiss': {
                    'index_type': ['flat', 'hnsw', 'lsh'],
                    'distance': str,
                    'k_search': int,
                    'path': str,

                    'hnsw_M': int,
                    'hnsw_ef_construction': int,
                    'hnsw_ef_search': int,

                    'lsh_nbits': int, (gets multiplied by dimensions)
                    'lsh_rotate_data': bool,
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
        Special preprocessing is applied for certain metrics:
        - For cosine similarity, vectors are L2-normalized
        - For Jensen-Shannon and Canberra metrics, small constant is added
          to prevent undefined values
        - For LSH index, the distance calculation is determined by the hash function,
          not directly by the selected distance metric

        """
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self.x_columns = list(x.cols)

        distance = controls["faiss"].get("distance")
        k_search = controls["faiss"].get("k_search")
        path = controls["faiss"].get("path")
        index_type = controls["faiss"].get("index_type", "hnsw")
        seed = controls.get("random_seed")

        if index_type not in {"flat", "hnsw", "lsh"}:
            raise ValueError(
                f"Invalid index_type '{index_type}'. Must be one of 'flat', 'hnsw', or 'lsh'."
            )

        X = x.to_dense()
        Y = y.to_dense()

        if distance == "cosine":
            faiss.normalize_L2(X)
            faiss.normalize_L2(Y)
        elif distance in {"jensen_shannon", "canberra"}:
            smooth = 1e-12
            X += smooth
            Y += smooth

        metric = self.METRIC_MAP[distance]

        if index_type == "flat":
            self.index = faiss.IndexFlat(X.shape[1], metric)
        elif index_type == "hnsw":
            M = controls["faiss"].get("hnsw_M")
            ef_construction = controls["faiss"].get("hnsw_ef_construction")
            ef_search = controls["faiss"].get("hnsw_ef_search")
            self.index = faiss.IndexHNSWFlat(X.shape[1], M, metric)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
            if seed is not None:
                self.index.hnsw.rng = faiss.RandomGenerator(int(seed))
        elif index_type == "lsh":
            nbits = controls["faiss"].get("lsh_nbits") * X.shape[1]
            if not isinstance(nbits, int):
                nbits = round(nbits)
            rotate_data = controls["faiss"].get("lsh_rotate_data")
            if seed is None:
                self.index = faiss.IndexLSH(X.shape[1], nbits, rotate_data)
            else:
                rot = faiss.RandomRotationMatrix(X.shape[1], X.shape[1])
                rot.init(int(seed))
                base = faiss.IndexLSH(X.shape[1], nbits, False, False)
                self.index = faiss.IndexPreTransform(rot, base)

        logger.info("Building index...")
        if distance == "cosine":
            self.index.add(x=X)
        else:
            self.index.add(x=X)

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

        if distance == "cosine":
            distances, indices = self.index.search(x=Y, k=k_search)
        else:
            distances, indices = self.index.search(x=Y, k=k_search)

        if distance == "cosine" and index_type != "lsh":
            distances = (1 - distances) / 2
        K_VAL = 2
        if k == K_VAL:
            indices, distances = rearrange_array(indices, distances)

        if path:
            self._save_index(path)

        result = pd.DataFrame(
            {
                "y": np.arange(Y.shape[0]),
                "x": indices[:, k - 1],
                "dist": distances[:, k - 1],
            }
        )

        logger.info("Process completed successfully.")

        return result

    def _save_index(self, path: str) -> None:
        """
        Save the FAISS index and column names to files.

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
            - 'index.faiss': The FAISS index file
            - 'index-colnames.txt': A text file with column names

        """
        if not os.path.exists(os.path.dirname(path)):
            raise ValueError("Provided path is incorrect")

        path_faiss = os.path.join(path, "index.faiss")
        path_faiss_cols = os.path.join(path, "index-colnames.txt")

        logger.info(f"Writing index to {path_faiss}")

        faiss.write_index(self.index, path_faiss)

        with open(path_faiss_cols, "w", encoding="utf-8") as f:
            f.write("\n".join(self.x_columns))
