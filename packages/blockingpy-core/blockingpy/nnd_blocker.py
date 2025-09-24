"""Contains the NNDBlocker class for blocking using the Nearest Neighbor Descent algorithm."""

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pynndescent

from .base import BlockingMethod
from .data_handler import DataHandler
from .helper_functions import rearrange_array

logger = logging.getLogger(__name__)


class NNDBlocker(BlockingMethod):
    """
    A blocker class that uses the Nearest Neighbor Descent (NND) algorithm.

    This class implements blocking functionality using the pynndescent library's
    NNDescent algorithm for efficient approximate nearest neighbor search.

    Parameters
    ----------
    None

    Attributes
    ----------
    index : pynndescent.NNDescent or None
        The NNDescent index used for querying

    See Also
    --------
    BlockingMethod : Abstract base class defining the blocking interface
    pynndescent.NNDescent : The underlying nearest neighbor descent implementation

    Notes
    -----
    For more details about the algorithm and implementation, see:
    https://pynndescent.readthedocs.io/en/latest/api.html
    https://github.com/lmcinnes/pynndescent

    """

    def __init__(self) -> None:
        """
        Initialize the NNDBlocker instance.

        Creates a new NNDBlocker with empty index.
        """
        self.index: pynndescent.NNDescent

    def block(
        self,
        x: DataHandler,
        y: DataHandler,
        k: int,
        verbose: bool | None,
        controls: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Perform blocking using the NND algorithm.

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
                'nnd': {
                    'metric': str,
                    'k_search': int,
                    'metric_kwds': dict,
                    'n_threads': int,
                    'tree_init': bool,
                    'n_trees': int,
                    'leaf_size': int,
                    'pruning_degree_multiplier': float,
                    'diversify_prob': float,
                    'init_graph': array-like or None,
                    'init_dist': array-like or None,
                    'low_memory': bool,
                    'max_candidates': int,
                    'max_rptree_depth': int,
                    'n_iters': int,
                    'delta': float,
                    'compressed': bool,
                    'parallel_batch_queries': bool,
                    'epsilon': float
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
        The algorithm builds an approximate nearest neighbor index using
        random projection trees and neighbor descent. The quality of the
        approximation can be controlled through various parameters such
        as n_trees, n_iters, and epsilon.

        """
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        distance = controls["nnd"].get("metric")
        k_search = controls["nnd"].get("k_search")

        X = x.data
        Y = y.data

        if k_search > X.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, X.shape[0])
            warnings.warn(
                f"k_search ({original_k_search}) is larger than the number of reference points "
                f"({X.shape[0]}). Adjusted k_search to {k_search}.",
                category=UserWarning,
                stacklevel=2,
            )

        logger.info(f"Initializing NND index with {distance} metric.")

        self.index = pynndescent.NNDescent(
            data=X,
            n_neighbors=k_search,
            metric=distance,
            metric_kwds=controls["nnd"].get("metric_kwds"),
            verbose=verbose,
            n_jobs=controls["nnd"].get("n_threads"),
            tree_init=controls["nnd"].get("tree_init"),
            n_trees=controls["nnd"].get("n_trees"),
            leaf_size=controls["nnd"].get("leaf_size"),
            pruning_degree_multiplier=controls["nnd"].get("pruning_degree_multiplier"),
            diversify_prob=controls["nnd"].get("diversify_prob"),
            init_graph=controls["nnd"].get("init_graph"),
            init_dist=controls["nnd"].get("init_dist"),
            low_memory=controls["nnd"].get("low_memory"),
            max_candidates=controls["nnd"].get("max_candidates"),
            max_rptree_depth=controls["nnd"].get("max_rptree_depth"),
            n_iters=controls["nnd"].get("n_iters"),
            delta=controls["nnd"].get("delta"),
            compressed=controls["nnd"].get("compressed"),
            parallel_batch_queries=controls["nnd"].get("parallel_batch_queries"),
            random_state=controls.get("random_seed"),
        )

        logger.info("Querying index...")

        l_1nn = self.index.query(query_data=Y, k=k_search, epsilon=controls["nnd"].get("epsilon"))
        indices = l_1nn[0]
        distances = l_1nn[1]

        K_VAL = 2
        if k == K_VAL:
            indices, distances = rearrange_array(indices, distances)

        result = pd.DataFrame(
            {
                "y": np.arange(Y.shape[0]),
                "x": indices[:, k - 1],
                "dist": distances[:, k - 1],
            }
        )

        logger.info("Process completed successfully.")

        return result
