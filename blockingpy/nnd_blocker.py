import numpy as np
import pandas as pd
import pynndescent
import logging
from typing import Dict, Any, Optional
from .base import BlockingMethod


class NNDBlocker(BlockingMethod):
    """
    A blocker class that uses the Nearest Neighbor Descent (NND).

    This class performs blocking using the pynndescent library's NNDescent algorithm.
    For details see: https://pynndescent.readthedocs.io/en/latest/api.html

    Attributes:
        index: The NNDescent index used for querying.
        logger: A logger instance for outputting information and warnings.

    The main method of this class is `block()`, which performs the actual
    blocking operation. Use the `controls` parameter in the `block()` method 
    to fine-tune the algorithm's behavior.

    This class inherits from the BlockingMethod abstract base class and
    implements its `block()` method.
    """
    def __init__(self):
        self.index = None
        self.logger = logging.getLogger(__name__)


    def block(self, x: pd.DataFrame, 
              y: pd.DataFrame, 
              k: int, 
              verbose: Optional[bool],
              controls: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform blocking using NND algorithm.

        Args:
            x (pd.DataFrame): Reference data.
            y (pd.DataFrame): Query data.
            k (int): Number of nearest neighbors to find.
            verbose (bool): control the level of verbosity.
            controls (Dict[str, Any]): Control parameters for the algorithm.

        Returns:
            pd.DataFrame: DataFrame containing the blocking results.

        Raises:
            ValueError: If an invalid distance metric is provided.
        """

        distance = controls['nnd'].get('metric', 'euclidean')
        verbose = verbose
        k_search = controls['nnd'].get('k_search', 30)

        if k_search > x.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, x.shape[0])
            self.logger.warning(f"k ({original_k_search}) is larger than the number of reference points ({x.shape[0]}). Adjusted k_search to {k_search}.")

        if verbose:
            self.logger.info(f"Initializing NND index with {distance} metric.")

        nnd_params = controls.get('nnd', {})
        self.index = pynndescent.NNDescent(
            data=x,
            n_neighbors=k_search,
            metric=distance,
            metric_kwds=nnd_params.get('metric_kwds', {}),
            verbose=verbose,
            n_jobs=nnd_params.get('n_threads', None),
            tree_init=nnd_params.get('tree_init', True),
            n_trees=nnd_params.get('n_trees', None),
            leaf_size=nnd_params.get('leaf_size', None),
            pruning_degree_multiplier=nnd_params.get('pruning_degree_multiplier', 1.5),
            diversify_prob=nnd_params.get('diversify_prob', 1.0),
            init_graph=nnd_params.get('init_graph', None),
            init_dist=nnd_params.get('init_dist', None),
            #algorithm=nnd_params.get('algorithm', 'standard'),
            low_memory=nnd_params.get('low_memory', True),
            max_candidates=nnd_params.get('max_candidates', None),
            max_rptree_depth=nnd_params.get('max_rptree_depth', 100),
            n_iters=nnd_params.get('n_iters', None),
            delta=nnd_params.get('delta', 0.001),
            compressed=nnd_params.get('compressed', False),
            parallel_batch_queries=nnd_params.get('parallel_batch_queries', False)
        )

        if verbose:
            self.logger.info("Querying index...")
        
        l_1nn = self.index.query(
            query_data=y,
            k=k_search,
            epsilon=nnd_params.get('epsilon', 0.1)
        )

        result = pd.DataFrame({
            'y': np.arange(y.shape[0]),
            'x': l_1nn[0][:, k-1],
            'dist': l_1nn[1][:, k-1]
        })

        if verbose:
            self.logger.info("Process completed successfully.")

        return result
        