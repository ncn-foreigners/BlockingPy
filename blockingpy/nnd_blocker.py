import numpy as np
import pandas as pd
import pynndescent
import logging
from typing import Dict, Any, Optional
from .base import BlockingMethod
import sys


class NNDBlocker(BlockingMethod):
    """
    A blocker class that uses the Nearest Neighbor Descent (NND).

    This class performs blocking using the pynndescent library's NNDescent algorithm.
    For details see: https://pynndescent.readthedocs.io/en/latest/api.html (https://github.com/lmcinnes/pynndescent)

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
        self.logger = logging.getLogger('__main__')


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
            controls (Dict[str, Any]): Control parameters for the algorithm. For details see: blockingpy/controls.py

        Returns:
            pd.DataFrame: DataFrame containing the blocking results.

        Raises:
            ValueError: If an invalid distance metric is provided.
        """ 
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        distance = controls.get('nnd').get('metric')
        k_search = controls.get('nnd').get('k_search')

        if k_search > x.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, x.shape[0])
            self.logger.warning(f"k_search ({original_k_search}) is larger than the number of reference points ({x.shape[0]}). Adjusted k_search to {k_search}.")
        
        self.logger.info(f"Initializing NND index with {distance} metric.")

        self.index = pynndescent.NNDescent(
            data=x,
            n_neighbors=k_search,
            metric=distance,
            metric_kwds=controls['nnd'].get('metric_kwds'),
            verbose=verbose,
            n_jobs=controls['nnd'].get('n_threads'),
            tree_init=controls['nnd'].get('tree_init'),
            n_trees=controls['nnd'].get('n_trees'),
            leaf_size=controls['nnd'].get('leaf_size'),
            pruning_degree_multiplier=controls['nnd'].get('pruning_degree_multiplier'),
            diversify_prob=controls['nnd'].get('diversify_prob'),
            init_graph=controls['nnd'].get('init_graph'),
            init_dist=controls['nnd'].get('init_dist'),
            #algorithm=nnd_params.get('algorithm'),
            low_memory=controls['nnd'].get('low_memory'),
            max_candidates=controls['nnd'].get('max_candidates'),
            max_rptree_depth=controls['nnd'].get('max_rptree_depth'),
            n_iters=controls['nnd'].get('n_iters'),
            delta=controls['nnd'].get('delta'),
            compressed=controls['nnd'].get('compressed'),
            parallel_batch_queries=controls['nnd'].get('parallel_batch_queries')
        )
        
        self.logger.info("Querying index...")
        
        l_1nn = self.index.query(
            query_data=y,
            k=k_search,
            epsilon=controls['nnd'].get('epsilon')
        )
        result = pd.DataFrame({
            'y': np.arange(y.shape[0]),
            'x': l_1nn[0][:, k-1],
            'dist': l_1nn[1][:, k-1]
        })

        self.logger.info("Process completed successfully.")

        return result
        