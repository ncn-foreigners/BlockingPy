import numpy as np
import pandas as pd
import pynndescent
from scipy.sparse import issparse, csr_matrix
import logging
from typing import Dict, Any, Union, Tuple, List
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


    def block(self, x: Union[np.ndarray, csr_matrix, pd.DataFrame], 
              y: Union[np.ndarray, csr_matrix, pd.DataFrame], 
              k: int, 
              controls: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform blocking using NND algorithm.

        Args:
            x (Union[np.ndarray, pd.DataFrame, csr_matrix]): Reference data.
            y (Union[np.ndarray, pd.DataFrame, csr_matrix]): Query data.
            k (int): Number of nearest neighbors to find.
            controls (Dict[str, Any]): Control parameters for the algorithm.

        Returns:
            pd.DataFrame: DataFrame containing the blocking results.

        Raises:
            ValueError: If an invalid distance metric is provided.
        """
        
        x, x_columns = self._prepare_input(x)
        y, _ = self._prepare_input(y)

        distance = controls['nnd'].get('metric')
        verbose = controls['nnd'].get('verbose', False)
        n_threads = controls['nnd'].get('n_threads', 1)
        k_search = controls['nnd'].get('k_search')

        if k_search > x.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, x.shape[0])
            self.logger.warning(f"k ({original_k_search}) is larger than the number of reference points ({x.shape[0]}). Adjusted k to {k_search}.")


        if k > x.shape[0]:
            original_k = k
            k = min(k, x.shape[0])
            self.logger.warning(f"k ({original_k}) is larger than the number of reference points ({x.shape[0]}). Adjusted k to {k}.")

        if verbose:
            self.logger.info("Initializing NND index...")

        nnd_params = controls.get('nnd', {})
        self.index = pynndescent.NNDescent(
            data=x,
            n_neighbors=k_search,
            metric=distance,
            metric_kwds=nnd_params.get('metric_kwds'),
            random_state=nnd_params.get('random_state'),
            n_jobs=n_threads,
            verbose=verbose,
            compressed=nnd_params.get('compressed'),
            tree_init=nnd_params.get('tree_init'),
            low_memory=nnd_params.get('low_memory'),
            max_candidates=nnd_params.get('max_candidates'),
            n_iters=nnd_params.get('n_iters'),
            delta=nnd_params.get('delta'),
            n_trees=nnd_params.get('n_trees'),
            leaf_size=nnd_params.get('leaf_size'),
            pruning_degree_multiplier=nnd_params.get('pruning_degree_multiplier'),
            diversify_prob=nnd_params.get('diversify_prob'),
            n_search_trees=nnd_params.get('n_search_trees'),
            init_dist=nnd_params.get('init_dist'),
            init_graph=nnd_params.get('init_graph'),
            parallel_batch_queries=nnd_params.get('parallel_batch_queries')
        )

        if verbose:
            self.logger.info("Querying index...")
        
        l_1nn = self.index.query(
            query_data=y,
            k=k_search,
            epsilon=nnd_params.get('epsilon')
        )

        result = pd.DataFrame({
            'y': np.arange(y.shape[0]),
            'x': l_1nn[0][:, k-1],
            'dist': l_1nn[1][:, k-1]
        })

        if verbose:
            self.logger.info("Process completed successfully.")

        return result

    def _prepare_input(self, data: Union[np.ndarray, csr_matrix, pd.DataFrame]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare input data for NND algorithm.

        Args:
            data Union[np.ndarray, csr_matrix, pd.DataFrame]: Input data in various formats.

        Returns:
            Tuple of numpy array and list of column names.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_numpy(), list(data.columns)
        elif issparse(data):
            return data.toarray(), [f'col_{i}' for i in range(data.shape[1])]
        else:
            return data, [f'col_{i}' for i in range(data.shape[1])]
        
def run_example():
    np.random.seed(42)
    x = np.random.rand(100, 10)
    y = np.random.rand(5, 10)

    nnd_blocker = NNDBlocker()
    result = nnd_blocker.block(x, y, k=3, controls={
        'nnd': {
            'metric': 'euclidean',
            'k_search': 5,
            'metric_kwds': {},  
            'tree_init': True,
            'n_trees': 10,
            'leaf_size': 30,
            'max_tree_depth': 20,
            'n_iters': 10,
            'delta': 0.001,
            'max_candidates': 60,
            'low_memory': False,
            'n_search_trees': 1,
            'pruning_degree_multiplier': 1.5,
            'diversify_prob': 0.1,
            'init_dist': 'euclidean',
            'init_graph': None,
            'compressed': False,
            'parallel_batch_queries': True,
            'epsilon': 0.1
        }
    })
    print(result.head())
