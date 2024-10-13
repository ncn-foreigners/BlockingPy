import logging
from mlpack import lsh, knn
import pandas as pd
import numpy as np
from scipy.sparse import issparse, csr_matrix
from typing import Dict, Any, Union, Tuple, List
import os
from .base import BlockingMethod



class MLPackBlocker(BlockingMethod):
    """
    A class for performing blocking using MLPack algorithms (LSH or k-d tree).
    For details see: https://github.com/mlpack

    Attributes:
        algo (Optional[str]): The selected algorithm ('lsh' or 'kd').
        logger (logging.Logger): Logger for the class.
        x_columns (Optional[List[str]]): Column names of the reference data.

    The main method of this class is `block()`, which performs the actual
    blocking operation. Use the `controls` parameter in the `block()` method 
    to fine-tune the algorithm's behavior.

    This class inherits from the BlockingMethod abstract base class and
    implements its `block()` method.
    """
    ALGO_MAP: Dict[str, str] = {
        "lsh": "lsh",
        "kd": "knn"
    }

    def __init__(self):
        self.algo = None
        self.logger = logging.getLogger(__name__)
        self.x_columns = None


    def block(self, x: Union[np.ndarray, csr_matrix, pd.DataFrame], 
              y: Union[np.ndarray, csr_matrix, pd.DataFrame], 
              k: int, 
              controls: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform blocking using MLPack algorithm (LSH or k-d tree).

        Args:
            x (Union[np.ndarray, csr_matrix, pd.DataFrame]): Reference data.
            y (Union[np.ndarray, csr_matrix, pd.DataFrame]): Query data.
            k (int): Number of nearest neighbors to find.
            controls (Dict[str, Any]): Control parameters for the algorithm.

        Returns:
            pd.DataFrame: DataFrame containing the blocking results.

        Raises:
            ValueError: If an invalid algorithm is specified in the controls.
        """
        self.algo = controls.get('algo', None)
        self._check_algo(self.algo)
        if self.algo == lsh:
            verbose = controls['lsh'].get('verbose', False)
            seed = controls['lsh'].get('seed', None)
            path = controls['lsh'].get('path', None)
            k_search = controls['lsh'].get('k_search', 5)
        else:
            verbose = controls['kd'].get('verbose', False)
            seed = controls['kd'].get('seed', None)
            path = controls['kd'].get('path', None)
            k_search = controls['kd'].get('k_search', 5)

        x, self.x_columns = self._prepare_input(x)
        y, _ = self._prepare_input(y)

        k_search = min(k_search, x.shape[0])

        if verbose:
            self.logger.info(f"Initializing MLPack {self.algo.upper()} index...")

        if self.algo == 'lsh':
            query_result = lsh(
                k=k_search,
                query=y,
                reference=x,
                verbose=verbose,
                seed=seed,
                bucket_size=controls['lsh']['bucket_size'],
                hash_width=controls['lsh']['hash_width'],
                num_probes=controls['lsh']['num_probes'],
                projections=controls['lsh']['projections'],
                tables=controls['lsh']['tables']
            )
        else:  
            query_result = knn(
                k=k_search,
                query=y,
                reference=x,
                verbose=verbose,
                seed=seed,
                algorithm=controls['kd']['algorithm'],
                leaf_size=controls['kd']['leaf_size'],
                tree_type=controls['kd']['tree_type'],
                epsilon=controls['kd']['epsilon'],
                rho=controls['kd']['rho'],
                tau=controls['kd']['tau'],
                random_basis=controls['kd']['random_basis']
            )
        
        if verbose:
            self.logger.info("MLPack index query completed.")

        if path:
            self._save_result(path, query_result, verbose)

        result = pd.DataFrame({
            'y': range(y.shape[0]),
            'x': query_result['neighbors'][:, k-1],
            'dist': query_result['distances'][:, k-1]
        })

        if verbose:
            self.logger.info("Blocking process completed successfully.")

        return result


    def _check_algo(self, algo: str) -> None:
        """
        Validate the provided algorithm.

        Args:
            algo (str): The algorithm to validate.

        Raises:
            ValueError: If the provided algorithm is not in the ALGO_MAP.
        """
        if algo not in self.ALGO_MAP:
            valid_algos = ", ".join(self.ALGO_MAP.keys())
            raise ValueError(f"Invalid algorithm '{algo}'. Accepted values are: {valid_algos}.")
        
    
    def _prepare_input(self, data: Union[np.ndarray, csr_matrix, pd.DataFrame]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare input data for MLPack algorithms.

        Args:
            data: Input data in various formats.

        Returns:
            Tuple of numpy array and list of column names.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_numpy(), list(data.columns)
        elif issparse(data):
            return data.toarray(), [f'col_{i}' for i in range(data.shape[1])]
        else:
            return data, [f'col_{i}' for i in range(data.shape[1])]
        
    
    def _save_result(self, path: str, query_result: Dict[str, np.ndarray], verbose: bool) -> None:
        """
        Save the MLPack result and column names to files.

        Args:
            path (str): Directory path where the files will be saved.
            query_result (Dict[str, np.ndarray]): The result from the MLPack algorithm.
            verbose (bool): If True, log the save operation.

        Notes:
            Creates two files:
            1. 'index.{algo}': The MLPack result file.
            2. 'index-colnames.txt': A text file with column names.
        """
        path_index = os.path.join(path, f"index.{self.algo}")
        path_colnames = os.path.join(path, "index-colnames.txt")

        if verbose:
            self.logger.info(f"Writing the index to {path_index}")

        np.save(path_index, query_result)

        if verbose:
            self.logger.info(f"Writing column names to {path_colnames}")

        with open(path_colnames, 'w') as f:
            f.write('\n'.join(self.x_columns))

        if verbose:
            self.logger.info("Index and column names saved successfully.")