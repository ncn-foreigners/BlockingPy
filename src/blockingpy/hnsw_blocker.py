import numpy as np
import pandas as pd
import hnswlib
import scipy
import logging
from typing import Dict, Any, Optional, Union
import os
from .base import BlockingMethod

class HNSWBlocker(BlockingMethod):
    SPACE_MAP: Dict[str, str] = {
        "l2": "l2",
        "euclidean": "l2",
        "cosine": "cosine",
        "ip": "ip"
    }

    def __init__(self):
        self.index: Optional[hnswlib.Index] = None
        self.logger = logging.getLogger(__name__)

    def block(self, x: Union[np.ndarray, scipy.sparse.csr_matrix], 
            y: Union[np.ndarray, scipy.sparse.csr_matrix], 
            k: int, 
            controls: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform blocking using HNSW algorithm.

        Args:
            x (Union[np.ndarray, scipy.sparse.csr_matrix]): Reference data.
            y (Union[np.ndarray, scipy.sparse.csr_matrix]): Query data.
            k (int): Number of nearest neighbors to find. If k is larger than the number of reference points,
                     it will be automatically adjusted.
            controls (Dict[str, Any]): Control parameters for the algorithm.

        Returns:
            pd.DataFrame: DataFrame containing the blocking results.

        Raises:
            ValueError: If an invalid distance metric is provided.
        """
        distance = controls['hnsw'].get('distance')
        verbose = controls['hnsw'].get('verbose', False)
        n_threads = controls['hnsw'].get('n_threads', 1)
        path = controls['hnsw'].get('path')

        if k > x.shape[0]:
            original_k = k
            k = min(k, x.shape[0])
            self.logger.warning(f"k ({original_k}) is larger than the number of reference points ({x.shape[0]}). Adjusted k to {k}.")

        self._check_distance(distance)
        space = self.SPACE_MAP[distance]

        if verbose:
            self.logger.info("Initializing HNSW index...")
        
        self.index = hnswlib.Index(space=space, dim=x.shape[1])
        self.index.init_index(
            max_elements=x.shape[0], 
            ef_construction=controls['hnsw']['ef_c'], 
            M=controls['hnsw']['M']
        )
        self.index.set_num_threads(n_threads)

        if verbose:
            self.logger.info("Adding items to index...")
            
        self.index.add_items(x)
        self.index.set_ef(controls['hnsw']['ef_s'])

        if verbose:
            self.logger.info("Querying index...")

        l_1nn = self.index.knn_query(y, k=k)

        if path:
            self._save_index(path, x, verbose)
        
        result = pd.DataFrame({
            'y': range(y.shape[0]),
            'x': l_1nn[0][:, k-1],
            'dist': l_1nn[1][:, k-1]
        })

        if verbose:
            self.logger.info("Process completed successfully.")

        return result
    
    def _check_distance(self, distance: str) -> None:
        """
        Validate the provided distance metric.

        Args:
            distance (str): The distance metric to validate.

        Raises:
            ValueError: If the provided distance is not in the SPACE_MAP.
        """
        if distance not in self.SPACE_MAP:
            valid_metrics = ", ".join(self.SPACE_MAP.keys())
            raise ValueError(f"Invalid distance metric '{distance}'. Accepted values are: {valid_metrics}.")

    def _save_index(self, path: str, x: Union[np.ndarray, 'scipy.sparse.csr_matrix'], verbose: bool) -> None:
        """
        Save the HNSW index and column names to files.

        Args:
            path (str): Directory path where the files will be saved.
            x (Union[np.ndarray, scipy.sparse.csr_matrix]): The reference data used to determine the number of columns.
            verbose (bool): If True, log the save operation.

        Notes:
            Creates two files:
            1. 'index.hnsw': The HNSW index file.
            2. 'index-colnames.txt': A text file with column names.
        """
        path_ann = os.path.join(path, "index.hnsw")
        path_ann_cols = os.path.join(path, "index-colnames.txt")
        
        if verbose:
            self.logger.info(f"Writing an index to {path_ann}")
        
        self.index.save_index(path_ann)
        with open(path_ann_cols, 'w') as f:
            f.write('\n'.join(x.columns if isinstance(x, pd.DataFrame) else [f'col_{i}' for i in range(x.shape[1])]))