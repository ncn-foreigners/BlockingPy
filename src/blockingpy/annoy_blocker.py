from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from tempfile import NamedTemporaryFile
from .base import BlockingMethod
import logging


class AnnoyBlocker(BlockingMethod):
    def __init__(self):
        self.index: Optional[AnnoyIndex] = None
        self.logger = logging.getLogger(__name__)
        self.metric_map:Dict[str, str] = {
            "euclidean": "euclidean",
            "manhattan": "manhattan",
            "hamming": "hamming",
            "angular": "angular"
        }
    
    def block(self, x: np.ndarray, y: np.ndarray, k: int, controls: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform blocking using Annoy algorithm.

        Args:
            x (np.ndarray): Reference data.
            y (np.ndarray): Query data.
            k (int): Number of nearest neighbors to find.
            controls (Dict[str, Any]): Control parameters for the algorithm.

        Returns:
            pd.DataFrame: DataFrame containing the blocking results.

        Raises:
            ValueError: If an invalid distance metric is provided.
        """
        distance = controls.annoy.get('distance', None)
        verbose = controls.annoy.get('verbose', False)
        seed = controls.annoy.get('seed', None)
        path = controls.annoy.get('path', None)
        n_trees = controls.annoy.get('n_trees', 10)
        build_on_disk = controls.annoy.get('build_on_disk', False)
        k_search = controls.get('k_search', 10) 

        self._check_distance(distance)    

        ncols = x.shape[1]
        metric = self.metric_map.get(distance)

        self.index = AnnoyIndex(ncols, metric)
        if seed is not None:
            self.index.set_seed(seed)

        if build_on_disk:
            if build_on_disk:
                with NamedTemporaryFile(prefix="annoy", suffix=".tree") as temp_file:
                    if verbose:
                        self.logger.info(f"Building index on disk: {temp_file.name}")
                    self.index.on_disk_build(temp_file.name)
        
        if verbose:
            self.index.verbose(True)
            self.logger.info("Building index...")
        
        for i in range(x.shape[0]):
            self.index.add_item(i, x[i])
        self.index.build(n_trees=n_trees)

        if verbose:
            self.logger.info("Querying index...")

        l_ind_nns = np.zeros(y.shape[0], dtype=int)
        l_ind_dist = np.zeros(y.shape[0])

        k_search = min(k_search, x.shape[0])
        for i  in range(y.shape[0]):
            annoy_res = self.index.get_nns_by_vector(y[i], k_search, include_distances=True)
            l_ind_nns[i] = annoy_res[0][k-1]
            l_ind_dist[i] = annoy_res[1][k-1]  

        if path:
            self._save_index(path, x, verbose)

        result = {
            'y': np.arange(y.shape[0]),  
            'x': l_ind_nns, 
            'dist': l_ind_dist,
        }

        result = pd.DataFrame(result)

        if verbose:
            self.logger.info("Process completed successfully.")

        return result

    def _check_distance(self, distance: str) -> None:
        """
        Validate the provided distance metric.

        Args:
            distance (str): The distance metric to validate.

        Raises:
            ValueError: If the provided distance is not in the metric map.
        """
        if distance not in self.metric_map.keys():
            valid_metrics = ", ".join(self.metric_map.keys())
            raise ValueError(f"Invalid distance metric '{distance}'. Accepted values are: {valid_metrics}.")
        
    def _save_index(self, path: str, x: np.ndarray, verbose: bool) -> None:
        """
        Save the Annoy index and column names to files.

        Args:
            path (str): Directory path where the files will be saved.
            x (np.ndarray): The reference data used to create column names.
            verbose (bool): If True, log the save operation.

        Notes:
            Creates two files:
            1. 'index.annoy': The Annoy index file.
            2. 'index-colnames.txt': A text file with column names.
        """
        import os
        path_ann = os.path.join(path, "index.annoy")
        path_ann_cols = os.path.join(path, "index-colnames.txt")

        if verbose:
             self.logger.info(f"Writing an index to {path_ann}")
        
        self.index.save(path_ann)

        with open(path_ann_cols, 'w') as f:
            f.write('\n'.join(x.columns if isinstance(x, pd.DataFrame) else [f'col_{i}' for i in range(x.shape[1])]))
