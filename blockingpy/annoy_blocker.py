import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from tempfile import NamedTemporaryFile
from typing import Dict, Any, Optional
import os
import logging
from .base import BlockingMethod


class AnnoyBlocker(BlockingMethod):
    """
    A class for performing blocking using the Annoy algorithm.
    For details see: https://github.com/spotify/annoy

    Attributes:
        index (Optional[AnnoyIndex]): The Annoy index used for nearest neighbor search.
        logger (logging.Logger): Logger for the class.
        x_columns: column names of x

    The main method of this class is `block()`, which performs the actual
    blocking operation. Use the `controls` parameter in the `block()` method 
    to fine-tune the algorithm's behavior.

    This class inherits from the BlockingMethod abstract base class and
    implements its `block()` method.
    """
    METRIC_MAP: Dict[str, str] = {
        "euclidean": "euclidean",
        "manhattan": "manhattan",
        "hamming": "hamming",
        "angular": "angular"
    }

    def __init__(self):
        self.index: Optional[AnnoyIndex] = None
        self.logger = logging.getLogger(__name__)
        self.x_columns = None
    
    def block(self, x: pd.DataFrame, 
              y: pd.DataFrame, 
              k: int,
              verbose: Optional[bool], 
              controls: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform blocking using Annoy algorithm.

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
        self.x_columns = x.columns

        distance = controls['annoy'].get('distance', None)
        verbose = verbose
        seed = controls['annoy'].get('seed', None)
        path = controls['annoy'].get('path', None)
        n_trees = controls['annoy'].get('n_trees', 10)
        build_on_disk = controls['annoy'].get('build_on_disk', False)
        k_search = controls['annoy'].get('k_search', 10)


        self._check_distance(distance)    

        ncols = x.shape[1]
        metric = self.METRIC_MAP[distance]

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

        if k_search > x.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, x.shape[0])
            self.logger.warning(f"k ({original_k_search}) is larger than the number of reference points ({x.shape[0]}). Adjusted k to {k_search}.")
        
        for i in range(y.shape[0]):
            annoy_res = self.index.get_nns_by_vector(y[i], k_search, include_distances=True)
            l_ind_nns[i] = annoy_res[0][k-1]
            l_ind_dist[i] = annoy_res[1][k-1]  

        if path:
            self._save_index(path, verbose)

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
            ValueError: If the provided distance is not in the METRIC_MAP.
        """
        if distance not in self.METRIC_MAP:
            valid_metrics = ", ".join(self.METRIC_MAP.keys())
            raise ValueError(f"Invalid distance metric '{distance}'. Accepted values are: {valid_metrics}.")
             
    def _save_index(self, path: str, verbose: bool) -> None:
        """
        Save the Annoy index and column names to files.

        Args:
            path (str): Directory path where the files will be saved.
            verbose (bool): If True, log the save operation.

        Notes:
            Creates two files:
            1. 'index.annoy': The Annoy index file.
            2. 'index-colnames.txt': A text file with column names.
        """
        path_ann = os.path.join(path, "index.annoy")
        path_ann_cols = os.path.join(path, "index-colnames.txt")

        if verbose:
             self.logger.info(f"Writing an index to {path_ann}")
        
        self.index.save(path_ann)

        with open(path_ann_cols, 'w') as f:
            f.write('\n'.join(self.x_columns))
