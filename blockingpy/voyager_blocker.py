import numpy as np
import pandas as pd
from voyager import Index, Space
from typing import Dict, Any, Optional
import os
import logging
from .base import BlockingMethod
import sys


class VoyagerBlocker(BlockingMethod):
    """
    A class for performing blocking using the Voyager algorithm from Spotify.
    For details see: https://github.com/spotify/voyager

    Attributes:
        index (Optional[Index]): The Voyager index used for nearest neighbor search.
        logger (logging.Logger): Logger for the class.
        x_columns: column names of x

    The main method of this class is `block()`, which performs the actual
    blocking operation. Use the `controls` parameter in the `block()` method 
    to fine-tune the algorithm's behavior.

    This class inherits from the BlockingMethod abstract base class and
    implements its `block()` method.
    """
    METRIC_MAP: Dict[str, Space] = {
        "euclidean": Space.Euclidean,
        "cosine": Space.Cosine,
        "inner_product": Space.InnerProduct,
    }

    def __init__(self):
        self.index: Optional[Index] = None
        self.logger = logging.getLogger('__main__')
        self.x_columns = None
    
    def block(self, x: pd.DataFrame, 
              y: pd.DataFrame, 
              k: int,
              verbose: Optional[bool], 
              controls: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform blocking using Voyager algorithm.

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
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        self.x_columns = x.columns

        distance = controls['voyager'].get('distance')
        self._check_distance(distance)
        space = self.METRIC_MAP[distance]
        k_search = controls['voyager'].get('k_search')
        path = controls['voyager'].get('path')      
        
        self.index = Index(
            space=space,
            num_dimensions=x.shape[1],
            M=controls['voyager'].get('M'),
            ef_construction=controls['voyager'].get('ef_construction'),
            random_seed=controls['voyager'].get('random_seed'),
            max_elements=controls['voyager'].get('max_elements'),
        )

        self.logger.info("Building index...")
        
        self.index.add_items(x.sparse.to_dense().values.tolist(),
                             num_threads=controls['voyager'].get('num_threads'),
                            )

        self.logger.info("Querying index...")

        l_ind_nns = np.zeros(y.shape[0], dtype=int)
        l_ind_dist = np.zeros(y.shape[0])

        if k_search > x.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, x.shape[0])
            self.logger.warning(f"k_search ({original_k_search}) is larger than the number of reference points ({x.shape[0]}). Adjusted k_search to {k_search}.")

        all_neighbor_ids, all_distances = self.index.query(vectors=y.sparse.to_dense().values.tolist(),
                         k=k_search,
                         num_threads=controls['voyager'].get('num_threads'),
                         query_ef=controls['voyager'].get('query_ef'),
                        )
    
        l_ind_nns = all_neighbor_ids[:, k-1]
        l_ind_dist = all_distances[:, k-1]

        if path:
            self._save_index(path, verbose)

        result = {
            'y': np.arange(y.shape[0]),  
            'x': l_ind_nns, 
            'dist': l_ind_dist,
        }

        result = pd.DataFrame(result)

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
             
    def _save_index(self, path: str) -> None:
        """
        Save the Voyager index and column names to files.

        Args:
            path (str): Directory path where the files will be saved.

        Notes:
            Creates two files:
            1. 'index.voyager': The Voyager index file.
            2. 'index-colnames.txt': A text file with column names.
        """
        path_voy = os.path.join(path, "index.voyager")
        path_voy_cols = os.path.join(path, "index-colnames.txt")

        self.logger.info(f"Writing an index to {path_voy}")
        
        self.index.save(path_voy)

        with open(path_voy_cols, 'w') as f:
            f.write('\n'.join(self.x_columns))