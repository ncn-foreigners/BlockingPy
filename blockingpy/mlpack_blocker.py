import logging
from mlpack import lsh, knn
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os
from .base import BlockingMethod


class MLPackBlocker(BlockingMethod):
    """
    A class for performing blocking using MLPack algorithms (LSH or k-d tree).
    For details see: https://github.com/mlpack

    Attributes:
        algo (Optional[str]): The selected algorithm ('lsh' or 'kd').
        logger (logging.Logger): Logger for the class.

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

    def block(self, x: pd.DataFrame, 
              y: pd.DataFrame, 
              k: int, 
              verbose: Optional[bool],
              controls: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform blocking using MLPack algorithm (LSH or k-d tree).

        Args:
            x (pd.DataFrame): Reference data.
            y (pd.DataFrame): Query data.
            k (int): Number of nearest neighbors to find.
            verbose (bool): control the level of verbosity.
            controls (Dict[str, Any]): Control parameters for the algorithm.

        Returns:
            pd.DataFrame: DataFrame containing the blocking results.

        Raises:
            ValueError: If an invalid algorithm is specified in the controls.
        """
        self.algo = controls.get('algo', 'lsh')
        self._check_algo(self.algo)
        if self.algo == 'lsh':
            verbose = verbose
            seed = controls['lsh'].get('seed', None)
            k_search = controls['lsh'].get('k_search', 10)
        else:
            verbose = verbose
            seed = controls['kd'].get('seed', None)
            k_search = controls['kd'].get('k_search', 10)

        if k_search > x.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, x.shape[0])
            self.logger.warning(f"k ({original_k_search}) is larger than the number of reference points ({x.shape[0]}). Adjusted k to {k_search}.")

        if verbose:
            self.logger.info(f"Initializing MLPack {self.algo.upper()} index...")

        if self.algo == 'lsh':
            query_result = lsh(
                k=k_search,
                query=y,
                reference=x,
                verbose=verbose,
                seed=seed,
                bucket_size=controls['lsh'].get('bucket_size', 500),
                hash_width=controls['lsh'].get('hash_width', 0.0),
                num_probes=controls['lsh'].get('num_probes', 0),
                projections=controls['lsh'].get('projections', 10),
                tables=controls['lsh'].get('tables', 30)
            )
        else:  
            query_result = knn(
                k=k_search,
                query=y,
                reference=x,
                verbose=verbose,
                seed=seed,
                algorithm=controls['kd'].get('algorithm', None),
                leaf_size=controls['kd'].get('leaf_size', None),
                tree_type=controls['kd'].get('tree_type', None),
                epsilon=controls['kd'].get('epsilon', None),
                rho=controls['kd'].get('rho', None),
                tau=controls['kd'].get('tau', None),
                random_basis=controls['kd'].get('random_basis', False)
            )
        
        if verbose:
            self.logger.info("MLPack index query completed.")

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
        