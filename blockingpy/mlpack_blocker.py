import logging
from mlpack import lsh, knn
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base import BlockingMethod


class MLPackBlocker(BlockingMethod):
    """
    A class for performing blocking using MLPack algorithms (LSH or k-d tree).

    This class implements blocking functionality using either Locality-Sensitive 
    Hashing (LSH) or k-d tree algorithms from the MLPack library for efficient 
    similarity search and nearest neighbor queries.

    Parameters
    ----------
    None

    Attributes
    ----------
    algo : str or None
        The selected algorithm ('lsh' or 'kd')
    logger : logging.Logger
        Logger instance for the class
    ALGO_MAP : dict
        Mapping of algorithm names to their MLPack implementations

    See Also
    --------
    BlockingMethod : Abstract base class defining the blocking interface

    Notes
    -----
    For more details about the MLPack library and its algorithms, see:
    https://github.com/mlpack
    """
    ALGO_MAP: Dict[str, str] = {
        "lsh": "lsh",
        "kd": "knn"
    }

    def __init__(self):
        """
        Initialize the MLPackBlocker instance.

        Creates a new MLPackBlocker with no algorithm selected and default 
        logger settings.
        """
        self.algo = None
        self.logger = logging.getLogger('__main__')

    def block(self, x: pd.DataFrame, 
              y: pd.DataFrame, 
              k: int, 
              verbose: Optional[bool],
              controls: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform blocking using MLPack algorithm (LSH or k-d tree).

        Parameters
        ----------
        x : pandas.DataFrame
            Reference dataset containing features for indexing
        y : pandas.DataFrame
            Query dataset to find nearest neighbors for
        k : int
            Number of nearest neighbors to find
        verbose : bool, optional
            If True, print detailed progress information
        controls : dict
            Algorithm control parameters with the following structure:
            {
                'algo': str,
                'lsh': {  # if using LSH
                    'seed': int,
                    'k_search': int,
                    'bucket_size': int,
                    'hash_width': float,
                    'num_probes': int,
                    'projections': int,
                    'tables': int
                },
                'kd': {   # if using k-d tree
                    'seed': int,
                    'k_search': int,
                    'algorithm': str,
                    'leaf_size': int,
                    'tree_type': str,
                    'epsilon': float,
                    'rho': float,
                    'tau': float,
                    'random_basis': bool
                }
            }

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the blocking results with columns:
            - 'y': indices from query dataset
            - 'x': indices of matched items from reference dataset
            - 'dist': distances to matched items

        Raises
        ------
        ValueError
            If an invalid algorithm is specified in the controls

        Notes
        -----
        The function supports two different algorithms:
        - LSH (Locality-Sensitive Hashing): Better for high-dimensional data
        - k-d tree: Better for low-dimensional data
        """
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self.algo = controls.get('algo')
        self._check_algo(self.algo)
        if self.algo == 'lsh':
            verbose = verbose
            seed = controls['lsh'].get('seed')
            k_search = controls['lsh'].get('k_search')
        else:
            verbose = verbose
            seed = controls['kd'].get('seed')
            k_search = controls['kd'].get('k_search')

        if k_search > x.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, x.shape[0])
            self.logger.warning(f"k_search ({original_k_search}) is larger than the number of reference points ({x.shape[0]}). Adjusted k_search to {k_search}.")

        self.logger.info(f"Initializing MLPack {self.algo.upper()} index...")

        if self.algo == 'lsh':
            query_result = lsh(
                k=k_search,
                query=y,
                reference=x,
                verbose=verbose,
                seed=seed,
                bucket_size=controls['lsh'].get('bucket_size'),
                hash_width=controls['lsh'].get('hash_width'),
                num_probes=controls['lsh'].get('num_probes'),
                projections=controls['lsh'].get('projections'),
                tables=controls['lsh'].get('tables')
            )
        else:  
            query_result = knn(
                k=k_search,
                query=y,
                reference=x,
                verbose=verbose,
                seed=seed,
                algorithm=controls['kd'].get('algorithm'),
                leaf_size=controls['kd'].get('leaf_size'),
                tree_type=controls['kd'].get('tree_type'),
                epsilon=controls['kd'].get('epsilon'),
                rho=controls['kd'].get('rho'),
                tau=controls['kd'].get('tau'),
                random_basis=controls['kd'].get('random_basis')
            )
        
        self.logger.info("MLPack index query completed.")

        result = pd.DataFrame({
            'y': range(y.shape[0]),
            'x': query_result['neighbors'][:, k-1],
            'dist': query_result['distances'][:, k-1]
        })

        self.logger.info("Blocking process completed successfully.")

        return result


    def _check_algo(self, algo: str) -> None:
        """
        Validate the provided algorithm.

        Parameters
        ----------
        algo : str
            The algorithm to validate

        Raises
        ------
        ValueError
            If the provided algorithm is not in the ALGO_MAP

        Notes
        -----
        Valid algorithms are defined in the ALGO_MAP class attribute.
        Currently supports 'lsh' for Locality-Sensitive Hashing and
        'kd' for k-d tree based search.
        """
        if algo not in self.ALGO_MAP:
            valid_algos = ", ".join(self.ALGO_MAP.keys())
            raise ValueError(f"Invalid algorithm '{algo}'. Accepted values are: {valid_algos}.")
        