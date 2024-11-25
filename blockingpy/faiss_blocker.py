"""Contains the FaissBlocker class for performing blocking using the FAISS algorithm from Meta."""

import logging
import os
from typing import Dict, Any, Optional

import faiss
import numpy as np
import pandas as pd

from .base import BlockingMethod


logger = logging.getLogger(__name__)

class FaissBlocker(BlockingMethod):
   """
    A class for performing blocking using the FAISS (Facebook AI Similarity Search) algorithm.

    This class implements blocking functionality using Facebook's FAISS library for
    efficient similarity search and nearest neighbor queries. It supports multiple
    distance metrics and is optimized for high-performance computing.

    Parameters
    ----------
    None

    Attributes
    ----------
    index : faiss.IndexFlat or None
        The FAISS index used for nearest neighbor search
    logger : logging.Logger
        Logger instance for the class
    x_columns : array-like or None
        Column names of the reference dataset
    METRIC_MAP : dict
        Mapping of distance metric names to FAISS metric types

    See Also
    --------
    BlockingMethod : Abstract base class defining the blocking interface
    faiss.IndexFlat : The underlying FAISS index implementation

    Notes
    -----
    For more details about the FAISS library and implementation, see:
    https://github.com/facebookresearch/faiss

    Some distance metrics require special handling:
    - Cosine similarity is implemented through L2 normalization
    - Jensen-Shannon and Canberra metrics require smoothing to handle zero values
    """
   METRIC_MAP: Dict[str, Any] = {
        "euclidean": faiss.METRIC_L2,
        "l2": faiss.METRIC_L2,
        "inner_product": faiss.METRIC_INNER_PRODUCT,
        "cosine": faiss.METRIC_INNER_PRODUCT,#note: later handled by vector normalisation (https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
        "l1": faiss.METRIC_L1,
        "manhattan": faiss.METRIC_L1,
        "linf": faiss.METRIC_Linf,
        #"lp" : faiss.METRIC_Lp,
        "canberra": faiss.METRIC_Canberra,#note: requires smoothing since 0/0 is undefined
        "bray_curtis": faiss.METRIC_BrayCurtis,
        "jensen_shannon": faiss.METRIC_JensenShannon,#note: requires smoothing since log(0) is undefined
    }
   
   def __init__(self) -> None:
       """
       Initialize the FaissBlocker instance.

       Creates a new FaissBlocker with empty index and default logger settings.
       """
       self.index: Optional[faiss.IndexFlat] = None
       self.x_columns = None

   def block(self, x: pd.DataFrame,
             y: pd.DataFrame,
             k: int,
             verbose: Optional[bool],
             controls: Dict[str, Any]) -> pd.DataFrame:
       """
        Perform blocking using the FAISS algorithm.

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
                'faiss': {
                    'distance': str,
                    'k_search': int,
                    'path': str
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
            If an invalid distance metric is provided or if path is provided but incorrect

        Notes
        -----
        Special preprocessing is applied for certain metrics:
        - For cosine similarity, vectors are L2-normalized
        - For Jensen-Shannon and Canberra metrics, small constant is added 
          to prevent undefined values
        """ 

       logger.setLevel(logging.INFO if verbose else logging.WARNING)
       self.x_columns = x.columns

       distance = controls['faiss'].get('distance')
       self._check_distance(distance)
       k_search = controls['faiss'].get('k_search')
       path = controls['faiss'].get('path')

       if distance == "cosine":
           faiss.normalize_L2(x=np.ascontiguousarray(x.sparse.to_dense().to_numpy(), dtype=np.float32))
           faiss.normalize_L2(x=np.ascontiguousarray(y.sparse.to_dense().to_numpy(), dtype=np.float32))
       elif distance in ['jensen_shannon', 'canberra']:
           smooth = 1e-12
           x = x + smooth
           y = y + smooth

       metric = self.METRIC_MAP[distance]

       self.index = faiss.IndexFlat(x.shape[1], metric)

       logger.info("Building index...")
       self.index.add(x=x)    
         
       logger.info("Querying index...")

       if k_search > x.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, x.shape[0])
            logger.warning(f"k_search ({original_k_search}) is larger than the number of reference points ({x.shape[0]}). Adjusted k_search to {k_search}.")

       distances, indices = self.index.search(x=y, k=k_search)

       if path:
           self._save_index(path)

       result = {
           'y': np.arange(y.shape[0]),
           'x': indices[:, k-1],
           'dist': distances[:, k-1]
       }

       result = pd.DataFrame(result)

       logger.info("Process completed successfully.")

       return result
    

   def _check_distance(self, distance: str) -> None:
       """
        Validate the provided distance metric.

        Parameters
        ----------
        distance : str
            The distance metric to validate

        Raises
        ------
        ValueError
            If the provided distance is not in the METRIC_MAP

        Notes
        -----
        Valid metrics are defined in the METRIC_MAP class attribute.
        Some metrics require special preprocessing (cosine, jensen_shannon, canberra).
        """
       if distance not in self.METRIC_MAP:
           valid_metrics = ", ".join(self.METRIC_MAP.keys())
           raise ValueError(f"Invalid distance metric '{distance}'. Accepted values are: {valid_metrics}.") 


   def _save_index(self, path: str) -> None:
       """
       Save the FAISS index and column names to files.

       Parameters
       ----------
       path : str
           Directory path where the files will be saved
       verbose : bool, optional
           If True, print information about the save operation (default True)
        
        Raises
        ------
        ValueError
            If the provided path is incorrect

       Notes
       -----
       Creates two files:
           - 'index.faiss': The FAISS index file
           - 'index-colnames.txt': A text file with column names
       """
       if not os.path.exists(os.path.dirname(path)):
            raise ValueError("Provided path is incorrect")
       
       path_faiss = os.path.join(path, "index.faiss")
       path_faiss_cols = os.path.join(path, "index-colnames.txt")

       logger.info(f"Writing index to {path_faiss}")
       
       faiss.write_index(self.index, path_faiss)

       with open(path_faiss_cols, 'w') as f:
           f.write('\n'.join(self.x_columns))    