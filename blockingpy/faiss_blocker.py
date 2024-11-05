import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional
import sys
from .base import BlockingMethod
import faiss
import os

class FaissBlocker(BlockingMethod):
   """
   A class for performing blocking using the FAISS algorithm.
   For details see: https://github.com/facebookresearch/faiss

   Attributes:
       index (Optional[IndexFlat]): The FAISS index used for nearest neighbor search.
       logger (logging.Logger): Logger for the class.
       x_columns: column names of x

   The main method of this class is `block()`, which performs the actual
   blocking operation. Use the `controls` parameter in the `block()` method 
   to fine-tune the algorithm's behavior.

   This class inherits from the BlockingMethod abstract base class and
   implements its `block()` method.
   """
   METRIC_MAP: Dict[str, any] = {
         "euclidean": faiss.METRIC_L2,
         'l2': faiss.METRIC_L2,
         "inner_product": faiss.METRIC_INNER_PRODUCT,
         "cosine": faiss.METRIC_INNER_PRODUCT, #note: later handled by vector normalisation (https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
         "l1" : faiss.METRIC_L1,
         "manhattan" : faiss.METRIC_L1,
         "linf" : faiss.METRIC_Linf,
         #"lp" : faiss.METRIC_Lp,
        "canberra" : faiss.METRIC_Canberra, #note: requires smoothing since 0/0 is undefined
        "bray_curtis" : faiss.METRIC_BrayCurtis,
        "jensen_shannon" : faiss.METRIC_JensenShannon, #note: requires smoothing since log(0) is undefined
    }
   
   def __init__(self):
       self.index: Optional[faiss.IndexFlat] = None
       self.logger = logging.getLogger(__name__)
       self.x_columns = None
       console_handler = logging.StreamHandler(sys.stdout)
       self.logger.addHandler(console_handler)

   def block(self, x: pd.DataFrame, 
             y: pd.DataFrame, 
             k: int,
             verbose: Optional[bool], 
             controls: Dict[str, Any]) -> pd.DataFrame:
       """
       Perform blocking using FAISS algorithm.

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

       self.logger.info("Building index...")
       self.index.add(x=x)    
         
       self.logger.info("Querying index...")

       if k_search > x.shape[0]:
            original_k_search = k_search
            k_search = min(k_search, x.shape[0])
            self.logger.warning(f"k_search ({original_k_search}) is larger than the number of reference points ({x.shape[0]}). Adjusted k_search to {k_search}.")

       distances, indices = self.index.search(x=y, k=k_search)

       if path:
           self._save_index(path)

       result = {
           'y': np.arange(y.shape[0]),
           'x': indices[:, k-1],
           'dist': distances[:, k-1]
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
       Save the FAISS index and column names to files.

       Args:
           path (str): Directory path where the files will be saved.

       Notes:
           Creates two files:
           1. 'index.faiss': The FAISS index file.
           2. 'index-colnames.txt': A text file with column names.
       """
       path_faiss = os.path.join(path, "index.faiss")
       path_faiss_cols = os.path.join(path, "index-colnames.txt")

       self.logger.info(f"Writing index to {path_faiss}")
       
       faiss.write_index(self.index, path_faiss)

       with open(path_faiss_cols, 'w') as f:
           f.write('\n'.join(self.x_columns))    