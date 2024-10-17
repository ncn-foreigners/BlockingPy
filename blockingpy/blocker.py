import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional, Union, List, Dict, Any
import logging

from .annoy_blocker import AnnoyBlocker
from .hnsw_blocker import HNSWBlocker
from .mlpack_blocker import MLPackBlocker
from .nnd_blocker import NNDBlocker
from .helper_functions import validate_input, validate_true_blocks

class Blocker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def block(self, 
              #x: Union[np.ndarray, pd.DataFrame, sparse.csr_matrix],
              x,
              y: Optional[Union[np.ndarray, pd.DataFrame, sparse.csr_matrix]] = None,
              deduplication: bool = True,
              on: Optional[List[str]] = None,
              on_blocking: Optional[List[str]] = None,
              ann: str = "nnd",
              distance: Optional[str] = None,
              ann_write: Optional[str] = None,
              ann_colnames: Optional[List[str]] = None,
              true_blocks: Optional[pd.DataFrame] = None,
              verbose: int = 0,
              graph: bool = False,
              seed: int = 2023,
              n_threads: int = 1,
              control_txt: Dict[str, Any] = None,
              control_ann: Dict[str, Any] = None):
        
        if distance is None:
            distance = {  
                "nnd": "cosine",
                "hnsw": "cosine",
                "annoy": "angular",
                "lsh": None,
                "kd": None
            }.get(ann)

        validate_input(x, ann, distance, ann_write)

        if y is not None:
            deduplication = False
            y_default = False
            k = 1
        else :
            y_default = y
            y = x
            k = 2
        
        validate_true_blocks(true_blocks, deduplication)
    