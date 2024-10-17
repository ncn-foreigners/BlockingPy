import numpy as np
import pandas as pd
from scipy import sparse
import os
from typing import Optional, Union, List

def validate_input(#x: Union[np.ndarray, pd.DataFrame, sparse.csr_matrix],
                   x,
                   ann: str,
                   distance: str,
                   ann_write: Optional[str],
                   ):
    """
    Validate input parameters for the block method in the Blocker class.

    Args:
        ???????x (Union[np.ndarray, pd.DataFrame, sparse.csr_matrix]): Reference data
        ann (str): Approximate Nearest Neighbor algorithm
        distance (str) : Distance metric
        ann_write (Optional[str]): Path to write ANN index

    Raises:
        ValueError: If any input validation fails
    """
    #if not (isinstance(x, (str, np.ndarray)) or sparse.issparse(x) or isinstance(x, pd.DataFrame)):
        #raise ValueError("Only character, dense or sparse (csr_matrix) matrix x is supported")
    
    if ann_write is not None and not os.path.exists(os.path.dirname(ann_write)):
        raise ValueError("Path provided in the `ann_write` is incorrect")
    
    if ann == "hnsw" and distance not in ["l2", "euclidean", "cosine", "ip"]:
        raise ValueError("Distance for HNSW should be `l2, euclidean, cosine, ip`")
    
    if ann == "annoy" and distance not in ["euclidean", "manhattan", "hamming", "angular"]:
        raise ValueError("Distance for Annoy should be `euclidean, manhattan, hamming, angular`")
    
def validate_true_blocks(true_blocks: Optional[pd.DataFrame],
                         deduplication: bool):
    """
    Validate true_blocks input parameter for the block method in the Blocker class.

    Args:
        true_blocks (Optional[pd.DataFrame]): True blocking information for evaluation
        deduplication (bool): Whether deduplication is being performed

    Raises:
        ValueError: If true_blocks validation fails
    """
    if true_blocks is not None:
        if not deduplication:
            if len(true_blocks.columns) != 3 or not all(col in true_blocks.columns for col in ["x", "y", "block"]):
                raise ValueError("`true blocks` should be a DataFrame with columns: x, y, block")
        else:
            if len(true_blocks.columns) != 2 or not all(col in true_blocks.columns for col in ["x", "block"]):
                raise ValueError("`true blocks` should be a DataFrame with columns: x, block")
    