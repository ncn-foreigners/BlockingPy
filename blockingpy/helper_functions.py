import numpy as np
import pandas as pd
from scipy import sparse
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
from typing import Optional, Union, List

def validate_input(x: Union[pd.Series, sparse.csr_matrix, np.ndarray, np.array, List[str]],
                   ann: str,
                   distance: str,
                   ann_write: Optional[str],
                   ):
    """
    Validate input parameters for the block method in the Blocker class.

    Args:
        x (Union[pd.Series, sparse.csr_matrix, np.ndarray, np.array, List[str]]): Reference data
        ann (str): Approximate Nearest Neighbor algorithm
        distance (str) : Distance metric
        ann_write (Optional[str]): Path to write ANN index

    Raises:
        ValueError: If any input validation fails
    """
    if not (isinstance(x, (str, np.ndarray)) or sparse.issparse(x) or isinstance(x, pd.Series)) or (isinstance(x, list) and all(isinstance(i, str) for i in x)):
        raise ValueError("Only character, dense (np.ndarray) or sparse (csr_matrix) matrix or pd.Series with str data is supported")
    
    if ann_write is not None and not os.path.exists(os.path.dirname(ann_write)):
        raise ValueError("Path provided in the `ann_write` is incorrect")
    
    if ann == "hnsw" and distance not in ["l2", "euclidean", "cosine", "ip"]:
        raise ValueError("Distance for HNSW should be `l2, euclidean, cosine, ip`")
    
    if ann == "annoy" and distance not in ["euclidean", "manhattan", "hamming", "angular"]:
        raise ValueError("Distance for Annoy should be `euclidean, manhattan, hamming, angular`")
    
    if ann == "voyager" and distance not in ["euclidean", "cosine", "inner_product"]:
        raise ValueError("Distance for Voyager should be `euclidean, cosine, inner_product`")
    
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
    
def tokenize_character_shingles(text, n=3, lowercase=True, strip_non_alphanum=True):
    """
    Generate character n-grams (shingles) from input text.
    
    Args:
        text (str): Input text to tokenize
        n (int): Size of character n-grams
        lowercase (bool): Whether to convert text to lowercase
        strip_non_alphanum (bool): Whether to remove non-alphanumeric characters
        
    Returns:
        List[str]: List of character n-grams
    """
    
    if not isinstance(text, str):
        raise TypeError(f"Expected string input, got {type(text)}")
    
    if lowercase:
        text = text.lower()
    if strip_non_alphanum:
        text = re.sub(r'[^a-z0-9]+', '', text)  
    shingles = [''.join(gram) for gram in ngrams(text, n)]
    return shingles

def create_sparse_dtm(x: Union[List[str], pd.Series], control_txt: dict, verbose: bool = False):
    """
    Create a sparse document-term matrix from input texts.
    
    Args:
        x (Union[List[str], pd.Series]): Input texts
        control_txt (dict): Configuration dictionary. For details see: blockingpy/controls.py
        verbose (bool): Whether to print additional information
        
    Returns:
        pd.DataFrame: Sparse dataframe containing the document-term matrix
    """
    x = x.tolist() if isinstance(x, pd.Series) else x

    vectorizer = CountVectorizer(
        tokenizer=lambda x: tokenize_character_shingles(
            x, 
            n=control_txt.get('n_shingles'), 
            lowercase=control_txt.get('lowercase'), 
            strip_non_alphanum=control_txt.get('strip_non_alphanum')
        ),
        max_features=control_txt.get('max_features'),
        token_pattern=None
    )
    x_dtm_sparse = vectorizer.fit_transform(x)
    x_voc = vectorizer.vocabulary_

    x_sparse_df = pd.DataFrame.sparse.from_spmatrix(
        x_dtm_sparse, columns=vectorizer.get_feature_names_out()
    )

    if verbose:
        print("Vocabulary:", x_voc)
        print("Sparse DataFrame shape:", x_sparse_df.shape)
        print("Sparse DataFrame:\n", x_sparse_df)

    return x_sparse_df