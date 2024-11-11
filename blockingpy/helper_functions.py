"""Contains helper functions for blocking operations such as input validation and Document Term Matrix (DTM) creation."""

import numpy as np
import pandas as pd
from scipy import sparse
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
from typing import Optional, Union

def validate_input(x: Union[pd.Series, sparse.csr_matrix, np.ndarray],
                   ann: str,
                   distance: str,
                   ):
    """
    Validate input parameters for the block method in the Blocker class.

    Parameters
    ----------
    x : pandas.Series or scipy.sparse.csr_matrix or numpy.ndarray
        Reference data for blocking
    ann : str
        Approximate Nearest Neighbor algorithm name
    distance : str
        Distance metric name

    Raises
    ------
    ValueError
        If any of the following conditions are met:
        - Input x is not a supported type
        - Distance metric is not supported for the chosen algorithm

    Notes
    -----
    Supported distance metrics per algorithm:
    - HNSW: l2, euclidean, cosine, ip
    - Annoy: euclidean, manhattan, hamming, angular
    - Voyager: euclidean, cosine, inner_product
    - FAISS: euclidean, l2, inner_product, cosine, l1, manhattan, linf, 
             canberra, bray_curtis, jensen_shannon
    """
    if not (isinstance(x, np.ndarray) or sparse.issparse(x) or isinstance(x, pd.Series)):
        raise ValueError("Only dense (np.ndarray) or sparse (csr_matrix) matrix or pd.Series with str data is supported")
    
    if ann == "hnsw" and distance not in ["l2", "euclidean", "cosine", "ip"]:
        raise ValueError("Distance for HNSW should be `l2, euclidean, cosine, ip`")
    
    if ann == "annoy" and distance not in ["euclidean", "manhattan", "hamming", "angular"]:
        raise ValueError("Distance for Annoy should be `euclidean, manhattan, hamming, angular`")
    
    if ann == "voyager" and distance not in ["euclidean", "cosine", "inner_product"]:
        raise ValueError("Distance for Voyager should be `euclidean, cosine, inner_product`")
    
    if ann == "faiss" and distance not in ["euclidean", "l2", "inner_product", "cosine", "l1", "manhattan", "linf", "canberra", "bray_curtis", "jensen_shannon"]:
        raise ValueError("Distance for Faiss should be `euclidean, l2, inner_product, cosine, l1, manhattan, linf, canberra, bray_curtis, jensen_shannon`")
    
def validate_true_blocks(true_blocks: Optional[pd.DataFrame],
                         deduplication: bool):
    """
    Validate true_blocks input parameter for the block method in the Blocker class.

    Parameters
    ----------
    true_blocks : pandas.DataFrame, optional
        True blocking information for evaluation
    deduplication : bool
        Whether deduplication is being performed

    Raises
    ------
    ValueError
        If true_blocks validation fails due to:
        - Missing required columns for regular blocking (x, y, block)
        - Missing required columns for deduplication (x, block)

    Notes
    -----
    Required columns:
    - For regular blocking: 'x', 'y', 'block'
    - For deduplication: 'x', 'block'
    """
    if true_blocks is not None:
        if not deduplication:
            if len(true_blocks.columns) != 3 or not all(col in true_blocks.columns for col in ["x", "y", "block"]):
                raise ValueError("`true blocks` should be a DataFrame with columns: x, y, block")
        else:
            if len(true_blocks.columns) != 2 or not all(col in true_blocks.columns for col in ["x", "block"]):
                raise ValueError("`true blocks` should be a DataFrame with columns: x, block")
    
def tokenize_character_shingles(text, n=2, lowercase=True, strip_non_alphanum=True):
    """
    Generate character n-grams (shingles) from input text.

    Parameters
    ----------
    text : str
        Input text to tokenize
    n : int, optional
        Size of character n-grams (default 2)
    lowercase : bool, optional
        Whether to convert text to lowercase (default True)
    strip_non_alphanum : bool, optional
        Whether to remove non-alphanumeric characters (default True)

    Returns
    -------
    list of str
        List of character n-grams

    Raises
    ------
    TypeError
        If input text is not a string

    Examples
    --------
    >>> tokenize_character_shingles("Hello", n=2)
    ['he', 'el', 'll', 'lo']

    Notes
    -----
    The function processes text in the following order:
    1. Converts to lowercase (if requested)
    2. Removes non-alphanumeric characters (if requested)
    3. Generates n-character shingles
    """
    
    if not isinstance(text, str):
        raise TypeError(f"Expected string input, got {type(text)}")
    
    if lowercase:
        text = text.lower()
    if strip_non_alphanum:
        text = re.sub(r'[^a-z0-9]+', '', text)  
    shingles = [''.join(gram) for gram in ngrams(text, n)]
    return shingles

def create_sparse_dtm(x: pd.Series, control_txt: dict, verbose: bool = False):
    """
    Create a sparse document-term matrix from input texts.

    Parameters
    ----------
    x : pandas.Series
        Input texts to process
    control_txt : dict
        Configuration dictionary with keys:
        - n_shingles : int
            Size of character n-grams
        - lowercase : bool
            Whether to convert text to lowercase
        - strip_non_alphanum : bool
            Whether to remove non-alphanumeric characters
        - max_features : int
            Maximum number of features to keep
    verbose : bool, optional
        Whether to print additional information (default False)

    Returns
    -------
    pandas.DataFrame
        Sparse dataframe containing the document-term matrix

    Notes
    -----
    The function uses CountVectorizer from scikit-learn with custom
    tokenization based on character n-grams. The resulting matrix
    is stored as a sparse pandas DataFrame.

    Examples
    --------
    >>> texts = pd.Series(['hello world', 'hello there'])
    >>> controls = {'n_shingles': 2, 'lowercase': True, 
    ...            'strip_non_alphanum': True, 'max_features': 100}
    >>> dtm = create_sparse_dtm(texts, controls)
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