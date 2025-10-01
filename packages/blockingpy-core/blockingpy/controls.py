"""Contains controls for ANN algorithms and text processing operations."""

from copy import deepcopy
from typing import Any


def deep_update(base_dict: dict, update_dict: dict) -> dict:
    """
    Update nested dictionaries while preserving default values.

    Parameters
    ----------
    base_dict : dict
        The base dictionary containing default values
    update_dict : dict
        The dictionary containing values to update

    Returns
    -------
    dict
        Updated dictionary with preserved nested structure

    Notes
    -----
    This function performs a deep copy and recursive update of nested dictionaries,
    ensuring that default values are preserved when not explicitly overridden.

    """
    result = deepcopy(base_dict)

    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def controls_ann(controls: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """
    Create configuration dictionary for Approximate Nearest Neighbor algorithms.

    This function provides a centralized configuration for multiple ANN algorithms,
    with sensible defaults and easy override capabilities.

    Parameters
    ----------
    controls : dict
        Dictionary of control parameters to override defaults
    **kwargs : dict
        Additional keyword arguments for direct parameter updates

    Returns
    -------
    dict
        Configuration dictionary with the following structure:
        {
            'random_seed': int,
            'nnd': {
                'metric': str,
                'k_search': int,
                'metric_kwds': dict or None,
                'n_threads': int or None,
                ...
            },
            'hnsw': {
                'distance': str,
                'n_threads': int,
                'M': int,
                ...
            },
            'lsh': {...},
            'kd': {...},
            'annoy': {...},
            'voyager': {...},
            'faiss': {...},
            'gpu_faiss': {...},
            'algo': str ['lsh' or 'kd']
        }

    Notes
    -----
    Supported algorithms and their documentation:
    - NND: https://github.com/lmcinnes/pynndescent
    - HNSW: https://github.com/nmslib/hnswlib
    - Annoy: https://github.com/spotify/annoy
    - LSH and KD: https://github.com/mlpack/mlpack
    - Voyager: https://github.com/spotify/voyager
    - FAISS: https://github.com/facebookresearch/faiss

    Examples
    --------
    >>> config = controls_ann(hnsw={"M": 30, "ef_c": 300})

    """
    defaults = {
        "random_seed": None,
        "nnd": {
            "metric": "euclidean",
            "k_search": 30,
            "metric_kwds": None,
            "n_threads": None,
            "tree_init": True,
            "n_trees": None,
            "leaf_size": None,
            "pruning_degree_multiplier": 1.5,
            "diversify_prob": 1.0,
            "init_graph": None,
            "init_dist": None,
            "low_memory": True,
            "max_candidates": None,
            "max_rptree_depth": 100,
            "n_iters": None,
            "delta": 0.001,
            "compressed": False,
            "parallel_batch_queries": False,
            "epsilon": 0.1,
        },
        "hnsw": {
            "k_search": 30,
            "distance": "cosine",
            "n_threads": 1,
            "path": None,
            "M": 25,
            "ef_c": 200,
            "ef_s": 200,
        },
        "lsh": {
            "k_search": 30,
            "bucket_size": 500,
            "hash_width": 10.0,
            "num_probes": 0,
            "projections": 10,
            "tables": 30,
        },
        "kd": {
            "k_search": 30,
            "algorithm": "dual_tree",
            "epsilon": 0.0,
            "leaf_size": 20,
            "random_basis": False,
            "rho": 0.7,
            "tau": 0.0,
            "tree_type": "kd",
        },
        "annoy": {
            "k_search": 30,
            "path": None,
            "distance": "angular",
            "n_trees": 250,
            "build_on_disk": False,
        },
        "voyager": {
            "k_search": 30,
            "path": None,
            "distance": "cosine",
            "M": 12,
            "ef_construction": 200,
            "max_elements": 1,
            "num_threads": -1,
            "query_ef": -1,
        },
        "faiss": {
            "index_type": "hnsw",
            "k_search": 30,
            "distance": "cosine",
            "path": None,
            "hnsw_M": 32,
            "hnsw_ef_construction": 200,
            "hnsw_ef_search": 200,
            "lsh_nbits": 2,
            "lsh_rotate_data": True,
        },
        "gpu_faiss": {
            "index_type": "flat",
            "k_search": 30,
            "distance": "cosine",
            "path": None,
            "ivf_nlist": 100,
            "ivf_nprobe": 10,
            "ivfpq_nlist": 100,
            "ivfpq_m": 8,
            "ivfpq_nbits": 8,
            "ivfpq_nprobe": 10,
            "ivfpq_useFloat16": False,
            "ivfpq_usePrecomputed": False,
            "ivfpq_reserveVecs": 0,
            "ivfpq_use_cuvs": False,
            "cagra": {
                "graph_degree": 64,
                "intermediate_graph_degree": 128,
                "build_algo": "ivf_pq",
                "nn_descent_niter": 20,
                "itopk_size": 64,
                "max_queries": 0,
                "algo": "auto",
                "team_size": 0,
                "search_width": 1,
                "min_iterations": 0,
                "max_iterations": 0,
                "thread_block_size": 0,
                "hashmap_mode": "auto",
                "hashmap_min_bitlen": 0,
                "hashmap_max_fill_rate": 0.5,
                "num_random_samplings": 1,
                "seed": 0x128394,
            },
        },
        "algo": "lsh",
    }

    updates = {}
    if controls is not None:
        updates.update(controls)
    if kwargs:
        updates.update(kwargs)

    return deep_update(defaults, updates)


def controls_txt(controls: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """
    Create configuration dictionary for text processing operations.

    Parameters
    ----------
    controls : dict
        Dictionary of control parameters to override defaults
    **kwargs : dict
        Additional keyword arguments for direct parameter updates

    Returns
    -------
    dict
        Configuration dictionary with the following structure:
        {
            'encoder': str,
            'embedding': {
                'model': str,
                'normalize': bool,
                'max_length': int,
                'emb_batch_size': int,
                'show_progress_bar': bool,
                'use_multiprocessing': bool,
                'multiprocessing_threshold': int,
            },
            'shingle': {
                'n_shingles': int,
                'lowercase': bool,
                'strip_non_alphanum': bool,
                'max_features': int,
            },
        }

    Notes
    -----
    Configuration options:
    - encoder: Type of text encoder ('shingle' or 'embedding')
    For 'embedding', additional parameters are required:
        - model: Pretrained model identifier or path
        - normalize: Normalize output vectors if True
        - max_length: Maximum sequence length for encoding
        - emb_batch_size: Batch size for encoding
        - show_progress_bar: Show progress bar if True
        - use_multiprocessing: Use multiprocessing if True
        - multiprocessing_threshold: Threshold for multiprocessing
    For 'shingle', additional parameters are required:
        - n_shingles: Number of consecutive characters to combine
        - max_features: Maximum number of features to keep
        - lowercase: Convert text to lowercase if True
        - strip_non_alphanum: Remove non-alphanumeric characters if True

    """
    defaults = {
        "encoder": "shingle",
        "embedding": {
            "model": "minishlab/potion-base-8M",
            "normalize": True,
            "max_length": 512,
            "emb_batch_size": 1024,
            "show_progress_bar": False,
            "use_multiprocessing": True,
            "multiprocessing_threshold": 10000,
        },
        "shingle": {
            "n_shingles": 2,
            "lowercase": True,
            "strip_non_alphanum": True,
            "max_features": 5000,
        },
    }

    updates = {}
    if controls is not None:
        updates.update(controls)
    if kwargs:
        updates.update(kwargs)

    return deep_update(defaults, updates)
