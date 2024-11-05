from typing import Dict, Any
from copy import deepcopy

def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """
    Update controls dictionaries while leaving default values intact.
    
    Args:
        base_dict (Dict): The base dictionary with default values
        update_dict (Dict): The dictionary with values to update       
    Returns:
        Dict: Updated dictionary with preserved nested structure
    """
    result = deepcopy(base_dict)
    
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result

def controls_ann(controls: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
    """
    Create controls dictionary for ANN algorithms.
    Handles nested dictionary updates while preserving defaults.

    For details on each algorithm's parameters, see:
    - NND: https://github.com/lmcinnes/pynndescent
    - HNSW: https://github.com/nmslib/hnswlib
    - Annoy: https://github.com/spotify/annoy
    - LSH and KD: https://github.com/mlpack/mlpack
    - Voyager: https://github.com/spotify/voyager
    - FAISS: https://github.com/facebookresearch/faiss (cpu only for now)

    Args:
        controls (Dict[str, Any], optional): Dictionary of controls
        **kwargs: Additional keyword arguments
    
    Returns:
        (Dict): Configuration dictionary for ANN algorithms with default values
    """
    defaults = {
        'nnd': {
            'metric': 'euclidean',
            'k_search': 30,
            'metric_kwds': None,
            'n_threads': None,
            'tree_init': True,
            'n_trees': None,
            'leaf_size': None,
            'pruning_degree_multiplier': 1.5,
            'diversify_prob': 1.0,
            'init_graph': None,
            'init_dist': None,
            'low_memory': True,
            'max_candidates': None,
            'max_rptree_depth': 100,
            'n_iters': None,
            'delta': 0.001,
            'compressed': False,
            'parallel_batch_queries': False,

            'epsilon': 0.1
        },
        'hnsw': {
            'distance': 'cosine',
            'n_threads': 1,
            'path': None,
            'M': 25,
            'ef_c': 200,
            'ef_s': 200,
        },
        'lsh': {
            'k_search': 30,
            'seed': None,
            'bucket_size': 500,
            'hash_width': 10.0,
            'num_probes': 0,
            'projections': 10,
            'tables': 30
        },
        'kd': {
            'k_search': 30,
            'seed': None,
            'algorithm': "dual_tree",
            'epsilon': 0,
            'leaf_size': 20,
            'random_basis': False,
            'rho': 0.7,
            'tau': 0,
            'tree_type': "kd"
        },
        'annoy': {
            'k_search': 30,
            'path': None,
            'seed': None,
            'distance': 'angular',
            'n_trees': 250,
            'build_on_disk': False
        },
        'voyager': {
            'k_search': 30,
            'path': None,
            'random_seed': 1,
            'distance': 'cosine',
            'M' : 12,
            'ef_construction': 200,
            'max_elements': 1,
            'num_threads': -1,
            'query_ef': -1

        },
        'faiss': {
            # to my knowledge, faiss only allows these parameters to be set
            'k_search': 30,
            'distance': 'euclidean',
            #'use_gpu': False, NOT SUPPORTED YET
            #'use_mutltiple_gpus': False, NOT SUPPORTED YET
            'path': None
        },
        'algo': 'lsh',
    }
    
    updates = {}
    if controls is not None:
        updates.update(controls)
    if kwargs:
        updates.update(kwargs)
    
    return deep_update(defaults, updates)

def controls_txt(controls: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
    """
    Create controls dictionary for text processing.

    Args:
        controls (Dict[str, Any], optional): Dictionary of controls
        **kwargs: Additional keyword arguments

        n_shingles (int): Number of shingles.
        n_chunks (int): Number of chunks.
        lowercase (bool): Wheather to convert text to lowercase.
        strip_non_alphanum (bool): Wheather to strip non-alphanumeric characters.
    
    Returns:
        (Dict): Configuration dictionary for text processing with default values
    """
    defaults = {
        'n_shingles': 2,
        'max_features': 5000,
        'lowercase': True,
        'strip_non_alphanum': True
    }
    
    updates = {}
    if controls is not None:
        updates.update(controls)
    if kwargs:
        updates.update(kwargs)
    
    return deep_update(defaults, updates)

# Example usage:
if __name__ == "__main__":
    config_dict = {
        'nnd': {
            'metric': 'euclidean',
            'n_threads': None,
            'tree_init': True,
            'n_trees': None,
            'leaf_size': None,
        
        },
        'algo': 'lsh'
    }
    custom_config = controls_ann(config_dict) 
    
    print("\nNND configuration:")
    for k, v in custom_config['nnd'].items():
        print(f"{k}: {v}")
    print(f"Algorithm: {custom_config['algo']}")
    
    another_config = controls_ann(hnsw={
        'M': 30,
        'ef_c': 300, 
    })
    
    print("\nHNSW configuration:")
    print(another_config['hnsw'])
    
    mixed_config = controls_ann(
        {
            'nnd': {'metric': 'cosine'}
        },
        algo='nnd'
    )