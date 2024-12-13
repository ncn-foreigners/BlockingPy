(configuration_tuning)=
# Configuration and Tuning

## Overview

BlockingPy provides two main configuration interfaces:

- control_txt: Text processing parameters
- control_ann: ANN algorithm parameters

## Text Processing Configuration (`control_txt`)

The `control_txt` dictionary controls how text data is processed before blocking:

```python
control_txt = {
    'n_shingles': 2,           # Size of character n-grams
    'max_features': 5000,      # Maximum number of features to keep
    'lowercase': True,         # Convert text to lowercase
    'strip_non_alphanum': True # Remove non-alphanumeric characters
}
```
### Parameter Details

`n_shingles` (default: `2`)

- Controls the size of character n-grams
- Larger values capture more context but increase dimensionality
- Common values: 2-4
- Impact: Higher values more precise but slower


`max_features` (default: `5000`)

- Maximum number of features in the document-term matrix
- Controls memory usage and processing speed
- Higher values may improve accuracy but increase memory usage
- Adjust based on your dataset size and available memory


`lowercase` (default: `True`)

- Whether to convert text to lowercase
- Usually keep True for better matching
- Set to False if case is meaningful for your data


`strip_non_alphanum` (default: `True`)

- Remove non-alphanumeric characters
- Usually keep True for cleaner matching
- Set to False if special characters are important

NOTE: `control_txt` is used only if the input is `pd.Series` as the other options were already processed.

## ANN Algorithm Configuration (`control_ann`)

Each algorithm has its own set of parameters in the `control_ann` dictionary. Overall `control_ann` should be in the following structure:

```python
control_ann = {
    "faiss" : {
        # parameters here
    },
    "voyager" : {},
    "annoy" : {},
    "lsh" : {},
    "kd" : {},
    "hnsw": {},
    # you can specify only the dict of the algorithm you are using

    "algo" : "lsh" or "kd" # specify if using lsh or kd

}
```


### FAISS Configuration
```python
control_ann = {
    'faiss': {
        'distance': 'euclidean', # Distance metric
        'k_search': 30,          # Number of neighbors to search
        'path': None             # Optional path to save index
    }
}
```

**Supported distance metrics**:

- `euclidean` (default)
- `cosine`
- `inner_product`
- `l1`
- `manhattan`
- `linf`
- `canberra`
- `bray_curtis`
- `jensen_shannon`

For more information about `faiss` see [here](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances).

## Voyager Configuration

```python
control_ann = {
    'voyager': {
        'distance': 'cosine',   # Distance metric
        'k_search': 30,         # Number of neighbors to search
        'path': None,           # Optional path to save index
        'random_seed': 1,       # Random seed
        'M': 12,                # Number of connections per element
        'ef_construction': 200, # Size of dynamic candidate list (construction)
        'max_elements': 1,      # Maximum number of elements
        'num_threads': -1,      # Number of threads (-1 for auto)
        'query_ef': -1          # Query expansion factor (-1 for auto)
    }
}
```

**Supported distance metrics**:

- `cosine`
- `inner_product`
- `euclidean` (default)

For more information about `voyager` see [here](https://github.com/spotify/voyager).

## HNSW Configuration

```python
control_ann = {
    'hnsw': {
        'distance': 'cosine', # Distance metric
        'k_search': 30,       # Number of neighbors to search
        'n_threads': 1,       # Number of threads
        'path': None,         # Optional path to save index
        'M': 25,              # Number of connections per element
        'ef_c': 200,          # Size of dynamic candidate list (construction)
        'ef_s': 200           # Size of dynamic candidate list (search)
    }
}
```
**Supported distance metrics**:

- `cosine` (default)
- `l2`
- `euclidean` (same as l2)
- `ip` (Inner Product)

For more information about `hnsw` configuration see [here](https://github.com/nmslib/hnswlib).

## Annoy Configuration

```python
control_ann = {
    'annoy': {
        'distance': 'angular', # Distance metric
        'k_search': 30,        # Number of neighbors to search
        'path': None,          # Optional path to save index
        'seed': None,          # Random seed
        'n_trees': 250,        # Number of trees
        'build_on_disk': False # Build index on disk
    }
}
```
**Supported distance metrics**:

- `angular`(default)
- `dot`
- `hamming`
- `manhattan`
- `euclidean`

For more information about `annoy` configuratino see [here](https://github.com/spotify/annoy).

## LSH Configuration

```python
control_ann = {
    'lsh': {
        'k_search': 30,        # Number of neighbors to search
        'seed': None,          # Random seed
        'bucket_size': 500,    # Hash bucket size
        'hash_width': 10.0,    # Hash function width
        'num_probes': 0,       # Number of probes
        'projections': 10,     # Number of projections
        'tables': 30           # Number of hash tables
    }
}
```
For more information about `lsh` see [here](https://github.com/mlpack).

### K-d Tree Configuration

```python
control_ann = {
    'kd': {
        'k_search': 30,           # Number of neighbors to search
        'seed': None,             # Random seed
        'algorithm': 'dual_tree', # Algorithm type
        'leaf_size': 20,          # Leaf size for tree
        'random_basis': False,    # Use random basis
        'rho': 0.7,               # Overlapping size
        'tau': 0.0,               # Early termination parameter
        'tree_type': 'kd',        # Type of tree to use
        'epsilon': 0.0            # Search approximation parameter
    }
}
```

For more information about `kd` see [here](https://github.com/mlpack).

## NND Configuration

```python
control_ann = {
    'nnd': {
        'metric': 'euclidean',  # Distance metric
        'k_search': 30,         # Number of neighbors to search
        'n_threads': None,      # Number of threads
        'leaf_size': None,      # Leaf size for tree building
        'n_trees': None,        # Number of trees
        'diversify_prob': 1.0,  # Probability of including diverse neighbors
        'low_memory': True      # Use low memory mode
    }
}
```

For more information about `nnd` see [here](https://pynndescent.readthedocs.io/en/latest/api.html).