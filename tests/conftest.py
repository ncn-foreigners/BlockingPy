import numpy as np
import pandas as pd
import pytest
from scipy import sparse

@pytest.fixture
def blocker():
    """Create a fresh MLPackBlocker instance for each test."""
    from blockingpy.mlpack_blocker import MLPackBlocker
    return MLPackBlocker()

@pytest.fixture
def small_sparse_data():
    """Create small sparse test datasets."""
    np.random.seed(42)
    x = sparse.csr_matrix(np.random.rand(5, 3))
    y = sparse.csr_matrix(np.random.rand(3, 3))
    return pd.DataFrame.sparse.from_spmatrix(x), pd.DataFrame.sparse.from_spmatrix(y)

@pytest.fixture
def large_sparse_data():
    """Create larger sparse test datasets."""
    np.random.seed(42)
    x = sparse.random(100, 10, density=0.1, format='csr')
    y = sparse.random(50, 10, density=0.1, format='csr')
    return pd.DataFrame.sparse.from_spmatrix(x), pd.DataFrame.sparse.from_spmatrix(y)

@pytest.fixture
def identical_sparse_data():
    """Create sparse datasets with identical points."""
    data = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    x = sparse.csr_matrix(data)
    y = sparse.csr_matrix(data[0:1])
    return pd.DataFrame.sparse.from_spmatrix(x), pd.DataFrame.sparse.from_spmatrix(y)

@pytest.fixture
def single_sparse_point():
    """Create sparse single point datasets."""
    x = sparse.csr_matrix([[1.0, 2.0]])
    y = sparse.csr_matrix([[1.5, 2.5]])
    return pd.DataFrame.sparse.from_spmatrix(x), pd.DataFrame.sparse.from_spmatrix(y)

@pytest.fixture
def lsh_controls():
    """Default LSH control parameters."""
    return {
        'algo': 'lsh',
        'lsh': {
            'seed': 42,
            'k_search': 5,
            'bucket_size': 500,
            'hash_width': 10.0,
            'num_probes': 0,
            'projections': 10,
            'tables': 30
        }
    }

@pytest.fixture
def kd_controls():
    """Default k-d tree control parameters."""
    return {
        'algo': 'kd',
        'kd': {
            'seed': 42,
            'k_search': 5,
            'algorithm': "dual_tree",
            'leaf_size': 20,
            'tree_type': "kd",
            'epsilon': 0.0,
            'rho': 0.7,
            'tau': 0.0,
            'random_basis': False
        }
    }