"""Fixtures for blockingpy tests."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from blockingpy import Blocker
from blockingpy.annoy_blocker import AnnoyBlocker
from blockingpy.data_handler import DataHandler

# from blockingpy.faiss_blocker import FaissBlocker
from blockingpy.hnsw_blocker import HNSWBlocker

# from blockingpy.mlpack_blocker import MLPackBlocker
from blockingpy.nnd_blocker import NNDBlocker
from blockingpy.voyager_blocker import VoyagerBlocker


@pytest.fixture
def blocker():
    """Create Blocker instance for each test."""
    return Blocker()


@pytest.fixture
def mlpack_blocker():
    """Create MLPackBlocker instance for each test."""
    pytest.importorskip("mlpack", reason="mlpack not installed")
    from blockingpy.mlpack_blocker import MLPackBlocker

    return MLPackBlocker()


@pytest.fixture
def nnd_blocker():
    """Create NNDBlocker instance for each test."""
    return NNDBlocker()


@pytest.fixture
def voyager_blocker():
    """Create VoyagerBlocker instance for each test."""
    return VoyagerBlocker()


@pytest.fixture
def faiss_blocker():
    """Create FaissBlocker instance for each test."""
    pytest.importorskip("faiss", reason="FAISS not installed")
    from blockingpy.faiss_blocker import FaissBlocker

    return FaissBlocker()


@pytest.fixture
def hnsw_blocker():
    """Create HNSWBlocker instance for each test."""
    return HNSWBlocker()


@pytest.fixture
def annoy_blocker():
    """Create AnnoyBlocker instance for each test."""
    return AnnoyBlocker()


@pytest.fixture
def small_sparse_data():
    """Create small sparse test datasets."""
    rng = np.random.default_rng()
    x = sparse.csr_matrix(rng.random((5, 3)))
    y = sparse.csr_matrix(rng.random((5, 3)))
    return DataHandler(data=x, cols=[f"x_col_{i}" for i in range(x.shape[1])]), DataHandler(
        data=y, cols=[f"y_col_{i}" for i in range(y.shape[1])]
    )


@pytest.fixture
def large_sparse_data():
    """Create larger sparse test datasets."""
    np.random.default_rng(42)
    x = sparse.random(2000, 20, density=0.1, format="csr")
    y = sparse.random(1000, 20, density=0.1, format="csr")
    return DataHandler(data=x, cols=[f"x_col_{i}" for i in range(x.shape[1])]), DataHandler(
        data=y, cols=[f"y_col_{i}" for i in range(y.shape[1])]
    )


@pytest.fixture
def small_named_csr_data():
    """Create small sparse test datasets with column names."""
    rng = np.random.default_rng()
    x = sparse.csr_matrix(rng.random((5, 3)))
    y = sparse.csr_matrix(rng.random((5, 3)))
    x_cols = ["ja", "nk", "ko"]
    y_cols = ["ja", "nk", "ok"]
    return x, y, x_cols, y_cols


@pytest.fixture
def small_named_ndarray_data():
    """Create small ndarray test datasets with column names."""
    rng = np.random.default_rng(42)
    x = rng.random((5, 3))
    y = rng.random((5, 3))
    x_cols = ["ja", "nk", "ko"]
    y_cols = ["ja", "nk", "ok"]
    return x, y, x_cols, y_cols


@pytest.fixture
def small_named_txt_data():
    """Create small text test datasets."""
    x = pd.DataFrame(
        {
            "txt": [
                "jankowalski",
                "kowalskijan",
                "kowalskimjan",
                "kowaljan",
                "montypython",
                "pythonmonty",
                "cyrkmontypython",
                "monty",
            ]
        }
    )

    y = pd.DataFrame(
        {
            "txt": [
                "montypython",
                "kowalskijan",
                "other",
            ]
        }
    )
    return x, y


@pytest.fixture
def identical_sparse_data():
    """Create sparse datasets with identical points."""
    data = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    x = sparse.csr_matrix(data)
    y = sparse.csr_matrix(data[0:1])
    return DataHandler(data=x, cols=[f"x_col_{i}" for i in range(x.shape[1])]), DataHandler(
        data=y, cols=[f"y_col_{i}" for i in range(y.shape[1])]
    )


@pytest.fixture
def single_sparse_point():
    """Create sparse single point datasets."""
    x = sparse.csr_matrix([[1.0, 2.0]])
    y = sparse.csr_matrix([[1.5, 2.5]])
    return DataHandler(data=x, cols=[f"x_col_{i}" for i in range(x.shape[1])]), DataHandler(
        data=y, cols=[f"y_col_{i}" for i in range(y.shape[1])]
    )


@pytest.fixture
def lsh_controls():
    """Default LSH control parameters."""
    return {
        "algo": "lsh",
        "lsh": {
            "seed": 42,
            "k_search": 5,
            "bucket_size": 500,
            "hash_width": 10.0,
            "num_probes": 0,
            "projections": 10,
            "tables": 30,
        },
    }


@pytest.fixture
def kd_controls():
    """Default k-d tree control parameters."""
    return {
        "algo": "kd",
        "kd": {
            "seed": 42,
            "k_search": 5,
            "algorithm": "dual_tree",
            "leaf_size": 20,
            "tree_type": "kd",
            "epsilon": 0.0,
            "rho": 0.7,
            "tau": 0.0,
            "random_basis": False,
        },
    }


def _has_faiss_gpu() -> bool:
    try:
        import faiss

        return hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
    except Exception:
        return False


def _has_mlpack() -> bool:
    try:

        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    have_gpu = _has_faiss_gpu()
    have_ml = _has_mlpack()
    for item in items:
        if "requires_faiss_gpu" in item.keywords and not have_gpu:
            item.add_marker(pytest.mark.skip(reason="FAISS GPU not available"))
        if "requires_mlpack" in item.keywords and not have_ml:
            item.add_marker(pytest.mark.skip(reason="mlpack not installed"))
