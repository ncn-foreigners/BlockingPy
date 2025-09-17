"""Tests for the NND blocking algorithm."""

import logging

import numpy as np
import pandas as pd
import pytest

from blockingpy.data_handler import DataHandler


@pytest.fixture
def nnd_controls():
    """Default NND control parameters."""
    return {
        "nnd": {
            "metric": "euclidean",
            "k_search": 5,
            "metric_kwds": None,
            "n_threads": 1,
            "tree_init": True,
            "n_trees": 5,
            "leaf_size": 30,
            "pruning_degree_multiplier": 1.5,
            "diversify_prob": 1.0,
            "init_graph": None,
            "init_dist": None,
            "low_memory": True,
            "max_candidates": 50,
            "max_rptree_depth": 100,
            "n_iters": 10,
            "delta": 0.001,
            "compressed": False,
            "parallel_batch_queries": False,
            "epsilon": 0.1,
        }
    }


def test_basic_blocking(nnd_blocker, small_sparse_data, nnd_controls):
    """Test basic blocking functionality."""
    x, y = small_sparse_data

    result = nnd_blocker.block(x=x, y=y, k=1, verbose=False, controls=nnd_controls)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x", "y", "dist"}
    assert len(result) == y.shape[0]
    assert result["dist"].notna().all()


@pytest.mark.parametrize(
    "metric",
    ["euclidean", "manhattan", "cosine", "correlation", "hamming", "minkowski"],
)
def test_different_metrics(nnd_blocker, small_sparse_data, nnd_controls, metric):
    """Test different distance metrics."""
    x, y = small_sparse_data

    controls = nnd_controls.copy()
    controls["nnd"]["metric"] = metric

    result = nnd_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)
    assert result["dist"].notna().all()


def test_result_reproducibility(nnd_blocker, small_sparse_data, nnd_controls):
    """Test result reproducibility with same parameters."""
    x, y = small_sparse_data

    result1 = nnd_blocker.block(x=x, y=y, k=1, verbose=False, controls=nnd_controls)
    result2 = nnd_blocker.block(x=x, y=y, k=1, verbose=False, controls=nnd_controls)
    print(result1)

    pd.testing.assert_frame_equal(result1, result2)


def test_k_search_warning(nnd_blocker, small_sparse_data, nnd_controls):
    """Test warning when k_search is larger than reference points."""
    x, y = small_sparse_data

    nnd_controls["nnd"]["k_search"] = x.shape[0] + 10

    with pytest.warns(UserWarning, match=r"k_search.*larger.*reference"):
        nnd_blocker.block(x=x, y=y, k=1, verbose=True, controls=nnd_controls)


def test_verbose_logging(nnd_blocker, small_sparse_data, nnd_controls, caplog):
    """Test verbose logging output."""
    x, y = small_sparse_data
    caplog.set_level(logging.INFO)

    nnd_blocker.block(x=x, y=y, k=1, verbose=True, controls=nnd_controls)

    assert any("Initializing NND index" in record.message for record in caplog.records)
    assert any("Querying index" in record.message for record in caplog.records)
    assert any("Process completed successfully" in record.message for record in caplog.records)


def test_identical_points(nnd_blocker, identical_sparse_data, nnd_controls):
    """Test blocking with identical points."""
    x, y = identical_sparse_data

    result = nnd_blocker.block(x=x, y=y, k=1, verbose=False, controls=nnd_controls)
    assert result["dist"].iloc[0] == pytest.approx(0.0, abs=1e-5)


def test_single_point(nnd_blocker, single_sparse_point, nnd_controls):
    """Test blocking with single point."""
    x, y = single_sparse_point

    result = nnd_blocker.block(x=x, y=y, k=1, verbose=False, controls=nnd_controls)
    assert len(result) == 1


def test_empty_data_handling(nnd_blocker, nnd_controls):
    """Test handling of empty datasets."""
    rng = np.random.default_rng()
    x = DataHandler(data=rng.random((0, 3)), cols=["col1", "col2", "col3"])
    y = DataHandler(data=rng.random((5, 3)), cols=["col1", "col2", "col3"])

    with pytest.raises(ValueError):
        nnd_blocker.block(x=x, y=y, k=1, verbose=False, controls=nnd_controls)


@pytest.mark.parametrize(
    "param_variation",
    [
        {"n_trees": 10},
        {"leaf_size": 20},
        {"pruning_degree_multiplier": 2.0},
        {"diversify_prob": 0.8},
        {"max_candidates": 100},
        {"n_iters": 15},
        {"delta": 0.002},
    ],
)
def test_parameter_variations(nnd_blocker, small_sparse_data, nnd_controls, param_variation):
    """Test NND with different parameters."""
    x, y = small_sparse_data

    controls = nnd_controls.copy()
    controls["nnd"].update(param_variation)

    result = nnd_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)


def test_metric_with_params(nnd_blocker, small_sparse_data, nnd_controls):
    """Test metric with additional parameters."""
    x, y = small_sparse_data

    controls = nnd_controls.copy()
    controls["nnd"].update({"metric": "minkowski", "metric_kwds": {"p": 3}})

    result = nnd_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)
