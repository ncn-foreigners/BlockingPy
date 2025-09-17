"""Tests for the HNSW blocker."""

import logging
import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from blockingpy.data_handler import DataHandler


@pytest.fixture
def hnsw_controls():
    """Default HNSW parameters."""
    return {
        "hnsw": {
            "k_search": 5,
            "distance": "cosine",
            "n_threads": 1,
            "path": None,
            "M": 25,
            "ef_c": 200,
            "ef_s": 200,
        }
    }


def test_basic_blocking(hnsw_blocker, small_sparse_data, hnsw_controls):
    """Test basic blocking functionality."""
    x, y = small_sparse_data

    result = hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=hnsw_controls)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x", "y", "dist"}
    assert len(result) == y.shape[0]
    assert result["dist"].notna().all()


@pytest.mark.parametrize("distance", ["l2", "euclidean", "ip"])
def test_different_metrics(hnsw_blocker, small_sparse_data, hnsw_controls, distance):
    """Test different distance metrics in HNSW."""
    x, y = small_sparse_data

    controls = hnsw_controls.copy()
    controls["hnsw"]["distance"] = distance

    result = hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)
    assert result["dist"].notna().all()


def test_result_reproducibility(hnsw_blocker, small_sparse_data, hnsw_controls):
    """Test result reproducibility with same parameters."""
    x, y = small_sparse_data

    result1 = hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=hnsw_controls)
    result2 = hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=hnsw_controls)

    pd.testing.assert_frame_equal(result1, result2)


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_threading(hnsw_blocker, large_sparse_data, hnsw_controls, n_threads):
    """Test HNSW with different thread counts."""
    x, y = large_sparse_data

    controls = hnsw_controls.copy()
    controls["hnsw"]["n_threads"] = n_threads

    result = hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)
    assert result["dist"].notna().all()


@pytest.mark.parametrize(
    "param_name,param_value",
    [("M", 16), ("M", 40), ("ef_c", 100), ("ef_c", 300), ("ef_s", 100), ("ef_s", 300)],
)
def test_parameter_variations(
    hnsw_blocker, small_sparse_data, hnsw_controls, param_name, param_value
):
    """Test HNSW with different M, ef_c, and ef_s values."""
    x, y = small_sparse_data

    controls = hnsw_controls.copy()
    controls["hnsw"][param_name] = param_value

    result = hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)
    assert result["dist"].notna().all()


def test_k_search_warning(hnsw_blocker, small_sparse_data, hnsw_controls):
    """Test warning when k_search is larger than reference points."""
    x, y = small_sparse_data

    hnsw_controls["hnsw"]["k_search"] = x.shape[0] + 10
    hnsw_blocker.block(x=x, y=y, k=1, verbose=True, controls=hnsw_controls)

    with pytest.warns(UserWarning, match=r"k_search.*larger.*reference"):
        hnsw_blocker.block(x=x, y=y, k=1, verbose=True, controls=hnsw_controls)


def test_verbose_logging(hnsw_blocker, small_sparse_data, hnsw_controls, caplog):
    """Test verbose logging."""
    x, y = small_sparse_data
    caplog.set_level(logging.INFO)

    hnsw_blocker.block(x=x, y=y, k=1, verbose=True, controls=hnsw_controls)

    assert any("Initializing HNSW index" in record.message for record in caplog.records)
    assert any("Adding items to index" in record.message for record in caplog.records)
    assert any("Querying index" in record.message for record in caplog.records)
    assert any("Process completed successfully" in record.message for record in caplog.records)


def test_identical_points(hnsw_blocker, identical_sparse_data, hnsw_controls):
    """Test blocking with identical points."""
    x, y = identical_sparse_data

    controls = hnsw_controls.copy()
    controls["hnsw"]["distance"] = "euclidean"

    result = hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert result["dist"].iloc[0] == pytest.approx(0.0, abs=1e-5)


def test_single_point(hnsw_blocker, single_sparse_point, hnsw_controls):
    """Test blocking with single point."""
    x, y = single_sparse_point

    result = hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=hnsw_controls)
    assert len(result) == 1


def test_empty_data_handling(hnsw_blocker, hnsw_controls):
    """Test handling of empty data."""
    rng = np.random.default_rng()
    x = DataHandler(data=rng.random((0, 3)), cols=["col1", "col2", "col3"])
    y = DataHandler(data=rng.random((5, 3)), cols=["col1", "col2", "col3"])

    with pytest.raises(IndexError):
        hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=hnsw_controls)


def test_large_input(hnsw_blocker, large_sparse_data, hnsw_controls):
    """Test blocking with larger input."""
    x, y = large_sparse_data

    result = hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=hnsw_controls)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x", "y", "dist"}
    assert len(result) == y.shape[0]
    assert result["dist"].notna().all()


def test_save_index(hnsw_blocker, small_sparse_data, hnsw_controls):
    """Test saving the HNSW index and colnames."""
    x, y = small_sparse_data

    with TemporaryDirectory() as temp_dir:
        controls = hnsw_controls.copy()
        controls["hnsw"]["path"] = temp_dir

        _ = hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)

        assert os.path.exists(os.path.join(temp_dir, "index.hnsw"))
        assert os.path.exists(os.path.join(temp_dir, "index-colnames.txt"))


def test_invalid_save_path(hnsw_blocker, small_sparse_data, hnsw_controls):
    """Test invalid save path."""
    x, y = small_sparse_data

    controls = hnsw_controls.copy()
    controls["hnsw"]["path"] = "/invalid/path/that/doesnt/exist"

    with pytest.raises(ValueError, match="Provided path is incorrect"):
        hnsw_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
