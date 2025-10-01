"""Tests for blocking with Voyager."""

import logging
import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from blockingpy.data_handler import DataHandler


@pytest.fixture
def voyager_controls():
    """Default Voyager parameters."""
    return {
        "voyager": {
            "k_search": 5,
            "path": None,
            "random_seed": 42,
            "distance": "cosine",
            "M": 12,
            "ef_construction": 200,
            "max_elements": 1000,
            "num_threads": 1,
            "query_ef": 50,
        }
    }


def test_basic_blocking(voyager_blocker, small_sparse_data, voyager_controls):
    """Test basic blocking functionality."""
    x, y = small_sparse_data

    result = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=voyager_controls)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x", "y", "dist"}
    assert len(result) == y.shape[0]
    assert result["dist"].notna().all()


@pytest.mark.parametrize("distance", ["euclidean", "cosine", "inner_product"])
def test_different_metrics(voyager_blocker, small_sparse_data, voyager_controls, distance):
    """Test different distance metrics in Voyager."""
    x, y = small_sparse_data

    controls = voyager_controls.copy()
    controls["voyager"]["distance"] = distance

    result = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)
    assert result["dist"].notna().all()


def test_result_reproducibility(voyager_blocker, small_sparse_data, voyager_controls):
    """Test result reproducibility with same random seed."""
    x, y = small_sparse_data

    result1 = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=voyager_controls)
    result2 = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=voyager_controls)

    pd.testing.assert_frame_equal(result1, result2)


def test_seed_reproducibility(voyager_blocker, large_sparse_data, voyager_controls):
    """Test that same seed gives reproducible results."""
    x, y = large_sparse_data

    controls = voyager_controls.copy()
    controls["voyager"]["random_seed"] = 42

    result1 = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    result2 = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)

    pd.testing.assert_frame_equal(result1, result2)


def test_k_search_warning(voyager_blocker, small_sparse_data, voyager_controls):
    """Test warning when k_search is larger than reference points."""
    x, y = small_sparse_data

    voyager_controls["voyager"]["k_search"] = x.shape[0] + 10

    with pytest.warns(UserWarning, match=r"k_search.*larger.*reference"):
        voyager_blocker.block(x=x, y=y, k=1, verbose=True, controls=voyager_controls)


def test_verbose_logging(voyager_blocker, small_sparse_data, voyager_controls, caplog):
    """Test verbose logging."""
    x, y = small_sparse_data
    caplog.set_level(logging.INFO)

    voyager_blocker.block(x=x, y=y, k=1, verbose=True, controls=voyager_controls)

    assert any("Building index" in record.message for record in caplog.records)
    assert any("Querying index" in record.message for record in caplog.records)
    assert any("Process completed successfully" in record.message for record in caplog.records)


def test_identical_points(voyager_blocker, identical_sparse_data, voyager_controls):
    """Test blocking with identical points."""
    x, y = identical_sparse_data

    controls = voyager_controls.copy()
    controls["voyager"]["distance"] = "euclidean"

    result = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert result["dist"].iloc[0] == pytest.approx(0.0, abs=1e-5)


def test_single_point(voyager_blocker, single_sparse_point, voyager_controls):
    """Test blocking with single point."""
    x, y = single_sparse_point

    result = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=voyager_controls)
    assert len(result) == 1


def test_empty_data_handling(voyager_blocker, voyager_controls):
    """Test handling of empty data."""
    rng = np.random.default_rng()

    x = DataHandler(data=rng.random((0, 3)), cols=["col1", "col2", "col3"])
    y = DataHandler(data=rng.random((5, 3)), cols=["col1", "col2", "col3"])

    with pytest.raises(ValueError):
        voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=voyager_controls)


@pytest.mark.parametrize(
    "param_variation",
    [
        {"M": 16},
        {"ef_construction": 300},
        {"query_ef": 100},
        {"max_elements": 2000},
    ],
)
def test_parameter_variations(
    voyager_blocker, small_sparse_data, voyager_controls, param_variation
):
    """Test Voyager with different parameters."""
    x, y = small_sparse_data

    controls = voyager_controls.copy()
    controls["voyager"].update(param_variation)

    result = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)


def test_large_input(voyager_blocker, large_sparse_data, voyager_controls):
    """Test blocking with larger input."""
    x, y = large_sparse_data

    result = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=voyager_controls)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x", "y", "dist"}
    assert len(result) == y.shape[0]
    assert result["dist"].notna().all()


@pytest.mark.parametrize("num_threads", [1, 2, -1])
def test_multithreading(voyager_blocker, large_sparse_data, voyager_controls, num_threads):
    """Test Voyager blocking with different num_threads."""
    x, y = large_sparse_data

    controls = voyager_controls.copy()
    controls["voyager"]["num_threads"] = num_threads

    result = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)


def test_save_index(voyager_blocker, small_sparse_data, voyager_controls):
    """Test saving the Voyager index and colnames."""
    x, y = small_sparse_data

    with TemporaryDirectory() as temp_dir:
        controls = voyager_controls.copy()
        controls["voyager"]["path"] = temp_dir

        _ = voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)

        assert os.path.exists(os.path.join(temp_dir, "index.voyager"))
        assert os.path.exists(os.path.join(temp_dir, "index-colnames.txt"))


def test_invalid_save_path(voyager_blocker, small_sparse_data, voyager_controls):
    """Test invalid save path."""
    x, y = small_sparse_data

    controls = voyager_controls.copy()
    controls["voyager"]["path"] = "/ten/dir/nie/istnieje"

    with pytest.raises(ValueError, match="Provided path is incorrect"):
        voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)


def test_invalid_file_creation(voyager_blocker, small_sparse_data, voyager_controls):
    """Test file creation fail."""
    x, y = small_sparse_data

    controls = voyager_controls.copy()
    controls["voyager"]["path"] = "./plik"

    with pytest.raises(RuntimeError, match="Failed to open file for writing"):
        voyager_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
