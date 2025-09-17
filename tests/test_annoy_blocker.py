"""Tests for the AnnoyBlocker class."""

import logging
import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from blockingpy.data_handler import DataHandler


@pytest.fixture
def annoy_controls():
    """Default Annoy parameters."""
    return {
        "annoy": {
            "k_search": 5,
            "distance": "angular",
            "seed": None,
            "path": None,
            "n_trees": 250,
            "build_on_disk": False,
        }
    }


def test_basic_blocking(annoy_blocker, small_sparse_data, annoy_controls):
    """Test basic blocking functionality."""
    x, y = small_sparse_data

    result = annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=annoy_controls)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x", "y", "dist"}
    assert len(result) == y.shape[0]
    assert result["dist"].notna().all()


@pytest.mark.parametrize("distance", ["euclidean", "manhattan", "hamming", "dot"])
def test_different_metrics(annoy_blocker, small_sparse_data, annoy_controls, distance):
    """Test different distance metrics in Annoy."""
    x, y = small_sparse_data

    controls = annoy_controls.copy()
    controls["annoy"]["distance"] = distance

    result = annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)
    assert result["dist"].notna().all()


def test_result_reproducibility(annoy_blocker, small_sparse_data, annoy_controls):
    """Test result reproducibility with same seed."""
    x, y = small_sparse_data

    controls = annoy_controls.copy()
    controls["annoy"]["seed"] = 42

    result1 = annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    result2 = annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)

    pd.testing.assert_frame_equal(result1, result2)


@pytest.mark.parametrize("n_trees", [10, 100, 500])
def test_n_trees_parameter(annoy_blocker, small_sparse_data, annoy_controls, n_trees):
    """Test Annoy with different numbers of trees."""
    x, y = small_sparse_data

    controls = annoy_controls.copy()
    controls["annoy"]["n_trees"] = n_trees

    result = annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)
    assert result["dist"].notna().all()


@pytest.mark.xfail(reason="Issue with OSError")
def test_build_on_disk(annoy_blocker, small_sparse_data, annoy_controls):
    """Test building index on disk."""
    x, y = small_sparse_data

    controls = annoy_controls.copy()
    controls["annoy"]["build_on_disk"] = True

    result = annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)
    assert result["dist"].notna().all()


def test_k_search_warning(annoy_blocker, small_sparse_data, annoy_controls):
    """Test warning when k_search is larger than reference points."""
    x, y = small_sparse_data

    annoy_controls["annoy"]["k_search"] = x.shape[0] + 10
    with pytest.warns(UserWarning, match=r"k_search.*larger.*reference"):
        annoy_blocker.block(x=x, y=y, k=1, verbose=True, controls=annoy_controls)


def test_verbose_logging(annoy_blocker, small_sparse_data, annoy_controls, caplog):
    """Test verbose logging."""
    x, y = small_sparse_data
    caplog.set_level(logging.INFO)

    annoy_blocker.block(x=x, y=y, k=1, verbose=True, controls=annoy_controls)

    assert any("Building index" in record.message for record in caplog.records)
    assert any("Querying index" in record.message for record in caplog.records)
    assert any("Process completed successfully" in record.message for record in caplog.records)


def test_identical_points(annoy_blocker, identical_sparse_data, annoy_controls):
    """Test blocking with identical points."""
    x, y = identical_sparse_data

    controls = annoy_controls.copy()
    controls["annoy"]["distance"] = "euclidean"

    result = annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert result["dist"].iloc[0] == pytest.approx(0.0, abs=1e-5)


def test_single_point(annoy_blocker, single_sparse_point, annoy_controls):
    """Test blocking with single point."""
    x, y = single_sparse_point

    result = annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=annoy_controls)
    assert len(result) == 1


def test_empty_data_handling(annoy_blocker, annoy_controls):
    """Test handling of empty data."""
    rng = np.random.default_rng()
    x = DataHandler(data=rng.random((0, 3)), cols=[f"x_col_{i}" for i in range(3)])
    y = DataHandler(data=rng.random((5, 3)), cols=[f"y_col_{i}" for i in range(3)])

    with pytest.raises(IndexError):
        annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=annoy_controls)


def test_large_input(annoy_blocker, large_sparse_data, annoy_controls):
    """Test blocking with larger input."""
    x, y = large_sparse_data

    result = annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=annoy_controls)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x", "y", "dist"}
    assert len(result) == y.shape[0]
    assert result["dist"].notna().all()


@pytest.mark.xfail(reason="Issue with NotADirectoryError")
def test_save_index(annoy_blocker, small_sparse_data, annoy_controls):
    """Test saving the Annoy index and colnames."""
    x, y = small_sparse_data
    x.columns = x.columns.astype(str)

    with TemporaryDirectory() as temp_dir:
        controls = annoy_controls.copy()
        os.makedirs(temp_dir, exist_ok=True)
        controls["annoy"]["path"] = temp_dir
        _ = annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)

        assert os.path.exists(os.path.join(temp_dir, "index.annoy"))
        assert os.path.exists(os.path.join(temp_dir, "index-colnames.txt"))


def test_invalid_save_path(annoy_blocker, small_sparse_data, annoy_controls):
    """Test invalid save path."""
    x, y = small_sparse_data

    controls = annoy_controls.copy()
    controls["annoy"]["path"] = "/invalid/path/that/doesnt/exist"

    with pytest.raises(ValueError, match="Provided path is incorrect"):
        annoy_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
