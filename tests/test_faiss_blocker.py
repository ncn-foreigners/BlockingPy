"""Tests for the FAISS blocker."""

import logging
import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from blockingpy.data_handler import DataHandler


@pytest.fixture
def faiss_controls():
    """Default FAISS parameters."""
    return {
        "faiss": {
            "index_type": "flat",
            "k_search": 5,
            "path": None,
            "distance": "cosine",
            "hnsw_M": 32,
            "hnsw_ef_construction": 200,
            "hnsw_ef_search": 200,
            "lsh_nbits": 8,
            "lsh_rotate_data": True,
        }
    }


def test_basic_blocking(faiss_blocker, small_sparse_data, faiss_controls):
    """Test basic blocking functionality."""
    x, y = small_sparse_data

    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x", "y", "dist"}
    assert len(result) == y.shape[0]
    assert result["dist"].notna().all()


@pytest.mark.parametrize(
    "distance",
    [
        "euclidean",
        "l2",
        "inner_product",
        "l1",
        "manhattan",
        "linf",
        "bray_curtis",
    ],
)
def test_different_metrics(faiss_blocker, small_sparse_data, faiss_controls, distance):
    """Test different distance metrics that do not need any additional actions in FAISS."""
    x, y = small_sparse_data

    controls = faiss_controls.copy()
    controls["faiss"]["distance"] = distance

    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)
    assert result["dist"].notna().all()


def test_cosine_normalization(faiss_blocker, small_sparse_data, faiss_controls):
    """Test the cosine normalization in FAISS."""
    x, y = small_sparse_data

    controls = faiss_controls.copy()
    controls["faiss"]["distance"] = "cosine"

    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    print(result)
    assert (result["dist"] >= -1.1).all() and (result["dist"] <= 1.1).all()


def test_smoothing_metrics(faiss_blocker, small_sparse_data, faiss_controls):
    """Test metrics that require smoothing (jensen_shannon, canberra)."""
    x, y = small_sparse_data

    for metric in ["jensen_shannon", "canberra"]:
        controls = faiss_controls.copy()
        controls["faiss"]["distance"] = metric

        result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
        assert result["dist"].notna().all()


def test_result_reproducibility(faiss_blocker, small_sparse_data, faiss_controls):
    """Test result reproducibility with same parameters."""
    x, y = small_sparse_data

    result1 = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)
    result2 = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)

    pd.testing.assert_frame_equal(result1, result2)


def test_k_search_warning(faiss_blocker, small_sparse_data, faiss_controls):
    """Warn when k_search exceeds reference size."""
    x, y = small_sparse_data
    faiss_controls["faiss"]["k_search"] = x.shape[0] + 10

    with pytest.warns(UserWarning, match=r"k_search.*larger.*reference"):
        faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)


def test_verbose_logging(faiss_blocker, small_sparse_data, faiss_controls, caplog):
    """Test verbose logging."""
    x, y = small_sparse_data

    with caplog.at_level(logging.DEBUG):
        faiss_blocker.block(x=x, y=y, k=1, verbose=True, controls=faiss_controls)

    assert "Building index..." in caplog.text
    assert "Querying index..." in caplog.text
    assert "Process completed successfully." in caplog.text


def test_identical_points(faiss_blocker, identical_sparse_data, faiss_controls):
    """Test blocking with identical points."""
    x, y = identical_sparse_data

    controls = faiss_controls.copy()
    controls["faiss"]["distance"] = "euclidean"

    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert result["dist"].iloc[0] == pytest.approx(0.0, abs=1e-5)


def test_single_point(faiss_blocker, single_sparse_point, faiss_controls):
    """Test blocking with single point."""
    x, y = single_sparse_point

    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)
    assert len(result) == 1


def test_empty_data_handling(faiss_blocker, faiss_controls):
    """Test handling of empty data."""
    rng = np.random.default_rng()
    x = DataHandler(data=rng.random((0, 3)), cols=[f"x_col_{i}" for i in range(3)])
    y = DataHandler(data=rng.random((5, 3)), cols=[f"y_col_{i}" for i in range(3)])

    with pytest.raises(AssertionError):
        faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)


def test_large_input(faiss_blocker, large_sparse_data, faiss_controls):
    """Test blocking with larger input."""
    x, y = large_sparse_data

    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x", "y", "dist"}
    assert len(result) == y.shape[0]
    assert result["dist"].notna().all()


def test_save_index(faiss_blocker, small_sparse_data, faiss_controls):
    """Test saving the FAISS index and colnames."""
    x, y = small_sparse_data

    with TemporaryDirectory() as temp_dir:
        controls = faiss_controls.copy()
        controls["faiss"]["path"] = temp_dir

        _ = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)

        assert os.path.exists(os.path.join(temp_dir, "index.faiss"))
        assert os.path.exists(os.path.join(temp_dir, "index-colnames.txt"))


def test_invalid_save_path(faiss_blocker, small_sparse_data, faiss_controls):
    """Test invalid save path."""
    x, y = small_sparse_data

    controls = faiss_controls.copy()
    controls["faiss"]["path"] = "/invalid/path/that/doesnt/exist"

    with pytest.raises(ValueError, match="Provided path is incorrect"):
        faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)


@pytest.mark.parametrize(
    "index_type,index_params",
    [
        ("flat", {}),
        ("hnsw", {"hnsw_M": 16, "hnsw_ef_construction": 100, "hnsw_ef_search": 50}),
        ("lsh", {"lsh_nbits": 8, "lsh_rotate_data": True}),
    ],
)
def test_index_types(faiss_blocker, small_sparse_data, faiss_controls, index_type, index_params):
    """Test different FAISS index types."""
    x, y = small_sparse_data

    controls = faiss_controls.copy()
    controls["faiss"]["index_type"] = index_type
    controls["faiss"].update(index_params)

    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x", "y", "dist"}
    assert len(result) == y.shape[0]
    assert result["dist"].notna().all()


def test_hnsw_parameters(faiss_blocker, small_sparse_data, faiss_controls):
    """Test HNSW with different parameter configurations."""
    x, y = small_sparse_data

    for M in [8, 16, 32]:
        controls = faiss_controls.copy()
        controls["faiss"]["index_type"] = "hnsw"
        controls["faiss"]["hnsw_M"] = M
        controls["faiss"]["hnsw_ef_construction"] = 100
        controls["faiss"]["hnsw_ef_search"] = 50

        result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
        assert len(result) == y.shape[0]

    for ef in [50, 100, 200]:
        controls = faiss_controls.copy()
        controls["faiss"]["index_type"] = "hnsw"
        controls["faiss"]["hnsw_M"] = 16
        controls["faiss"]["hnsw_ef_construction"] = ef
        controls["faiss"]["hnsw_ef_search"] = 50

        result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
        assert len(result) == y.shape[0]


def test_lsh_parameters(faiss_blocker, small_sparse_data, faiss_controls):
    """Test LSH with different parameter configurations."""
    x, y = small_sparse_data

    for nbits in [0.5, 1, 0.3]:
        controls = faiss_controls.copy()
        controls["faiss"]["index_type"] = "lsh"
        controls["faiss"]["lsh_nbits"] = nbits
        controls["faiss"]["lsh_rotate_data"] = True

        result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
        assert len(result) == y.shape[0]

    for rotate in [True, False]:
        controls = faiss_controls.copy()
        controls["faiss"]["index_type"] = "lsh"
        controls["faiss"]["lsh_nbits"] = 1
        controls["faiss"]["lsh_rotate_data"] = rotate

        result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
        assert len(result) == y.shape[0]


def test_index_type_reproducibility(faiss_blocker, small_sparse_data, faiss_controls):
    """Test result reproducibility with different index types."""
    x, y = small_sparse_data

    for index_type in ["flat", "hnsw", "lsh"]:
        controls = faiss_controls.copy()
        controls["faiss"]["index_type"] = index_type

        if index_type == "hnsw":
            controls["faiss"]["hnsw_M"] = 16
            controls["faiss"]["hnsw_ef_construction"] = 100
            controls["faiss"]["hnsw_ef_search"] = 50
        elif index_type == "lsh":
            controls["faiss"]["lsh_nbits"] = 8
            controls["faiss"]["lsh_rotate_data"] = True

        result1 = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
        result2 = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)

        pd.testing.assert_frame_equal(result1, result2)
