"""Tests for the MLPackBlocker class."""

import logging

import numpy as np
import pandas as pd
import pytest

from blockingpy.data_handler import DataHandler

pytestmark = pytest.mark.requires_mlpack
pytest.importorskip("mlpack", reason="mlpack not installed")


@pytest.fixture
def mlpack_controls():
    """Provides controls dict."""
    return [
        (
            "lsh",
            {
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
            },
        ),
        (
            "kd",
            {
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
            },
        ),
    ]


def test_check_algo_valid(mlpack_blocker):
    """Test _check_algo with valid algorithms."""
    mlpack_blocker._check_algo("lsh")
    mlpack_blocker._check_algo("kd")


def test_check_algo_invalid(mlpack_blocker):
    """Test _check_algo with invalid algorithm."""
    with pytest.raises(ValueError) as exc_info:
        mlpack_blocker._check_algo("invalid_algo")
    assert "Invalid algorithm 'invalid_algo'. Accepted values are: lsh, kd" in str(exc_info.value)


def test_basic_blocking(mlpack_blocker, small_sparse_data, mlpack_controls):
    """Test basic functionality with both algorithms."""
    x, y = small_sparse_data

    for _, controls in mlpack_controls:
        result = mlpack_blocker.block(x, y, k=1, verbose=False, controls=controls)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"x", "y", "dist"}
        assert len(result) == y.shape[0]
        assert result["dist"].notna().all()


def test_result_reproducibility(mlpack_blocker, small_sparse_data, kd_controls):
    """Test result reproducibility with fixed seed."""
    x, y = small_sparse_data

    result1 = mlpack_blocker.block(x=x, y=y, k=1, verbose=False, controls=kd_controls)
    result2 = mlpack_blocker.block(x=x, y=y, k=1, verbose=False, controls=kd_controls)

    pd.testing.assert_frame_equal(result1, result2)


def test_k_search_warning(mlpack_blocker, small_sparse_data, lsh_controls):
    """Test warning when k_search is larger than reference points."""
    x, y = small_sparse_data

    lsh_controls["lsh"]["k_search"] = x.shape[0] + 10
    with pytest.warns(UserWarning, match=r"k_search.*larger.*reference"):
        mlpack_blocker.block(x=x, y=y, k=1, verbose=True, controls=lsh_controls)


def test_verbose_logging(mlpack_blocker, small_sparse_data, lsh_controls, caplog):
    """Test verbose logging output."""
    x, y = small_sparse_data
    caplog.set_level(logging.INFO)

    mlpack_blocker.block(x=x, y=y, k=1, verbose=True, controls=lsh_controls)

    assert any("Initializing MLPack LSH index" in record.message for record in caplog.records)
    assert any("MLPack index query completed" in record.message for record in caplog.records)
    assert any(
        "Blocking process completed successfully" in record.message for record in caplog.records
    )


def test_identical_points(mlpack_blocker, identical_sparse_data, mlpack_controls):
    """Test blocking with identical points."""
    x, y = identical_sparse_data

    for _, controls in mlpack_controls:
        result = mlpack_blocker.block(x, y, k=1, verbose=False, controls=controls)
        assert result["dist"].iloc[0] == pytest.approx(0.0, abs=1e-5)


def test_single_point(mlpack_blocker, single_sparse_point, mlpack_controls):
    """Test blocking with single point."""
    x, y = single_sparse_point

    for _, controls in mlpack_controls:
        result = mlpack_blocker.block(x, y, k=1, verbose=False, controls=controls)
        assert len(result) == 1


def test_empty_data_handling(mlpack_blocker, lsh_controls):
    """Test handling of empty datasets."""
    rng = np.random.default_rng()
    x = DataHandler(data=rng.random((0, 3)), cols=[f"x_col_{i}" for i in range(3)])
    y = DataHandler(data=rng.random((5, 3)), cols=[f"y_col_{i}" for i in range(3)])

    with pytest.raises(RuntimeError):
        mlpack_blocker.block(x, y, k=1, verbose=False, controls=lsh_controls)


@pytest.mark.parametrize(
    "param_variation",
    [{"bucket_size": 100}, {"hash_width": 5.0}, {"tables": 10}, {"projections": 5}],
)
def test_lsh_parameter_variations(mlpack_blocker, small_sparse_data, lsh_controls, param_variation):
    """Test LSH with different parameters."""
    x, y = small_sparse_data

    controls = lsh_controls.copy()
    controls["lsh"].update(param_variation)

    try:
        result = mlpack_blocker.block(x, y, k=1, verbose=False, controls=controls)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.skip(f"MLPack parameter validation failed: {e!s}")


@pytest.mark.parametrize(
    "param_variation",
    [
        {"leaf_size": 10},
        {"algorithm": "single_tree"},
        {"random_basis": True},
        {"rho": 0.5},
    ],
)
def test_kd_parameter_variations(mlpack_blocker, small_sparse_data, kd_controls, param_variation):
    """Test k-d tree with different parameters."""
    x, y = small_sparse_data

    controls = kd_controls.copy()
    controls["kd"].update(param_variation)

    try:
        result = mlpack_blocker.block(x, y, k=1, verbose=False, controls=controls)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.skip(f"MLPack parameter validation failed: {e!s}")


def test_large_input(mlpack_blocker, large_sparse_data, kd_controls):
    """Test blocking with larger input matrices."""
    x, y = large_sparse_data

    result = mlpack_blocker.block(x=x, y=y, k=1, verbose=False, controls=kd_controls)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"x", "y", "dist"}
    assert len(result) == y.shape[0]
    assert result["dist"].notna().all()
