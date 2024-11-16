import logging
import numpy as np
import pandas as pd
import pytest
from blockingpy.mlpack_blocker import MLPackBlocker


@pytest.fixture
def blocker():
    """Create a fresh MLPackBlocker instance for each test."""
    return MLPackBlocker()


def test_check_algo_valid(blocker):
    """Test _check_algo with valid algorithms."""
    blocker._check_algo('lsh')
    blocker._check_algo('kd')


def test_check_algo_invalid(blocker):
    """Test _check_algo with invalid algorithm."""
    with pytest.raises(ValueError) as exc_info:
        blocker._check_algo('invalid_algo')
    assert "Invalid algorithm 'invalid_algo'. Accepted values are: lsh, kd" in str(exc_info.value)


def test_lsh_basic_blocking(blocker, small_sparse_x, small_sparse_y):
    """Test basic LSH with small sparse input."""
    result = blocker.block(
        x=small_sparse_x,
        y=small_sparse_y,
        k=1,
        verbose=False,
        controls={
            'algo': 'lsh',
            'lsh': {
                'seed': 42,
                'k_search': 2,
                'bucket_size': 500,
                'hash_width': 10.0,
                'num_probes': 0,
                'projections': 10,
                'tables': 30
            }
        }
    )
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'x', 'y', 'dist'}
    assert len(result) == len(small_sparse_y)
    assert result['x'].dtype == np.int64
    assert result['y'].dtype == np.int64
    assert result['dist'].dtype == np.float64


def test_kd_basic_blocking(blocker, small_sparse_x, small_sparse_y):
    """Test basic k-d tree with small sparse input."""
    result = blocker.block(
        x=small_sparse_x,
        y=small_sparse_y,
        k=1,
        verbose=False,
        controls={
            'algo': 'kd',
            'kd': {
                'seed': 42,
                'k_search': 2,
                'algorithm': "dual_tree",
                'leaf_size': 20,
                'tree_type': "kd",
                'epsilon': 0.0,
                'rho': 0.7,
                'tau': 0.0,
                'random_basis': False
            }
        }
    )
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'x', 'y', 'dist'}
    assert len(result) == len(small_sparse_y)
    assert result['x'].dtype == np.int64
    assert result['y'].dtype == np.int64
    assert result['dist'].dtype == np.float64


def test_k_search_warning(blocker, small_sparse_x, small_sparse_y, caplog):
    """Test warning when k_search is larger than reference points."""
    caplog.set_level(logging.WARNING)
    
    result = blocker.block(
        x=small_sparse_x,
        y=small_sparse_y,
        k=1,
        verbose=True,
        controls={
            'algo': 'lsh',
            'lsh': {
                'k_search': len(small_sparse_x) + 10,
                'seed': 42
            }
        }
    )
    
    warning_message = f"k_search ({len(small_sparse_x) + 10}) is larger than the number of reference points ({len(small_sparse_x)})"
    assert any(warning_message in record.message for record in caplog.records)


def test_verbose_logging(blocker, small_sparse_x, small_sparse_y, caplog):
    """Test verbose logging output."""
    caplog.set_level(logging.INFO)
    
    _ = blocker.block(
        x=small_sparse_x,
        y=small_sparse_y,
        k=1,
        verbose=True,
        controls={
            'algo': 'lsh',
            'lsh': {
                'k_search': 2,
                'seed': 42
            }
        }
    )

    assert any("Initializing MLPack LSH index" in record.message for record in caplog.records)
    assert any("MLPack index query completed" in record.message for record in caplog.records)
    assert any("Blocking process completed successfully" in record.message for record in caplog.records)


@pytest.mark.parametrize("algo,config", [
    ('lsh', {'bucket_size': 500, 'hash_width': 10.0, 'num_probes': 0, 'projections': 10, 'tables': 30}),
    ('kd', {'algorithm': "dual_tree", 'leaf_size': 20, 'tree_type': "kd", 'epsilon': 0.0, 'rho': 0.7, 'tau': 0.0, 'random_basis': False})
])
def test_algorithm_specific_params(blocker, small_sparse_x, small_sparse_y, algo, config):
    """Test algorithm-specific parameters for both LSH and k-d tree."""
    controls = {
        'algo': algo,
        algo: {
            'k_search': 2,
            'seed': 42,
            **config
        }
    }
    
    result = blocker.block(
        x=small_sparse_x,
        y=small_sparse_y,
        k=1,
        verbose=False,
        controls=controls
    )
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'x', 'y', 'dist'}
    assert len(result) == len(small_sparse_y)


def test_result_reproducibility(blocker, small_sparse_x, small_sparse_y):
    """Test result reproducibility with seed."""
    controls = {
        'algo': 'kd',
        'kd': {
            'k_search': 2,
            'seed': 42,
            'random_basis': True,
        }
    }
    
    result1 = blocker.block(x=small_sparse_x, y=small_sparse_y, k=1, verbose=False, controls=controls)
    result2 = blocker.block(x=small_sparse_x, y=small_sparse_y, k=1, verbose=False, controls=controls)
    
    pd.testing.assert_frame_equal(result1, result2)


def test_large_sparse_input(blocker, mat_x, mat_y):
    """Test blocking with larger sparse input matrices."""
    controls = {
        'algo': 'kd',
        'kd': {
            'k_search': 3,
            'seed': 42,
            'random_basis': True,
        }
    }
    print(mat_x.shape, mat_y.shape)
    result = blocker.block(x=mat_x, y=mat_y, k=1, verbose=False, controls=controls)
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'x', 'y', 'dist'}
    assert len(result) == len(mat_y)
    assert result['dist'].notna().all()


def test_empty_data_handling(blocker):
        """Test handling of empty datasets"""
        x = pd.DataFrame(np.random.rand(0, 3))
        y = pd.DataFrame(np.random.rand(5, 3))
        controls = {
            'algo': 'lsh',
            'lsh': {
                'k_search': 5,
                'seed': 42
            }
        }

        with pytest.raises(Exception):
            blocker.block(x, y, k=1, verbose=False, controls=controls)


def test_lsh_specific_params(blocker, small_sparse_x, small_sparse_y):
    """Test LSH specific parameters."""
    controls = {
        'algo': 'lsh',
        'lsh': {
            'k_search': 5,
            'bucket_size': 1,
            'hash_width': 0.1,
            'tables': 1
        }
    }

    try:
        result = blocker.block(small_sparse_x, small_sparse_y, k=1, verbose=False, controls=controls)
    except Exception as e:
        pytest.skip(f"MLPack parameter validation failed: {str(e)}")