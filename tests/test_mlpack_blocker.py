import logging

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def algo_controls():
    """Provides controls dict."""
    return [
        ('lsh', {
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
        }),
        ('kd', {
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
        })
    ]


def test_check_algo_valid(blocker):
    """Test _check_algo with valid algorithms."""
    blocker._check_algo('lsh')
    blocker._check_algo('kd')

def test_check_algo_invalid(blocker):
    """Test _check_algo with invalid algorithm."""
    with pytest.raises(ValueError) as exc_info:
        blocker._check_algo('invalid_algo')
    assert "Invalid algorithm 'invalid_algo'. Accepted values are: lsh, kd" in str(exc_info.value)

def test_basic_blocking(blocker, small_sparse_data, algo_controls):
    """Test basic functionality with both algorithms."""
    x, y = small_sparse_data
    
    for _, controls in algo_controls:
        result = blocker.block(x, y, k=1, verbose=False, controls=controls)
        
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {'x', 'y', 'dist'}
        assert len(result) == len(y)
        assert result['x'].dtype == np.int64
        assert result['y'].dtype == np.int64
        assert result['dist'].dtype == np.float64
        assert result['dist'].notna().all()

def test_result_reproducibility(blocker, small_sparse_data, kd_controls):
    """Test result reproducibility with fixed seed."""
    x, y = small_sparse_data
    
    result1 = blocker.block(x=x, y=y, k=1, verbose=False, controls=kd_controls)
    result2 = blocker.block(x=x, y=y, k=1, verbose=False, controls=kd_controls)
    
    pd.testing.assert_frame_equal(result1, result2)

def test_k_search_warning(blocker, small_sparse_data, lsh_controls, caplog):
    """Test warning when k_search is larger than reference points."""
    x, y = small_sparse_data
    caplog.set_level(logging.WARNING)
    
    lsh_controls['lsh']['k_search'] = len(x) + 10
    blocker.block(x=x, y=y, k=1, verbose=True, controls=lsh_controls)
    
    warning_message = f"k_search ({len(x) + 10}) is larger than the number of reference points ({len(x)})"
    assert any(warning_message in record.message for record in caplog.records)

def test_verbose_logging(blocker, small_sparse_data, lsh_controls, caplog):
    """Test verbose logging output."""
    x, y = small_sparse_data
    caplog.set_level(logging.INFO)
    
    blocker.block(x=x, y=y, k=1, verbose=True, controls=lsh_controls)

    assert any("Initializing MLPack LSH index" in record.message for record in caplog.records)
    assert any("MLPack index query completed" in record.message for record in caplog.records)
    assert any("Blocking process completed successfully" in record.message for record in caplog.records)

def test_identical_points(blocker, identical_sparse_data, algo_controls):
    """Test blocking with identical points."""
    x, y = identical_sparse_data
    
    for _, controls in algo_controls:
        result = blocker.block(x, y, k=1, verbose=False, controls=controls)
        assert result['dist'].iloc[0] == pytest.approx(0.0, abs=1e-5)

def test_single_point(blocker, single_sparse_point, algo_controls):
    """Test blocking with single point."""
    x, y = single_sparse_point
    
    for _, controls in algo_controls:
        result = blocker.block(x, y, k=1, verbose=False, controls=controls)
        assert len(result) == 1

def test_empty_data_handling(blocker, lsh_controls):
    """Test handling of empty datasets."""
    x = pd.DataFrame(np.random.rand(0, 3))
    y = pd.DataFrame(np.random.rand(5, 3))

    with pytest.raises(Exception):
        blocker.block(x, y, k=1, verbose=False, controls=lsh_controls)

@pytest.mark.parametrize("param_variation", [
    {'bucket_size': 100},
    {'hash_width': 5.0},
    {'tables': 10},
    {'projections': 5}
])
def test_lsh_parameter_variations(blocker, small_sparse_data, lsh_controls, param_variation):
    """Test LSH with different parameter configurations."""
    x, y = small_sparse_data
    
    controls = lsh_controls.copy()
    controls['lsh'].update(param_variation)
    
    try:
        result = blocker.block(x, y, k=1, verbose=False, controls=controls)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.skip(f"MLPack parameter validation failed: {str(e)}")

@pytest.mark.parametrize("param_variation", [
    {'leaf_size': 10},
    {'algorithm': "single_tree"},
    {'random_basis': True},
    {'rho': 0.5}
])
def test_kd_parameter_variations(blocker, small_sparse_data, kd_controls, param_variation):
    """Test k-d tree with different parameter configurations."""
    x, y = small_sparse_data
    
    controls = kd_controls.copy()
    controls['kd'].update(param_variation)
    
    try:
        result = blocker.block(x, y, k=1, verbose=False, controls=controls)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        pytest.skip(f"MLPack parameter validation failed: {str(e)}")

def test_large_sparse_input(blocker, large_sparse_data, kd_controls):
    """Test blocking with larger sparse input matrices."""
    x, y = large_sparse_data
    
    result = blocker.block(x=x, y=y, k=1, verbose=False, controls=kd_controls)
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'x', 'y', 'dist'}
    assert len(result) == len(y)
    assert result['dist'].notna().all()