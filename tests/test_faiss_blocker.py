import logging
import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def faiss_controls():
    """Default FAISS parameters."""
    return {
        'faiss': {
            'k_search': 5,
            'path': None,
            'distance': 'euclidean',
        }
    }


def test_basic_blocking(faiss_blocker, small_sparse_data, faiss_controls):
    """Test basic blocking functionality."""
    x, y = small_sparse_data
    
    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'x', 'y', 'dist'}
    assert len(result) == len(y)
    assert result['dist'].notna().all()


@pytest.mark.parametrize("distance", [
    'euclidean',
    'l2',
    'inner_product',
    'l1',
    'manhattan',
    'linf',
    'bray_curtis',
])
def test_different_metrics(faiss_blocker, small_sparse_data, faiss_controls, distance):
    """Test different distance metrics that do not need any additional actions in FAISS."""
    x, y = small_sparse_data
    
    controls = faiss_controls.copy()
    controls['faiss']['distance'] = distance
    
    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(result, pd.DataFrame)
    assert result['dist'].notna().all()


def test_cosine_normalization(faiss_blocker, small_sparse_data, faiss_controls):
    """Test the cosine normalization in FAISS."""
    x, y = small_sparse_data
    
    controls = faiss_controls.copy()
    controls['faiss']['distance'] = 'cosine'
    
    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert (result['dist'] >= -1).all() and (result['dist'] <= 1).all()


def test_smoothing_metrics(faiss_blocker, small_sparse_data, faiss_controls):
    """Test metrics that require smoothing (jensen_shannon, canberra)."""
    x, y = small_sparse_data
    
    for metric in ['jensen_shannon', 'canberra']:
        controls = faiss_controls.copy()
        controls['faiss']['distance'] = metric
        
        result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
        assert result['dist'].notna().all()


def test_invalid_metric(faiss_blocker, small_sparse_data, faiss_controls):
    """Test error handling for invalid distance metric."""
    x, y = small_sparse_data
    
    controls = faiss_controls.copy()
    controls['faiss']['distance'] = 'invalid_metric'
    
    with pytest.raises(ValueError, match="Invalid distance metric"):
        faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)


def test_result_reproducibility(faiss_blocker, small_sparse_data, faiss_controls):
    """Test result reproducibility with same parameters."""
    x, y = small_sparse_data

    result1 = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)
    result2 = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)
    
    pd.testing.assert_frame_equal(result1, result2)


def test_k_search_warning(faiss_blocker, small_sparse_data, faiss_controls, caplog):
    """Test warning when k_search is larger than reference points."""
    x, y = small_sparse_data
    
    faiss_controls['faiss']['k_search'] = len(x) + 10
    with caplog.at_level(logging.WARNING):
        faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)
    
    warning_message = f"k_search ({len(x) + 10}) is larger than the number of reference points ({len(x)}). Adjusted k_search to {len(x)}."
    assert warning_message in caplog.text


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
    controls['faiss']['distance'] = 'euclidean'
    
    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert result['dist'].iloc[0] == pytest.approx(0.0, abs=1e-5)


def test_single_point(faiss_blocker, single_sparse_point, faiss_controls):
    """Test blocking with single point."""
    x, y = single_sparse_point
    
    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)
    assert len(result) == 1


def test_empty_data_handling(faiss_blocker, faiss_controls):
    """Test handling of empty data."""
    x = pd.DataFrame(np.random.rand(0, 3))
    y = pd.DataFrame(np.random.rand(5, 3))
    
    with pytest.raises(Exception):
        faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)


def test_large_input(faiss_blocker, large_sparse_data, faiss_controls):
    """Test blocking with larger input."""
    x, y = large_sparse_data
    
    result = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=faiss_controls)
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'x', 'y', 'dist'}
    assert len(result) == len(y)
    assert result['dist'].notna().all()


def test_save_index(faiss_blocker, small_sparse_data, faiss_controls):
    """Test saving the FAISS index and colnames."""
    x, y = small_sparse_data
    x.columns = x.columns.astype(str)
    
    with TemporaryDirectory() as temp_dir:
        controls = faiss_controls.copy()
        controls['faiss']['path'] = temp_dir
      
        result1 = faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
        
        assert os.path.exists(os.path.join(temp_dir, "index.faiss"))
        assert os.path.exists(os.path.join(temp_dir, "index-colnames.txt"))


def test_invalid_save_path(faiss_blocker, small_sparse_data, faiss_controls):
    """Test invalid save path."""
    x, y = small_sparse_data
    
    controls = faiss_controls.copy()
    controls['faiss']['path'] = "/invalid/path/that/doesnt/exist"
    
    with pytest.raises(ValueError, match="Provided path is incorrect"):
        faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)