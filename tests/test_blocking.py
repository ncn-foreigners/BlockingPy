import logging
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from blockingpy.blocker import (
    Blocker,
    BlockingResult,
)

def test_input_validation_types(small_named_csr_data, small_named_ndarray_data, small_named_txt_data):
    """Test input validation for different data types."""
    blocker = Blocker()
    x_csr, y_csr, x_cols_csr, y_cols_csr = small_named_csr_data
    x_ndarray, y_ndarray, x_cols_ndarray, y_cols_ndarray = small_named_ndarray_data
    x_txt, y_txt = small_named_txt_data
    
    result_csr = blocker.block(
        x_csr, 
        y=y_csr,
        x_colnames=x_cols_csr,
        y_colnames=y_cols_csr,
        ann="faiss"
    )
    assert isinstance(result_csr, BlockingResult)
    
    result_ndarray = blocker.block(
        x_ndarray,
        y=y_ndarray,
        x_colnames=x_cols_ndarray,
        y_colnames=y_cols_ndarray,
        ann="faiss"
    )
    assert isinstance(result_ndarray, BlockingResult)
    
    result_txt = blocker.block(x_txt['txt'], y=y_txt['txt'], ann="faiss")
    assert isinstance(result_txt, BlockingResult)
    
    with pytest.raises(ValueError):
        blocker.block([1, 2, 3], ann="faiss")
    with pytest.raises(ValueError):
        blocker.block(pd.DataFrame({'a': [1, 2, 3]}), ann="faiss")

@pytest.mark.parametrize("algo", ["nnd", "hnsw", "annoy", "faiss", "voyager"])
def test_algorithm_selection(algo, small_named_csr_data, small_named_txt_data):
    """Test different algorithms with both matrix and text inputs."""
    blocker = Blocker()
    x_csr, y_csr, x_cols, y_cols = small_named_csr_data
    x_txt, y_txt = small_named_txt_data
    
    result_csr = blocker.block(
        x_csr,
        y=y_csr,
        x_colnames=x_cols,
        y_colnames=y_cols,
        ann=algo
    )
    assert isinstance(result_csr, BlockingResult)
    assert result_csr.method == algo
    
    result_txt = blocker.block(
        x_txt['txt'],
        y=y_txt['txt'],
        ann=algo
    )
    assert isinstance(result_txt, BlockingResult)
    assert result_txt.method == algo

def test_deduplication_vs_linkage(small_named_csr_data, small_named_txt_data):
    """Test deduplication and linkage with both matrix and text data."""
    blocker = Blocker()
    x_csr, y_csr, x_cols, y_cols = small_named_csr_data
    x_txt, y_txt = small_named_txt_data
    
    dedup_result_csr = blocker.block(
        x_csr,
        x_colnames=x_cols,
        y_colnames=x_cols,
        deduplication=True
    )
    assert isinstance(dedup_result_csr, BlockingResult)
    assert dedup_result_csr.deduplication == True
    
    link_result_csr = blocker.block(
        x_csr,
        y=y_csr,
        x_colnames=x_cols,
        y_colnames=y_cols,
        deduplication=False
    )
    assert isinstance(link_result_csr, BlockingResult)
    assert link_result_csr.deduplication == False
    
    dedup_result_txt = blocker.block(
        x_txt['txt'],
        deduplication=True
    )
    assert isinstance(dedup_result_txt, BlockingResult)
    assert dedup_result_txt.deduplication == True
    
    link_result_txt = blocker.block(
        x_txt['txt'],
        y=y_txt['txt'],
        deduplication=False
    )
    assert isinstance(link_result_txt, BlockingResult)
    assert link_result_txt.deduplication == False

def test_graph_creation(small_named_csr_data, small_named_txt_data):
    """Test graph creation with both matrix and text data."""
    blocker = Blocker()
    x_csr, y_csr, x_cols, y_cols = small_named_csr_data
    x_txt, y_txt = small_named_txt_data
    
    result_csr = blocker.block(
        x_csr,
        x_colnames=x_cols,
        y_colnames=x_cols,
        graph=True
    )
    assert isinstance(result_csr.graph, nx.Graph)
    assert result_csr.graph.number_of_nodes() > 0
    
    result_txt = blocker.block(
        x_txt['txt'],
        graph=True
    )
    assert isinstance(result_txt.graph, nx.Graph)
    assert result_txt.graph.number_of_nodes() > 0

def test_column_intersection(small_named_csr_data):
    """Test handling of column intersections with named columns."""
    blocker = Blocker()
    x_csr, y_csr, x_cols, y_cols = small_named_csr_data
    
    result = blocker.block(
        x_csr,
        y=y_csr,
        x_colnames=x_cols,
        y_colnames=y_cols
    )
    colnames_test = np.intersect1d(x_cols, y_cols)
    assert isinstance(result, BlockingResult)
    assert len(result.colnames) <= len(x_cols)
    assert all(col in result.colnames for col in colnames_test)

def test_verbosity_levels(small_named_txt_data, caplog):
    """Test different verbosity levels."""
    blocker = Blocker()
    x_txt, _ = small_named_txt_data
    
    with caplog.at_level(logging.INFO):
        blocker.block(x_txt['txt'], verbose=0, ann='faiss', control_ann={'faiss': {'k_search': 3}})
    assert len(caplog.records) == 0
    
    caplog.clear()
    with caplog.at_level(logging.INFO):
        blocker.block(x_txt['txt'], verbose=1)
    assert len(caplog.records) > 0

def test_text_data_with_names(small_named_txt_data, small_named_csr_data):
    """Test that text data ignores colnames parameters while matrix data requires them."""
    blocker = Blocker()
    x_txt, _ = small_named_txt_data
    _, _, x_cols, _ = small_named_csr_data
    
    result_txt = blocker.block(
        x_txt['txt'],
        x_colnames=x_cols,
        y_colnames=x_cols
    )
    assert isinstance(result_txt, BlockingResult)

def test_true_blocks_linkage(small_named_txt_data):
    """Test true blocks validation and metrics calculation for linkage."""
    blocker = Blocker()
    x_txt, y_txt = small_named_txt_data

    true_blocks_linkage = pd.DataFrame({
        'x': [0, 1],
        'y': [0, 1],
        'block': [0, 1]
    })

    result = blocker.block(
        x_txt['txt'],
        y=y_txt['txt'],
        true_blocks=true_blocks_linkage,
        deduplication=False
    )

    assert hasattr(result, 'metrics')
    assert hasattr(result, 'confusion')
    assert isinstance(result.metrics, pd.Series)
    assert isinstance(result.confusion, pd.DataFrame)

    expected_metrics = ['recall', 'precision', 'f1_score', 'accuracy', 
                        'specificity', 'fpr', 'fnr']
    assert all(metric in result.metrics for metric in expected_metrics)

    assert result.confusion.shape == (2, 2) 
    assert set(result.confusion.index) == {True, False}
    assert set(result.confusion.columns) == {True, False}

def test_true_blocks_deduplication(small_named_txt_data):
    """Test true blocks validation and metrics calculation for deduplication."""
    blocker = Blocker()
    x_txt, _ = small_named_txt_data
    
    true_blocks_dedup = pd.DataFrame({
        'x': [0, 1, 2, 3],  
        'block': [0, 0, 0, 0]  
    })
    
    result = blocker.block(
        x_txt['txt'],
        true_blocks=true_blocks_dedup,
        deduplication=True
    )
    
    assert hasattr(result, 'metrics')
    assert hasattr(result, 'confusion')
    assert isinstance(result.metrics, pd.Series)
    assert isinstance(result.confusion, pd.DataFrame)
    
    expected_metrics = ['recall', 'precision', 'f1_score', 'accuracy', 
                       'specificity', 'fpr', 'fnr']
    assert all(metric in result.metrics for metric in expected_metrics)
    
    assert result.confusion.shape == (2, 2) 
    assert set(result.confusion.index) == {True, False}
    assert set(result.confusion.columns) == {True, False}

def test_true_blocks_validation_errors(small_named_txt_data):
    """Test error handling for invalid true blocks format."""
    blocker = Blocker()
    x_txt, y_txt = small_named_txt_data
    
    invalid_linkage = pd.DataFrame({
        'x': [0, 1],
        'y': [0, 1]
    })
    
    with pytest.raises(ValueError):
        blocker.block(
            x_txt['txt'],
            y=y_txt['txt'],
            true_blocks=invalid_linkage,
            deduplication=False
        )
    
    invalid_dedup = pd.DataFrame({
        'x': [0, 1],
        'y': [1, 2],
        'block': [0, 1]
    })
    
    with pytest.raises(ValueError):
        blocker.block(
            x_txt['txt'],
            true_blocks=invalid_dedup,
            deduplication=True
        )

@pytest.mark.parametrize("algo", ["hnsw", "annoy", "faiss", "voyager"])
def test_validations(small_named_txt_data, algo):
    """Test error handling for invalid distance metric."""
    blocker = Blocker()
    x_txt, _ = small_named_txt_data
    
    with pytest.raises(ValueError, match="Distance for"):
        blocker.block(
            x_txt['txt'],
            ann=algo,
            control_ann={algo: {'distance': 'bad_distance'}}
        )
