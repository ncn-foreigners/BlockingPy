"""Tests for the main blocker."""

import importlib.util
import logging

import numpy as np
import pandas as pd
import pytest

from blockingpy.blocker import (
    Blocker,
    BlockingResult,
)

HAS_FAISS = importlib.util.find_spec("faiss") is not None


def available_algos():
    algos = ["nnd", "hnsw", "annoy", "voyager"]
    if HAS_FAISS:
        algos.append("faiss")
    return algos


def test_input_validation_types(
    small_named_csr_data, small_named_ndarray_data, small_named_txt_data
):
    """Test input validation for different data types."""
    blocker = Blocker()
    x_csr, y_csr, x_cols_csr, y_cols_csr = small_named_csr_data
    x_ndarray, y_ndarray, x_cols_ndarray, y_cols_ndarray = small_named_ndarray_data
    x_txt, y_txt = small_named_txt_data

    result_csr = blocker.block(
        x_csr, y=y_csr, x_colnames=x_cols_csr, y_colnames=y_cols_csr, ann="hnsw"
    )
    assert isinstance(result_csr, BlockingResult)

    result_ndarray = blocker.block(
        x_ndarray,
        y=y_ndarray,
        x_colnames=x_cols_ndarray,
        y_colnames=y_cols_ndarray,
        ann="hnsw",
    )
    assert isinstance(result_ndarray, BlockingResult)

    result_txt = blocker.block(x_txt["txt"], y=y_txt["txt"], ann="hnsw")
    assert isinstance(result_txt, BlockingResult)

    with pytest.raises(ValueError):
        blocker.block([1, 2, 3], ann="hnsw")
    with pytest.raises(ValueError):
        blocker.block(pd.DataFrame({"a": [1, 2, 3]}), ann="hnsw")


@pytest.mark.parametrize("algo", available_algos())
def test_algorithm_selection(algo, small_named_csr_data, small_named_txt_data):
    """Test different algorithms with both matrix and text inputs."""
    blocker = Blocker()
    x_csr, y_csr, x_cols, y_cols = small_named_csr_data
    x_txt, y_txt = small_named_txt_data

    result_csr = blocker.block(x_csr, y=y_csr, x_colnames=x_cols, y_colnames=y_cols, ann=algo)
    assert isinstance(result_csr, BlockingResult)
    assert result_csr.method == algo

    result_txt = blocker.block(x_txt["txt"], y=y_txt["txt"], ann=algo)
    assert isinstance(result_txt, BlockingResult)
    assert result_txt.method == algo


@pytest.mark.parametrize("algo", available_algos())
def test_algos_with_embedding(algo, small_named_txt_data):
    """Test different algorithms with embeddings."""
    blocker = Blocker()
    x_txt, y_txt = small_named_txt_data

    control_txt = {"encoder": "embedding", "embedding": {"model": "minishlab/potion-base-8M"}}
    result_txt = blocker.block(x_txt["txt"], y=y_txt["txt"], ann=algo, control_txt=control_txt)
    assert isinstance(result_txt, BlockingResult)
    assert result_txt.method == algo


def test_deduplication_vs_linkage(small_named_csr_data, small_named_txt_data):
    """Test deduplication and linkage with both matrix and text data."""
    blocker = Blocker()
    x_csr, y_csr, x_cols, y_cols = small_named_csr_data
    x_txt, y_txt = small_named_txt_data

    dedup_result_csr = blocker.block(
        x_csr, x_colnames=x_cols, y_colnames=x_cols, deduplication=True
    )
    assert isinstance(dedup_result_csr, BlockingResult)
    assert dedup_result_csr.deduplication

    link_result_csr = blocker.block(
        x_csr, y=y_csr, x_colnames=x_cols, y_colnames=y_cols, deduplication=False
    )
    assert isinstance(link_result_csr, BlockingResult)
    assert not link_result_csr.deduplication

    dedup_result_txt = blocker.block(x_txt["txt"], deduplication=True)
    assert isinstance(dedup_result_txt, BlockingResult)
    assert dedup_result_txt.deduplication

    link_result_txt = blocker.block(x_txt["txt"], y=y_txt["txt"], deduplication=False)
    assert isinstance(link_result_txt, BlockingResult)
    assert not link_result_txt.deduplication


def test_column_intersection(small_named_csr_data):
    """Test handling of column intersections with named columns."""
    blocker = Blocker()
    x_csr, y_csr, x_cols, y_cols = small_named_csr_data

    result = blocker.block(x_csr, y=y_csr, x_colnames=x_cols, y_colnames=y_cols)
    colnames_test = np.intersect1d(x_cols, y_cols)
    assert isinstance(result, BlockingResult)
    assert len(result.colnames) <= len(x_cols)
    assert all(col in result.colnames for col in colnames_test)


def test_verbosity_levels(small_named_txt_data, caplog):
    blocker = Blocker()
    x_txt, _ = small_named_txt_data

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="blockingpy"):
        blocker.block(x_txt["txt"], verbose=0, ann="hnsw", control_ann={"hnsw": {"k_search": 3}})
    assert not any(
        r.name.startswith("blockingpy") and r.levelno == logging.INFO for r in caplog.records
    )

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="blockingpy"):
        blocker.block(x_txt["txt"], verbose=1)
    assert any(
        r.name.startswith("blockingpy") and r.levelno == logging.INFO for r in caplog.records
    )


def test_text_data_with_names(small_named_txt_data, small_named_csr_data):
    """Test that text data ignores colnames parameters while matrix data requires them."""
    blocker = Blocker()
    x_txt, _ = small_named_txt_data
    _, _, x_cols, _ = small_named_csr_data

    result_txt = blocker.block(x_txt["txt"], x_colnames=x_cols, y_colnames=x_cols)
    assert isinstance(result_txt, BlockingResult)


def test_true_blocks_linkage(small_named_txt_data):
    """Test true blocks validation and metrics calculation for linkage."""
    blocker = Blocker()
    x_txt, y_txt = small_named_txt_data

    true_blocks_linkage = pd.DataFrame({"x": [0, 1], "y": [0, 1], "block": [0, 1]})

    result = blocker.block(
        x_txt["txt"],
        y=y_txt["txt"],
        true_blocks=true_blocks_linkage,
        deduplication=False,
    )

    assert hasattr(result, "metrics")
    assert hasattr(result, "confusion")
    assert isinstance(result.metrics, pd.Series)
    assert isinstance(result.confusion, pd.DataFrame)

    expected_metrics = [
        "recall",
        "precision",
        "f1_score",
        "accuracy",
        "specificity",
        "fpr",
        "fnr",
    ]
    assert all(metric in result.metrics for metric in expected_metrics)

    assert result.confusion.shape == (2, 2)
    assert set(result.confusion.index) == {"Actual Positive", "Actual Negative"}
    assert set(result.confusion.columns) == {"Predicted Positive", "Predicted Negative"}


def test_true_blocks_deduplication(small_named_txt_data):
    """Test true blocks validation and metrics calculation for deduplication."""
    blocker = Blocker()
    x_txt, _ = small_named_txt_data

    true_blocks_dedup = pd.DataFrame({"x": [0, 1, 2, 3], "block": [0, 0, 0, 0]})

    result = blocker.block(x_txt["txt"], true_blocks=true_blocks_dedup, deduplication=True)

    assert hasattr(result, "metrics")
    assert hasattr(result, "confusion")
    assert isinstance(result.metrics, pd.Series)
    assert isinstance(result.confusion, pd.DataFrame)

    expected_metrics = [
        "recall",
        "precision",
        "f1_score",
        "accuracy",
        "specificity",
        "fpr",
        "fnr",
    ]
    assert all(metric in result.metrics for metric in expected_metrics)

    assert result.confusion.shape == (2, 2)
    assert set(result.confusion.index) == {"Actual Positive", "Actual Negative"}
    assert set(result.confusion.columns) == {"Predicted Positive", "Predicted Negative"}


def test_true_blocks_validation_errors(small_named_txt_data):
    """Test error handling for invalid true blocks format."""
    blocker = Blocker()
    x_txt, y_txt = small_named_txt_data

    invalid_linkage = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    with pytest.raises(ValueError):
        blocker.block(
            x_txt["txt"],
            y=y_txt["txt"],
            true_blocks=invalid_linkage,
            deduplication=False,
        )

    invalid_dedup = pd.DataFrame({"x": [0, 1], "y": [1, 2], "block": [0, 1]})

    with pytest.raises(ValueError):
        blocker.block(x_txt["txt"], true_blocks=invalid_dedup, deduplication=True)


@pytest.mark.parametrize(
    "algo", ["hnsw", "annoy", "faiss", "voyager"] if HAS_FAISS else ["hnsw", "annoy", "voyager"]
)
def test_metric_validations_error(small_named_txt_data, algo):
    """Test error handling for invalid distance metric."""
    blocker = Blocker()
    x_txt, _ = small_named_txt_data

    with pytest.raises(ValueError):
        blocker.block(x_txt["txt"], ann=algo, control_ann={algo: {"distance": "bad_distance"}})


def test_algo_validation_error(small_named_txt_data):
    """Test validation for different distance metrics."""
    blocker = Blocker()
    x_txt, _ = small_named_txt_data

    with pytest.raises(ValueError, match="Unsupported algorithm"):
        blocker.block(
            x_txt["txt"],
            ann="bad_algo",
        )


@pytest.mark.parametrize("bad_input", [1, "bad_x", True])
def test_input_x_validation(bad_input):
    """Test input validation for x."""
    with pytest.raises(ValueError):
        blocker = Blocker()
        blocker.block(bad_input, ann="hnsw")


# Tests for eval method:


def test_eval_basic_functionality(small_named_txt_data):
    """Test basic functionality of eval method."""
    blocker = Blocker()
    x_txt, y_txt = small_named_txt_data
    true_blocks = pd.DataFrame({"x": [0, 1], "y": [0, 1], "block": [0, 1]})

    result_with_eval = blocker.block(
        x_txt["txt"], y=y_txt["txt"], true_blocks=true_blocks, deduplication=False
    )

    result_no_eval = blocker.block(x_txt["txt"], y=y_txt["txt"], deduplication=False)
    eval_result = blocker.eval(result_no_eval, true_blocks)

    assert result_no_eval.metrics is None
    assert result_no_eval.confusion is None

    pd.testing.assert_series_equal(eval_result.metrics, result_with_eval.metrics)
    pd.testing.assert_frame_equal(eval_result.confusion, result_with_eval.confusion)

    assert eval_result.method == result_no_eval.method
    assert eval_result.deduplication == result_no_eval.deduplication
    assert eval_result.n_original_records == result_no_eval.n_original_records
    pd.testing.assert_frame_equal(eval_result.result, result_no_eval.result)


def test_eval_input_validation(small_named_txt_data):
    """Test input type validation for eval method."""
    blocker = Blocker()
    x_txt, _ = small_named_txt_data

    _ = blocker.block(x_txt["txt"])

    with pytest.raises(ValueError, match="must be a BlockingResult instance"):
        blocker.eval(pd.DataFrame(), pd.DataFrame({"x": [0], "block": [0]}))


@pytest.mark.parametrize("deduplication", [True, False])
def test_eval_true_blocks_validation(small_named_txt_data, deduplication):
    """Test true_blocks format validation for eval method."""
    blocker = Blocker()
    x_txt, y_txt = small_named_txt_data

    block_result = blocker.block(
        x_txt["txt"], y=y_txt["txt"] if not deduplication else None, deduplication=deduplication
    )

    if deduplication:
        invalid_true_blocks = pd.DataFrame({"x": [0], "y": [0], "block": [0]})
    else:
        invalid_true_blocks = pd.DataFrame({"x": [0], "block": [0]})

    with pytest.raises(ValueError):
        blocker.eval(block_result, invalid_true_blocks)


def test_eval_metrics_structure(small_named_txt_data):
    """Test structure and content of evaluation metrics."""
    blocker = Blocker()
    x_txt, y_txt = small_named_txt_data
    true_blocks = pd.DataFrame({"x": [0, 1], "y": [0, 1], "block": [0, 1]})

    result = blocker.block(x_txt["txt"], y=y_txt["txt"], deduplication=False)
    eval_result = blocker.eval(result, true_blocks)

    expected_metrics = {"recall", "precision", "fpr", "fnr", "accuracy", "specificity", "f1_score"}
    assert set(eval_result.metrics.index) == expected_metrics
    assert all(0 <= val <= 1 for val in eval_result.metrics.to_numpy())

    assert eval_result.confusion.shape == (2, 2)
    assert list(eval_result.confusion.index) == ["Actual Positive", "Actual Negative"]
    assert list(eval_result.confusion.columns) == ["Predicted Positive", "Predicted Negative"]
    assert (eval_result.confusion >= 0).all().all()


@pytest.mark.parametrize("algo", available_algos())
def test_eval_different_algorithms(small_named_txt_data, algo):
    """Test eval method works with different blocking algos."""
    blocker = Blocker()
    x_txt, y_txt = small_named_txt_data
    true_blocks = pd.DataFrame({"x": [0, 1], "y": [0, 1], "block": [0, 1]})

    result = blocker.block(x_txt["txt"], y=y_txt["txt"], ann=algo, deduplication=False)
    eval_result = blocker.eval(result, true_blocks)

    assert isinstance(eval_result, BlockingResult)
    assert eval_result.method == algo
    assert isinstance(eval_result.metrics, pd.Series)
    assert isinstance(eval_result.confusion, pd.DataFrame)


def test_eval_empty_blocks(small_named_txt_data):
    """Test eval method with empty blocking results."""
    blocker = Blocker()
    x_txt, y_txt = small_named_txt_data

    true_blocks = pd.DataFrame({"x": [], "y": [], "block": []})

    result = blocker.block(x_txt["txt"], y=y_txt["txt"], deduplication=False)
    eval_result = blocker.eval(result, true_blocks)

    assert eval_result.metrics is not None
    assert eval_result.confusion is not None
    assert eval_result.confusion.shape == (2, 2)
