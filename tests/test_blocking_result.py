"""Tests for BlockingResult class."""

import numpy as np
import pandas as pd
import pytest

from blockingpy.blocking_result import BlockingResult


def _mk(df, *, deduplication, metrics, cols=("a", "b"), rr=0.123456):
    """Helper for some tests."""
    br = BlockingResult.__new__(BlockingResult)
    br.result = df
    br.deduplication = deduplication
    br.method = "faiss"
    br.colnames = list(cols)
    br.reduction_ratio = rr
    br.metrics = metrics
    if not hasattr(br, "_format_metrics"):

        def _format_metrics():
            if br.metrics is None:
                return {}
            return {k: float(f"{v * 100:.4f}") for k, v in br.metrics.items()}

        br._format_metrics = _format_metrics
    return br


@pytest.fixture
def make_br_dedup():
    res_df = pd.DataFrame({"x": [1, 2], "y": [0, 3], "block": [0, 1], "dist": [0.1, 0.2]})

    return BlockingResult(
        x_df=res_df,
        ann="test",
        deduplication=True,
        n_original_records=(4, 4),
        true_blocks=None,
        eval_metrics=None,
        confusion=None,
        colnames_xy=np.array([0]),
    )


def test_add_block_column_dedup_orphans(make_br_dedup):
    br = make_br_dedup
    df = pd.DataFrame({"val": ["a", "b", "c", "d"]})
    out = br.add_block_column(df)

    assert list(out["block"]) == [0, 0, 1, 1]
    assert out["block"].dtype == np.int64


def test_add_block_column_reclink_orphans():
    res_df = pd.DataFrame({"x": [0, 2], "y": [1, 2], "block": [0, 1], "dist": [0.1, 0.2]})
    br = BlockingResult(
        x_df=res_df,
        ann="test",
        deduplication=False,
        n_original_records=(3, 3),
        true_blocks=None,
        eval_metrics=None,
        confusion=None,
        colnames_xy=np.array([0]),
    )

    left = pd.DataFrame({"L": ["a", "b", "c"]})
    right = pd.DataFrame({"R": ["x", "y", "z"]})
    out_l, out_r = br.add_block_column(left, right)

    assert list(out_l["block"]) == [0, 2, 1]
    assert out_l["block"].dtype == np.int64

    assert list(out_r["block"]) == [3, 0, 1]
    assert out_r["block"].dtype == np.int64


def test_str_dedupulation_branch_with_metrics():
    df = pd.DataFrame(
        {
            "block": [1, 1, 2, 2],
            "x": [1, 2, 10, 10],
            "y": [2, 3, 10, 11],
        }
    )
    metrics = pd.Series({"precision": 0.9, "recall": 0.8})
    br = _mk(df, deduplication=True, metrics=metrics, cols=("fname", "lname"), rr=0.5)

    s = str(br)

    assert "Blocking based on the faiss method." in s
    assert "Number of columns created for blocking: 2" in s
    assert "Reduction ratio: 0.500000" in s

    assert "Block Size" in s and "Number of Blocks" in s
    assert "         2 | 1" in s
    assert "         3 | 1" in s

    assert "Evaluation metrics (standard):" in s
    fm = br._format_metrics()
    assert fm == {"precision": 90.0, "recall": 80.0}
    assert "precision : 90.0" in s
    assert "recall : 80.0" in s


def test_str_linkage_branch_without_metrics():
    df = pd.DataFrame(
        {
            "block": [1, 1, 2, 2, 2],
            "x": [1, 1, 2, 3, 3],
            "y": [4, 5, 5, 5, 6],
        }
    )
    br = _mk(df, deduplication=False, metrics=None, cols=("txt",), rr=0.123456)

    s = str(br)

    assert "Blocking based on the faiss method." in s
    assert "Number of blocks: 2" in s
    assert "Number of columns created for blocking: 1" in s
    assert "Reduction ratio: 0.123456" in s

    assert "         3 | 1" in s
    assert "         4 | 1" in s

    assert "Evaluation metrics" not in s


def test__format_metrics_none_and_values():
    df = pd.DataFrame({"block": [], "x": [], "y": []})
    br_none = _mk(df, deduplication=True, metrics=None)
    assert br_none._format_metrics() == {}

    br_vals = _mk(df, deduplication=True, metrics=pd.Series({"f1": 0.3333333}))
    out = br_vals._format_metrics()
    assert out["f1"] == 33.3333
