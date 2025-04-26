"""Tests for BlockingResult class."""

import numpy as np
import pandas as pd
import pytest

from blockingpy.blocking_result import BlockingResult

@pytest.fixture
def make_br_dedup():
    res_df = pd.DataFrame({
        "x": [1, 2],
        "y": [0, 3],
        "block": [0, 1],
        "dist": [0.1, 0.2]
    })

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
    res_df = pd.DataFrame({
        "x": [0, 2],
        "y": [1, 2],
        "block": [0, 1],
        "dist": [0.1, 0.2]
    })
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