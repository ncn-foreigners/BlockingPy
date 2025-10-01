"""Tests for the GPU FAISS blocker."""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from blockingpy.data_handler import DataHandler

pytestmark = pytest.mark.requires_faiss_gpu
pytest.importorskip("faiss", reason="faiss [gpu] not installed")


@pytest.fixture
def gpu_faiss_blocker():
    from blockingpy.gpu_faiss_blocker import GPUFaissBlocker

    return GPUFaissBlocker()


@pytest.fixture
def gpu_faiss_controls():
    """Default GPU FAISS parameters"""
    return {
        "gpu_faiss": {
            "index_type": "flat",
            "k_search": 5,
            "distance": "cosine",
            "path": None,
            "ivf_nlist": 64,
            "ivf_nprobe": 8,
            "ivfpq_nlist": 64,
            "ivfpq_m": 8,
            "ivfpq_nbits": 8,
            "ivfpq_nprobe": 8,
            "ivfpq_useFloat16": False,
            "ivfpq_usePrecomputed": False,
            "ivfpq_reserveVecs": 0,
            "ivfpq_use_cuvs": False,
            "train_size": None,
            "cagra": {
                "graph_degree": 64,
                "intermediate_graph_degree": 96,
                "build_algo": "ivf_pq",
                "itopk_size": 64,
            },
        }
    }


def test_basic_blocking_gpu(gpu_faiss_blocker, small_sparse_data, gpu_faiss_controls):
    x, y = small_sparse_data
    res = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=gpu_faiss_controls)
    assert isinstance(res, pd.DataFrame)
    assert set(res.columns) == {"x", "y", "dist"}
    assert len(res) == y.shape[0]
    assert res["dist"].notna().all()


@pytest.mark.parametrize("distance", ["euclidean", "l2", "inner_product", "cosine"])
def test_supported_metrics_gpu(gpu_faiss_blocker, small_sparse_data, gpu_faiss_controls, distance):
    x, y = small_sparse_data
    controls = gpu_faiss_controls.copy()
    controls["gpu_faiss"]["distance"] = distance
    res = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert isinstance(res, pd.DataFrame)
    assert res["dist"].notna().all()
    if distance == "cosine":
        assert (res["dist"] > -1.1).all() and (res["dist"] < 2.1).all()


def test_result_reproducibility_gpu(gpu_faiss_blocker, small_sparse_data, gpu_faiss_controls):
    x, y = small_sparse_data
    r1 = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=gpu_faiss_controls)
    r2 = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=gpu_faiss_controls)
    pd.testing.assert_frame_equal(r1, r2)


def test_k_search_guard_gpu(gpu_faiss_blocker, small_sparse_data, gpu_faiss_controls):
    x, y = small_sparse_data
    controls = gpu_faiss_controls.copy()
    controls["gpu_faiss"]["k_search"] = x.shape[0] + 10

    with pytest.warns(UserWarning, match=r"k_search.*exceeds.*clipping"):
        _ = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)


@pytest.mark.parametrize("index_type", ["flat"])
def test_index_flat_gpu(gpu_faiss_blocker, large_sparse_data, gpu_faiss_controls, index_type):
    x, y = large_sparse_data
    controls = gpu_faiss_controls.copy()
    controls["gpu_faiss"]["index_type"] = index_type
    res = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert len(res) == y.shape[0]


def test_index_ivf_gpu(gpu_faiss_blocker, large_sparse_data, gpu_faiss_controls):
    x, y = large_sparse_data
    controls = gpu_faiss_controls.copy()
    controls["gpu_faiss"].update(
        {"index_type": "ivf", "ivf_nlist": 32, "ivf_nprobe": 4, "train_size": min(x.shape[0], 50)}
    )
    res = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert len(res) == y.shape[0]


@pytest.mark.parametrize(
    "pq_kwargs",
    [
        {"ivfpq_m": 4, "ivfpq_nbits": 8, "ivfpq_useFloat16": False, "ivfpq_usePrecomputed": False},
        {"ivfpq_m": 4, "ivfpq_nbits": 8, "ivfpq_useFloat16": True, "ivfpq_usePrecomputed": False},
        {"ivfpq_m": 4, "ivfpq_nbits": 4, "ivfpq_useFloat16": False, "ivfpq_usePrecomputed": True},
    ],
)
def test_index_ivfpq_gpu(gpu_faiss_blocker, large_sparse_data, gpu_faiss_controls, pq_kwargs):
    x, y = large_sparse_data
    controls = gpu_faiss_controls.copy()
    controls["gpu_faiss"].update(
        {
            "index_type": "ivfpq",
            "ivfpq_nlist": 32,
            "ivfpq_nprobe": 4,
            **pq_kwargs,
        }
    )
    if pq_kwargs.get("ivfpq_nbits", 8) != 8:
        pytest.xfail("FAISS GPU IVFPQ supports only 8-bit codes.")
    res = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert len(res) == y.shape[0]


def test_index_cagra_gpu(gpu_faiss_blocker, small_sparse_data, gpu_faiss_controls):
    x, y = small_sparse_data
    controls = gpu_faiss_controls.copy()
    controls["gpu_faiss"]["index_type"] = "cagra"
    controls["gpu_faiss"]["k_search"] = 5
    try:
        res = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    except NotImplementedError:
        pytest.xfail("CAGRA path not implemented in current build.")
    else:
        assert len(res) == y.shape[0]


def test_identical_points_gpu(gpu_faiss_blocker, identical_sparse_data, gpu_faiss_controls):
    x, y = identical_sparse_data
    controls = gpu_faiss_controls.copy()
    controls["gpu_faiss"]["distance"] = "l2"
    res = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert res["dist"].iloc[0] == pytest.approx(0.0, abs=1e-5)


def test_single_point_gpu(gpu_faiss_blocker, single_sparse_point, gpu_faiss_controls):
    x, y = single_sparse_point
    res = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=gpu_faiss_controls)
    assert len(res) == 1


def test_empty_reference_gpu(gpu_faiss_blocker, gpu_faiss_controls):
    rng = np.random.default_rng(0)
    x = DataHandler(data=rng.random((0, 3)), cols=["a", "b", "c"])
    y = DataHandler(data=rng.random((5, 3)), cols=["a", "b", "c"])

    with pytest.raises(AssertionError):
        gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=gpu_faiss_controls)


def test_save_index_gpu(gpu_faiss_blocker, small_sparse_data, gpu_faiss_controls):
    x, y = small_sparse_data
    with TemporaryDirectory() as tmp:
        controls = gpu_faiss_controls.copy()
        controls["gpu_faiss"]["path"] = tmp
        _ = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
        assert os.path.exists(os.path.join(tmp, "index.faiss"))
        assert os.path.exists(os.path.join(tmp, "index-colnames.txt"))


def test_invalid_save_path_gpu(gpu_faiss_blocker, small_sparse_data, gpu_faiss_controls):
    x, y = small_sparse_data
    controls = gpu_faiss_controls.copy()
    controls["gpu_faiss"]["path"] = "/this/path/should/not/exist"
    with pytest.raises(ValueError, match="Provided path is not a directory"):
        gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)


@pytest.mark.parametrize("train_frac", [None, 0.2, 0.5, 1.0])
def test_train_size_respected_gpu(
    gpu_faiss_blocker, large_sparse_data, gpu_faiss_controls, train_frac
):
    """
    Ensure we can pass a train subset size for IVF/IVFPQ without errors.
    This doesn't validate internal FAISS state—just that our code path works.
    """
    x, y = large_sparse_data
    controls = gpu_faiss_controls.copy()
    controls["gpu_faiss"]["index_type"] = "ivfpq"
    controls["gpu_faiss"]["ivfpq_nlist"] = 16
    controls["gpu_faiss"]["ivfpq_nprobe"] = 4

    controls["gpu_faiss"]["ivfpq_m"] = 4

    if train_frac is None:
        controls["gpu_faiss"]["train_size"] = None
    else:
        controls["gpu_faiss"]["train_size"] = max(1, int(x.shape[0] * train_frac))

    res = gpu_faiss_blocker.block(x=x, y=y, k=1, verbose=False, controls=controls)
    assert len(res) == y.shape[0]
