import os
import sys
import types

if os.getenv("BPY_GPU_ONLY") == "1":

    def _stub(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules.setdefault(name, m)

    _stub("igraph", Graph=type("Graph", (), {}))
    _stub("tokenizers", Encoding=object, Tokenizer=object)
    _stub("safetensors")
    _stub("sentencepiece")
    _stub("model2vec", StaticModel=type("StaticModel", (), {}))

import pytest

GPU_ONLY = os.getenv("BPY_GPU_ONLY") == "1"
GPU_MOCK = os.getenv("BPY_GPU_MOCK") == "1"

if os.getenv("BPY_GPU_MOCK") == "1":
    try:
        import faiss

        if not hasattr(faiss, "get_num_gpus"):
            faiss.get_num_gpus = lambda: 1
        if not hasattr(faiss, "index_cpu_to_all_gpus"):
            faiss.index_cpu_to_all_gpus = lambda idx: idx
        if not hasattr(faiss, "index_cpu_to_gpu"):
            faiss.index_cpu_to_gpu = lambda res, dev, idx: idx
        if not hasattr(faiss, "index_gpu_to_cpu"):
            faiss.index_gpu_to_cpu = lambda idx: idx
    except Exception:
        pass

if GPU_ONLY:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

if GPU_MOCK:
    try:
        import faiss

        if not hasattr(faiss, "get_num_gpus"):
            faiss.get_num_gpus = lambda: 1
        else:
            faiss.get_num_gpus = lambda: 1
        if not hasattr(faiss, "index_cpu_to_all_gpus"):
            faiss.index_cpu_to_all_gpus = lambda idx: idx
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _gpu_mock(monkeypatch):
    if os.getenv("BPY_GPU_MOCK") != "1":
        return
    import faiss

    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 1, raising=False)

    def _cpu_to_gpu(res, dev, idx, *args, **kwargs):
        return idx

    def _cpu_to_all_gpus(idx, *args, **kwargs):
        return idx

    def _gpu_to_cpu(idx, *args, **kwargs):
        return idx

    monkeypatch.setattr(faiss, "index_cpu_to_gpu", _cpu_to_gpu, raising=False)
    monkeypatch.setattr(faiss, "index_cpu_to_all_gpus", _cpu_to_all_gpus, raising=False)
    monkeypatch.setattr(faiss, "index_gpu_to_cpu", _gpu_to_cpu, raising=False)

    if not hasattr(faiss, "StandardGpuResources"):

        class _StdRes: ...

        monkeypatch.setattr(faiss, "StandardGpuResources", _StdRes, raising=False)

    if not hasattr(faiss, "GpuClonerOptions"):

        class GpuClonerOptions:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        monkeypatch.setattr(faiss, "GpuClonerOptions", GpuClonerOptions, raising=False)

    if not hasattr(faiss, "INDICES_CPU"):
        monkeypatch.setattr(faiss, "INDICES_CPU", 0, raising=False)
    if not hasattr(faiss, "INDICES_32_BIT"):
        monkeypatch.setattr(faiss, "INDICES_32_BIT", 1, raising=False)

    if not hasattr(faiss, "GpuIndexCagra"):

        class GpuIndexCagra:
            def __init__(self, *args, **kwargs):
                self.ntotal = 0
                self.d = kwargs.get("d")

            def is_trained(self):
                return True

            def train(self, x):
                pass

            def add(self, x):
                try:
                    self.ntotal += x.shape[0]
                except Exception:
                    self.ntotal += len(x)

            def search(self, x, k):
                import numpy as np

                n = getattr(x, "shape", (len(x),))[0]
                return np.zeros((n, k), dtype="float32"), np.zeros((n, k), dtype="int64")

        monkeypatch.setattr(faiss, "GpuIndexCagra", GpuIndexCagra, raising=False)

    if not hasattr(faiss, "GpuParameterSpace"):

        class GpuParameterSpace:
            def initialize(self, index):
                pass

            def set_index_parameter(self, index, key, val):
                pass

        monkeypatch.setattr(faiss, "GpuParameterSpace", GpuParameterSpace, raising=False)


if not GPU_ONLY:

    @pytest.fixture
    def blocker():
        from blockingpy import Blocker

        return Blocker()

    @pytest.fixture
    def mlpack_blocker():
        pytest.importorskip("mlpack", reason="mlpack not installed")
        from blockingpy.mlpack_blocker import MLPackBlocker

        return MLPackBlocker()

    @pytest.fixture
    def nnd_blocker():
        from blockingpy.nnd_blocker import NNDBlocker

        return NNDBlocker()

    @pytest.fixture
    def voyager_blocker():
        from blockingpy.voyager_blocker import VoyagerBlocker

        return VoyagerBlocker()

    @pytest.fixture
    def faiss_blocker():
        pytest.importorskip("faiss", reason="FAISS not installed")
        from blockingpy.faiss_blocker import FaissBlocker

        return FaissBlocker()

    @pytest.fixture
    def hnsw_blocker():
        from blockingpy.hnsw_blocker import HNSWBlocker

        return HNSWBlocker()

    @pytest.fixture
    def annoy_blocker():
        from blockingpy.annoy_blocker import AnnoyBlocker

        return AnnoyBlocker()

    @pytest.fixture
    def lsh_controls():
        return {
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
        }

    @pytest.fixture
    def kd_controls():
        return {
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
        }


def _has_faiss_gpu() -> bool:
    try:
        import faiss

        return hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
    except Exception:
        return False


def _has_mlpack() -> bool:
    try:
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    have_gpu = _has_faiss_gpu()
    have_ml = _has_mlpack()
    for item in items:
        if "requires_faiss_gpu" in item.keywords and not have_gpu:
            item.add_marker(pytest.mark.skip(reason="FAISS GPU not available"))
        if "requires_mlpack" in item.keywords and not have_ml:
            item.add_marker(pytest.mark.skip(reason="mlpack not installed"))


@pytest.fixture
def small_sparse_data():
    import numpy as np
    from scipy import sparse

    from blockingpy.data_handler import DataHandler

    rng = np.random.default_rng()
    x = sparse.csr_matrix(rng.random((5, 3)))
    y = sparse.csr_matrix(rng.random((5, 3)))
    return DataHandler(x, [f"x_col_{i}" for i in range(x.shape[1])]), DataHandler(
        y, [f"y_col_{i}" for i in range(y.shape[1])]
    )


@pytest.fixture
def large_sparse_data():
    import numpy as np
    from scipy import sparse

    from blockingpy.data_handler import DataHandler

    np.random.default_rng(42)
    x = sparse.random(2000, 20, density=0.1, format="csr")
    y = sparse.random(1000, 20, density=0.1, format="csr")
    return DataHandler(x, [f"x_col_{i}" for i in range(x.shape[1])]), DataHandler(
        y, [f"y_col_{i}" for i in range(y.shape[1])]
    )


@pytest.fixture
def small_named_csr_data():
    import numpy as np
    from scipy import sparse

    rng = np.random.default_rng()
    x = sparse.csr_matrix(rng.random((5, 3)))
    y = sparse.csr_matrix(rng.random((5, 3)))
    return x, y, ["ja", "nk", "ko"], ["ja", "nk", "ok"]


@pytest.fixture
def small_named_ndarray_data():
    import numpy as np

    rng = np.random.default_rng(42)
    x = rng.random((5, 3))
    y = rng.random((5, 3))
    return x, y, ["ja", "nk", "ko"], ["ja", "nk", "ok"]


@pytest.fixture
def small_named_txt_data():
    import pandas as pd

    x = pd.DataFrame(
        {
            "txt": [
                "jankowalski",
                "kowalskijan",
                "kowalskimjan",
                "kowaljan",
                "montypython",
                "pythonmonty",
                "cyrkmontypython",
                "monty",
            ]
        }
    )
    y = pd.DataFrame({"txt": ["montypython", "kowalskijan", "other"]})
    return x, y


@pytest.fixture
def identical_sparse_data():
    import numpy as np
    from scipy import sparse

    from blockingpy.data_handler import DataHandler

    data = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    x = sparse.csr_matrix(data)
    y = sparse.csr_matrix(data[0:1])
    return DataHandler(x, [f"x_col_{i}" for i in range(x.shape[1])]), DataHandler(
        y, [f"y_col_{i}" for i in range(y.shape[1])]
    )


@pytest.fixture
def single_sparse_point():
    from scipy import sparse

    from blockingpy.data_handler import DataHandler

    x = sparse.csr_matrix([[1.0, 2.0]])
    y = sparse.csr_matrix([[1.5, 2.5]])
    return DataHandler(x, [f"x_col_{i}" for i in range(x.shape[1])]), DataHandler(
        y, [f"y_col_{i}" for i in range(y.shape[1])]
    )
