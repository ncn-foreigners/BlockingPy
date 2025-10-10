"""GPU Faiss Blocker."""

from __future__ import annotations

import gc
import logging
import os
import random
import warnings
from typing import Any, Literal

import faiss
import numpy as np
import pandas as pd

from .base import BlockingMethod
from .data_handler import DataHandler
from .helper_functions import rearrange_array

logger = logging.getLogger(__name__)

_IndexType = Literal["flat", "ivf", "ivfpq", "cagra"]


class GPUFaissBlocker(BlockingMethod):
    """
    A class for performing blocking using the FAISS (Facebook AI Similarity Search) algorithms
    that are GPU-accelerated.

    Parameters
    ----------
    None

    Attributes
    ----------
    index : faiss.Index (gpu)
        The FAISS index used for nearest neighbor search
    x_columns : array-like or None
        Column names of the reference dataset
    METRIC_MAP : dict
        Mapping of distance metric names to FAISS metric types

    See Also
    --------
    BlockingMethod : Abstract base class defining the blocking interface

    Notes
    -----
    The available Index types from FAISS are: 'flat', 'ivf', 'ivfpq' and 'cagra'.

    For more details about the FAISS library and implementation, see:
    https://github.com/facebookresearch/faiss

    """

    def __init__(self) -> None:
        self.index: faiss.Index
        self.x_columns: list[str]

        self.METRIC_MAP: dict[str, int] = {
            "euclidean": faiss.METRIC_L2,
            "l2": faiss.METRIC_L2,
            "inner_product": faiss.METRIC_INNER_PRODUCT,
            "cosine": faiss.METRIC_INNER_PRODUCT,
        }

    def block(  # noqa: PLR0915, PLR0912
        self,
        x: DataHandler,
        y: DataHandler,
        k: int,
        verbose: bool | None,
        controls: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Perform blocking using the GPU FAISS algorithms.

        Parameters
        ----------
        x : DataHandler
            Reference dataset containing features for indexing
        y : DataHandler
            Query dataset to find nearest neighbors for
        k : int
            Number of nearest neighbors to find
        verbose : bool, optional
            If True, print detailed progress information
        controls : dict
            Algorithm control parameters.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the blocking results with columns:
            - 'y': indices from query dataset
            - 'x': indices of matched items from reference dataset
            - 'dist': distances to matched items

        """
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        faiss_ctl = controls["gpu_faiss"]
        distance: str = faiss_ctl.get("distance", "cosine")
        index_type: _IndexType = faiss_ctl.get("index_type", "flat")
        k_search: int = faiss_ctl.get("k_search", 30)
        path: str | None = faiss_ctl.get("path")
        train_size: int | None = faiss_ctl.get("train_size")
        seed: int | None = controls.get("random_seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if index_type not in {"flat", "ivf", "ivfpq", "cagra"}:
            raise ValueError("index_type must be one of 'flat', 'ivf', 'ivfpq', 'cagra'")

        self.x_columns = list(x.cols)
        metric = self.METRIC_MAP[distance]

        x_arr = x.to_dense(dtype=np.float32)
        y_arr = y.to_dense(dtype=np.float32)

        del x, y
        gc.collect()

        if distance == "cosine":
            faiss.normalize_L2(x_arr)
            faiss.normalize_L2(y_arr)

        d = x_arr.shape[1]
        index_cpu = None
        trainable = False

        gpu_res = faiss.StandardGpuResources()

        if index_type == "flat":
            index_cpu = faiss.IndexFlat(d, metric)
        elif index_type == "ivf":
            nlist = faiss_ctl.get("ivf_nlist", 100)
            quantiser = faiss.IndexFlat(d, metric)
            index_cpu = faiss.IndexIVFFlat(quantiser, d, nlist, metric)
            trainable = True
            if seed is not None:
                self._apply_faiss_seeds(index_cpu, seed)
        elif index_type == "ivfpq":
            nlist = faiss_ctl.get("ivfpq_nlist", 100)
            m = faiss_ctl.get("ivfpq_m", 8)
            nbits = faiss_ctl.get("ivfpq_nbits", 8)
            NBITS_SUPPORTED_VAL = 8
            if nbits != NBITS_SUPPORTED_VAL:
                raise ValueError("FAISS GPU IVFPQ requires ivfpq_nbits == 8.")
            if d % m != 0:
                raise ValueError(f"Dimension d={d} must be divisible by m={m} for IVFPQ index.")
            quantiser = faiss.IndexFlat(d, metric)
            index_cpu = faiss.IndexIVFPQ(quantiser, d, nlist, m, nbits)
            trainable = True
            if seed is not None:
                self._apply_faiss_seeds(index_cpu, seed)
        elif index_type == "cagra":
            cctl = faiss_ctl.get("cagra", {})
            CfgCls = getattr(faiss, "GpuIndexCagraConfig", None)
            if CfgCls is not None:
                cagra_cfg = CfgCls()
                if hasattr(cagra_cfg, "graph_degree"):
                    cagra_cfg.graph_degree = int(cctl.get("graph_degree", 64))
                if hasattr(cagra_cfg, "intermediate_graph_degree"):
                    cagra_cfg.intermediate_graph_degree = int(
                        cctl.get("intermediate_graph_degree", 128)
                    )
                build_algo = str(cctl.get("build_algo", "ivf_pq")).lower()
                if hasattr(cagra_cfg, "build_algo"):
                    if build_algo == "nn_descent" and hasattr(CfgCls, "NN_DESCENT"):
                        cagra_cfg.build_algo = CfgCls.NN_DESCENT
                        if hasattr(cagra_cfg, "nn_descent_niter"):
                            cagra_cfg.nn_descent_niter = int(cctl.get("nn_descent_niter", 20))
                    elif hasattr(CfgCls, "IVF_PQ"):
                        cagra_cfg.build_algo = CfgCls.IVF_PQ
                try:
                    self.index = faiss.GpuIndexCagra(gpu_res, d, metric, cagra_cfg)
                except TypeError:
                    self.index = faiss.GpuIndexCagra(gpu_res, d, metric)
            else:
                self.index = faiss.GpuIndexCagra(gpu_res, d, metric)
        else:
            raise ValueError("Unsupported index_type")

        if trainable and index_cpu is not None and not index_cpu.is_trained:
            if train_size is not None and train_size < x_arr.shape[0]:
                sample_idx = np.random.choice(x_arr.shape[0], train_size, replace=False)
                train_data = x_arr[sample_idx]
            else:
                train_data = x_arr
            index_cpu.train(train_data)
            if train_size is not None and train_size < x_arr.shape[0]:
                del train_data
                gc.collect()

        if index_type in {"flat", "ivf", "ivfpq"}:
            if index_type == "flat":
                self.index = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)
            else:
                co = faiss.GpuClonerOptions()
                if index_type == "ivfpq":
                    ivfpq_useFloat16 = faiss_ctl.get("ivfpq_useFloat16", False)
                    ivfpq_usePrecomputed = faiss_ctl.get("ivfpq_usePrecomputed", False)
                    ivfpq_reserveVecs = faiss_ctl.get("ivfpq_reserveVecs", 0)
                    ivfpq_use_cuvs = faiss_ctl.get("ivfpq_use_cuvs", False)
                    if hasattr(co, "useFloat16"):
                        co.useFloat16 = ivfpq_useFloat16
                    if hasattr(co, "usePrecomputed"):
                        co.usePrecomputed = ivfpq_usePrecomputed
                    if hasattr(co, "reserveVecs"):
                        co.reserveVecs = ivfpq_reserveVecs
                    if hasattr(co, "use_cuvs"):
                        co.use_cuvs = ivfpq_use_cuvs
                self.index = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu, co)
                if index_type == "ivf":
                    self.index.nprobe = faiss_ctl.get("ivf_nprobe", 10)
                if index_type == "ivfpq":
                    self.index.nprobe = faiss_ctl.get("ivfpq_nprobe", 10)

            del index_cpu
            gc.collect()

        if index_type == "cagra":
            if (
                hasattr(self.index, "is_trained")
                and hasattr(self.index, "train")
                and not self.index.is_trained
            ):
                self.index.train(x_arr)
            if getattr(self.index, "ntotal", 0) == 0 and hasattr(self.index, "add"):
                self.index.add(x_arr)
        else:
            self.index.add(x_arr)

        x_total = x_arr.shape[0]
        del x_arr
        gc.collect()

        if k_search > x_total:
            warnings.warn(
                f"k_search={k_search} exceeds reference size ({x_total}); clipping.",
                category=UserWarning,
                stacklevel=2,
            )
            k_search = x_total

        if index_type == "cagra":
            SPCls = getattr(faiss, "SearchParametersCagra", None)
            if SPCls is not None:
                sp = SPCls()

                def _set(name: str, val: Any) -> None:
                    if hasattr(sp, name):
                        setattr(sp, name, val)

                cctl = faiss_ctl.get("cagra", {})
                _set("itopk_size", int(cctl.get("itopk_size", 64)))
                _set("max_queries", int(cctl.get("max_queries", 0)))
                _set("team_size", int(cctl.get("team_size", 0)))
                _set("search_width", int(cctl.get("search_width", 1)))
                _set("min_iterations", int(cctl.get("min_iterations", 0)))
                _set("max_iterations", int(cctl.get("max_iterations", 0)))
                _set("thread_block_size", int(cctl.get("thread_block_size", 0)))
                _set("hashmap_min_bitlen", int(cctl.get("hashmap_min_bitlen", 0)))
                _set(
                    "hashmap_max_fill_rate",
                    float(cctl.get("hashmap_max_fill_rate", 0.5)),
                )
                _set("num_random_samplings", int(cctl.get("num_random_samplings", 1)))
                _set("seed", int(cctl.get("seed", seed if seed is not None else 0x128394)))

                algo_s = str(cctl.get("algo", "auto")).lower()
                if hasattr(SPCls, "AUTO"):
                    if algo_s == "single_cta" and hasattr(SPCls, "SINGLE_CTA"):
                        sp.algo = SPCls.SINGLE_CTA
                    elif algo_s == "multi_cta" and hasattr(SPCls, "MULTI_CTA"):
                        sp.algo = SPCls.MULTI_CTA
                    else:
                        sp.algo = SPCls.AUTO

                hm_s = str(cctl.get("hashmap_mode", "auto")).lower()
                if hasattr(SPCls, "AUTO"):
                    if hm_s == "small" and hasattr(SPCls, "SMALL"):
                        sp.hashmap_mode = SPCls.SMALL
                    elif hm_s == "hash" and hasattr(SPCls, "HASH"):
                        sp.hashmap_mode = SPCls.HASH
                    else:
                        sp.hashmap_mode = SPCls.AUTO

                distances, indices = self.index.search(y_arr, k_search, params=sp)
            else:
                distances, indices = self.index.search(y_arr, k_search)
        else:
            distances, indices = self.index.search(y_arr, k_search)

        y_n = y_arr.shape[0]
        del y_arr
        gc.collect()

        if distance == "cosine" and index_type != "ivfpq":
            distances = (1 - distances) / 2

        K_VAL = 2
        if k == K_VAL:
            indices, distances = rearrange_array(indices, distances)

        if path:
            self._save_index(path)

        result_df = pd.DataFrame(
            {
                "y": np.arange(y_n),
                "x": indices[:, k - 1],
                "dist": distances[:, k - 1],
            }
        )

        del distances, indices
        gc.collect()

        return result_df

    def _save_index(self, path: str) -> None:
        if not os.path.isdir(path):
            raise ValueError("Provided path is not a directory")

        if isinstance(self.index, getattr(faiss, "GpuIndexCagra", ())):
            raise ValueError("Saving CAGRA indices is not supported in this build.")

        index_cpu = faiss.index_gpu_to_cpu(self.index)

        faiss_path = os.path.join(path, "index.faiss")
        cols_path = os.path.join(path, "index-colnames.txt")

        faiss.write_index(index_cpu, faiss_path)

        with open(cols_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.x_columns))

        del index_cpu
        gc.collect()

    def _apply_faiss_seeds(self, idx: faiss.Index, s: int) -> None:
        try:
            if hasattr(idx, "cp") and hasattr(idx.cp, "seed"):
                idx.cp.seed = int(s)
            if hasattr(idx, "pq") and hasattr(idx.pq, "cp") and hasattr(idx.pq.cp, "seed"):
                idx.pq.cp.seed = int(s)
        except Exception:
            pass
