"""Annoy-based ANN blocker compatible with DataHandler."""

from __future__ import annotations

import logging
import os
import warnings
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import numpy as np
import pandas as pd
from annoy import AnnoyIndex

from .base import BlockingMethod
from .data_handler import DataHandler
from .helper_functions import rearrange_array

logger = logging.getLogger(__name__)
MetricType = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]


class AnnoyBlocker(BlockingMethod):
    """Blocking with Spotify *Annoy* (Approximate Nearest Neighbors Oh Yeah)."""

    METRIC_MAP: dict[str, MetricType] = {
        "euclidean": "euclidean",
        "manhattan": "manhattan",
        "hamming": "hamming",
        "angular": "angular",
        "dot": "dot",
    }

    def __init__(self) -> None:
        self.index: AnnoyIndex | None = None
        self.x_columns: list[str] | None = None

    def block(
        self,
        x: DataHandler,
        y: DataHandler,
        k: int,
        verbose: bool | None,
        controls: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Perform blocking using the Annoy algorithm.

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
            Algorithm control parameters with the following structure:
            {
                'random_seed': int,
                'annoy': {
                    'distance': str,
                    'seed': int,
                    'path': str,
                    'n_trees': int,
                    'build_on_disk': bool,
                    'k_search': int
                }
            }

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the blocking results with columns:
            - 'y': indices from query dataset
            - 'x': indices of matched items from reference dataset
            - 'dist': distances to matched items

        Notes
        -----
        The function builds an Annoy index from the reference dataset
        and finds the k-nearest neighbors for each point in the query dataset.

        """
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self.x_columns = list(x.cols)

        distance = controls["annoy"].get("distance")
        seed = controls.get("random_seed")
        path = controls["annoy"].get("path")
        n_trees = controls["annoy"].get("n_trees")
        build_on_disk = controls["annoy"].get("build_on_disk")
        k_search = controls["annoy"].get("k_search")

        X = x.to_dense()
        Y = y.to_dense()

        ncols = X.shape[1]
        metric = self.METRIC_MAP[distance]

        self.index = AnnoyIndex(ncols, metric)
        if seed is not None:
            self.index.set_seed(seed)

        if build_on_disk:
            with NamedTemporaryFile(prefix="annoy", suffix=".tree") as tmp:
                if verbose:
                    logger.info(f"Building index on disk: {tmp.name}")
                self.index.on_disk_build(tmp.name)

        if verbose:
            self.index.verbose(True)

        logger.info("Building index…")
        for i in range(X.shape[0]):
            self.index.add_item(i, X[i])
        self.index.build(n_trees=n_trees)

        logger.info("Querying index…")
        if k_search > X.shape[0]:
            k_search = X.shape[0]
            warnings.warn(
                "k_search larger than reference set; adjusted.", category=UserWarning, stacklevel=2
            )

        ind_nns = np.empty((Y.shape[0], k_search), dtype=int)
        ind_dist = np.empty((Y.shape[0], k_search), dtype=float)

        for i in range(Y.shape[0]):
            ids, dists = self.index.get_nns_by_vector(Y[i], k_search, include_distances=True)
            ind_nns[i] = ids
            ind_dist[i] = dists

        K_VAL = 2
        if k == K_VAL:
            ind_nns, ind_dist = rearrange_array(ind_nns, ind_dist)

        if path:
            self._save_index(path)

        result = pd.DataFrame(
            {
                "y": np.arange(Y.shape[0]),
                "x": ind_nns[:, k - 1],
                "dist": ind_dist[:, k - 1],
            }
        )
        logger.info("Process completed successfully.")
        return result

    def _save_index(self, path: str) -> None:
        if not os.path.exists(os.path.dirname(path)):
            raise ValueError("Provided path is incorrect")
        path_ann = os.path.join(path, "index.annoy")
        path_cols = os.path.join(path, "index-colnames.txt")
        logger.info(f"Writing an index to {path_ann}")
        if self.index is not None:
            self.index.save(path_ann)
        with open(path_cols, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.x_columns or []))
