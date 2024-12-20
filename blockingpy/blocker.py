"""
Contains the main Blocker class for record linkage
and deduplication blocking.
"""

import itertools
import logging
from collections import OrderedDict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from .annoy_blocker import AnnoyBlocker
from .blocking_result import BlockingResult
from .controls import controls_ann, controls_txt
from .faiss_blocker import FaissBlocker
from .helper_functions import (
    DistanceMetricValidator,
    InputValidator,
    create_sparse_dtm,
)
from .hnsw_blocker import HNSWBlocker
from .mlpack_blocker import MLPackBlocker
from .nnd_blocker import NNDBlocker
from .voyager_blocker import VoyagerBlocker

logger = logging.getLogger(__name__)


class Blocker:

    """
    A class implementing various blocking methods for record linkage and deduplication.

    This class provides a unified interface to multiple approximate nearest neighbor
    search algorithms for blocking in record linkage and deduplication tasks.

    Parameters
    ----------
    None

    Attributes
    ----------
    eval_metrics : pandas.Series or None
        Evaluation metrics when true blocks are provided
    confusion : pandas.DataFrame or None
        Confusion matrix when true blocks are provided
    x_colnames : list of str or None
        Column names for reference dataset
    y_colnames : list of str or None
        Column names for query dataset
    control_ann : dict
        Control parameters for ANN algorithms
    control_txt : dict
        Control parameters for text processing
    BLOCKER_MAP : dict
        Mapping of blocking algorithm names to their implementations


    Notes
    -----
    Supported algorithms:
    - Annoy (Spotify)
    - HNSW (Hierarchical Navigable Small World)
    - MLPack (LSH and k-d tree)
    - NND (Nearest Neighbor Descent)
    - Voyager (Spotify)
    - FAISS (Facebook AI Similarity Search)

    """

    def __init__(self) -> None:
        """
        Initialize the Blocker instance.

        Initializes empty state variables.
        """
        self.eval_metrics = None
        self.confusion = None
        self.x_colnames = None
        self.y_colnames = None
        self.control_ann: dict[str, Any] = {}
        self.control_txt: dict[str, Any] = {}
        self.BLOCKER_MAP = {
            "annoy": AnnoyBlocker,
            "hnsw": HNSWBlocker,
            "lsh": MLPackBlocker,
            "kd": MLPackBlocker,
            "nnd": NNDBlocker,
            "voyager": VoyagerBlocker,
            "faiss": FaissBlocker,
        }

    def block(
        self,
        x: pd.Series | sparse.csr_matrix | np.ndarray,
        y: np.ndarray | pd.Series | sparse.csr_matrix | None = None,
        deduplication: bool = True,
        ann: str = "faiss",
        true_blocks: pd.DataFrame | None = None,
        verbose: int = 0,
        graph: bool = False,
        control_txt: dict[str, Any] = {},
        control_ann: dict[str, Any] = {},
        x_colnames: list[str] | None = None,
        y_colnames: list[str] | None = None,
    ) -> BlockingResult:
        """
        Perform blocking using the specified algorithm.

        Parameters
        ----------
        x : pandas.Series or scipy.sparse.csr_matrix or numpy.ndarray
            Reference dataset for blocking
        y : numpy.ndarray or pandas.Series or scipy.sparse.csr_matrix, optional
            Query dataset (defaults to x for deduplication)
        deduplication : bool, default True
            Whether to perform deduplication instead of record linkage
        ann : str, default "faiss"
            Approximate Nearest Neighbor algorithm to use
        true_blocks : pandas.DataFrame, optional
            True blocking information for evaluation
        verbose : int, default 0
            Verbosity level (0-3). Controls logging level:
            - 0: WARNING level
            - 1-3: INFO level with increasing detail
        graph : bool, default False
            Whether to create a graph representation of blocks
        control_txt : dict, default {}
            Text processing parameters
        control_ann : dict, default {}
            ANN algorithm parameters
        x_colnames : list of str, optional
            Column names for reference dataset used with csr_matrix or np.ndarray
        y_colnames : list of str, optional
            Column names for query dataset used with csr_matrix or np.ndarray

        Raises
        ------
        ValueError
            If one of the input validations fails

        Returns
        -------
        BlockingResult
            Object containing blocking results and evaluation metrics

        Notes
        -----
        The function supports three input types:
        1. Text data (pandas.Series)
        2. Sparse matrices (scipy.sparse.csr_matrix) as a Document-Term Matrix (DTM)
        3. Dense matrices (numpy.ndarray) as a Document-Term Matrix (DTM)

        For text data, additional preprocessing is performed using
        the parameters in control_txt.

        See Also
        --------
        BlockingResult : Class containing blocking results
        controls_ann : Function to create ANN control parameters
        controls_txt : Function to create text control parameters

        """
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

        self.x_colnames = x_colnames
        self.y_colnames = y_colnames
        self.control_ann = controls_ann(control_ann)
        self.control_txt = controls_txt(control_txt)

        if deduplication:
            self.y_colnames = self.x_colnames

        if ann == "nnd":
            distance = self.control_ann.get("nnd").get("metric")
        elif ann in {"annoy", "voyager", "hnsw", "faiss"}:
            distance = self.control_ann.get(ann).get("distance")
        else:
            distance = None

        if distance is None:
            distance = {
                "nnd": "cosine",
                "hnsw": "cosine",
                "annoy": "angular",
                "voyager": "cosine",
                "faiss": "euclidean",
                "lsh": None,
                "kd": None,
            }.get(ann)

        InputValidator.validate_data(x)
        DistanceMetricValidator.validate_metric(ann, distance)

        if y is not None:
            deduplication = False
            k = 1
        else:
            y = x
            k = 2

        InputValidator.validate_true_blocks(true_blocks, deduplication)

        len_x = x.shape[0]
        # TOKENIZATION
        if sparse.issparse(x):
            if self.x_colnames is None:
                raise ValueError("Input is sparse, but x_colnames is None.")
            if self.y_colnames is None:
                raise ValueError("Input is sparse, but y_colnames is None.")

            x_dtm = pd.DataFrame.sparse.from_spmatrix(x, columns=self.x_colnames)
            y_dtm = pd.DataFrame.sparse.from_spmatrix(y, columns=self.y_colnames)
        elif isinstance(x, np.ndarray):
            if self.x_colnames is None:
                raise ValueError("Input is np.ndarray, but x_colnames is None.")
            if self.y_colnames is None:
                raise ValueError("Input is np.ndarray, but y_colnames is None.")

            x_dtm = pd.DataFrame(x, columns=self.x_colnames).astype(
                pd.SparseDtype("int", fill_value=0)
            )
            y_dtm = pd.DataFrame(y, columns=self.y_colnames).astype(
                pd.SparseDtype("int", fill_value=0)
            )
        else:
            FULL_VERBOSE = 3
            logger.info("===== creating tokens =====")
            x_dtm = create_sparse_dtm(
                x, self.control_txt, verbose=True if verbose == FULL_VERBOSE else False
            )
            y_dtm = create_sparse_dtm(
                y, self.control_txt, verbose=True if verbose == FULL_VERBOSE else False
            )
        # TOKENIZATION

        colnames_xy = np.intersect1d(x_dtm.columns, y_dtm.columns)

        logger.info(
            f"===== starting search ({ann}, x, y: {x_dtm.shape[0]},"
            f"{y_dtm.shape[0]}, t: {len(colnames_xy)}) ====="
        )

        blocker = self.BLOCKER_MAP[ann]
        x_df = blocker().block(
            x=x_dtm[colnames_xy],
            y=y_dtm[colnames_xy],
            k=k,
            verbose=True if verbose in {2, 3} else False,
            controls=self.control_ann,
        )

        logger.info("===== creating graph =====\n")

        if deduplication:
            x_df["pair"] = x_df.apply(lambda row: tuple(sorted([row["y"], row["x"]])), axis=1)
            x_df = x_df.loc[x_df.groupby("pair")["dist"].idxmin()]
            x_df = x_df.drop("pair", axis=1)

            x_df["query_g"] = "q" + x_df["y"].astype(str)
            x_df["index_g"] = "q" + x_df["x"].astype(str)
        else:
            x_df["query_g"] = "q" + x_df["y"].astype(str)
            x_df["index_g"] = "i" + x_df["x"].astype(str)

        # IGRAPH PART IN R
        x_gr = nx.from_pandas_edgelist(
            x_df, source="query_g", target="index_g", create_using=nx.Graph()
        )
        components = nx.connected_components(x_gr)
        x_block = {}
        for component_id, component in enumerate(components):
            for node in component:
                x_block[node] = component_id

        unique_query_g = x_df["query_g"].unique()
        unique_index_g = x_df["index_g"].unique()
        combined_keys = list(unique_query_g) + [
            node for node in unique_index_g if node not in unique_query_g
        ]

        sorted_dict = OrderedDict()
        for key in combined_keys:
            if key in x_block:
                sorted_dict[key] = x_block[key]

        x_df["block"] = x_df["query_g"].apply(lambda x: x_block[x] if x in x_block else None)
        ###

        if true_blocks is not None:
            if not deduplication:
                candidate_pairs = list(itertools.product(list(range(len(x_dtm))), true_blocks["y"]))
                cp_df = pd.DataFrame(candidate_pairs, columns=["x", "y"])
                cp_df = cp_df.astype(int)
                comparison_df = (
                    cp_df.merge(true_blocks, on=["x", "y"], how="left")
                    .rename(columns={"block": "block_true"})
                    .merge(x_df, on=["x", "y"], how="left")
                    .rename(columns={"block": "block_pred"})
                )
                comparison_df["TP"] = (comparison_df["block_true"].notna()) & (
                    comparison_df["block_pred"].notna()
                )
                # CNL -> Correct Non-Links / True Negative
                comparison_df["CNL"] = (comparison_df["block_true"].isna()) & (
                    comparison_df["block_pred"].isna()
                )
                comparison_df["FP"] = (comparison_df["block_true"].isna()) & (
                    comparison_df["block_pred"].notna()
                )
                comparison_df["FN"] = (comparison_df["block_true"].notna()) & (
                    comparison_df["block_pred"].isna()
                )
                self.confusion = pd.DataFrame(
                    [
                        [comparison_df["CNL"].sum(), comparison_df["FN"].sum()],
                        [comparison_df["FP"].sum(), comparison_df["TP"].sum()],
                    ],
                    index=["Predicted Negative", "Predicted Positive"],
                    columns=["Actual Negative", "Actual Positive"],
                )

            else:
                pairs_to_eval_long = (
                    pd.melt(x_df[["x", "y", "block"]], id_vars=["block"])[["block", "value"]]
                    .drop_duplicates()
                    .rename(columns={"block": "block_id", "value": "x"})
                    .merge(true_blocks[["x", "block"]], on="x", how="left")
                    .rename(columns={"block": "true_id"})
                )

                candidate_pairs = np.array(
                    list(itertools.combinations(range(pairs_to_eval_long.shape[0]), 2))
                )

                block_id_array = pairs_to_eval_long["block_id"].to_numpy()
                true_id_array = pairs_to_eval_long["true_id"].to_numpy()
                same_block = (
                    block_id_array[candidate_pairs[:, 0]] == block_id_array[candidate_pairs[:, 1]]
                )
                same_truth = (
                    true_id_array[candidate_pairs[:, 0]] == true_id_array[candidate_pairs[:, 1]]
                )

                self.confusion = pd.crosstab(same_block, same_truth)
                self.confusion.index = ["Predicted Negative", "Predicted Positive"]
                self.confusion.columns = ["Actual Negative", "Actual Positive"]

            fp = self.confusion.iloc[1, 0]
            fn = self.confusion.iloc[0, 1]
            tp = self.confusion.iloc[1, 1]
            tn = self.confusion.iloc[0, 0]

            recall = tp / (fn + tp) if (fn + tp) != 0 else 0
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            f1_score = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            )
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

            self.eval_metrics = {
                "recall": recall,
                "precision": precision,
                "fpr": fpr,
                "fnr": fnr,
                "accuracy": accuracy,
                "specificity": specificity,
                "f1_score": f1_score,
            }
            self.eval_metrics = pd.Series(self.eval_metrics)

        x_df = x_df.sort_values(["y", "x", "block"]).reset_index(drop=True)

        return BlockingResult(
            x_df=x_df,
            ann=ann,
            deduplication=deduplication,
            len_x=len_x,
            true_blocks=true_blocks,
            eval_metrics=self.eval_metrics,
            confusion=self.confusion,
            colnames_xy=colnames_xy,
            graph=graph,
        )
