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

        For evaluation of larger datasets, we recommend using the separate eval() method
            since it allows you to set the batch size for evaluation.

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
        self.EVAL_BATCH_SIZE = 10_000

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

        if true_blocks is not None:
            if not deduplication:
                total_tn = total_fp = total_fn = total_tp = 0
                batch_size = 10_000
                n_records = len(x_dtm)

                for start_idx in range(0, n_records, batch_size):
                    tn, fp, fn, tp = self._process_record_linkage_batch(
                        start_idx, batch_size, n_records, true_blocks, x_df
                    )
                    total_tn += tn
                    total_fp += fp
                    total_fn += fn
                    total_tp += tp

                self.confusion = pd.DataFrame(
                    [[total_tn, total_fn], [total_fp, total_tp]],
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

                total_tn = total_fp = total_fn = total_tp = 0
                batch_size = 10_000
                n_pairs = pairs_to_eval_long.shape[0]

                for start_idx in range(0, n_pairs, batch_size):
                    tn, fp, fn, tp = self._process_dedup_batch(
                        pairs_to_eval_long, start_idx, batch_size
                    )
                    total_tn += tn
                    total_fp += fp
                    total_fn += fn
                    total_tp += tp

                self.confusion = pd.DataFrame(
                    [[total_tn, total_fn], [total_fp, total_tp]],
                    index=["Predicted Negative", "Predicted Positive"],
                    columns=["Actual Negative", "Actual Positive"],
                )

            recall = total_tp / (total_fn + total_tp) if (total_fn + total_tp) != 0 else 0
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0
            f1_score = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            )
            accuracy = (
                (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
                if (total_tp + total_tn + total_fp + total_fn) != 0
                else 0
            )
            specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) != 0 else 0
            fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) != 0 else 0
            fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) != 0 else 0

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

    def eval(
        self, blocking_result: BlockingResult, true_blocks: pd.DataFrame, batch_size: int = 10_000
    ) -> BlockingResult:
        """
        Evaluate blocking results against true block assignments and return new BlockingResult.

        This method calculates evaluation metrics and confusion matrix
        by comparing predicted blocks with known true blocks and returns
        a new BlockingResult instance containing the evaluation results
        along with the original blocking results. It allows you to set
        the batch size for evaluation of larger datasets.

        Parameters
        ----------
        blocking_result : BlockingResult
            Original blocking result to evaluate
        true_blocks : pandas.DataFrame
            DataFrame with true block assignments
            For deduplication: columns ['x', 'block']
            For record linkage: columns ['x', 'y', 'block']
        batch_size : int
            Number of records from dataset X to evaluate in each batch.
            Each batch will be compared with all records from dataset Y.
            Defaults to 10,000.

        Returns
        -------
        BlockingResult
            A new BlockingResult instance with added evaluation results
            and original blocking results

        Examples
        --------
        >>> blocker = Blocker()
        >>> result = blocker.block(x, y)
        >>> evaluated = blocker.eval(result, true_blocks)
        >>> print(evaluated.metrics)

        See Also
        --------
        block : Main blocking method that includes evaluation
        BlockingResult : Class for analyzing blocking results

        """
        if not isinstance(blocking_result, BlockingResult):
            raise ValueError(
                "blocking_result must be a BlockingResult instance obtained from `block` method."
            )
        InputValidator.validate_true_blocks(true_blocks, blocking_result.deduplication)

        if not blocking_result.deduplication:
            total_tn = total_fp = total_fn = total_tp = 0
            n_records = blocking_result.len_x

            for start_idx in range(0, n_records, batch_size):
                tn, fp, fn, tp = self._process_record_linkage_batch(
                    start_idx, batch_size, n_records, true_blocks, blocking_result.result
                )
                total_tn += tn
                total_fp += fp
                total_fn += fn
                total_tp += tp

            confusion = pd.DataFrame(
                [[total_tn, total_fn], [total_fp, total_tp]],
                index=["Predicted Negative", "Predicted Positive"],
                columns=["Actual Negative", "Actual Positive"],
            )

        else:
            pairs_to_eval_long = (
                pd.melt(blocking_result.result[["x", "y", "block"]], id_vars=["block"])[
                    ["block", "value"]
                ]
                .drop_duplicates()
                .rename(columns={"block": "block_id", "value": "x"})
                .merge(true_blocks[["x", "block"]], on="x", how="left")
                .rename(columns={"block": "true_id"})
            )

            total_tn = total_fp = total_fn = total_tp = 0
            n_pairs = pairs_to_eval_long.shape[0]

            for start_idx in range(0, n_pairs, batch_size):
                tn, fp, fn, tp = self._process_dedup_batch(
                    pairs_to_eval_long, start_idx, batch_size
                )
                total_tn += tn
                total_fp += fp
                total_fn += fn
                total_tp += tp

            confusion = pd.DataFrame(
                [[total_tn, total_fn], [total_fp, total_tp]],
                index=["Predicted Negative", "Predicted Positive"],
                columns=["Actual Negative", "Actual Positive"],
            )

        recall = total_tp / (total_fn + total_tp) if (total_fn + total_tp) != 0 else 0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        )
        accuracy = (
            (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
            if (total_tp + total_tn + total_fp + total_fn) != 0
            else 0
        )
        specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) != 0 else 0
        fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) != 0 else 0
        fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) != 0 else 0

        eval_metrics = {
            "recall": recall,
            "precision": precision,
            "fpr": fpr,
            "fnr": fnr,
            "accuracy": accuracy,
            "specificity": specificity,
            "f1_score": f1_score,
        }
        eval_metrics = pd.Series(eval_metrics)

        return BlockingResult(
            x_df=blocking_result.result,
            ann=blocking_result.method,
            deduplication=blocking_result.deduplication,
            len_x=blocking_result.len_x,
            true_blocks=true_blocks,
            eval_metrics=eval_metrics,
            confusion=confusion,
            colnames_xy=blocking_result.colnames,
            graph=blocking_result.graph is not None,
        )

    def _process_dedup_batch(
        self, pairs_to_eval_long: pd.DataFrame, start_idx: int, batch_size: int
    ) -> tuple[int, int, int, int]:
        """
        Process a batch of candidate pairs for deduplication evaluation.
        This method processes a subset of record pairs to compute confusion matrix elements
        for evaluating blocking quality.

        Parameters
        ----------
        pairs_to_eval_long : pd.DataFrame
            DataFrame containing record pairs with their block_id and true_id assignments.
        start_idx : int
            Starting index in the pairs DataFrame to process.
        batch_size : int
            Number of records to process in this batch.

        Returns
        -------
        tuple[int, int, int, int]
            A tuple containing partial confusion matrix counts:
            - tn (true negatives): Pairs correctly not blocked together
            - fp (false positives): Pairs incorrectly blocked together
            - fn (false negatives): Pairs incorrectly not blocked together
            - tp (true positives): Pairs correctly blocked together

        Notes
        -----
        The method generates all possible pairs within the batch range and compares
        their blocking assignments against their true matching status.

        """
        n_pairs = pairs_to_eval_long.shape[0]
        pairs_range = range(start_idx, min(start_idx + batch_size, n_pairs))
        candidate_pairs = np.array([(i, j) for i in pairs_range for j in range(i + 1, n_pairs)])

        if len(candidate_pairs) == 0:
            return 0, 0, 0, 0

        block_id_array = pairs_to_eval_long["block_id"].to_numpy()
        true_id_array = pairs_to_eval_long["true_id"].to_numpy()

        same_block = block_id_array[candidate_pairs[:, 0]] == block_id_array[candidate_pairs[:, 1]]
        same_truth = true_id_array[candidate_pairs[:, 0]] == true_id_array[candidate_pairs[:, 1]]

        tn = np.sum(~same_block & ~same_truth)
        fp = np.sum(same_block & ~same_truth)
        fn = np.sum(~same_block & same_truth)
        tp = np.sum(same_block & same_truth)

        return tn, fp, fn, tp

    def _process_record_linkage_batch(
        self,
        start_idx: int,
        batch_size: int,
        n_records: int,
        true_blocks: pd.DataFrame,
        blocking_df: pd.DataFrame,
    ) -> tuple[int, int, int, int]:
        """
        Process a batch of record pairs and compute confusion matrix counts.
        This method processes a subset of record pairs for record linkage evaluation.

        Parameters
        ----------
        start_idx : int
            Starting index of the batch in the record set
        batch_size : int
            Size of the batch to process
        n_records : int
            Total number of records
        true_blocks : pd.DataFrame
            DataFrame containing true blocking assignments with columns ['x', 'y', 'block']
        blocking_df : pd.DataFrame
            DataFrame containing predicted blocking assignments with columns ['x', 'y', 'block']

        Returns
        -------
        tuple[int, int, int, int]
            A tuple containing (true_negatives, false_positives, false_negatives, true_positives)

        Notes
        -----
        - The method generates all possible pairs between records in the batch and compares
          their blocking assignments.

        """
        end_idx = min(start_idx + batch_size, n_records)
        batch_pairs = list(itertools.product(range(start_idx, end_idx), true_blocks["y"]))

        if not batch_pairs:
            return 0, 0, 0, 0

        cp_df = pd.DataFrame(batch_pairs, columns=["x", "y"])
        cp_df = cp_df.astype(int)

        comparison_df = (
            cp_df.merge(true_blocks, on=["x", "y"], how="left")
            .rename(columns={"block": "block_true"})
            .merge(blocking_df, on=["x", "y"], how="left")
            .rename(columns={"block": "block_pred"})
        )

        tp = np.sum((comparison_df["block_true"].notna()) & (comparison_df["block_pred"].notna()))
        tn = np.sum((comparison_df["block_true"].isna()) & (comparison_df["block_pred"].isna()))
        fp = np.sum((comparison_df["block_true"].isna()) & (comparison_df["block_pred"].notna()))
        fn = np.sum((comparison_df["block_true"].notna()) & (comparison_df["block_pred"].isna()))

        return tn, fp, fn, tp
