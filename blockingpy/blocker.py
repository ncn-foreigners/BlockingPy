"""
Contains the main Blocker class for record linkage
and deduplication blocking.
"""

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
)
from .hnsw_blocker import HNSWBlocker
from .mlpack_blocker import MLPackBlocker
from .nnd_blocker import NNDBlocker
from .text_encoders.text_transformer import TextTransformer
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
    - FAISS (LSH, HNSW and Flat Indexes)

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
        random_seed: int | None = 2025,
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
        random_seed : int, optional
            Random seed for reproducibility (default is None)

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

        Evaluation of larger datasets can be done separately using the `eval` method.

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
        InputValidator.validate_controls_txt(control_txt)
        self.control_ann = controls_ann(control_ann)
        self.control_txt = controls_txt(control_txt)

        if deduplication:
            self.y_colnames = self.x_colnames

        if self.control_ann["random_seed"] is None:
            self.control_ann["random_seed"] = random_seed

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
                "faiss": "cosine",
                "lsh": None,
                "kd": None,
            }.get(ann)

        InputValidator.validate_data(x)
        DistanceMetricValidator.validate_metric(ann, distance)

        if y is not None:
            deduplication = False
            k = 1
            len_y = y.shape[0]
        else:
            y = x
            k = 2
            len_y = None

        InputValidator.validate_true_blocks(true_blocks, deduplication)

        len_x = x.shape[0]
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
            logger.info(f"===== creating tokens: {self.control_txt.get('encoder', 'Error')} =====")
            transformer = TextTransformer(**self.control_txt)
            x_dtm = transformer.transform(x)
            y_dtm = transformer.transform(y)

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
        logger.info("===== creating graph =====")

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
                TP, FP, FN, TN = self._eval_rl(x_df, true_blocks)
            else:
                TP, FP, FN, TN = self._eval_dedup(x_df, true_blocks)

            self.confusion = self._get_confusion(TP, FP, FN, TN)
            self.eval_metrics = self._get_metrics(TP, FP, FN, TN)

        x_df = x_df.sort_values(["y", "x", "block"]).reset_index(drop=True)

        return BlockingResult(
            x_df=x_df,
            ann=ann,
            deduplication=deduplication,
            n_original_records=(len_x, len_y),
            true_blocks=true_blocks,
            eval_metrics=self.eval_metrics,
            confusion=self.confusion,
            colnames_xy=colnames_xy,
            graph=graph,
        )

    def eval(self, blocking_result: BlockingResult, true_blocks: pd.DataFrame) -> BlockingResult:
        """
        Evaluate blocking results against true block assignments and return new BlockingResult.

        This method calculates evaluation metrics and confusion matrix
        by comparing predicted blocks with known true blocks and returns
        a new BlockingResult instance containing the evaluation results
        along with the original blocking results.

        Parameters
        ----------
        blocking_result : BlockingResult
            Original blocking result to evaluate
        true_blocks : pandas.DataFrame
            DataFrame with true block assignments
            For deduplication: columns ['x', 'block']
            For record linkage: columns ['x', 'y', 'block']

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
            TP, FP, FN, TN = self._eval_rl(blocking_result.result, true_blocks)
        else:
            TP, FP, FN, TN = self._eval_dedup(blocking_result.result, true_blocks)

        confusion = self._get_confusion(TP, FP, FN, TN)
        eval_metrics = self._get_metrics(TP, FP, FN, TN)

        return BlockingResult(
            x_df=blocking_result.result,
            ann=blocking_result.method,
            deduplication=blocking_result.deduplication,
            n_original_records=blocking_result.n_original_records,
            true_blocks=true_blocks,
            eval_metrics=eval_metrics,
            confusion=confusion,
            colnames_xy=blocking_result.colnames,
            graph=blocking_result.graph is not None,
        )

    def _eval_rl(
        self,
        pred_df: pd.DataFrame,
        true_df: pd.DataFrame,
    ) -> tuple[int, int, int, int]:
        """
        Get confusion matrix from *record linkage*.

        Parameters
        ----------
        pred_df : pd.DataFrame
            output from the algorithm (or BlockingResult.result)
        true_df : pd.DataFrame
            ground-truth links (may be subset)

        Returns
        -------
        TP, FP, FN, TN   (pair counts, integers)

        """
        pred_x_map = pred_df[["x", "block"]].drop_duplicates().set_index("x")["block"]
        pred_y_map = pred_df[["y", "block"]].drop_duplicates().set_index("y")["block"]

        true_x = true_df[["x", "block"]].drop_duplicates().set_index("x")["block"]
        true_y = true_df[["y", "block"]].drop_duplicates().set_index("y")["block"]

        pred_x = pred_x_map.reindex(true_x.index)
        pred_y = pred_y_map.reindex(true_y.index)

        n_missing = pred_x.isna().sum() + pred_y.isna().sum()
        if n_missing:
            start = (pred_df["block"].max() + 1) if len(pred_df) else 0
            fresh_ids = pd.Series(
                np.arange(start, start + n_missing, dtype="int64"),
                index=pred_x[pred_x.isna()].index.tolist() + pred_y[pred_y.isna()].index.tolist(),
            )
            pred_x = pred_x.fillna(fresh_ids)
            pred_y = pred_y.fillna(fresh_ids)

        pred_x = pred_x.astype("int64")
        pred_y = pred_y.astype("int64")

        all_pred = pd.concat([pred_x, pred_y], ignore_index=True)
        codes_pred, uniq_pred = pd.factorize(all_pred, sort=False)

        all_true = pd.concat([true_x, true_y], ignore_index=True)
        codes_true, uniq_true = pd.factorize(all_true, sort=False)

        n_pred = len(uniq_pred)
        n_true = len(uniq_true)

        cp_x = codes_pred[: len(pred_x)]
        cp_y = codes_pred[len(pred_x) :]
        ct_x = codes_true[: len(true_x)]
        ct_y = codes_true[len(true_x) :]

        cx = sparse.coo_matrix(
            (np.ones_like(cp_x, dtype=np.int64), (cp_x, ct_x)),
            shape=(n_pred, n_true),
        ).tocsr()

        cy = sparse.coo_matrix(
            (np.ones_like(cp_y, dtype=np.int64), (cp_y, ct_y)),
            shape=(n_pred, n_true),
        ).tocsr()

        TP = int(cx.multiply(cy).sum())

        row_sum_x = np.asarray(cx.sum(axis=1)).ravel()
        row_sum_y = np.asarray(cy.sum(axis=1)).ravel()
        pred_pairs = int((row_sum_x * row_sum_y).sum())

        col_sum_x = np.asarray(cx.sum(axis=0)).ravel()
        col_sum_y = np.asarray(cy.sum(axis=0)).ravel()
        true_pairs = int((col_sum_x * col_sum_y).sum())

        FP = pred_pairs - TP
        FN = true_pairs - TP
        NX, NY = len(true_x), len(true_y)
        TN = NX * NY - TP - FP - FN

        return TP, FP, FN, TN

    def _eval_dedup(
        self,
        pred_pairs_df: pd.DataFrame,
        true_blocks: pd.DataFrame,
    ) -> tuple[int, int, int, int]:
        """
        Get confusion matrix from *deduplication*.

        Parameters
        ----------
        pred_pairs_df : pd.DataFrame
            output from the algorithm or BlockingResult.result
        true_blocks : pd.DataFrame
            ground-truth links (may be subset)

        Returns
        -------
        TP, FP, FN, TN    (pair counts, ints)

        """
        pred_lbl = (
            pred_pairs_df.melt(id_vars="block", value_vars=["x", "y"], value_name="rec")
            .drop_duplicates("rec")
            .set_index("rec")["block"]
            .astype("int64")
        )

        true_lbl = true_blocks.drop_duplicates("x").set_index("x")["block"].astype("int64")

        pred_lbl = pred_lbl.reindex(true_lbl.index)

        if pred_lbl.isna().any():
            start = pred_pairs_df["block"].max() + 1 if len(pred_pairs_df) else 0
            n_miss = pred_lbl.isna().sum()
            pred_lbl.loc[pred_lbl.isna()] = np.arange(start, start + n_miss, dtype="int64")
            pred_lbl = pred_lbl.astype("int64")

        both_df = pd.DataFrame({"pred": pred_lbl.values, "true": true_lbl.values})

        g = both_df.groupby(["pred", "true"]).size().astype("int64")

        TP = int((g * (g - 1) // 2).sum())

        row_sum = g.groupby(level=0).sum()
        col_sum = g.groupby(level=1).sum()

        pred_pairs = int((row_sum * (row_sum - 1) // 2).sum())
        true_pairs = int((col_sum * (col_sum - 1) // 2).sum())

        FP = pred_pairs - TP
        FN = true_pairs - TP

        N = len(both_df)
        total_pairs = N * (N - 1) // 2
        TN = total_pairs - TP - FP - FN

        return TP, FP, FN, TN

    def _get_confusion(self, tp: int, fp: int, fn: int, tn: int) -> pd.DataFrame:
        """
        Build a confusion matrix DataFrame from raw counts.

        Parameters
        ----------
        tp : int
            True positives.
        fp : int
            False positives.
        fn : int
            False negatives.
        tn : int
            True negatives.

        Returns
        -------
        pd.DataFrame
            2 x 2 confusion matrix with
             - rows = Actual Positive / Actual Negative
             - columns = Predicted Positive / Predicted Negative

        """
        cm = pd.DataFrame(
            [
                [tp, fn],
                [fp, tn],
            ],
            index=["Actual Positive", "Actual Negative"],
            columns=["Predicted Positive", "Predicted Negative"],
        ).astype(int)
        return cm

    def _get_metrics(self, tp: int, fp: int, fn: int, tn: int) -> pd.Series:
        """
        Compute standard evaluation metrics from raw counts.

        Parameters
        ----------
        tp : int
            True positives.
        fp : int
            False positives.
        fn : int
            False negatives.
        tn : int
            True negatives.

        Returns
        -------
        pd.Series
            Series with index
            ['recall', 'precision', 'fpr', 'fnr', 'accuracy', 'specificity', 'f1_score']

        """
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

        return pd.Series(
            {
                "recall": recall,
                "precision": precision,
                "fpr": fpr,
                "fnr": fnr,
                "accuracy": accuracy,
                "specificity": specificity,
                "f1_score": f1_score,
            }
        )
