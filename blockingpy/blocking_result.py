"""
Contains the BlockingResult class for analyzing and printing
blocking results.
"""

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd


class BlockingResult:

    """
    A class to represent and analyze the results of a blocking operation.

    This class provides functionality to analyze and evaluate blocking results,
    including calculation of reduction ratios, metrics evaluation, and optional
    graph representation of the blocking structure.

    Parameters
    ----------
    x_df : pandas.DataFrame
        DataFrame containing blocking results with columns ['x', 'y', 'block', 'dist']
    ann : str
        The blocking method used (e.g., 'nnd', 'hnsw', 'annoy', etc.)
    deduplication : bool
        Whether the blocking was performed for deduplication
    true_blocks : pandas.DataFrame, optional
        DataFrame with true blocks to calculate evaluation metrics
    n_original_records : tuple[int, int]
        Number of records in the original dataset(s)
    eval_metrics : pandas.Series, optional
        Evaluation metrics if true blocks were provided
    confusion : pandas.DataFrame, optional
        Confusion matrix if true blocks were provided
    colnames_xy : numpy.ndarray
        Column names used in the blocking process
    graph : bool, optional
        Whether to create a graph from the blocking results (default False)
    reduction_ratio : float, optional
        Pre-calculated reduction ratio (default None)

    Attributes
    ----------
    result : pandas.DataFrame
        The blocking results containing ['x', 'y', 'block', 'dist'] columns
    method : str
        Name of the blocking method used
    deduplication : bool
        Indicates if this was a deduplication operation
    metrics : pandas.Series or None
        Evaluation metrics if true blocks were provided
    confusion : pandas.DataFrame or None
        Confusion matrix if true blocks were provided
    colnames : numpy.ndarray
        Names of columns used in blocking
    graph : networkx.Graph or None
        Network representation of blocking results if requested
    n_original_records : tuple[int, int]
        Number of records in the original dataset(s)
    reduction_ratio : float
        Reduction ratio calculated for the blocking method

    Notes
    -----
    The class provides methods for calculating reduction ratio and formatting
    evaluation metrics for blocking quality assessment.

    """

    def __init__(
        self,
        x_df: pd.DataFrame,
        ann: str,
        deduplication: bool,
        n_original_records: tuple[int, int],
        true_blocks: pd.DataFrame | None,
        eval_metrics: pd.Series | None,
        confusion: pd.DataFrame | None,
        colnames_xy: np.ndarray,
        graph: bool | None = False,
        reduction_ratio: float | None = None,
    ) -> None:
        """Initialize a BlockingResult instance."""
        self.result = x_df[["x", "y", "block", "dist"]]
        self.method = ann
        self.deduplication = deduplication
        self.metrics = eval_metrics if true_blocks is not None else None
        self.confusion = confusion if true_blocks is not None else None
        self.colnames = colnames_xy
        self.graph = (
            nx.from_pandas_edgelist(x_df[["x", "y"]], source="x", target="y") if graph else None
        )
        self.n_original_records = n_original_records

        if reduction_ratio is None:
            self.reduction_ratio = self._calculate_reduction_ratio()
        else:
            self.reduction_ratio = reduction_ratio

    def __repr__(self) -> str:
        """
        Provide a concise representation of the blocking result.

        Returns
        -------
        str
            A string representation showing method and deduplication status

        """
        return f"BlockingResult(method={self.method}, deduplication={self.deduplication})"

    def __str__(self) -> str:
        """
        Create a detailed string representation of the blocking result.

        Returns
        -------
        str
            A formatted string containing:
            - Basic information about the blocking
            - Block size distribution
            - Evaluation metrics (if available)

        Notes
        -----
        The output includes reduction ratio and detailed block size statistics.
        If evaluation metrics are available, they are included in the output.

        """
        if self.deduplication:
            block_sizes = self.result.groupby("block").apply(
                lambda x: len(pd.concat([x["x"], x["y"]]).unique())
            )
        else:
            block_sizes = (
                self.result.groupby("block").agg({"x": "nunique", "y": "nunique"}).sum(axis=1)
            )
        block_size_dist = Counter(block_sizes.values)

        output = []
        output.append("=" * 56)
        output.append(f"Blocking based on the {self.method} method.")
        output.append(f"Number of blocks: {len(block_sizes)}")
        output.append(f"Number of columns used for blocking: {len(self.colnames)}")
        output.append(f"Reduction ratio: {self.reduction_ratio:.6f}")
        output.append("=" * 56)

        output.append("Distribution of the size of the blocks:")
        output.append(f"{'Block Size':>10} | {'Number of Blocks':<15}")
        for size, count in sorted(block_size_dist.items()):
            output.append(f"{size:>10} | {count:<15}")

        if self.metrics is not None:
            output.append("=" * 56)
            output.append("Evaluation metrics (standard):")
            metrics = self._format_metrics()
            for name, value in metrics.items():
                output.append(f"{name} : {value}")

        return "\n".join(output)

    def _calculate_reduction_ratio(self) -> float:
        """
        Calculate the reduction ratio for the blocking method.

        The reduction ratio measures how much the blocking method reduces
        the number of comparisons needed compared to all possible pairs.

        Returns
        -------
        float
            The reduction ratio, where:
            - 1.0 means maximum reduction (minimal comparisons)
            - 0.0 means no reduction (all pairs compared)

        Notes
        -----
        The ratio is calculated as:
        1 - (number of comparisons after blocking / total possible comparisons)

        """
        if self.deduplication:
            denominator = self.n_original_records[0] * (self.n_original_records[0] - 1) / 2
            block_sizes = self.result.groupby("block")[["x", "y"]].apply(
                lambda x: len(pd.concat([x["x"], x["y"]]).unique())
            )
            numerator = (block_sizes * (block_sizes - 1) / 2).sum() if len(block_sizes) > 0 else 0
        else:
            denominator = self.n_original_records[0] * self.n_original_records[1]
            block_comparisons = self.result.groupby("block").agg({"x": "nunique", "y": "nunique"})
            numerator = (block_comparisons["x"] * block_comparisons["y"]).sum()

        return 1 - (numerator / denominator if denominator > 0 else 0)

    def _format_metrics(self) -> dict[str, float]:
        """
        Format the evaluation metrics for display.

        Returns
        -------
        dict
            Dictionary of metric names and formatted values as percentages,
            rounded to 4 decimal places

        Notes
        -----
        Returns an empty dictionary if no metrics are available.
        Values are multiplied by 100 to convert to percentages.

        """
        if self.metrics is None:
            return {}

        return {name: float(f"{value * 100:.4f}") for name, value in self.metrics.items()}
