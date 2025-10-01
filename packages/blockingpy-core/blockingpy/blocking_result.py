"""
Contains the BlockingResult class for analyzing and printing
blocking results.
"""

from collections import Counter

import numpy as np
import pandas as pd


class BlockingResult:
    """
    A class to represent and analyze the results of a blocking operation.

    This class provides functionality to analyze and evaluate blocking results,
    including calculation of reduction ratios, metrics evaluation.

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
    n_original_records : tuple[int, int]
        Number of records in the original dataset(s)
    reduction_ratio : float
        Reduction ratio calculated for the blocking method

    Notes
    -----
    The class provides methods for calculating reduction ratio and formatting
    evaluation metrics for blocking quality assessment.

    """

    def __init__(  # noqa: PLR0913
        self,
        x_df: pd.DataFrame,
        ann: str,
        deduplication: bool,
        n_original_records: tuple[int, int | None],
        true_blocks: pd.DataFrame | None,
        eval_metrics: pd.Series | None,
        confusion: pd.DataFrame | None,
        colnames_xy: np.ndarray,
        reduction_ratio: float | None = None,
    ) -> None:
        """Initialize a BlockingResult instance."""
        self.result = x_df[["x", "y", "block", "dist"]]
        self.method = ann
        self.deduplication = deduplication
        self.metrics = eval_metrics if true_blocks is not None else None
        self.confusion = confusion if true_blocks is not None else None
        self.colnames = colnames_xy
        self.n_original_records = n_original_records
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
        output.append(f"Number of columns created for blocking: {len(self.colnames)}")
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

    def add_block_column(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame | None = None,
        id_col_left: str | None = None,
        id_col_right: str | None = None,
        block_col: str = "block",
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Attach block IDs back onto the original DataFrame(s), filling any
        records with no assignment into their own singleton blocks.

        - **Deduplication**: pass only `df_left`; returns one DataFrame.
        - **Record-linkage**: pass both `df_left` and `df_right`; returns
          a tuple `(left_with_blocks, right_with_blocks)`.

        Parameters
        ----------
        df_left
            If dedup: your input DataFrame. If rec-lin: the “x” DataFrame.
        df_right
            If rec-lin: the “y” DataFrame. Otherwise None.
        id_col_left
            Column in `df_left` matching integer index into `self.result.x`;
            if None, uses the DataFrame's positional index.
        id_col_right
            Column in `df_right` matching integer index into `self.result.y`;
            if None, uses that DataFrame's positional index.
        block_col
            Name of the new block-ID column.

        Returns
        -------
        Single DataFrame (dedup) or tuple of two DataFrames (rec-lin).

        Examples
        --------
        >>> x = blocking_result.add_block_column(org_x_df)  # dedup
        >>> x, y = blocking_result.add_block_column(org_x_df, org_y_df)  # rec-lin

        """

        def _fill_orphans(out: pd.DataFrame, col: str, start_id: int) -> int:
            mask = out[col].isna()
            n = int(mask.sum())
            if n > 0:
                new_ids = range(start_id, start_id + n)
                out.loc[mask, col] = list(new_ids)
                start_id += n
            out[col] = out[col].astype("int64")
            return start_id

        max_block = int(self.result["block"].max()) + 1

        if df_right is None:
            mapping = (
                self.result.melt(id_vars="block", value_vars=["x", "y"], value_name="rec-id-map")
                .drop_duplicates("rec-id-map")
                .set_index("rec-id-map")["block"]
            )
            out = df_left.copy()
            if id_col_left:
                out[block_col] = out[id_col_left].map(mapping)
            else:
                out[block_col] = out.index.map(mapping)

            out[block_col] = out[block_col].astype("Int64")
            _fill_orphans(out, block_col, max_block)
            return out

        map_x = self.result[["x", "block"]].drop_duplicates("x").set_index("x")["block"]
        map_y = self.result[["y", "block"]].drop_duplicates("y").set_index("y")["block"]

        left = df_left.copy()
        if id_col_left:
            left[block_col] = left[id_col_left].map(map_x)
        else:
            left[block_col] = left.index.map(map_x)
        left[block_col] = left[block_col].astype("Int64")
        max_block = _fill_orphans(left, block_col, max_block)

        right = df_right.copy()
        if id_col_right:
            right[block_col] = right[id_col_right].map(map_y)
        else:
            right[block_col] = right.index.map(map_y)
        right[block_col] = right[block_col].astype("Int64")
        _fill_orphans(right, block_col, max_block)

        return left, right

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

        self.metrics.index = self.metrics.index.map(str)
        self.metrics = self.metrics.astype(float) if self.metrics is not None else None
        return {
            name: round(value * 100.0, 4)
            for name, value in zip(self.metrics.index, self.metrics.values, strict=False)
        }
