import networkx as nx
import numpy as np
from math import comb
from collections import Counter
from typing import Dict, Optional
import pandas as pd

class BlockingResult:
    """
    A class to represent and analyze the results of a blocking operation.

    Args:
        x_df (pd.DataFrame): DataFrame containing blocking results with columns ['x', 'y', 'block', 'dist'].
        ann (str): The blocking method used (e.g., 'nnd', 'hnsw', 'annoy', etc.).
        deduplication (bool): Whether the blocking was performed for deduplication.
        true_blocks (Optional[pd.DataFrame]): DataFrame with true blocks to calculate evaluation metrics.
        eval_metrics (Optional[pd.Series]): Evaluation metrics if true blocks were provided.
        confusion (Optional[pd.DataFrame]): Confusion matrix if true blocks were provided.
        colnames_xy (np.ndarray): Column names used in the blocking process.
        graph (Optional[bool]): Whether to create a graph from the blocking results.
    """
    def __init__(self, x_df,
                ann: str,
                deduplication: bool,
                true_blocks: Optional[pd.DataFrame],
                eval_metrics: Optional[pd.Series],
                confusion: Optional[pd.DataFrame], 
                colnames_xy: np.ndarray, 
                graph: Optional[bool] = False):
        
        self.result = x_df[["x", "y", "block", "dist"]]
        self.method = ann
        self.deduplication = deduplication
        self.metrics = eval_metrics if true_blocks is not None else None
        self.confusion = confusion if true_blocks is not None else None
        self.colnames = colnames_xy
        self.graph = nx.from_pandas_edgelist(x_df[["x", "y"]], source="x", target="y") if graph else None

    def __repr__(self):
        """Provide a concise representation of the blocking result."""
        return f"BlockingResult(method={self.method}, deduplication={self.deduplication})"
    

    def __str__(self) -> str:
        """Create a string representation of the blocking result."""
        
        blocks_tab = self.result['block'].value_counts()   
        block_sizes = Counter(blocks_tab.values)
        reduction_ratio = self._calculate_reduction_ratio()

        output = []
        output.append("=" * 56)
        output.append(f"Blocking based on the {self.method} method.")
        output.append(f"Number of blocks: {len(blocks_tab)}")
        output.append(f"Number of columns used for blocking: {len(self.colnames)}")
        output.append(f"Reduction ratio: {reduction_ratio:.4f}")
        output.append("=" * 56)

        output.append("Distribution of the size of the blocks:")
        for size, count in sorted(block_sizes.items()):
            output.append(f"Size -> {size} : {count} <- Number of blocks with this size")

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
        
        Returns:
            float: Reduction ratio
        """
        blocks_tab = self.result['block'].value_counts()

        block_ids = np.repeat(blocks_tab.index.values, blocks_tab.values + 1)

        block_id_counts = (Counter(block_ids))
        numerator = sum(comb(count, 2) for count in block_id_counts.values())
        denominator = comb(len(block_ids), 2)

        return 1 - (numerator / denominator if denominator > 0 else 0)
    
    def _format_metrics(self) -> Dict[str, float]:
        """
        Format the evaluation metrics for the blocking method.
        
        Returns:
            Dict[str, float]: Formatted evaluation metrics
        """
        if self.metrics is None:
            return {}
        
        return {name: float(f"{value*100:.4f}") for name, value in self.metrics.items()}