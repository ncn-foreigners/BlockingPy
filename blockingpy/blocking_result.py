import networkx as nx

class BlockingResult:
    def __init__(self, x_df, ann, deduplication, true_blocks, eval_metrics, confusion, colnames_xy, graph):
        self.result = x_df[["x", "y", "block", "dist"]]
        self.method = ann
        self.deduplication = deduplication
        self.metrics = eval_metrics if true_blocks is not None else None
        self.confusion = confusion if true_blocks is not None else None
        self.colnames = colnames_xy
        self.graph = nx.from_pandas_edgelist(x_df[["x", "y"]], source="x", target="y") if graph else None

    def __repr__(self):
        return f"BlockingResult(method={self.method}, deduplication={self.deduplication})"