"""Module containing the EmbeddingEncoder class."""
import pandas as pd
from model2vec import StaticModel

from .base import TextEncoder


class EmbeddingEncoder(TextEncoder):

    """
    Encoder that produces dense embeddings for text using the
    StaticModel from the model2vec library.
    """

    def __init__(
        self,
        model: str = "minishlab/potion-base-8M",
        normalize: bool | None = None,
        max_length: int | None = 512,
        emb_batch_size: int = 1024,
        show_progress_bar: bool = False,
        use_multiprocessing: bool = True,
        multiprocessing_threshold: int = 10000,
    ) -> None:
        """
        Initialize the embedding encoder.

        Parameters
        ----------
        model : str, optional
            Identifier or path for the pretrained model.
        normalize : bool, optional
            Whether to normalize output vectors (default True).
        max_length : int, optional
            Maximum sequence length for encoding (default 512).
        emb_batch_size : int, optional
            Batch size for encoding (default 1024).
        show_progress_bar : bool, optional
            If True, display a progress bar (default False).
        use_multiprocessing : bool, optional
            If True, use multiprocessing (default True).
        multiprocessing_threshold : int, optional
            Threshold for multiprocessing (default 10000).

        """
        self.model = model
        self.normalize = normalize
        self.max_length = max_length
        self.emb_batch_size = emb_batch_size
        self.show_progress_bar = show_progress_bar
        self.use_multiprocessing = use_multiprocessing
        self.multiprocessing_threshold = multiprocessing_threshold

    def fit(self, X: pd.Series, y: pd.Series | None = None) -> "EmbeddingEncoder":
        """
        Placeholder for fit method. This encoder does not learn any state
        from the data, but this method is included for API consistency.
        """
        return self

    def transform(self, x: pd.Series) -> pd.DataFrame:
        """
        Encode texts into dense embeddings.

        Parameters
        ----------
        x : pandas.Series
            Series of text strings to encode.

        Returns
        -------
        pandas.DataFrame
            DataFrame of shape (n_samples, embedding_dim) with column names
            'emb_0', 'emb_1', ..., representing each embedding dimension.

        """
        x = x.tolist()
        model = StaticModel.from_pretrained(self.model, normalize=self.normalize)
        embedding = model.encode(
            x,
            max_length=self.max_length,
            batch_size=self.emb_batch_size,
            show_progress_bar=self.show_progress_bar,
            use_multiprocessing=self.use_multiprocessing,
            multiprocessing_threshold=self.multiprocessing_threshold,
        )
        x_df = pd.DataFrame(embedding, columns=[f"emb_{i}" for i in range(embedding.shape[1])])
        return x_df
