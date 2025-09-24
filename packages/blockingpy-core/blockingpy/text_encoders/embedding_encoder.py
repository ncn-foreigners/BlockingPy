"""Module containing the EmbeddingEncoder class using DataHandler."""

from __future__ import annotations

import numpy as np
from model2vec import StaticModel
from pandas import Series

from ..data_handler import DataHandler
from .base import TextEncoder


class EmbeddingEncoder(TextEncoder):
    """
    Dense-vector encoder that wraps `model2vec.StaticModel`.

    The encoder converts a :class:`pandas.Series` of text strings into a
    :class:`DataHandler` whose ``data`` attribute is a C-contiguous
    ``np.ndarray`` of shape ``(n_samples, embedding_dim)`` and whose ``cols``
    are the synthetic column names ``emb_0 … emb_{d-1}``.
    """

    def __init__(  # noqa: PLR0913
        self,
        model: str = "minishlab/potion-base-8M",
        normalize: bool | None = None,
        max_length: int | None = 512,
        emb_batch_size: int = 1024,
        show_progress_bar: bool = False,
        use_multiprocessing: bool = True,
        multiprocessing_threshold: int = 10_000,
    ) -> None:
        self.model = model
        self.normalize = normalize
        self.max_length = max_length
        self.emb_batch_size = emb_batch_size
        self.show_progress_bar = show_progress_bar
        self.use_multiprocessing = use_multiprocessing
        self.multiprocessing_threshold = multiprocessing_threshold

    def fit(self, X: Series, y: Series | None = None) -> EmbeddingEncoder:
        """No-op fit for scikit-learn compatibility."""
        return self

    def transform(self, X: Series) -> DataHandler:
        """
        Encode *X* into dense numeric vectors.

        Parameters
        ----------
        X
            Series of raw text strings.

        Returns
        -------
        DataHandler
            ``data`` is ``np.ndarray`` ``(n_samples, d)`` in ``float32``;
            ``cols`` contains synthetic names ``emb_0 … emb_{d-1}``.

        """
        model = StaticModel.from_pretrained(self.model, normalize=self.normalize)
        embeddings: np.ndarray = model.encode(
            X.tolist(),
            max_length=self.max_length,
            batch_size=self.emb_batch_size,
            show_progress_bar=self.show_progress_bar,
            use_multiprocessing=self.use_multiprocessing,
            multiprocessing_threshold=self.multiprocessing_threshold,
        )

        emb_arr = np.ascontiguousarray(embeddings, dtype=np.float32)

        dim = emb_arr.shape[1]
        colnames = [f"emb_{i}" for i in range(dim)]

        return DataHandler(data=emb_arr, cols=colnames)
