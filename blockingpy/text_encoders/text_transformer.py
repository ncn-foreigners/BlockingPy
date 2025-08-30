"""Facade for selecting a concrete TextEncoder based on configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pandas import Series

from ..data_handler import DataHandler
from .base import TextEncoder
from .embedding_encoder import EmbeddingEncoder
from .shingle_encoder import NgramEncoder

_ENCODER_MAP: dict[str, type[TextEncoder]] = {
    "shingle": NgramEncoder,
    "embedding": EmbeddingEncoder,
}


class TextTransformer(TextEncoder):

    """
    Facade for selecting a concrete :class:`TextEncoder` based on a control
    dictionary.

    Parameters
    ----------
    **control_txt
        Configuration mapping. Must contain key ``encoder`` set to one of the
        registry keys (``'shingle'`` or ``'embedding'``). Additional
        sub‑mappings with the same names may provide encoder‑specific keyword
        arguments.

    """

    def __init__(self, **control_txt: Mapping[str, Any]) -> None:
        encoder_key = control_txt.get("encoder", "shingle")
        if encoder_key not in _ENCODER_MAP:
            raise ValueError(
                f"Unknown encoder '{encoder_key}'. Valid options: {list(_ENCODER_MAP)}"
            )
        encoder_cls = _ENCODER_MAP[encoder_key]
        specific: Mapping[str, Any] = control_txt.get(encoder_key, {})
        self.encoder: TextEncoder = encoder_cls(**specific)

    def fit(self, X: Series, y: Series | None = None) -> TextTransformer:
        self.encoder.fit(X, y)
        return self

    def transform(self, X: Series) -> DataHandler:
        return self.encoder.transform(X)

    def fit_transform(self, X: Series, y: Series | None = None) -> DataHandler:
        return self.encoder.fit(X, y).transform(X)
