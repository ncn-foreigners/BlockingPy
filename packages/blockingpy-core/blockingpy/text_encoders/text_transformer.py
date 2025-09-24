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

    def __init__(self, **control_txt: Mapping[str, Any] | str) -> None:
        enc_val = control_txt.get("encoder", "shingle")

        if isinstance(enc_val, str):
            name = enc_val
            inline_cfg: Mapping[str, Any] = {}
        elif isinstance(enc_val, Mapping):
            n = enc_val.get("name", "shingle")
            if not isinstance(n, str):
                raise TypeError("encoder.name must be str")
            name = n
            inline_cfg = enc_val
        else:
            raise TypeError("encoder must be str or mapping")

        if name not in _ENCODER_MAP:
            raise ValueError(f"Unknown encoder '{name}'. Valid options: {list(_ENCODER_MAP)}")

        encoder_cls = _ENCODER_MAP[name]

        spec_from_control = control_txt.get(name)
        if not isinstance(spec_from_control, Mapping):
            spec_from_control = {}

        specific: dict[str, Any] = {**spec_from_control, **inline_cfg}
        self.encoder: TextEncoder = encoder_cls(**specific)

    def fit(self, X: Series, y: Series | None = None) -> TextTransformer:
        self.encoder.fit(X, y)
        return self

    def transform(self, X: Series) -> DataHandler:
        return self.encoder.transform(X)

    def fit_transform(self, X: Series, y: Series | None = None) -> DataHandler:
        return self.encoder.fit(X, y).transform(X)
