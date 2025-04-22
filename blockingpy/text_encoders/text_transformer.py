"""Facade for selecting a concrete TextEncoder based on configuration."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .base import TextEncoder
from .embedding_encoder import EmbeddingEncoder
from .shingle_encoder import NgramEncoder

_ENCODER_MAP: dict[str, type[TextEncoder]] = {
    "shingle": NgramEncoder,
    "embedding": EmbeddingEncoder,
}


class TextTransformer(TextEncoder):

    """
    Facade for selecting a concrete TextEncoder based on configuration.

    Parameters
    ----------
    **control_txt : Mapping[str, Any]
        Configuration dictionary. Must contain key 'encoder' specifying
        one of the registry keys (e.g., 'shingle', 'embedding').
        Additional sub-dicts may supply encoder-specific params:
        e.g. {'encoder': 'embedding', 'embedding': {...}, 'shingle': {...}}

    """

    def __init__(self, **control_txt: Mapping[str, Any]) -> None:
        """Initialize the TextTransformer with a specific encoder."""
        encoder_key = control_txt.get("encoder", "shingle")
        if encoder_key not in _ENCODER_MAP:
            raise ValueError(
                f"Unknown encoder '{encoder_key}'. Valid options: {list(_ENCODER_MAP)}"
            )
        encoder_cls = _ENCODER_MAP[encoder_key]
        specific = control_txt.get(encoder_key, {})
        self.encoder: TextEncoder = encoder_cls(**specific)

    def fit(self, X: pd.Series, y: pd.Series|None = None) -> "TextTransformer":
        """Learn encoder-specific state (e.g., vocabulary)."""
        self.encoder.fit(X, y)
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        """Transform input texts into a feature matrix."""
        return self.encoder.transform(X)

    def fit_transform(self, X: pd.Series, y: pd.Series | None=None) -> pd.DataFrame:
        """Fit on X then transform X."""
        return self.encoder.fit(X, y).transform(X)
