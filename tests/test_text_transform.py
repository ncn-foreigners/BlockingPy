"""Tests for TextTransformer, NgramEncoder and EmbeddingEncoder classes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blockingpy.text_encoders.text_transformer import TextTransformer
from blockingpy.text_encoders.shingle_encoder import NgramEncoder
from blockingpy.text_encoders.embedding_encoder import EmbeddingEncoder


@pytest.fixture
def sample_text_series() -> pd.Series:
    """Small text sample for encoder tests."""
    return pd.Series(["Monty Python", "python monty!!", "MONTY-PYTHON"])

def test_ngram_encoder_basic(sample_text_series: pd.Series) -> None:
    """NgramEncoder should return a sparse DataFrame with expected shape."""
    encoder = NgramEncoder(n_shingles=2, lowercase=True, strip_non_alphanum=True)
    dtm = encoder.transform(sample_text_series)

    assert dtm.shape[0] == len(sample_text_series)

    assert (dtm.fillna(0) >= 0).all().all()

    assert dtm.shape[1] > 0


def test_ngram_encoder_token_contents() -> None:
    """Check that stripping non‑alphanum and lowercasing works as expected."""
    series = pd.Series(["AbC!!"])
    encoder = NgramEncoder(n_shingles=3, lowercase=True, strip_non_alphanum=True)
    dtm = encoder.transform(series)
    assert list(dtm.columns) == ["abc"]
    assert dtm.iloc[0, 0] == 1

@pytest.fixture
def dummy_static_model(monkeypatch):
    """Patch `StaticModel` used in EmbeddingEncoder with a lightweight dummy."""

    class _DummyModel:
        dim: int = 4

        @classmethod
        def from_pretrained(cls, model: str, normalize: bool | None = None):
            return cls()

        def encode(self, texts, *_, **__):  # noqa: ANN001, D401
            """Return ones for each text with fixed dimensionality."""
            return np.ones((len(texts), self.dim), dtype=np.float32)

    import blockingpy.text_encoders.embedding_encoder as _emb_mod

    monkeypatch.setattr(_emb_mod, "StaticModel", _DummyModel)
    yield _DummyModel


def test_embedding_encoder_basic(sample_text_series: pd.Series, dummy_static_model) -> None:  # noqa: D401
    """EmbeddingEncoder should return dense DataFrame with embedding columns."""
    encoder = EmbeddingEncoder(model="dummy/unused", normalize=True)
    df = encoder.transform(sample_text_series)

    assert df.shape == (len(sample_text_series), dummy_static_model.dim)
    assert all(encoder.transform(sample_text_series)) == all(encoder.fit_transform(sample_text_series))
    assert encoder == encoder.fit(sample_text_series)
    expected_cols = [f"emb_{i}" for i in range(dummy_static_model.dim)]
    assert list(df.columns) == expected_cols
    assert (df.values == 1).all()

def test_text_transformer_shingle_equivalence(sample_text_series: pd.Series) -> None:
    """TextTransformer('shingle') output should equal direct NgramEncoder output."""
    control_txt = {"encoder": "shingle", "shingle": {"n_shingles": 2}}
    transformer = TextTransformer(**control_txt)
    transformed_via_transformer = transformer.transform(sample_text_series)

    direct_encoder = NgramEncoder(n_shingles=2)
    transformed_direct = direct_encoder.transform(sample_text_series)

    pd.testing.assert_frame_equal(
        transformed_via_transformer.sort_index(axis=1),
        transformed_direct.sort_index(axis=1),
        check_dtype=False,
    )


def test_text_transformer_embedding_selection(sample_text_series: pd.Series, dummy_static_model) -> None:  # noqa: D401
    """TextTransformer should select EmbeddingEncoder when requested."""
    transformer = TextTransformer(encoder="embedding", embedding={"model": "irrelevant"})
    result = transformer.transform(sample_text_series)
    assert all(result) == all(transformer.fit_transform(sample_text_series))
    assert transformer == transformer.fit(sample_text_series)
    assert result.shape[1] == dummy_static_model.dim


def test_text_transformer_invalid_encoder() -> None:
    """Using an unknown encoder key should raise ValueError."""
    with pytest.raises(ValueError):
        TextTransformer(encoder="unknown_encoder")
