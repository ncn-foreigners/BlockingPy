"""Tests for TextTransformer, NgramEncoder and EmbeddingEncoder classes with DataHandler output only (no pandas conversions)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blockingpy.text_encoders.text_transformer import TextTransformer
from blockingpy.text_encoders.shingle_encoder import NgramEncoder
from blockingpy.text_encoders.embedding_encoder import EmbeddingEncoder
from blockingpy.data_handler import DataHandler


@pytest.fixture

def sample_text_series() -> pd.Series:
    """Small text sample for encoder tests."""
    return pd.Series(["Monty Python", "python monty!!", "MONTY-PYTHON"])


def _assert_equal_handlers(lhs: DataHandler, rhs: DataHandler) -> None:  # noqa: D401
    """Assert two DataHandlers hold identical dense data with the same columns (order‑agnostic)."""
    # Ensure they contain exactly the same column set
    assert set(lhs.cols) == set(rhs.cols), "Column sets differ"

    # Establish a canonical column order (sorted) to align both matrices
    common_cols = sorted(lhs.cols)
    lhs_indices = [lhs.cols.index(c) for c in common_cols]
    rhs_indices = [rhs.cols.index(c) for c in common_cols]

    np.testing.assert_array_equal(
        lhs.to_dense()[:, lhs_indices],
        rhs.to_dense()[:, rhs_indices],
    )



def test_ngram_encoder_basic(sample_text_series: pd.Series) -> None:
    """NgramEncoder should return a DataHandler with expected shape and non‑negative values."""
    encoder = NgramEncoder(n_shingles=2, lowercase=True, strip_non_alphanum=True)
    dh = encoder.transform(sample_text_series)

    # Shape check
    assert dh.shape[0] == len(sample_text_series)
    assert dh.shape[1] > 0

    # Non‑negative values in dense view
    assert np.all(dh.to_dense() >= 0)



def test_ngram_encoder_token_contents() -> None:
    """Check that stripping non‑alphanum and lower‑casing works as expected."""
    series = pd.Series(["AbC!!"])
    encoder = NgramEncoder(n_shingles=3, lowercase=True, strip_non_alphanum=True)
    dh = encoder.transform(series)

    assert list(dh.cols) == ["abc"]
    assert dh.to_dense()[0, 0] == 1


@pytest.fixture
def dummy_static_model(monkeypatch):
    """Patch `StaticModel` used in EmbeddingEncoder with a lightweight dummy."""

    class _DummyModel:
        dim: int = 4

        @classmethod
        def from_pretrained(cls, model: str, normalize: bool | None = None):  # noqa: D401
            return cls()

        def encode(self, texts, *_, **__):  # noqa: ANN001, D401
            """Return ones for each text with fixed dimensionality."""
            return np.ones((len(texts), self.dim), dtype=np.float32)

    import blockingpy.text_encoders.embedding_encoder as _emb_mod

    monkeypatch.setattr(_emb_mod, "StaticModel", _DummyModel)
    yield _DummyModel



def test_embedding_encoder_basic(sample_text_series: pd.Series, dummy_static_model) -> None:  # noqa: D401
    """EmbeddingEncoder should return DataHandler with expected embedding columns."""
    encoder = EmbeddingEncoder(model="dummy/unused", normalize=True)
    dh = encoder.transform(sample_text_series)

    assert dh.shape == (len(sample_text_series), dummy_static_model.dim)

    # fit_transform parity
    np.testing.assert_array_equal(
        encoder.transform(sample_text_series).to_dense(),
        encoder.fit_transform(sample_text_series).to_dense(),
    )
    # fit returns self
    assert encoder is encoder.fit(sample_text_series)

    expected_cols = [f"emb_{i}" for i in range(dummy_static_model.dim)]
    assert list(dh.cols) == expected_cols
    assert np.all(dh.to_dense() == 1)



def test_text_transformer_shingle_equivalence(sample_text_series: pd.Series) -> None:
    """TextTransformer('shingle') output should equal direct NgramEncoder output."""
    transformer = TextTransformer(encoder="shingle", shingle={"n_shingles": 2})
    dh_trans = transformer.transform(sample_text_series)
    dh_direct = NgramEncoder(n_shingles=2).transform(sample_text_series)

    _assert_equal_handlers(dh_trans, dh_direct)



def test_text_transformer_embedding_selection(sample_text_series: pd.Series, dummy_static_model) -> None:  # noqa: D401
    """TextTransformer should select EmbeddingEncoder when requested."""
    transformer = TextTransformer(encoder="embedding", embedding={"model": "irrelevant"})

    result = transformer.transform(sample_text_series)

    # fit_transform parity
    np.testing.assert_array_equal(
        transformer.transform(sample_text_series).to_dense(),
        transformer.fit_transform(sample_text_series).to_dense(),
    )
    assert transformer.encoder is transformer.fit(sample_text_series).encoder

    assert result.shape[1] == dummy_static_model.dim



def test_text_transformer_invalid_encoder() -> None:
    """Using an unknown encoder key should raise ValueError."""
    with pytest.raises(ValueError):
        TextTransformer(encoder="unknown_encoder")
