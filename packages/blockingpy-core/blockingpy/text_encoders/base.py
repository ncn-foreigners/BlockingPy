"""Base class for text-to-matrix transformers using DataHandler."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pandas import Series

from ..data_handler import DataHandler


class TextEncoder(ABC):
    """
    Abstract base class for text-to-matrix transformers.

    Concrete subclasses turn a :class:`pandas.Series` of raw strings into a
    numeric feature matrix packaged in a :class:`DataHandler`.  The actual
    container may be a dense ``np.ndarray`` (e.g. embeddings) or a sparse
    ``scipy.sparse.csr_matrix`` (e.g. n-gram token counts), but that detail is
    hidden behind the common :class:`DataHandler` interface.
    """

    def fit(self, X: Series, y: Series | None = None) -> TextEncoder:
        """
        Learn stateful parameters from *X*.

        The default implementation is a no-op that returns *self*; override in
        subclasses that need to build a vocabulary or train a model.

        Parameters
        ----------
        X
            Series of input strings to learn from.
        y
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        TextEncoder
            ``self`` to allow method chaining.

        """
        return self

    def fit_transform(self, X: Series, y: Series | None = None) -> DataHandler:
        """
        Fit the encoder on *X* and return the transformed matrix.

        Equivalent to calling :py:meth:`fit` followed by :py:meth:`transform`.

        Parameters
        ----------
        X
            Series of input strings.
        y
            Ignored.

        Returns
        -------
        DataHandler
            The encoded feature matrix together with its column names.

        """
        return self.fit(X, y).transform(X)

    @abstractmethod
    def transform(self, X: Series) -> DataHandler:
        """
        Convert raw strings into a numeric feature matrix.

        Subclasses must implement this method.

        Parameters
        ----------
        X
            Series of raw text to encode.

        Returns
        -------
        DataHandler
            Wrapper containing the encoded matrix and its feature names.

        """
        ...
