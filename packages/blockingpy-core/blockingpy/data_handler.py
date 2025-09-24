"""Custom DataHandler class."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import DTypeLike
from scipy import sparse as _sp

ArrayLike: TypeAlias = np.ndarray | _sp.csr_matrix


@dataclass(slots=True, frozen=True)
class DataHandler:
    """Lightweight wrapper for a feature matrix and its column names."""

    data: ArrayLike
    cols: Sequence[str]

    @property
    def shape(self) -> tuple[int, int]:
        NDIM_VAL = 2
        if getattr(self.data, "ndim", 2) != NDIM_VAL:
            raise ValueError("DataHandler.data must be 2-D")
        return cast(tuple[int, int], self.data.shape)

    @property
    def is_sparse(self) -> bool:
        return _sp.issparse(self.data)

    def to_dense(
        self,
        *,
        dtype: DTypeLike | None = np.float32,
        contiguous: bool = True,
    ) -> np.ndarray:
        """
        Convert the feature matrix to a dense NumPy array.
        """
        arr = self.data.toarray() if _sp.issparse(self.data) else self.data

        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)

        if contiguous:
            arr = np.ascontiguousarray(arr)
        return arr
