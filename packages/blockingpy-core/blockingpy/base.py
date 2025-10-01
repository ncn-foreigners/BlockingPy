"""Contains the abstract base class for blocking methods."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .data_handler import DataHandler


class BlockingMethod(ABC):
    """
    Abstract base class for blocking methods.

    This class defines the interface for all blocking method implementations.

    Parameters
    ----------
    None

    Notes
    -----
    All blocking method implementations must inherit from this class and
    implement the `block` method.

    See Also
    --------
    AnnoyBlocker : Blocking using Annoy algorithm
    FaissBlocker : Blocking using FAISS algorithm
    HNSWBlocker : Blocking using HNSW algorithm
    MLPackBlocker : Blocking using MLPack algorithms
    NNDBlocker : Blocking using Nearest Neighbor Descent
    VoyagerBlocker : Blocking using Voyager algorithm

    """

    @abstractmethod
    def block(
        self,
        x: DataHandler,
        y: DataHandler,
        k: int,
        verbose: bool | None,
        controls: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Perform blocking operation to identify potential matches.

        This abstract method must be implemented by all blocking method classes.
        It should efficiently find approximate nearest neighbors for each query
        point in the input dataset.

        Parameters
        ----------
        x : DataHandler
            Reference dataset containing features for indexing
        y : DataHandler
            Query dataset to find nearest neighbors for
        k : int
            Number of nearest neighbors to find
        verbose : bool, optional
            If True, print detailed progress information
        controls : dict
            Algorithm-specific control parameters. The structure varies by
            implementation but typically includes:
            - Distance metric
            - Search parameters
            - Index construction parameters
            - Performance tuning options

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the blocking results with columns:
            - 'y': indices from query dataset
            - 'x': indices of matched items from reference dataset
            - 'dist': distances to matched items

        Raises
        ------
        NotImplementedError
            If the child class does not implement this method
        ValueError
            If invalid parameters are provided (implementation specific)

        Notes
        -----
        Different implementations may have different performance characteristics
        and trade-offs. Some may be better suited for high-dimensional data,
        others for specific distance metrics or data distributions.

        """
