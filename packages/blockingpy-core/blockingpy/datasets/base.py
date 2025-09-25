from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import fetch_example_file


def _read_csv_any(pathlike: str | Path, **read_csv_kw: Any) -> pd.DataFrame:
    return pd.read_csv(pathlike, **read_csv_kw)


def load_census_cis_data(
    as_frame: bool = True,
    data_home: str | None = None,
    **read_csv_kw: Any,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
    """
    Returns (census, cis) in the same shapes as before.
    If data_home is provided, read from there. Otherwise download via Pooch.
    """
    census_path: str | Path
    cis_path: str | Path

    if data_home is None:
        census_path = fetch_example_file("census.csv")
        cis_path = fetch_example_file("cis.csv")
    else:
        census_path = Path(data_home) / "census.csv"
        cis_path = Path(data_home) / "cis.csv"

    census = _read_csv_any(census_path, **read_csv_kw)
    cis = _read_csv_any(cis_path, **read_csv_kw)

    if as_frame:
        return census, cis
    return census.to_numpy(), cis.to_numpy()


def load_deduplication_data(
    as_frame: bool = True,
    data_home: str | None = None,
    **read_csv_kw: Any,
) -> pd.DataFrame | np.ndarray:
    """
    Returns RLdata10000 in the same shape as before.
    Accepts legacy filename in data_home but downloads 'rldata10000.csv' by default.
    """
    path: str | Path

    if data_home is None:
        path = fetch_example_file("rldata10000.csv")
    else:
        for name in ("RL_data_10000.csv", "rldbata10000.csv", "rldata10000.csv"):
            candidate = Path(data_home) / name
            if candidate.exists():
                path = candidate
                break
        else:
            path = Path(data_home) / "rldata10000.csv"

    df = _read_csv_any(path, **read_csv_kw)
    return df if as_frame else df.to_numpy()
