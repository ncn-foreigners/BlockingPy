"""Functions to load built-in BlockingPy datasets."""

import pandas as pd
from .utils import open_package_data, resolve_external_file

def _read_csv_any(pathlike):
    return pd.read_csv(pathlike)

def load_census_cis_data(as_frame: bool = True, data_home: str | None = None):
    census_names = ("census.csv.gz", "census.csv")
    cis_names = ("cis.csv.gz", "cis.csv")

    if data_home is None:
        for name in census_names:
            try:
                with open_package_data(name) as p:
                    census = _read_csv_any(p)
                    break
            except FileNotFoundError:
                continue
        for name in cis_names:
            try:
                with open_package_data(name) as p:
                    cis = _read_csv_any(p)
                    break
            except FileNotFoundError:
                continue
    else:
        census = _read_csv_any(resolve_external_file("census.csv", data_home))
        cis    = _read_csv_any(resolve_external_file("cis.csv", data_home))

    if not as_frame:
        census, cis = census.to_numpy(), cis.to_numpy()
    return census, cis

def load_deduplication_data(as_frame: bool = True, data_home: str | None = None):
    candidates = ("RL_data_10000.csv.gz", "rldata10000.csv.gz",
                  "RL_data_10000.csv",    "rldata10000.csv")

    if data_home is None:
        for name in candidates:
            try:
                with open_package_data(name) as p:
                    data = _read_csv_any(p)
                    break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError("Bundled RLdata file not found in package data/")
    else:
        data = _read_csv_any(resolve_external_file("RL_data_10000.csv", data_home))

    if not as_frame:
        data = data.to_numpy()
    return data
