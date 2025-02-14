"""Functions to load built-in BlockingPy datasets."""

import pandas as pd

from .utils import get_data_file


def load_census_cis_data(
    as_frame: bool = True, data_home: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CIS and census datasets for record linkage example.

    This data was created by Paula McLeod, Dick Heasman and Ian Forbes, ONS,
    for the ESSnet DI on-the-job training course, Southampton,
    25-28 January 2011
    https://wayback.archive-it.org/12090/20231221144450/https://cros-legacy.ec.europa.eu/content/job-training_en

    This dataset contains census data in one table and CIS (Customer Information
    System) data in another, useful for demonstrating record linkage tasks.

    Parameters
    ----------
    as_frame : bool, default=True
        If True, returns pandas DataFrames, otherwise numpy arrays
    data_home : str, optional
        Alternative directory to look for the data

    Returns
    -------
    census : pandas.DataFrame
        Census dataset
    cis : pandas.DataFrame
        CIS dataset

    """
    census_file = get_data_file("census.csv", data_home)
    cis_file = get_data_file("cis.csv", data_home)

    census = pd.read_csv(census_file)
    cis = pd.read_csv(cis_file)

    if not as_frame:
        census = census.to_numpy()
        cis = cis.to_numpy()

    return census, cis


def load_deduplication_data(
    as_frame: bool = True, data_home: str | None = None
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the RLdata10000 dataset for deduplication examples.

    This data is taken from "RecordLinkage" R package developed by Murat Sariyar
    and Andreas Borg. Package is licensed under GPL-3 license.
    https://cran.r-project.org/package=RecordLinkage

    This dataset contains 10,000 artificial patient records with introduced
    typographical errors, making it suitable for deduplication tasks.

    Parameters
    ----------
    as_frame : bool, default=True
        If True, returns pandas DataFrame, otherwise numpy array
    data_home : str, optional
        Alternative directory to look for the data

    Returns
    -------
    data : pandas.DataFrame
        The full dataset with potential duplicates

    Examples
    --------
    >>> from blockingpy.datasets import load_deduplication_data
    >>> data = load_deduplication_data()
    >>> print(data.head())

    """
    data_file = get_data_file("rldata10000.csv", data_home)
    data = pd.read_csv(data_file)

    if not as_frame:
        data = data.to_numpy()

    return data
