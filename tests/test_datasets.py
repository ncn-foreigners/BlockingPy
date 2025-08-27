import pandas as pd

from blockingpy.datasets import load_census_cis_data, load_deduplication_data


def test_load_census_cis_data():
    census, cis = load_census_cis_data()
    assert isinstance(census, pd.DataFrame)
    assert isinstance(cis, pd.DataFrame)


def test_load_deduplication_data():
    data = load_deduplication_data()
    assert isinstance(data, pd.DataFrame)
    expected_cols = ["fname_c1", "fname_c2", "lname_c1", "lname_c2", "by", "bm", "bd"]
    assert all(col in data.columns for col in expected_cols)
