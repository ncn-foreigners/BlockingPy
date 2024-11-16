import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mat_x():
    """Create sparse DataFrame X for testing."""
    columns = [
        "cy", "ij", "im", "km", "lj", "mj", "nk", "nm", "rk", "yr", "yp",
        "ho", "ki", "ls", "py", "sk", "th", "yt", "al", "an", "ja", "ko",
        "mo", "nt", "ow", "ty", "wa", "on"
    ]
    

    data = np.zeros((3, 28))

    data[0, [columns.index(col) for col in ["ij", "ho", "ki", "ls", "nt", "py", "sk", "ty", "al", "on"]]] = 1

    data[1, [columns.index(col) for col in ["ij", "ho", "ko", "mo", "py", "ty", "al", "an"]]] = 1

    data[2, [columns.index(col) for col in ["ij", "ho"]]] = 2

    df = pd.DataFrame(data, columns=columns)
    return df.astype(pd.SparseDtype("float", np.nan))

@pytest.fixture
def mat_y():
    """Create sparse DataFrame Y for testing."""
    data = np.zeros((8, 28))
    columns = [
        "cy", "ij", "im", "km", "lj", "mj", "nk", "nm", "rk", "yr", "yp",
        "ho", "ki", "ls", "py", "sk", "th", "yt", "al", "an", "ja", "ko",
        "mo", "nt", "ow", "ty", "wa", "on"
    ]

    group1_cols = ["al", "an", "ja", "ko", "mo", "nt", "ow", "ty", "wa", "on"]
    group1_start = 18
    for i in range(4):
        for j in range(len(group1_cols)):
            data[i, group1_start + j] = 1

    for i in range(4, 7):
        for j in range(len(group1_cols)):
            data[i, group1_start + j] = 1

    data[7, group1_start:group1_start+len(group1_cols)] = 2

    df = pd.DataFrame(data, columns=columns)
    return df.astype(pd.SparseDtype("float", np.nan))

@pytest.fixture
def small_sparse_x():
    """Create small sparse DataFrame for basic testing."""
    data = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0]
    ]
    columns = ["feat1", "feat2", "feat3"]
    df = pd.DataFrame(data, columns=columns)
    return df.astype(pd.SparseDtype("float", np.nan))

@pytest.fixture
def small_sparse_y():
    """Create small sparse DataFrame for basic testing."""
    data = [
        [1, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0]
    ]
    columns = ["feat1", "feat2", "feat3"]
    df = pd.DataFrame(data, columns=columns)
    return df.astype(pd.SparseDtype("float", np.nan))