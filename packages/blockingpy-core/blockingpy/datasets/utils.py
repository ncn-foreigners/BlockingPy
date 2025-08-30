"""Utility functions for handling dataset files."""

from contextlib import contextmanager
from importlib.resources import files, as_file
from pathlib import Path


@contextmanager
def open_package_data(name: str):
    """
    Yield a filesystem path to a bundled dataset file inside
    blockingpy.datasets/data, working for wheels and zipped installs.
    """
    ref = files("blockingpy.datasets").joinpath("data").joinpath(name)
    if not ref.is_file():
        raise FileNotFoundError(f"Bundled dataset not found: {name}")
    with as_file(ref) as p:
        yield p  

def resolve_external_file(filename: str, data_home: str) -> Path:
    """
    Return a Path to filename (or filename.gz) inside <data_home>/data.
    No decompression is performed; pandas can read .gz directly.
    """
    base = Path(data_home).expanduser()
    data_dir = base / "data"
    for cand in (filename, f"{filename}.gz"):
        p = data_dir / cand
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Neither {data_dir/filename} nor {data_dir/(filename+'.gz')} found."
    )
