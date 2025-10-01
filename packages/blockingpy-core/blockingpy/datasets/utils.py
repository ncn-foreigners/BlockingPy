"""Utility functions for handling dataset files."""

import os
from pathlib import Path

import pooch

_BASE_URL = "https://github.com/ncn-foreigners/BlockingPy/releases/download/data-0.2.4/"

_REGISTRY = {
    "rldata10000.csv": "sha256:69c3617051d551251e5bcd5a98005d6e20af3661252390055b397618d2a5142d",
    "cis.csv": "sha256:d7917de312d3847fcfa28bf3dfd0f49367a2d412b1b30b8af23813ef9151eac2",
    "census.csv": "sha256:5c103f0a556363321cfda7705c1b67c6b66517e9b7905f8217c11df49b2f2f4e",
}


def _create_pooch(data_dir: str | None = None) -> pooch.Pooch:
    """
    Create and return a Pooch instance for managing example data files.
    """
    cache_dir = (
        Path(data_dir)
        if data_dir
        else Path(os.environ.get("BLOCKINGPY_DATA", pooch.os_cache("blockingpy")))
    )
    return pooch.create(
        path=str(cache_dir),
        base_url=_BASE_URL,
        version=None,
        version_dev="main",
        registry=_REGISTRY,
        env="BLOCKINGPY_DATA",
    )


def fetch_example_file(name: str, data_dir: str | None = None) -> str:
    """
    Download `name` on first use, verify hash, return local path.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown example file '{name}'. Available: {sorted(_REGISTRY)}")
    try:
        return _create_pooch(data_dir).fetch(name)
    except Exception as e:
        raise RuntimeError(
            "Could not download example data. Check internet "
            "or pre-populate the cache at $BLOCKINGPY_DATA."
        ) from e
