"""Utility functions for handling dataset files."""

import gzip
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def get_data_home(data_home: str | None = None) -> Path:
    """Return the path of the BlockingPy data directory."""
    if data_home is None:
        data_home = Path(__file__).parent
    else:
        data_home = Path(data_home)

    data_home = data_home.expanduser()
    data_home.mkdir(parents=True, exist_ok=True)

    return data_home


def decompress_gzfile(gz_path: Path, target_path: Path) -> None:
    """
    Decompress a gzipped file.

    Parameters
    ----------
    gz_path : Path
        Path to the compressed .gz file
    target_path : Path
        Path where to save the decompressed file

    """
    logger.info(f"Decompressing {gz_path} to {target_path}")
    with gzip.open(gz_path, "rb") as f_in:
        with open(target_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def get_data_file(filename: str, data_home: str | None = None) -> Path:
    """
    Get path to a data file, decompressing if necessary.

    Parameters
    ----------
    filename : str
        Name of the data file (without .gz extension)
    data_home : str, optional
        Alternative directory to look for the data

    Returns
    -------
    Path
        Path to the decompressed data file

    Raises
    ------
    FileNotFoundError
        If neither the compressed nor decompressed file is found

    """
    data_dir = get_data_home(data_home) / "data"
    file_path = data_dir / filename
    gz_path = data_dir / f"{filename}.gz"

    if not file_path.exists() and gz_path.exists():
        logger.info(f"Found compressed file {gz_path}")
        decompress_gzfile(gz_path, file_path)
    elif not file_path.exists() and not gz_path.exists():
        raise FileNotFoundError(
            f"Neither {file_path} nor {gz_path} found. "
            "Ensure the package is installed correctly."
        )

    return file_path
