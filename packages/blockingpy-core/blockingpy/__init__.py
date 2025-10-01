"""Public API for blockingpy."""

import logging

from .blocker import Blocker
from .blocking_result import BlockingResult
from .datasets import load_census_cis_data, load_deduplication_data

__all__ = ["Blocker", "BlockingResult", "load_census_cis_data", "load_deduplication_data"]
logger = logging.getLogger("blockingpy")
