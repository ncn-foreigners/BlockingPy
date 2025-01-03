"""Set up logging for blockingpy package."""

import logging
import sys

from .blocker import Blocker

__all__ = ["Blocker"]

logger = logging.getLogger("blockingpy")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.propagate = True
