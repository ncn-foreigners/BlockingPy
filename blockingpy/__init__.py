"""Set up logging for blockingpy package."""

import logging
import sys

logger = logging.getLogger("blockingpy")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.propagate = True
