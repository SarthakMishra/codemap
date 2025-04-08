"""Entry point for the CodeMap CLI."""

from __future__ import annotations

import logging
import sys

# Import the run function directly
from codemap.cli.commit import RunConfig, run
from codemap.utils.git_utils import GitError

logger = logging.getLogger(__name__)


def main() -> int:
    """Command-line entry point."""
    try:
        # Create a default config with string strategy
        config = RunConfig(split_strategy="file")  # Use string value
        return run(config)
    except (GitError, ValueError, OSError, TypeError):
        logger.exception("Error running CodeMap commit")
        return 1


if __name__ == "__main__":
    sys.exit(main())
