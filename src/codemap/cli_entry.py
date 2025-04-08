"""Entry point for the CodeMap CLI."""

from __future__ import annotations

import sys

# Import the run function directly
from codemap.cli.commit import RunConfig, run


def main() -> int:
    """Command-line entry point."""
    try:
        # Create a default config with string strategy
        config = RunConfig(split_strategy="file")  # Use string value
        return run(config)
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
