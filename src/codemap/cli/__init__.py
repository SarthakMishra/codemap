"""Command-line interface package for CodeMap."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer  # type: ignore[import]

from codemap import __version__
from codemap.cli.commit_cmd import commit_command
from codemap.cli.gen_cmd import gen_command
from codemap.cli.pr_cmd import pr_command

logger = logging.getLogger(__name__)

# Load environment variables from .env files
try:
	from dotenv import load_dotenv

	# Try to load from .env.local first, then fall back to .env
	env_local = Path(".env.local")
	if env_local.exists():
		load_dotenv(dotenv_path=env_local)
		logging.getLogger(__name__).debug("Loaded environment variables from %s", env_local)
	else:
		env_file = Path(".env")
		if env_file.exists():
			load_dotenv(dotenv_path=env_file)
			logging.getLogger(__name__).debug("Loaded environment variables from %s", env_file)
except ImportError as err:
	# python-dotenv is required for loading environment variables from .env files
	# Log an error and raise an exception to halt execution if it's missing.
	error_msg = (
		"The 'python-dotenv' package is required but not installed.Please install it using: pip install python-dotenv"
	)
	# Log the error message along with the original exception information
	logger.exception(error_msg)
	# Raise a new RuntimeError, explicitly chaining the original ImportError
	raise RuntimeError(error_msg) from err

# Configure logging
logger = logging.getLogger(__name__)

# Determine the invoked command name for help message customization
invoked_command = Path(sys.argv[0]).name
if invoked_command == "cm":
	alias_note = "\n\nNote: 'cm' is an alias for 'codemap'."
elif invoked_command == "codemap":
	alias_note = "\n\nNote: You can also use 'cm' as a shorter alias."
else:
	alias_note = ""  # Handle cases where script might be called differently

# Initialize the main CLI app
app = typer.Typer(
	help=f"CodeMap - Developer tools powered by AI\n\nVersion: {__version__}{alias_note}",
	no_args_is_help=True,
)

# Register commands
app.command(name="gen")(gen_command)
app.command(name="commit")(commit_command)
app.command(name="pr")(pr_command)


def main() -> int:
	"""Run the CLI application."""
	return app()


if __name__ == "__main__":
	sys.exit(main())
