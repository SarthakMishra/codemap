"""Command-line interface package for CodeMap."""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from codemap import __version__
from codemap.utils.cli_utils import check_for_updates
from codemap.utils.log_setup import setup_logging

# Import registration functions first
from .ask_cmd import register_command as register_ask_command
from .commit_cmd import register_command as register_commit_command
from .gen_cmd import register_command as register_gen_command
from .index_cmd import register_command as register_index_command
from .pr_cmd import register_command as register_pr_command

# Configure logging early for import error handling
logger = logging.getLogger(__name__)

# Load environment variables from .env files
try:
	from dotenv import load_dotenv

	# Try to load from .env.local first, then fall back to .env
	env_local = Path(".env.local")
	if env_local.exists():
		load_dotenv(dotenv_path=env_local)
		logger.debug("Loaded environment variables from %s", env_local)
	else:
		env_file = Path(".env")
		if env_file.exists():
			load_dotenv(dotenv_path=env_file)
			logger.debug("Loaded environment variables from %s", env_file)
except ImportError as err:
	# python-dotenv is required for loading environment variables from .env files
	# Log an error and raise an exception to halt execution if it's missing.
	error_msg = (
		"The 'python-dotenv' package is required but not installed.Please install it using: pip install python-dotenv"
	)
	# Use the named logger instead of root logger
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
	context_settings={"help_option_names": ["-h", "--help"]},
)

# --- Global Options Callback ---


def _version_callback(value: bool) -> None:
	"""Callback for --version option."""
	if value:
		typer.echo(f"CodeMap version: {__version__}")
		raise typer.Exit


@app.callback(invoke_without_command=True)
def global_options(
	ctx: typer.Context,
	is_verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging.")] = False,
	is_output_log: Annotated[
		bool,
		typer.Option(
			"--save-log",
			help="Enable logging to a file. Logs to logs/codemap_{datetime}.log.",
		),
	] = False,
	_version: Annotated[
		bool | None,
		typer.Option("--version", help="Show version and exit.", callback=_version_callback, is_eager=True),
	] = None,
) -> None:
	"""Global CLI options and logging setup."""
	ctx.meta["is_verbose"] = is_verbose
	ctx.meta["is_output_log"] = is_output_log

	log_file_path_to_use: Path | None = None
	if is_output_log:
		log_dir = Path("logs")
		log_dir.mkdir(parents=True, exist_ok=True)
		current_time = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
		log_file_path_to_use = log_dir / f"codemap_{current_time}.log"

	setup_logging(is_verbose=is_verbose or is_output_log, log_file_path=log_file_path_to_use)

	# Perform update check (after logging is set up so we can log debug info)
	check_for_updates(is_verbose)


# --- Register commands using lazy-loading pattern ---

register_gen_command(app)
register_ask_command(app)
register_commit_command(app)
register_index_command(app)
register_pr_command(app)


# --- Main Entry Point ---
def main() -> int:
	"""Run the CLI application."""
	return app()


if __name__ == "__main__":
	sys.exit(main())
