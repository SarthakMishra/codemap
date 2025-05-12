"""
Logging setup for CodeMap.

This module configures logging for different parts of the CodeMap
application, ensuring logs are stored in the appropriate directories.

"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# Initialize console for rich output
console = Console()


def setup_logging(
	is_verbose: bool = False,
	log_to_console: bool = True,
	log_file_path: Path | str | None = None,
	log_level_default: int = logging.ERROR,
	log_level_verbose: int = logging.DEBUG,
) -> None:
	"""
	Set up logging configuration.

	Args:
	    is_verbose: Enable verbose logging
	    log_to_console: Whether to log to the console
	    log_file_path: Optional path to a file for logging. If None, no file logging.
	    log_level_default: The log level to use for default logging.
	    log_level_verbose: The log level to use for verbose logging.
	"""
	# Determine log level
	log_level = log_level_default if not is_verbose else log_level_verbose

	# Root logger configuration
	root_logger = logging.getLogger()
	root_logger.setLevel(log_level)

	# Clear existing handlers to avoid duplicate logs if called multiple times
	for handler in root_logger.handlers[:]:
		root_logger.removeHandler(handler)

	# Setup console logging if requested
	if log_to_console:
		console_handler = RichHandler(
			level=log_level,
			rich_tracebacks=True,
			show_time=True,
			show_path=is_verbose,
		)
		root_logger.addHandler(console_handler)

	# Setup file logging if a path is provided
	if log_file_path:
		try:
			file_handler_path = Path(log_file_path)
			# Ensure the directory for the log file exists
			file_handler_path.parent.mkdir(parents=True, exist_ok=True)

			file_handler = logging.FileHandler(file_handler_path, mode="a", encoding="utf-8")
			file_handler.setLevel(logging.DEBUG)  # Log all debug messages and above to file
			file_formatter = logging.Formatter(
				"%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
			)
			file_handler.setFormatter(file_formatter)
			root_logger.addHandler(file_handler)
			root_logger.debug(f"Logging to file: {file_handler_path}")  # Log where file logs are going
		except (OSError, PermissionError, TypeError) as e:
			# If file logging setup fails, log error to console (if available) and continue
			console_error_msg = f"[CODEMAP CLI CRITICAL] Failed to set up file logging to {log_file_path}: {e}"
			# Use root logger to print critical setup error to console if possible
			# This avoids dependency on whether RichHandler was successfully added.
			crit_logger = logging.getLogger("codemap.cli.critical_setup")
			crit_logger.handlers.clear()  # Ensure it only goes to basic stderr if no console handler
			console_err_handler = logging.StreamHandler()
			console_err_handler.setFormatter(logging.Formatter("%(message)s"))
			crit_logger.addHandler(console_err_handler)
			crit_logger.propagate = False
			crit_logger.critical(console_error_msg)
