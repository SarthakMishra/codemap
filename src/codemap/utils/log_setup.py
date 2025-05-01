"""
Logging setup for CodeMap.

This module configures logging for different parts of the CodeMap
application, ensuring logs are stored in the appropriate directories.

"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule
from rich.text import Text

# Initialize console for rich output
console = Console()


def setup_logging(
	is_verbose: bool = False,
	log_to_console: bool = True,
) -> None:
	"""
	Set up logging configuration.

	Args:
	    log_type: Type of log ('daemon', 'cli', 'error')
	    log_name: Specific name for the log file (default: based on log_type)
	    is_verbose: Enable verbose logging
	    log_to_file: Whether to log to a file
	    log_to_console: Whether to log to the console

	"""
	# Determine log level
	log_level = logging.DEBUG if is_verbose else logging.INFO

	# Root logger configuration
	root_logger = logging.getLogger()
	root_logger.setLevel(log_level)

	# Clear existing handlers
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
		formatter = logging.Formatter("%(message)s")
		console_handler.setFormatter(formatter)
		root_logger.addHandler(console_handler)


def log_environment_info() -> None:
	"""Log information about the execution environment."""
	logger = logging.getLogger(__name__)

	try:
		import platform

		from codemap import __version__

		logger.info("CodeMap version: %s", __version__)
		logger.info("Python version: %s", platform.python_version())
		logger.info("Platform: %s", platform.platform())

	except Exception:
		# logger.exception automatically handles exception info
		logger.exception("Error logging environment info:")


def display_error_summary(error_message: str) -> None:
	"""
	Display an error summary with a divider and a title.

	Args:
	        error_message: The error message to display

	"""
	title = Text("Error Summary", style="bold red")

	console.print()
	console.print(Rule(title, style="red"))
	console.print(f"\n{error_message}\n")
	console.print(Rule(style="red"))
	console.print()


def display_warning_summary(warning_message: str) -> None:
	"""
	Display a warning summary with a divider and a title.

	Args:
	        warning_message: The warning message to display

	"""
	title = Text("Warning Summary", style="bold yellow")

	console.print()
	console.print(Rule(title, style="yellow"))
	console.print(f"\n{warning_message}\n")
	console.print(Rule(style="yellow"))
	console.print()
