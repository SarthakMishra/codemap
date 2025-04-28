"""
Logging setup for CodeMap.

This module configures logging for different parts of the CodeMap
application, ensuring logs are stored in the appropriate directories.

"""

from __future__ import annotations

import logging
import logging.handlers

from rich.console import Console
from rich.logging import RichHandler

from codemap.utils.directory_manager import get_directory_manager

# Initialize console for rich output
console = Console()


def setup_logging(
	log_type: str = "cli",
	log_name: str | None = None,
	is_verbose: bool = False,
	log_to_file: bool = True,
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

	# Setup file logging if requested
	if log_to_file:
		# Get directory manager and ensure directories exist
		dir_manager = get_directory_manager()
		dir_manager.ensure_directories()

		# Get log file path
		log_path = dir_manager.get_log_file_path(log_type, log_name)

		try:
			# Configure rotating file handler
			file_handler = logging.handlers.RotatingFileHandler(
				log_path,
				maxBytes=10 * 1024 * 1024,  # 10MB
				backupCount=5,
				encoding="utf-8",
			)

			# More detailed formatting for file logs
			file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
			file_handler.setFormatter(file_formatter)
			file_handler.setLevel(log_level)
			root_logger.addHandler(file_handler)

			logging.getLogger(__name__).debug("Log file: {log_path}")
		except (PermissionError, FileNotFoundError) as e:
			console.print(f"[yellow]Warning: Could not set up file logging: {e}[/yellow]")


def setup_daemon_logging(log_name: str | None = None, is_verbose: bool = False) -> None:
	"""
	Configure logging specifically for the daemon process.

	Args:
	    log_name: Specific name for the log file
	    is_verbose: Enable verbose logging

	"""
	setup_logging(
		log_type="daemon",
		log_name=log_name,
		is_verbose=is_verbose,
		log_to_file=True,
		log_to_console=False,  # Daemon typically doesn't log to console
	)


def log_environment_info() -> None:
	"""Log information about the execution environment."""
	logger = logging.getLogger(__name__)

	try:
		import platform

		from codemap import __version__

		logger.info("CodeMap version: %s", __version__)
		logger.info("Python version: %s", platform.python_version())
		logger.info("Platform: %s", platform.platform())
		logger.info("Directory structure initialized")

		# Log directory paths
		dir_manager = get_directory_manager()
		logger.debug("User data directory: %s", dir_manager.user_data_dir)
		logger.debug("User config directory: %s", dir_manager.user_config_dir)
		logger.debug("User cache directory: %s", dir_manager.user_cache_dir)
		logger.debug("User log directory: %s", dir_manager.user_log_dir)

	except Exception:
		logger.exception("Error logging environment info: %s")
