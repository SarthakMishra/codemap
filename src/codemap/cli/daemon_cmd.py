"""Command-line interface for the daemon functionality."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003
from typing import Any

import typer

from codemap.daemon.command import list_or_show_jobs, show_daemon_logs, show_daemon_status, start_daemon, stop_daemon
from codemap.utils.cli_utils import console, progress_indicator, setup_logging, show_error
from codemap.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Define option defaults as module-level variables
CONFIG_OPTION = typer.Option(default=None, help="Path to configuration file", show_default=False)
FOREGROUND_OPTION = typer.Option(default=False, help="Run in foreground (for debugging)")
DETAILED_OPTION = typer.Option(default=False, help="Show detailed status information")
JSON_OPTION = typer.Option(default=False, help="Output as JSON")
STATUS_FILTER_OPTION = typer.Option(
	default=None, help="Filter jobs by status", metavar="[active|completed|failed]", show_default=False
)
TIMEOUT_OPTION = typer.Option(default=30, help="Timeout in seconds for startup/shutdown operations")

# Create Typer command group
daemon_cmd = typer.Typer(
	help="Manage the CodeMap daemon.",
	name="daemon",
)


def load_config(config_path: Path | None) -> dict[str, Any]:
	"""
	Load configuration for the daemon.

	Args:
	        config_path: Path to the configuration file

	Returns:
	        Configuration data dictionary

	"""
	config_loader = ConfigLoader(str(config_path) if config_path else None)
	return config_loader.config


def start_daemon_process(config_path: Path | None, foreground: bool, timeout: int) -> int:
	"""
	Start the daemon process.

	Args:
	        config_path: Path to configuration file
	        foreground: Whether to run in foreground
	        timeout: Timeout for startup operation

	Returns:
	        Exit code (0 for success)

	"""
	with progress_indicator("Starting daemon", style="progress", total=1) as advance:
		try:
			load_config(config_path)
			result = start_daemon(config_path=config_path, foreground=foreground, timeout=timeout)
			advance(1)

			if result == 0:
				console.print("[green]Daemon started successfully")
			else:
				show_error("Failed to start daemon")

			return result
		except (FileNotFoundError, PermissionError, OSError) as e:
			advance(1)
			show_error(f"File system error when starting daemon: {e!s}")
			return 1
		except ValueError as e:
			advance(1)
			show_error(f"Configuration error when starting daemon: {e!s}")
			return 1


def stop_daemon_process(config_path: Path | None, timeout: int) -> int:
	"""
	Stop the daemon process.

	Args:
	        config_path: Path to configuration file
	        timeout: Timeout for shutdown operation

	Returns:
	        Exit code (0 for success)

	"""
	with progress_indicator("Stopping daemon", style="progress", total=1) as advance:
		try:
			load_config(config_path)
			result = stop_daemon(config_path=config_path, timeout=timeout)
			advance(1)

			if result == 0:
				console.print("[green]Daemon stopped successfully")
			else:
				show_error("Failed to stop daemon")

			return result
		except (FileNotFoundError, PermissionError, OSError) as e:
			advance(1)
			show_error(f"File system error when stopping daemon: {e!s}")
			return 1
		except ValueError as e:
			advance(1)
			show_error(f"Configuration error when stopping daemon: {e!s}")
			return 1


def get_daemon_status(config_path: Path | None, detailed: bool, output_json: bool) -> int:
	"""
	Get and display the daemon status.

	Args:
	        config_path: Path to configuration file
	        detailed: Whether to show detailed information
	        output_json: Whether to output as JSON

	Returns:
	        Exit code (0 for success)

	"""
	with progress_indicator("Fetching daemon status", style="progress", total=1) as advance:
		try:
			load_config(config_path)
			advance(1)
			console.print()
			return show_daemon_status(config_path=config_path, detailed=detailed, output_json=output_json)
		except (FileNotFoundError, PermissionError, OSError) as e:
			advance(1)
			console.print()
			show_error(f"File system error when fetching daemon status: {e!s}")
			return 1
		except ValueError as e:
			advance(1)
			console.print()
			show_error(f"Configuration error when fetching daemon status: {e!s}")
			return 1


def fetch_jobs_information(
	job_id: str | None, config_path: Path | None, status_filter: str | None, output_json: bool
) -> int:
	"""
	Fetch and display jobs information.

	Args:
	        job_id: ID of the job to show details for
	        config_path: Path to configuration file
	        status_filter: Filter jobs by status
	        output_json: Whether to output as JSON

	Returns:
	        Exit code (0 for success)

	"""
	with progress_indicator("Fetching jobs", style="progress", total=1) as advance:
		try:
			load_config(config_path)
			result = list_or_show_jobs(
				job_id=job_id, config_path=config_path, status_filter=status_filter, output_json=output_json
			)
			advance(1)
			return result
		except (FileNotFoundError, PermissionError, OSError) as e:
			advance(1)
			show_error(f"File system error when fetching jobs information: {e!s}")
			return 1
		except ValueError as e:
			advance(1)
			show_error(f"Configuration error when fetching jobs information: {e!s}")
			return 1


def restart_daemon_process(config_path: Path | None, timeout: int) -> int:
	"""
	Restart the daemon process.

	Args:
	        config_path: Path to configuration file
	        timeout: Timeout for restart operation

	Returns:
	        Exit code (0 for success)

	"""
	with progress_indicator("Restarting daemon", style="progress", total=2) as advance:
		try:
			load_config(config_path)

			# Stop the daemon
			stop_result = stop_daemon(config_path=config_path, timeout=timeout)
			advance(1)

			if stop_result != 0:
				show_error("Failed to stop daemon during restart")
				return stop_result

			# Start the daemon
			start_result = start_daemon(config_path=config_path, foreground=False, timeout=timeout)
			advance(1)

			if start_result == 0:
				console.print("[green]Daemon restarted successfully")
			else:
				show_error("Failed to start daemon during restart")

			return start_result
		except (FileNotFoundError, PermissionError, OSError) as e:
			advance(2)  # Complete the progress
			show_error(f"File system error when restarting daemon: {e!s}")
			return 1
		except ValueError as e:
			advance(2)  # Complete the progress
			show_error(f"Configuration error when restarting daemon: {e!s}")
			return 1


def fetch_daemon_logs(config_path: Path | None, lines: int, follow: bool = False) -> int:
	"""
	Fetch and display daemon logs.

	Args:
	        config_path: Path to configuration file
	        lines: Number of lines to show
	        follow: Whether to follow log output

	Returns:
	        Exit code (0 for success)

	"""
	with progress_indicator("Fetching daemon logs", style="spinner"):
		try:
			load_config(config_path)
			return show_daemon_logs(config_path=config_path, tail=lines, follow=follow)
		except (FileNotFoundError, PermissionError, OSError) as e:
			show_error(f"File system error when fetching daemon logs: {e!s}")
			return 1
		except ValueError as e:
			show_error(f"Configuration error when fetching daemon logs: {e!s}")
			return 1


@daemon_cmd.command(name="start", help="Start the daemon")
def daemon_start(
	config: Path | None = CONFIG_OPTION,
	foreground: bool = FOREGROUND_OPTION,
	timeout: int = TIMEOUT_OPTION,
	is_verbose: bool = typer.Option(default=False, help="Enable verbose logging"),
) -> int:
	"""Start the CodeMap daemon."""
	setup_logging(is_verbose=is_verbose)
	return start_daemon_process(config_path=config, foreground=foreground, timeout=timeout)


@daemon_cmd.command(name="stop", help="Stop the daemon")
def daemon_stop(
	config: Path | None = CONFIG_OPTION,
	timeout: int = TIMEOUT_OPTION,
	is_verbose: bool = typer.Option(default=False, help="Enable verbose logging"),
) -> int:
	"""Stop the CodeMap daemon."""
	setup_logging(is_verbose=is_verbose)
	return stop_daemon_process(config_path=config, timeout=timeout)


@daemon_cmd.command(name="status", help="Show daemon status")
def daemon_status(
	config: Path | None = CONFIG_OPTION,
	detailed: bool = DETAILED_OPTION,
	output_json: bool = JSON_OPTION,
	is_verbose: bool = typer.Option(default=False, help="Enable verbose logging"),
) -> int:
	"""Show the daemon status."""
	setup_logging(is_verbose=is_verbose)
	return get_daemon_status(config_path=config, detailed=detailed, output_json=output_json)


@daemon_cmd.command(name="jobs", help="List and manage jobs")
def daemon_jobs(
	job_id: str | None = typer.Argument(default=None, help="Show specific job details"),
	config: Path | None = CONFIG_OPTION,
	status_filter: str | None = STATUS_FILTER_OPTION,
	output_json: bool = JSON_OPTION,
	is_verbose: bool = typer.Option(default=False, help="Enable verbose logging"),
) -> int:
	"""List or get info about processing jobs."""
	setup_logging(is_verbose=is_verbose)
	return fetch_jobs_information(
		job_id=job_id, config_path=config, status_filter=status_filter, output_json=output_json
	)


@daemon_cmd.command(name="restart", help="Restart the daemon")
def daemon_restart(
	config: Path | None = CONFIG_OPTION,
	timeout: int = TIMEOUT_OPTION,
	is_verbose: bool = typer.Option(default=False, help="Enable verbose logging"),
) -> int:
	"""Restart the CodeMap daemon."""
	setup_logging(is_verbose=is_verbose)
	return restart_daemon_process(config_path=config, timeout=timeout)


@daemon_cmd.command(name="logs", help="View daemon logs")
def daemon_logs(
	config: Path | None = CONFIG_OPTION,
	tail: int = typer.Option(default=10, help="Number of lines to show"),
	follow: bool = typer.Option(default=False, help="Follow log output"),
	is_verbose: bool = typer.Option(default=False, help="Enable verbose logging"),
) -> int:
	"""View the daemon logs."""
	setup_logging(is_verbose=is_verbose)
	return fetch_daemon_logs(config_path=config, lines=tail, follow=follow)
