"""Command-line interface for the daemon functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from codemap.daemon.command import (
	list_or_show_jobs,
	restart_daemon,
	show_daemon_logs,
	show_daemon_status,
	start_daemon,
	stop_daemon,
)

if TYPE_CHECKING:
	from pathlib import Path

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


@daemon_cmd.command(name="start", help="Start the daemon")
def daemon_start(
	config: Path | None = CONFIG_OPTION,
	foreground: bool = FOREGROUND_OPTION,
	timeout: int = TIMEOUT_OPTION,
) -> int:
	"""
	Start the CodeMap daemon.

	Args:
	    config: Path to configuration file
	    foreground: Run in foreground mode (for debugging)
	    timeout: Timeout in seconds for daemon startup

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	return start_daemon(config_path=config, foreground=foreground, timeout=timeout)


@daemon_cmd.command(name="stop", help="Stop the daemon")
def daemon_stop(
	config: Path | None = CONFIG_OPTION,
	timeout: int = TIMEOUT_OPTION,
) -> int:
	"""
	Stop the CodeMap daemon.

	Args:
	    config: Path to configuration file
	    timeout: Timeout in seconds for daemon shutdown

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	return stop_daemon(config_path=config, timeout=timeout)


@daemon_cmd.command(name="status", help="Show daemon status")
def daemon_status(
	config: Path | None = CONFIG_OPTION,
	detailed: bool = DETAILED_OPTION,
	output_json: bool = JSON_OPTION,
) -> int:
	"""
	Show the daemon status.

	Args:
	    config: Path to configuration file
	    detailed: Show detailed status information
	    output_json: Output status as JSON

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	return show_daemon_status(config_path=config, detailed=detailed, output_json=output_json)


@daemon_cmd.command(name="jobs", help="List and manage jobs")
def daemon_jobs(
	job_id: str | None = typer.Argument(default=None, help="Show specific job details"),
	config: Path | None = CONFIG_OPTION,
	status_filter: str | None = STATUS_FILTER_OPTION,
	output_json: bool = JSON_OPTION,
) -> int:
	"""
	List or get info about processing jobs.

	Args:
	    job_id: ID of the job to show details for (optional)
	    config: Path to configuration file
	    status_filter: Filter jobs by status
	    output_json: Output as JSON

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	return list_or_show_jobs(job_id=job_id, config_path=config, status_filter=status_filter, output_json=output_json)


@daemon_cmd.command(name="restart", help="Restart the daemon")
def daemon_restart(
	config: Path | None = CONFIG_OPTION,
	timeout: int = TIMEOUT_OPTION,
) -> int:
	"""
	Restart the CodeMap daemon.

	Args:
	    config: Path to configuration file
	    timeout: Timeout in seconds for daemon shutdown/startup

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	return restart_daemon(config_path=config, timeout=timeout)


@daemon_cmd.command(name="logs", help="View daemon logs")
def daemon_logs(
	config: Path | None = CONFIG_OPTION,
	tail: int = typer.Option(default=10, help="Number of lines to show"),
	follow: bool = typer.Option(default=False, help="Follow log output"),
) -> int:
	"""
	View the daemon logs.

	Args:
	    config: Path to configuration file
	    tail: Number of log lines to show
	    follow: Whether to follow the log output

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	return show_daemon_logs(config_path=config, tail=tail, follow=follow)


def daemon_command() -> int:
	"""
	Entry point for the daemon command in the main CLI app.

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	return daemon_cmd()
