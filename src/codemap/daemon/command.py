"""Command functions for the daemon functionality."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from codemap.daemon.client import HTTP_NOT_FOUND, DaemonClient
from codemap.daemon.service import CodeMapDaemon

# Set up logger
logger = logging.getLogger(__name__)

# Set up console
console = Console()


def wait_for_daemon_api(timeout: int = 30) -> bool:
	"""
	Wait for the daemon API to become available.

	Args:
	    timeout: Maximum number of seconds to wait

	Returns:
	    bool: True if the API became available, False if timeout reached

	"""
	client = DaemonClient()
	start_time = time.time()

	with Progress(
		SpinnerColumn(),
		TextColumn("[bold blue]Waiting for daemon API to become available..."),
		transient=True,
	) as progress:
		progress.add_task("wait", total=None)

		while time.time() - start_time < timeout:
			try:
				client.check_status()
				return True
			except (RuntimeError, ConnectionError):
				time.sleep(0.5)
				continue

	return False


def gracefully_stop_daemon(daemon: CodeMapDaemon, timeout: int = 30) -> bool:
	"""
	Gracefully stop the daemon with progress indicator and timeout.

	Args:
	    daemon: The CodeMapDaemon instance
	    timeout: Maximum number of seconds to wait for shutdown

	Returns:
	    bool: True if daemon stopped successfully, False otherwise

	"""
	if not daemon.is_running():
		console.print("[yellow]Daemon is not running")
		return True

	pid = daemon.get_pid()

	with Progress(
		SpinnerColumn(),
		TextColumn(f"[bold blue]Stopping daemon (PID: {pid})..."),
		transient=True,
	) as progress:
		progress.add_task("stop", total=None)

		if daemon.stop():
			# Wait for the process to actually terminate
			start_time = time.time()
			while time.time() - start_time < timeout:
				if not daemon.is_running():
					return True
				time.sleep(0.5)

	# If we get here, either stop() failed or timeout reached
	if daemon.is_running():
		console.print("[red]Timeout reached while stopping daemon")
		return False
	return True


def start_daemon(config_path: Path | None = None, foreground: bool = False, timeout: int = 30) -> int:
	"""
	Start the CodeMap daemon.

	Args:
	    config_path: Path to configuration file
	    foreground: Run in foreground mode (for debugging)
	    timeout: Timeout in seconds for daemon startup

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	daemon = CodeMapDaemon(config_path=config_path)

	if daemon.is_running():
		pid = daemon.get_pid()
		console.print(f"[yellow]Daemon already running (PID: {pid})")
		return 0

	if foreground:
		console.print(Panel("[bold]Starting daemon in foreground mode", style="blue"))
		try:
			daemon.run_foreground()
		except KeyboardInterrupt:
			console.print("\n[yellow]Stopping daemon due to keyboard interrupt")
			return 0
	else:
		with console.status("[bold blue]Starting daemon...", spinner="dots"):
			if not daemon.start():
				console.print("[red]Failed to start daemon")
				return 1

		# Wait for the API to become available
		if not wait_for_daemon_api(timeout):
			console.print("[red]Daemon started but API is not responsive")
			console.print("[yellow]Check the logs for details")
			return 1

		# If we get here, the daemon is running and the API is responsive
		pid = daemon.get_pid()
		console.print(f"[green]Daemon started successfully (PID: {pid})")

		# Show connection details
		client = DaemonClient()
		console.print(f"[blue]API available at: [bold]{client.base_url}[/bold]")

	return 0


def stop_daemon(config_path: Path | None = None, timeout: int = 30) -> int:
	"""
	Stop the CodeMap daemon.

	Args:
	    config_path: Path to configuration file
	    timeout: Timeout in seconds for daemon shutdown

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	daemon = CodeMapDaemon(config_path=config_path)

	if not daemon.is_running():
		console.print("[yellow]Daemon is not running")
		return 0

	if gracefully_stop_daemon(daemon, timeout):
		console.print("[green]Daemon stopped successfully")
		return 0

	console.print("[red]Failed to stop daemon")
	return 1


def show_daemon_status(config_path: Path | None = None, detailed: bool = False, output_json: bool = False) -> int:
	"""
	Show the daemon status.

	Args:
	    config_path: Path to configuration file
	    detailed: Show detailed status information
	    output_json: Output status as JSON

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	daemon = CodeMapDaemon(config_path=config_path)
	status_data = daemon.status()

	if output_json:
		console.print(json.dumps(status_data, indent=2))
		return 0

	# Print status in a nice format
	status = status_data["status"]
	if status == "running":
		pid = status_data["pid"]
		console.print(Panel(f"[bold green]Daemon is running[/bold green]\nPID: {pid}", title="Daemon Status"))

		# If detailed and running, query the API for more info
		if detailed:
			try:
				client = DaemonClient()
				api_status = client.check_status()

				# Print job statistics
				console.print("\n[bold blue]Job Statistics:[/bold blue]")
				jobs = api_status.get("jobs", {})
				console.print(f"  Active: {jobs.get('active', 0)}")
				console.print(f"  Completed: {jobs.get('completed', 0)}")
				console.print(f"  Failed: {jobs.get('failed', 0)}")

				# Print API connection details
				console.print("\n[bold blue]API Details:[/bold blue]")
				console.print(f"  URL: {client.base_url}")
			except (RuntimeError, ConnectionError) as e:
				console.print(f"[red]Could not connect to API server: {e}")
	else:
		console.print(Panel("[bold red]Daemon is not running", title="Daemon Status"))

	return 0


def list_or_show_jobs(
	job_id: str | None = None,
	config_path: Path | None = None,
	status_filter: str | None = None,
	output_json: bool = False,
) -> int:
	"""
	List or get info about processing jobs.

	Args:
	    job_id: ID of the job to show details for (optional)
	    config_path: Path to configuration file
	    status_filter: Filter jobs by status
	    output_json: Output as JSON

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	# Check if daemon is running
	daemon = CodeMapDaemon(config_path=config_path)
	if not daemon.is_running():
		console.print("[red]Daemon is not running")
		return 1

	client = DaemonClient()

	try:
		if job_id:
			# Get specific job details
			try:
				job = client.get_job_status(job_id)

				if output_json:
					console.print(json.dumps(job, indent=2))
					return 0

				# Print job details in a nice format
				created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.get("created_at", 0)))

				completed_msg = ""
				if job.get("completed_at"):
					completed_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.get("completed_at", 0)))
					completed_msg = f"[bold]Completed:[/bold] {completed_time}\n"

				error_msg = ""
				if job.get("error"):
					error_msg = f"[bold]Error:[/bold] {job.get('error')}"

				panel_content = (
					f"[bold]Job ID:[/bold] {job.get('id')}\n"
					f"[bold]Status:[/bold] {job.get('status')}\n"
					f"[bold]Created:[/bold] {created_time}\n"
					f"{completed_msg}"
					f"{error_msg}"
				)

				console.print(Panel(panel_content, title=f"Job Details: {job_id}"))
			except RuntimeError as e:
				if str(HTTP_NOT_FOUND) in str(e):
					console.print(f"[red]Job {job_id} not found")
				else:
					console.print(f"[red]Error: {e}")
				return 1
		else:
			# List all jobs
			try:
				with console.status("[blue]Fetching jobs...", spinner="dots"):
					jobs = client.list_jobs(status_filter)

				if output_json:
					console.print(json.dumps(jobs, indent=2))
					return 0

				if not jobs:
					console.print("[yellow]No jobs found")
					return 0

				# Print job list in a nice table
				from rich.table import Table

				table = Table(title=f"Jobs ({len(jobs)})")
				table.add_column("ID", style="dim")
				table.add_column("Status", style="green")
				table.add_column("Created", style="blue")

				for job in jobs:
					# Format the creation time
					created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.get("created_at", 0)))
					# Set status style based on the status value
					status = job.get("status", "N/A")
					status_style = "[green]" if status == "completed" else "[yellow]" if status == "active" else "[red]"

					table.add_row(job.get("id", "N/A"), f"{status_style}{status}[/]", created)

				console.print(table)
			except RuntimeError as e:
				console.print(f"[red]Error fetching jobs: {e}")
				return 1

	except (RuntimeError, ConnectionError) as e:
		console.print(f"[red]Error connecting to daemon: {e}")
		return 1

	return 0


def restart_daemon(config_path: Path | None = None, timeout: int = 30) -> int:
	"""
	Restart the CodeMap daemon.

	Args:
	    config_path: Path to configuration file
	    timeout: Timeout in seconds for daemon shutdown/startup

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	daemon = CodeMapDaemon(config_path=config_path)

	# If not running, just start
	if not daemon.is_running():
		console.print("[yellow]Daemon is not running, starting...")
		return start_daemon(config_path=config_path, foreground=False, timeout=timeout)

	# First, try to stop the daemon gracefully
	console.print("[bold blue]Restarting daemon...[/bold blue]")
	if not gracefully_stop_daemon(daemon, timeout):
		console.print("[red]Failed to stop daemon for restart")
		return 1

	# Give it a moment to fully shut down
	time.sleep(1)

	# Now start it again
	with console.status("[bold blue]Starting daemon...", spinner="dots"):
		if not daemon.start():
			console.print("[red]Failed to start daemon after stopping")
			return 1

	# Wait for the API to become available
	if not wait_for_daemon_api(timeout):
		console.print("[red]Daemon started but API is not responsive")
		console.print("[yellow]Check the logs for details")
		return 1

	# Success!
	pid = daemon.get_pid()
	console.print(f"[green]Daemon restarted successfully (PID: {pid})")
	return 0


def show_daemon_logs(config_path: Path | None = None, tail: int = 10, follow: bool = False) -> int:
	"""
	View the daemon logs.

	Args:
	    config_path: Path to configuration file
	    tail: Number of log lines to show
	    follow: Whether to follow the log output

	Returns:
	    int: Exit code (0 for success, non-zero for failure)

	"""
	# Load daemon to get config
	daemon = CodeMapDaemon(config_path=config_path)
	config_data = daemon.config

	# Get log file path from daemon config
	log_file = config_data.get("server", {}).get("log_file", "~/.codemap/daemon.log")
	log_file = Path(log_file).expanduser().resolve()

	if not log_file.exists():
		console.print(f"[red]Log file not found: {log_file}")
		return 1

	console.print(f"[blue]Showing logs from: {log_file}")

	if follow:
		# Use the system's 'tail' command for following logs
		cmd = ["tail", f"-n{tail}", "-f", str(log_file)]
		console.print("[yellow]Press Ctrl+C to stop following logs[/yellow]")

		try:
			# Command is constructed with fixed parameters and a validated file path
			# Security risk is minimal since shell=False and we have validated log_file
			subprocess.run(cmd, shell=False, check=True)  # noqa: S603
		except KeyboardInterrupt:
			return 0
		except subprocess.CalledProcessError as e:
			console.print(f"[red]Error running tail command: {e}")
			return 1
	else:
		# Just read the last N lines
		with Path(log_file).open(encoding="utf-8") as f:
			# Read all lines and get the tail
			lines = f.readlines()
			tail_lines = lines[-tail:] if len(lines) > tail else lines

			for line in tail_lines:
				console.print(line.rstrip())

	return 0
