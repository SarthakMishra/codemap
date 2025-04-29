"""Package management commands for CodeMap."""

from __future__ import annotations

import logging
import platform

import typer
from rich.table import Table

from codemap import __version__
from codemap.utils.cli_utils import console, exit_with_error, show_warning
from codemap.utils.config_manager import get_config_manager
from codemap.utils.directory_manager import get_directory_manager
from codemap.utils.log_setup import log_environment_info, setup_logging
from codemap.utils.package_utils import check_for_updates

# Configure logger
logger = logging.getLogger(__name__)

# Create package management subcommand
pkg_cmd = typer.Typer(
	name="pkg",
	help="Package management commands for CodeMap.",
)


@pkg_cmd.command(name="version")
def version_command(
	check_updates: bool = typer.Option(default=False, help="Check for updates"),
) -> None:
	"""Show CodeMap version information."""
	# Get platform and Python version
	py_version = platform.python_version()
	platform_info = platform.platform()

	# Create version info table
	table = Table(title="CodeMap Version Information")
	table.add_column("", style="green")
	table.add_column("", style="white")

	table.add_row("CodeMap version:", f"v{__version__}")
	table.add_row("Python version:", py_version)
	table.add_row("Platform:", platform_info)

	console.print(table)

	# Check for updates if requested
	if check_updates:
		is_latest, latest_version = check_for_updates()
		if is_latest:
			console.print("You are using the latest version.")
		else:
			show_warning(
				f"A new version is available: {latest_version}\nRun 'uv pip install --upgrade codemap' to update."
			)


@pkg_cmd.command(name="dirs")
def dirs_command(
	is_verbose: bool = typer.Option(default=False, help="Show detailed directory information"),
) -> None:
	"""Display CodeMap directory structure information."""
	setup_logging(is_verbose=is_verbose)
	dir_manager = get_directory_manager()
	dir_manager.ensure_directories()

	# Create directory info table
	table = Table(title="CodeMap Directory Information")
	table.add_column("Directory Type", style="green")
	table.add_column("Path", style="white")
	table.add_column("Status", style="cyan")

	# Add main directories
	directories = [
		("User Data", dir_manager.user_data_dir),
		("User Config", dir_manager.user_config_dir),
		("User Cache", dir_manager.user_cache_dir),
		("User Logs", dir_manager.user_log_dir),
	]

	if is_verbose:
		# Add subdirectories in verbose mode
		directories.extend(
			[
				("Config Files", dir_manager.config_dir),
				("Vector Database", dir_manager.vector_db_dir),
				("SQLite Database", dir_manager.sqlite_db_dir),
				("Key-Value Store", dir_manager.kv_db_dir),
				("Daemon Logs", dir_manager.daemon_logs_dir),
				("CLI Logs", dir_manager.cli_logs_dir),
				("Error Logs", dir_manager.error_logs_dir),
				("Model Cache", dir_manager.models_cache_dir),
				("Provider Cache", dir_manager.providers_cache_dir),
			]
		)

	# Check directory status and add to table
	for dir_type, dir_path in directories:
		status = "✅ Exists" if dir_path.exists() else "❌ Missing"
		table.add_row(dir_type, str(dir_path), status)

	console.print(table)

	# Log details about the environment
	log_environment_info()


@pkg_cmd.command(name="clean")
def clean_command(
	cache: bool = typer.Option(default=True, help="Clean cache directories"),
	logs: bool = typer.Option(default=False, help="Clean log files"),
	force: bool = typer.Option(default=False, help="Don't ask for confirmation"),
) -> None:
	"""Clean CodeMap directories and files."""
	dir_manager = get_directory_manager()

	if not force:
		if cache:
			show_warning("This will delete all cached data.")
		if logs:
			show_warning("This will delete all log files.")

		confirm = typer.confirm("Do you want to continue?")
		if not confirm:
			console.print("Operation cancelled.")
			return

	try:
		if cache:
			dir_manager.clear_cache("all")
			console.print("[green]✅ Cleaned cache directories[/green]")

		if logs:
			# Clear log directories
			for log_dir in [dir_manager.daemon_logs_dir, dir_manager.cli_logs_dir, dir_manager.error_logs_dir]:
				if log_dir.exists():
					for log_file in log_dir.glob("*.log*"):
						log_file.unlink()
			console.print("[green]✅ Cleaned log files[/green]")

	except OSError as e:
		exit_with_error(f"Error cleaning directories: {e!s}", exception=e)


@pkg_cmd.command(name="setup")
def setup_command(
	force: bool = typer.Option(default=False, help="Force setup even if directories already exist"),
) -> None:
	"""Set up the CodeMap directory structure."""
	try:
		# Get managers
		dir_manager = get_directory_manager()
		config_manager = get_config_manager()

		# Check if directories already exist
		if not force and all(
			[
				dir_manager.user_data_dir.exists(),
				dir_manager.user_config_dir.exists(),
				dir_manager.user_cache_dir.exists(),
				dir_manager.user_log_dir.exists(),
			]
		):
			warning_message = "CodeMap directory structure already exists.\n"
			warning_message += "Use --force to recreate the directory structure.\n"
			warning_message += "Note: This will not delete any existing data."
			show_warning(warning_message)
			return

		# Create directories
		with console.status("Setting up CodeMap directory structure..."):
			dir_manager.ensure_directories()

			# Initialize empty global config if it doesn't exist
			if not (dir_manager.config_dir / "settings.yml").exists():
				config_manager.update_config("global", {})

		console.print("[green]✅ CodeMap directory structure has been set up.[/green]")
		console.print(f"[green]User data directory: {dir_manager.user_data_dir}[/green]")
		console.print(f"[green]User config directory: {dir_manager.user_config_dir}[/green]")
		console.print(f"[green]User cache directory: {dir_manager.user_cache_dir}[/green]")
		console.print(f"[green]User logs directory: {dir_manager.user_log_dir}[/green]")

	except OSError as e:
		exit_with_error(f"Error setting up directories: {e!s}", exception=e)
