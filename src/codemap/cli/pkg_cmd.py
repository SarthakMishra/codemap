"""Package management commands for CodeMap."""

from __future__ import annotations

import logging
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from codemap.utils.cli_utils import console, setup_logging
from codemap.utils.package_utils import (
	get_current_version,
	get_system_info,
	is_update_available,
	uninstall_package,
	update_package,
)

logger = logging.getLogger(__name__)

# Create package command group
pkg_cmd = typer.Typer(
	help="Package management commands for CodeMap.",
	name="pkg",
)


@pkg_cmd.command(name="update", help="Update CodeMap to the latest version")
def update_command(
	check_only: Annotated[
		bool,
		typer.Option(
			"--check-only",
			help="Only check for updates without updating",
		),
	] = False,
	is_verbose: Annotated[
		bool,
		typer.Option(
			"--verbose",
			"-v",
			help="Enable verbose logging",
		),
	] = False,
) -> int:
	"""Update CodeMap to the latest version."""
	setup_logging(is_verbose=is_verbose)

	current_version = get_current_version()
	console.print(f"[bold]Current version:[/] v{current_version}")

	update_available, latest_version, release_url = is_update_available()

	if update_available and latest_version:
		console.print(f"[green]Update available:[/] v{latest_version}")
		if release_url:
			console.print(f"[blue]Release notes:[/] {release_url}")

		if check_only:
			console.print("[yellow]Run 'codemap pkg update' to update to the latest version.[/]")
			return 0

		# Confirm update
		if typer.confirm("Do you want to update now?", default=True):
			if update_package():
				console.print(f"[green]Successfully updated to v{latest_version}[/]")
				return 0
			console.print("[red]Update failed. Please try again or update manually with pip.[/]")
			return 1
		console.print("[yellow]Update cancelled.[/]")
		return 0
	console.print("[green]You are already using the latest version.[/]")
	return 0


@pkg_cmd.command(name="version", help="Show version information")
def version_command(
	check_updates: Annotated[
		bool,
		typer.Option(
			"--check-updates",
			help="Check for updates",
		),
	] = True,
) -> None:
	"""Show version information and optionally check for updates."""
	current_version = get_current_version()

	# Create a table for version info
	table = Table(show_header=False, box=None)
	table.add_row("[bold]CodeMap version:[/]", f"v{current_version}")

	# Add system info
	info = get_system_info()
	table.add_row("[bold]Python version:[/]", info["python_version"])
	table.add_row("[bold]Platform:[/]", info["platform"])

	if "pip_version" in info:
		table.add_row("[bold]Pip version:[/]", info["pip_version"])

	# Show in a panel
	console.print(Panel(table, title="CodeMap Version Information", border_style="blue"))

	# Check for updates if requested
	if check_updates:
		update_available, latest_version, release_url = is_update_available()
		if update_available and latest_version:
			console.print(f"[yellow]Update available:[/] v{latest_version}")
			console.print("[yellow]Run 'codemap pkg update' to update to the latest version.[/]")
			if release_url:
				console.print(f"[blue]Release notes:[/] {release_url}")
		else:
			console.print("[green]You are using the latest version.[/]")


@pkg_cmd.command(name="uninstall", help="Uninstall CodeMap")
def uninstall_command() -> int:
	"""Uninstall CodeMap from your system."""
	console.print("[yellow]This will uninstall CodeMap from your system.[/]")
	console.print("[yellow]Your configuration files will not be removed.[/]")

	# Confirm uninstallation
	if typer.confirm("Are you sure you want to uninstall CodeMap?", default=False):
		if uninstall_package():
			console.print("[green]CodeMap has been uninstalled successfully.[/]")
			return 0
		console.print("[red]Uninstallation failed. Please try uninstalling manually with pip.[/]")
		return 1
	console.print("[yellow]Uninstallation cancelled.[/]")
	return 0


@pkg_cmd.command(name="info", help="Show system information for debugging")
def info_command() -> None:
	"""Display system information for debugging purposes."""
	info = get_system_info()

	# Create a table for system info
	table = Table(show_header=False, box=None)

	for key, value in info.items():
		# Format key with underscores to spaces and title case
		formatted_key = key.replace("_", " ").title()
		table.add_row(f"[bold]{formatted_key}:[/]", str(value))

	# Show in a panel
	console.print(Panel(table, title="CodeMap System Information", border_style="blue"))
