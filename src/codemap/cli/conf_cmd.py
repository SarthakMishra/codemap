"""Configuration management commands."""

from __future__ import annotations

import logging
from typing import Annotated

import typer

logger = logging.getLogger(__name__)

# --- Command Argument Annotations (Keep these lightweight) ---

ForceOpt = Annotated[
	bool,
	typer.Option("--force", "-f", help="Overwrite existing configuration file."),
]

PathOpt = Annotated[
	str | None,
	typer.Option("--path", "-p", help="Path to the configuration file. Defaults to .codemap.yml in the repo root."),
]

# --- Registration Function ---


def register_command(app: typer.Typer) -> None:
	"""Register the configuration commands with the main app."""

	@app.command("conf")
	def conf_command(
		force: ForceOpt = False,
	) -> None:
		"""Create a default .codemap.yml configuration file in the project root."""
		# Defer heavy imports and logic to the implementation function
		_conf_command_impl(force=force)


# --- Implementation Function (Heavy imports deferred here) ---


def _conf_command_impl(
	force: bool,
) -> None:
	"""Actual implementation of the config command."""
	# --- Heavy Imports ---
	from rich.console import Console

	from codemap.config.default_config_template import DEFAULT_CONFIG_TEMPLATE
	from codemap.git.utils import GitRepoContext

	console = Console()

	project_root = GitRepoContext().get_repo_root()
	config_file_path = project_root / ".codemap.yml"

	if config_file_path.exists() and not force:
		console.print(
			f"[yellow]Configuration file '{config_file_path}' already exists. Use --force to overwrite.[/yellow]"
		)
		raise typer.Exit(code=1)

	try:
		with config_file_path.open("w", encoding="utf-8") as f:
			f.write(DEFAULT_CONFIG_TEMPLATE)
		console.print(f"[green]Created configuration file: {config_file_path.resolve()}[/green]")
		console.print("[yellow]Review and edit the file to customize your CodeMap settings.[/yellow]")
		console.print("[blue]Run 'codemap conf check' to validate your new configuration.[/blue]")
	except OSError as e:
		logger.exception(f"Failed to write configuration file to {config_file_path}")
		console.print(f"[red]Error writing configuration file: {e}[/red]")
		raise typer.Exit(code=1) from e
