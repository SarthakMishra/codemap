"""Command-line interface package for CodeMap."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer  # type: ignore[import]

# Load environment variables from .env files
try:
	from dotenv import load_dotenv

	# Try to load from .env.local first, then fall back to .env
	env_local = Path(".env.local")
	if env_local.exists():
		load_dotenv(dotenv_path=env_local)
		logging.getLogger(__name__).debug("Loaded environment variables from %s", env_local)
	else:
		env_file = Path(".env")
		if env_file.exists():
			load_dotenv(dotenv_path=env_file)
			logging.getLogger(__name__).debug("Loaded environment variables from %s", env_file)
except ImportError:
	pass  # dotenv not installed, skip loading

from codemap import __version__
from codemap.cli.commit_cmd import commit_command
from codemap.cli.gen_cmd import gen_command
from codemap.cli.init_cmd import init_command
from codemap.cli.pkg_cmd import pkg_cmd
from codemap.cli.pr_cmd import pr_command
from codemap.utils.package_utils import notify_update_available

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the main CLI app
app = typer.Typer(
	help=f"CodeMap - Developer tools powered by AI\n\nVersion: {__version__}",
	no_args_is_help=True,
)

# Register commands
app.command(name="init")(init_command)
app.command(name="gen")(gen_command)
app.command(name="commit")(commit_command)
app.command(name="pr")(pr_command)
app.add_typer(pkg_cmd)


# App callback to check for updates and handle version flag
@app.callback(invoke_without_command=True)
def app_callback(
	ctx: typer.Context,
	version: bool = typer.Option(False, "--version", "-V", help="Show version and exit"),  # noqa: FBT003
) -> None:
	"""Callback that runs before any command."""
	if version:
		"""Show CodeMap version information."""
		from codemap.cli.pkg_cmd import version_command as pkg_version_command

		pkg_version_command(check_updates=True)
		raise typer.Exit

	# If no command was invoked and no version flag, show help
	if ctx.invoked_subcommand is None:
		typer.echo(ctx.get_help())
		raise typer.Exit

	# Check for updates (non-blocking, just notification)
	notify_update_available()


def main() -> int:
	"""Run the CLI application."""
	return app()


if __name__ == "__main__":
	sys.exit(main())
