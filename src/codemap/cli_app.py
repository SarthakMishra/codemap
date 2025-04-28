"""Command-line interface for the codemap tool."""

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

from .cli.commit_cmd import commit_command
from .cli.daemon_cmd import daemon_cmd
from .cli.generate_cmd import generate_command
from .cli.init_cmd import init_command
from .cli.pr_cmd import pr_command

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the main CLI app
app = typer.Typer(
	help="CodeMap - Generate optimized markdown documentation from your codebase.",
)

# Register commands
app.command(name="init")(init_command)
app.command(name="generate")(generate_command)
app.command(name="commit")(commit_command)
app.command(name="pr")(pr_command)
app.add_typer(daemon_cmd)


def main() -> int:
	"""Run the CLI application."""
	return app()


if __name__ == "__main__":
	sys.exit(main())
