"""Command for generating conventional commit messages from Git diffs."""

import logging
from pathlib import Path
from typing import Annotated

import asyncer
import typer

logger = logging.getLogger(__name__)

# --- Command Argument Annotations (Keep these lightweight) ---

PathArg = Annotated[
	Path | None,
	typer.Argument(
		help="Path to repository or file to commit",
		exists=True,
	),
]

NonInteractiveFlag = Annotated[bool, typer.Option("--non-interactive", "-y", help="Run in non-interactive mode")]

BypassHooksFlag = Annotated[
	bool, typer.Option("--bypass-hooks", "--no-verify", help="Bypass git hooks with --no-verify")
]

# --- Registration Function ---


def register_command(app: typer.Typer) -> None:
	"""Register the commit commands with the CLI app."""

	@app.command(name="commit")
	@asyncer.runnify
	async def semantic_commit_command(
		path: PathArg = None,
		non_interactive: NonInteractiveFlag = False,
		bypass_hooks: BypassHooksFlag = False,
		pathspecs: list[str] | None = None,
	) -> None:
		"""
		Generate semantic commits by grouping related changes.

		This command analyzes your changes, groups them semantically, and
		creates multiple focused commits with AI-generated messages.

		"""
		# Defer heavy imports and logic to the implementation function
		await _semantic_commit_command_impl(
			path=path,
			non_interactive=non_interactive,
			bypass_hooks=bypass_hooks,
			pathspecs=pathspecs,
		)


# --- Implementation Function (Heavy imports deferred here) ---


async def _semantic_commit_command_impl(
	path: Path | None,
	non_interactive: bool,
	bypass_hooks: bool,
	pathspecs: list[str] | None = None,
) -> None:
	"""Actual implementation of the semantic commit command."""
	# --- Heavy Imports ---

	from codemap.config import ConfigLoader
	from codemap.git.commit_generator.command import SemanticCommitCommand
	from codemap.git.utils import (
		validate_repo_path,
	)
	from codemap.utils.cli_utils import exit_with_error, handle_keyboard_interrupt

	# --- Setup & Logic ---

	try:
		# Validate repo path (optional, defaults to cwd if None)
		repo_path = validate_repo_path(path) if path else Path.cwd()

		# Use get_instance to avoid creating multiple ConfigLoader instances
		config = ConfigLoader.get_instance(repo_root=repo_path)

		# Determine parameters (CLI > Config > Default)
		is_non_interactive = non_interactive or config.get.commit.is_non_interactive
		should_bypass_hooks = bypass_hooks or config.get.commit.bypass_hooks

		# Create the semantic commit command
		semantic_workflow = SemanticCommitCommand(
			path=repo_path,
			bypass_hooks=should_bypass_hooks,
		)

		# Run the semantic commit process
		success = await semantic_workflow.run(
			interactive=not is_non_interactive,
			pathspecs=pathspecs,
		)

		if not success:
			exit_with_error("Semantic commit process failed.")

	except KeyboardInterrupt:
		handle_keyboard_interrupt()
	except Exception as e:
		logger.exception("An unexpected error occurred during the semantic commit command.")
		exit_with_error(f"An unexpected error occurred: {e}", exception=e)
