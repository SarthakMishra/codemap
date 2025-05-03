"""CLI command for manually synchronizing the CodeMap databases."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from codemap.processor.pipeline import ProcessingPipeline
from codemap.utils.cli_utils import progress_indicator, setup_logging
from codemap.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


def sync_command(
	ctx: typer.Context,
	verbose: Annotated[
		bool,
		typer.Option(
			"-v",
			"--verbose",
			help="Enable verbose logging.",
			show_default=False,
		),
	] = False,
) -> None:
	"""
	Manually synchronize the CodeMap databases with the Git repository state.

	This command compares the current Git state with the stored state in
	the Kuzu graph database and updates the graph accordingly by adding new
	files, updating changed files, and removing deleted files. Vector
	embeddings are updated as part of this process.

	"""
	setup_logging(is_verbose=verbose)

	if ctx.invoked_subcommand is not None:
		return  # Only run sync logic if no subcommand is called

	logger.info("Starting manual database synchronization...")

	try:
		# Create pipeline without progress tracking initially
		pipeline = ProcessingPipeline(
			repo_path=None,  # Let pipeline determine repo_path
			sync_on_init=False,  # Don't sync yet
			progress=None,
			task_id=None,
			config_loader=ConfigLoader.get_instance(repo_root=Path.cwd()),
		)

		# Check if pipeline initialization succeeded
		if not pipeline.kuzu_manager:
			logger.error("Pipeline initialization failed.")
			raise typer.Exit(1)

		typer.echo("Pipeline initialized. Starting synchronization...")

		# Use the progress_indicator from utils
		with progress_indicator(
			message="Synchronizing databases...",
			style="progress",
			total=100,  # 100% progress
			transient=False,
		) as advance:
			# Track progress position with a closure
			current_position = 0

			# Create a simple callback function for our sync_databases_simple method
			def update_progress(percent: int) -> None:
				nonlocal current_position
				# Calculate the amount to advance
				amount_to_advance = percent - current_position
				if amount_to_advance > 0:
					advance(amount_to_advance)
					current_position = percent

			# Run sync with our progress callback
			pipeline.sync_databases_simple(progress_callback=update_progress)

	except Exception as e:
		logger.exception("An unexpected error occurred during synchronization.")
		typer.secho(f"Error during synchronization: {e}", fg=typer.colors.RED)
		# Chain the original exception
		raise typer.Exit(1) from e

	logger.info("Synchronization command finished.")
	typer.secho("Synchronization complete.", fg=typer.colors.GREEN)
