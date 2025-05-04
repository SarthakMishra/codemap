"""
CLI command for indexing repositories.

Contains the logic for the 'codemap index' command, including
initialization, synchronization, and the optional file watching mode.

"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console

from codemap.processor.pipeline import ProcessingPipeline
from codemap.utils.cli_utils import exit_with_error, loading_spinner, setup_logging
from codemap.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)
console = Console()

# Command line argument annotations
PathArg = Annotated[
	Path,
	typer.Argument(
		help="Path to the repository root directory.",
		exists=True,
		file_okay=False,
		dir_okay=True,
		readable=True,
		resolve_path=True,
	),
]

SyncOpt = Annotated[
	bool,
	typer.Option(
		"--sync/--no-sync",
		help="Synchronize the vector database with the current Git state on startup.",
	),
]

WatchOpt = Annotated[
	bool,
	typer.Option(
		"--watch",
		"-w",
		help="Keep running and watch for file changes, automatically syncing the index.",
	),
]

LogLevelOpt = Annotated[
	str,
	typer.Option(
		"--log-level",
		"-L",
		help="Set the logging level (e.g., DEBUG, INFO, WARNING).",
	),
]

VerboseFlag = Annotated[
	bool,
	typer.Option(
		"--verbose",
		"-v",
		help="Enable verbose logging",
	),
]

ConfigOpt = Annotated[
	Path | None,
	typer.Option(
		"--config",
		"-c",
		help="Path to config file",
	),
]


# --- Helper async function for the main logic ---
async def _index_repo_async(
	path: Path,
	sync: bool,
	watch: bool,
	config_loader: ConfigLoader,
) -> None:
	"""Asynchronous part of the index command logic."""
	pipeline: ProcessingPipeline | None = None

	try:
		# --- Initialize Pipeline --- #
		with loading_spinner("Initializing indexing pipeline..."):
			try:
				pipeline = ProcessingPipeline(repo_path=path, config_loader=config_loader)
				logger.info(f"Pipeline initialized for {path}")
			except ValueError:
				logger.exception("Initialization failed")
				exit_with_error("Failed to initialize the processing pipeline")
			except Exception:
				logger.exception("Unexpected initialization error")
				exit_with_error("An unexpected error occurred during pipeline initialization")

		# --- Run the pipeline operations --- #
		# Ensure pipeline is not None before using it
		if not pipeline:
			exit_with_error("Pipeline initialization failed unexpectedly")

		# async_init handles the initial sync if sync is True
		with loading_spinner("Initializing vector database..."):
			# Use type narrowing to ensure pipeline is not None
			pipeline = cast("ProcessingPipeline", pipeline)
			await pipeline.async_init(sync_on_init=sync, progress=None, task_id=None)
			logger.info("Vector database initialized")

		# --- Watch Mode --- #
		if watch:
			logger.info("Watch mode enabled. Initializing file watcher...")
			# Get debounce delay from config_loader
			watcher_config = config_loader.get("watcher", {})
			debounce_delay = float(watcher_config.get("debounce_delay", 2.0))

			with loading_spinner("Starting file watcher..."):
				pipeline.initialize_watcher(debounce_delay=debounce_delay)
				await pipeline.start_watcher()

			console.print(f"[green]✓[/green] File watcher started with {debounce_delay}s debounce delay")
			console.print("[blue]Monitoring for file changes...[/blue] (Press Ctrl+C to exit)")

			# Use an Event to wait for cancellation instead of a sleep loop
			cancel_event = asyncio.Event()
			try:
				await cancel_event.wait()  # Wait until cancelled
			except asyncio.CancelledError:
				logger.info("Watch mode cancelled.")
			except KeyboardInterrupt:
				logger.info("Watch mode interrupted by user (Ctrl+C).")
		elif sync:
			console.print("[green]✓[/green] Initial synchronization complete.")
		else:
			console.print("[green]✓[/green] Initialization complete (sync skipped).")

	except Exception:
		logger.exception("An error occurred during the index operation")
		exit_with_error("An error occurred during the indexing operation. Check logs for details.")
	finally:
		# --- Cleanup --- #
		if pipeline and pipeline.is_async_initialized:
			logger.info("Shutting down pipeline...")
			await pipeline.stop()
			logger.info("Pipeline shutdown complete.")


def index_command(
	path: PathArg = Path(),
	sync: SyncOpt = True,
	watch: WatchOpt = False,
	log_level: LogLevelOpt = "INFO",
	is_verbose: VerboseFlag = False,
	config: ConfigOpt = None,
) -> None:
	"""
	Index the repository: Process files, generate embeddings, and store in the vector database.

	Optionally, use --sync (default) to synchronize with the Git state on startup,
	and --watch (-w) to keep running and sync automatically on file changes.
	"""
	# Configure logging based on CLI options
	setup_logging(is_verbose=is_verbose)

	# Override log level if explicitly specified
	if log_level and log_level != "INFO":
		logging.getLogger().setLevel(log_level.upper())
		logger.info(f"Log level set to {log_level.upper()}")

	try:
		target_path = path.resolve()

		# Load config directly instead of getting from context
		config_loader = ConfigLoader(str(config) if config else None)

		# Run the indexing operation
		asyncio.run(_index_repo_async(target_path, sync, watch, config_loader))
	except KeyboardInterrupt:
		console.print("\n[yellow]Operation cancelled by user.[/yellow]")
		sys.exit(1)
	except RuntimeError as e:
		# Handle specific runtime errors like event loop issues
		exit_with_error(f"Runtime error: {e}", exception=e)
