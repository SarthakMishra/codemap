"""CLI command for indexing repositories."""

import logging
from pathlib import Path
from typing import Annotated, cast

import typer

from codemap.utils.cli_utils import progress_indicator

logger = logging.getLogger(__name__)

# --- Command Argument Annotations (Keep these lightweight) ---

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

# --- Registration Function ---


def register_command(app: typer.Typer) -> None:
	"""Register the index command with the CLI app."""

	@app.command(name="index")
	def index_command(
		path: PathArg = Path(),
		sync: SyncOpt = True,
		watch: WatchOpt = False,
	) -> None:
		"""
		Index the repository: Process files, generate embeddings, and store in the vector database.

		Optionally, use --sync (default) to synchronize with the Git state on startup,
		and --watch (-w) to keep running and sync automatically on file changes.
		"""
		# Defer heavy imports and logic to the implementation function
		_index_command_impl(
			path=path,
			sync=sync,
			watch=watch,
		)


# --- Implementation Function (Heavy imports deferred here) ---


def _index_command_impl(
	path: Path,
	sync: bool,
	watch: bool,
) -> None:
	"""Actual implementation of the index command."""
	# --- Heavy Imports ---
	with progress_indicator("Initializing index command..."):
		import asyncio

		from rich.console import Console

		from codemap.config import ConfigLoader
		from codemap.processor.pipeline import ProcessingPipeline
		from codemap.utils.cli_utils import exit_with_error, handle_keyboard_interrupt

	console = Console()  # Initialize Console here

	# --- Helper async function (nested inside implementation) ---
	async def _index_repo_async(
		target_path: Path,  # Renamed from path to avoid conflict
		sync_flag: bool,  # Renamed from sync
		watch_flag: bool,  # Renamed from watch
		config_loader: ConfigLoader,
	) -> None:
		"""Asynchronous part of the index command logic."""
		pipeline: ProcessingPipeline | None = None

		try:
			# --- Initialize Pipeline --- #
			with progress_indicator("Initializing indexing pipeline..."):
				try:
					pipeline = ProcessingPipeline(config_loader=config_loader)
					logger.info(f"Pipeline initialized for {target_path}")
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

			# async_init handles the initial sync if sync_flag is True
			with progress_indicator("Initializing vector database..."):
				pipeline = cast("ProcessingPipeline", pipeline)
				await pipeline.async_init(sync_on_init=sync_flag)
				logger.info("Vector database initialized")

			# --- Watch Mode --- #
			if watch_flag:
				logger.info("Watch mode enabled. Initializing file watcher...")
				# Get debounce delay from config_loader
				watcher_config = config_loader.get.processor.watcher
				debounce_delay = float(watcher_config.debounce_delay)

				with progress_indicator("Starting file watcher..."):
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
			elif sync_flag:
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

	# --- Main logic of _index_command_impl ---

	try:
		target_path = path.resolve()

		# Load config directly instead of getting from context
		config_loader = ConfigLoader.get_instance()

		# Run the indexing operation using the nested async helper
		asyncio.run(_index_repo_async(target_path, sync, watch, config_loader))
	except KeyboardInterrupt:
		handle_keyboard_interrupt()
	except RuntimeError as e:
		# Handle specific runtime errors like event loop issues
		exit_with_error(f"Runtime error: {e}", exception=e)
