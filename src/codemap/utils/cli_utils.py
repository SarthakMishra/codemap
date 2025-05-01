"""Utility functions for CLI operations in CodeMap."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING, Self

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from codemap.utils.log_setup import display_error_summary, display_warning_summary

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator
	from pathlib import Path

console = Console()
logger = logging.getLogger(__name__)


# Singleton class to track spinner state
class SpinnerState:
	"""Singleton class to track spinner state."""

	_instance = None
	is_active = False

	def __new__(cls) -> Self:
		"""
		Create or return the singleton instance.

		Returns:
		    The singleton instance of SpinnerState

		"""
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance


def setup_logging(*, is_verbose: bool) -> None:
	"""
	Configure logging based on verbosity.

	Args:
	    is_verbose: Whether to enable debug logging.

	"""
	# Set log level based on verbosity
	# In verbose mode, use DEBUG or the level specified in LOG_LEVEL env var
	# In non-verbose mode, only show ERROR and above
	log_level = "DEBUG" if is_verbose else os.environ.get("LOG_LEVEL", "INFO").upper()

	# When not in verbose mode, override to only show errors
	if not is_verbose:
		log_level = "ERROR"

	logging.basicConfig(
		level=log_level,
		format="%(message)s",
		datefmt="[%X]",
		handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_path=True)],
	)

	# Also set specific loggers to ERROR when not in verbose mode
	if not is_verbose:
		# Suppress logs from these packages completely unless in verbose mode
		logging.getLogger("litellm").setLevel(logging.ERROR)
		logging.getLogger("httpx").setLevel(logging.ERROR)
		logging.getLogger("httpcore").setLevel(logging.ERROR)
		logging.getLogger("urllib3").setLevel(logging.ERROR)
		logging.getLogger("requests").setLevel(logging.ERROR)
		logging.getLogger("openai").setLevel(logging.ERROR)
		logging.getLogger("tqdm").setLevel(logging.ERROR)

		# Set codemap loggers to ERROR level
		logging.getLogger("codemap").setLevel(logging.ERROR)

		# Specifically suppress git-related logs
		logging.getLogger("codemap.utils.git_utils").setLevel(logging.ERROR)
		logging.getLogger("codemap.git").setLevel(logging.ERROR)


def create_spinner_progress() -> Progress:
	"""
	Create a spinner progress bar.

	Returns:
	    A Progress instance with a spinner

	"""
	return Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
	)


@contextlib.contextmanager
def loading_spinner(message: str = "Processing...") -> Iterator[None]:
	"""
	Display a loading spinner while executing a task.

	Args:
	    message: Message to display alongside the spinner

	Yields:
	    None

	"""
	# In test environments, don't display a spinner
	if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
		yield
		return

	# Check if a spinner is already active
	spinner_state = SpinnerState()
	if spinner_state.is_active:
		# If there's already an active spinner, don't create a new one
		yield
		return

	# Only use spinner in interactive environments
	try:
		spinner_state.is_active = True
		# Use rich.console.Console.status which is designed for this purpose
		# and provides the spinner animation
		with console.status(message):
			yield
	finally:
		spinner_state.is_active = False


def ensure_directory_exists(directory_path: Path) -> None:
	"""
	Ensure a directory exists, creating it if necessary.

	Args:
	    directory_path: Path to ensure exists

	"""
	try:
		directory_path.mkdir(parents=True, exist_ok=True)
	except (PermissionError, OSError) as e:
		console.print(f"[red]Unable to create directory {directory_path}: {e!s}")
		raise


@contextlib.contextmanager
def progress_indicator(
	message: str,
	style: str = "spinner",
	total: int | None = None,
	transient: bool = False,
) -> Iterator[Callable[[int], None]]:
	"""
	Standardized progress indicator that supports different styles uniformly.

	Args:
	    message: The message to display with the progress indicator
	    style: The style of progress indicator - options:
	           - "spinner": Shows an indeterminate spinner
	           - "progress": Shows a determinate progress bar
	           - "step": Shows simple step-by-step progress
	    total: For determinate progress, the total units of work
	    transient: Whether the progress indicator should disappear after completion

	Yields:
	    A callable that accepts an integer amount to advance the progress

	"""
	# Skip visual indicators in testing/CI environments
	if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
		# Return a no-op advance function
		yield lambda _: None
		return

	# Check if a spinner is already active
	spinner_state = SpinnerState()
	if spinner_state.is_active and style == "spinner":
		# If there's already an active spinner, don't create a new one for spinner style
		yield lambda _: None
		return

	try:
		# Mark spinner as active if using spinner style
		if style == "spinner":
			spinner_state.is_active = True

		# Handle different progress styles
		if style == "spinner":
			# Indeterminate spinner using console.status
			with console.status(message):
				# Return a no-op advance function since spinners don't advance
				yield lambda _: None

		elif style == "progress":
			# Determinate progress bar using rich.progress.Progress
			progress = Progress(
				SpinnerColumn(),
				TextColumn("[progress.description]{task.description}"),
				transient=transient,
			)
			with progress:
				task_id = progress.add_task(message, total=total or 1)
				# Return a function that advances the progress
				yield lambda amount=1: progress.update(task_id, advance=amount)

		elif style == "step":
			# Simple step progress like typer.progressbar
			steps_completed = 0
			total_steps = total or 1

			console.print(f"{message} [0/{total_steps}]")

			# Function to advance and display steps
			def advance_step(amount: int = 1) -> None:
				nonlocal steps_completed
				steps_completed += amount
				steps_completed = min(steps_completed, total_steps)
				console.print(f"{message} [{steps_completed}/{total_steps}]")

			yield advance_step

			# Print completion if not transient
			if not transient and steps_completed >= total_steps:
				console.print(f"{message} [green]Complete![/green]")
	finally:
		# Reset spinner state if we were using spinner style
		if style == "spinner":
			spinner_state.is_active = False


def show_error(message: str, exception: Exception | None = None) -> None:
	"""
	Display an error summary with standardized formatting.

	Args:
	        message: The error message to display
	        exception: Optional exception that caused the error

	"""
	error_text = message
	if exception:
		error_text += f"\n\nDetails: {exception!s}"
		logger.exception("Error occurred", exc_info=exception)

	display_error_summary(error_text)


def show_warning(message: str) -> None:
	"""
	Display a warning summary with standardized formatting.

	Args:
	        message: The warning message to display

	"""
	display_warning_summary(message)


def exit_with_error(message: str, exit_code: int = 1, exception: Exception | None = None) -> None:
	"""
	Display an error message and exit.

	Args:
	        message: Error message to display
	        exit_code: Exit code to use
	        exception: Optional exception that caused the error

	"""
	show_error(message, exception)
	if exception is None:
		raise typer.Exit(exit_code)
	raise typer.Exit(exit_code) from exception
