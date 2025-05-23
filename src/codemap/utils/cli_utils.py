"""Utility functions for CLI operations in CodeMap."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from http import HTTPStatus
from typing import TYPE_CHECKING, Literal, Self

import typer
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from codemap import __version__

if TYPE_CHECKING:
	from collections.abc import Iterator

	from rich.status import Status

console = Console()
logger = logging.getLogger(__name__)

# Type for the progress update function with multiple parameters
ProgressUpdater = Callable[[str | None, int | None, int | None], None]


# Singleton class to track spinner state and manage active spinner display
class SpinnerState:
	"""Singleton class to manage the stack and display of active spinners."""

	_instance: Self | None = None

	def __init__(self) -> None:
		"""Initialize the spinner state attributes."""
		if not hasattr(self, "spinner_message_stack"):
			self.spinner_message_stack: list[str] = []
			self.active_rich_status_cm: Status | None = None
			self.tree_display_active: bool = False

	def __new__(cls) -> Self:
		"""Create or return the singleton instance.

		Returns:
		    Self: The singleton instance of SpinnerState
		"""
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance

	def _stop_active_status_cm(self) -> None:
		"""Safely stops (exits) the currently active Rich Status context manager."""
		if self.active_rich_status_cm:
			try:
				self.active_rich_status_cm.__exit__(None, None, None)
			except (RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover
				logger.debug("Error stopping previous status context manager", exc_info=exc)
			self.active_rich_status_cm = None
			self.tree_display_active = False

	def _format_tree_display(self) -> str:
		"""Format the spinner messages as a tree structure.

		Returns:
		    str: A formatted tree representation of all spinners in the stack
		"""
		if not self.spinner_message_stack:
			return ""

		result = []
		static_spinner_char = "▸"  # Static indicator for child lines

		for i, message in enumerate(self.spinner_message_stack):
			line_parts = []
			if i == 0:  # Root message
				line_parts.append(message)
			else:  # Child messages
				# Add indentation and vertical bars for ancestors
				line_parts.extend(["   "] * (i - 1))

				# Add connector from direct parent
				line_parts.append("[green]└─ [/green]")
				line_parts.append(f"{static_spinner_char} ")
				line_parts.append(message)

			result.append("".join(line_parts))

		return "\n".join(result)

	def _start_status_cm_for_tree(self) -> None:
		"""Creates or updates a status display showing all spinners in a tree structure."""
		if not self.spinner_message_stack:
			self._stop_active_status_cm()  # Ensure spinner stops if stack is empty
			return

		tree_display_text = self._format_tree_display()

		if self.active_rich_status_cm and self.tree_display_active:
			# If a tree display is already active, just update its content
			try:
				self.active_rich_status_cm.update(tree_display_text)
			except (RuntimeError, TypeError, ValueError) as e:  # pragma: no cover
				logger.debug(f"Error updating tree status: {e}", exc_info=True)
				# If update fails, fall back to recreating (e.g., if status was manually stopped)
				self._stop_active_status_cm()  # Clean up before recreating
				self._create_new_tree_status(tree_display_text)
		else:
			# Otherwise, stop any existing (non-tree) status and create a new tree status
			self._stop_active_status_cm()
			self._create_new_tree_status(tree_display_text)

	def _create_new_tree_status(self, tree_display_text: str) -> None:
		"""Helper to create and start a new status cm for the tree display."""
		new_status_cm = console.status(tree_display_text)  # Using default Rich spinner
		try:
			new_status_cm.__enter__()
			self.active_rich_status_cm = new_status_cm
			self.tree_display_active = True
		except (RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover
			logger.debug("Error starting new tree status context manager", exc_info=exc)

	def start_new_spinner(self, message: str) -> None:
		"""Handles the start of a new spinner.

		Adds the new spinner message to the stack and updates the tree display.
		"""
		self.spinner_message_stack.append(message)
		self._start_status_cm_for_tree()

	def stop_current_spinner_and_resume_parent(self) -> None:
		"""Handles the end of the current spinner.

		Pops the top spinner from the stack and updates the tree display.
		"""
		if self.spinner_message_stack:
			self.spinner_message_stack.pop()  # Remove the spinner that just ended

		if self.spinner_message_stack:  # If any spinners remain
			self._start_status_cm_for_tree()
		else:  # No spinners left on stack
			self._stop_active_status_cm()

	def temporarily_halt_visual_spinner(self) -> bool:
		"""Stops the current visual spinner if one is active.

		Used when a progress bar is about to take over.
		Returns True if a visual spinner was halted, False otherwise.
		"""
		if self.active_rich_status_cm:
			self._stop_active_status_cm()  # This already sets tree_display_active to False
			return True
		return False

	def resume_visual_spinner_if_needed(self) -> None:
		"""Resumes a visual spinner for the top message on the stack.

		If the stack is not empty and no visual spinner is currently active.
		Used after a progress bar (that might have halted a spinner) finishes.
		"""
		if self.spinner_message_stack and not self.active_rich_status_cm:
			self._start_status_cm_for_tree()


@contextlib.contextmanager
def progress_indicator(
	message: str,
	style: Literal["spinner", "progress"] = "spinner",
	total: int | None = None,
	transient: bool = False,
) -> Iterator[ProgressUpdater]:
	"""Standardized progress indicator that supports different styles uniformly.

	Manages nested spinners and interaction between spinners and progress bars
	to prevent UI flickering and ensure a clear display.

	Args:
	    message: The message to display with the progress indicator.
	    style: The style of progress indicator ('spinner' or 'progress').
	    total: For determinate progress, the total units of work.
	    transient: Whether the progress indicator should disappear after completion.

	Yields:
	    A callable (ProgressUpdater) for updating the progress/spinner.
	    For spinners, the callable is a no-op accepting three ignored arguments.
	    For progress bars, it accepts description, completed, and total (all optional).
	"""
	if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
		yield lambda _d=None, _c=None, _t=None: None
		return

	spinner_state = SpinnerState()

	if style == "spinner":
		spinner_state.start_new_spinner(message)
		try:
			yield lambda _d=None, _c=None, _t=None: None  # No-op for spinner
		finally:
			spinner_state.stop_current_spinner_and_resume_parent()
	elif style == "progress":
		was_spinner_visually_active = spinner_state.temporarily_halt_visual_spinner()
		try:
			progress = Progress(
				SpinnerColumn(),
				TextColumn("[progress.description]{task.description}"),
				BarColumn(),
				TextColumn("{task.completed}/{task.total}"),
				TimeElapsedColumn(),
				transient=transient,
				console=console,  # Ensure it uses the same console
			)
			with progress:  # Progress context manager handles its own start/stop
				task_id = progress.add_task(message, total=total)  # total can be None for indeterminate
				yield lambda description=None, completed=None, new_total=None: progress.update(
					task_id,
					description=description,
					completed=completed,
					total=new_total if new_total is not None else total,  # Use new_total if provided
				)
		finally:
			# Progress bar's 'with' context has exited.
			# If a spinner was active before this progress bar, try to resume it.
			if was_spinner_visually_active:
				spinner_state.resume_visual_spinner_if_needed()
	else:
		# Should not happen due to Literal type hint, but as a fallback
		logger.warning(f"Unknown progress_indicator style: {style}")
		yield lambda _d=None, _c=None, _t=None: None


def exit_with_error(message: str, exit_code: int = 1, exception: BaseException | None = None) -> None:
	"""
	Display an error message and exit.

	Args:
	        message: Error message to display
	        exit_code: Exit code to use
	        exception: Optional exception that caused the error

	"""
	logger.error(message, exc_info=exception)
	if exception is not None:
		raise typer.Exit(exit_code) from exception
	raise typer.Exit(exit_code)


def handle_keyboard_interrupt() -> None:
	"""Handles KeyboardInterrupt by printing a message and exiting cleanly."""
	console.print("\n[yellow]Operation cancelled by user.[/yellow]")
	raise typer.Exit(130)  # Standard exit code for SIGINT


def check_for_updates(is_verbose_param: bool) -> None:
	"""Check PyPI for a new version of CodeMap and warn if available."""
	try:
		package_name = "codemap"
		logger.debug(f"Checking for updates for package: {package_name}")

		current_v = parse_version(__version__)
		is_current_prerelease = current_v.is_prerelease
		logger.debug(f"Current version: {current_v} (Is pre-release: {is_current_prerelease})")

		req = urllib.request.Request(
			f"https://pypi.org/pypi/{package_name}/json",
			headers={"User-Agent": f"CodeMap-CLI-Update-Check/{__version__}"},
		)
		with urllib.request.urlopen(req, timeout=5) as response:  # noqa: S310
			if response.status == HTTPStatus.OK:
				data = json.load(response)
				pypi_releases = data.get("releases", {})
				if not pypi_releases:
					logger.debug("No releases found in PyPI response.")
					return

				valid_pypi_versions_str = []
				for version_str, release_files_list in pypi_releases.items():
					if not release_files_list:  # Skip if no files for this version
						continue
					# Consider version yanked if all its files are yanked
					version_is_yanked = all(file_info.get("yanked", False) for file_info in release_files_list)
					if not version_is_yanked:
						try:
							# Ensure the version string can be parsed and has a release segment
							if parse_version(version_str).release is not None:
								valid_pypi_versions_str.append(version_str)
						except InvalidVersion:  # Catch specific exception
							logger.debug(f"Could not parse version string from PyPI: {version_str}")

				if not valid_pypi_versions_str:
					logger.debug("No valid, non-yanked releases found on PyPI after filtering.")
					return

				all_pypi_versions = sorted(
					[parse_version(v) for v in valid_pypi_versions_str],
					reverse=True,
				)

				if not all_pypi_versions:
					logger.debug("No valid parseable releases found on PyPI after filtering.")
					return

				latest_candidate_v = None
				if is_current_prerelease:
					# If current is pre-release, consider the absolute latest version from PyPI
					latest_candidate_v = all_pypi_versions[0]
				else:
					# If current is stable, consider the latest stable version from PyPI
					stable_pypi_versions = [v for v in all_pypi_versions if not v.is_prerelease]
					if stable_pypi_versions:
						latest_candidate_v = stable_pypi_versions[0]

				if latest_candidate_v:
					logger.debug(f"Latest candidate version for comparison: {latest_candidate_v}")
					if latest_candidate_v > current_v:
						typer.secho(
							f"\n[!] A new version of CodeMap is available: {latest_candidate_v} (You have {current_v})",
							fg=typer.colors.YELLOW,
						)
						typer.secho(
							f"[!] To update, run: pip install --upgrade {package_name}",
							fg=typer.colors.YELLOW,
						)
					else:
						logger.debug("No newer version found on PyPI for current version type (stable/prerelease).")
			else:
				logger.debug(f"Failed to fetch update info from PyPI. Status: {response.status}")

	except urllib.error.URLError as e:
		logger.debug(f"Could not connect to PyPI to check for updates (URLError): {e.reason}")
	except json.JSONDecodeError:
		logger.debug("Could not parse PyPI response as JSON.")
	except TimeoutError:
		logger.debug("Timeout while checking for updates on PyPI.")
	except Exception as e:  # noqa: BLE001
		logger.debug(f"An unexpected error occurred during update check: {e}", exc_info=is_verbose_param)
