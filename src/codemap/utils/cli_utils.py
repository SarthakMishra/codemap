"""Utility functions for CLI operations in CodeMap."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import urllib.error
import urllib.request
from http import HTTPStatus
from typing import TYPE_CHECKING, Literal, Self

import typer
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version  # For version comparison
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from codemap import __version__

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator

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


@contextlib.contextmanager
def progress_indicator(
	message: str,
	style: Literal["spinner", "progress", "step"] = "spinner",
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
				"""Advances the step progress by the specified amount and updates the display.

				Args:
					amount: The number of steps to advance. Defaults to 1.

				Returns:
					None
				"""
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


def exit_with_error(message: str, exit_code: int = 1, exception: Exception | None = None) -> None:
	"""
	Display an error message and exit.

	Args:
	        message: Error message to display
	        exit_code: Exit code to use
	        exception: Optional exception that caused the error

	"""
	logger.error(message, exc_info=exception)
	raise typer.Exit(exit_code) from exception


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
