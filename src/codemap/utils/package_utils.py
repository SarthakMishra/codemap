"""Utilities for package management and updates."""

from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
import sys
from typing import TYPE_CHECKING

import pkg_resources
import requests
from packaging import version

from codemap import __version__
from codemap.utils.cli_utils import console

if TYPE_CHECKING:
	from rich.console import Console

logger = logging.getLogger(__name__)

GITHUB_REPO = "SarthakMishra/codemap"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
HTTP_OK = 200  # Status code for successful HTTP request

# PyPI package name
PACKAGE_NAME = "codemap"
PYPI_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"


def get_current_version() -> str:
	"""
	Get the current installed version of CodeMap.

	Returns:
	    The version string

	"""
	return __version__


def get_latest_version() -> tuple[str | None, str | None]:
	"""
	Check the latest version available on GitHub.

	Returns:
	    A tuple of (version, release_url) or (None, None) if check fails

	"""
	try:
		response = requests.get(GITHUB_API_URL, timeout=5)
		if response.status_code == HTTP_OK:
			data = response.json()
			# Remove 'v' prefix if present
			latest_version = data["tag_name"].lstrip("v")
			html_url = data.get("html_url")
			return latest_version, html_url
	except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
		logger.warning("Failed to check for updates: %s", str(e))

	return None, None


def is_update_available() -> tuple[bool, str | None, str | None]:
	"""
	Check if an update is available.

	Returns:
	    Tuple of (update_available, latest_version, release_url)

	"""
	current = get_current_version()
	latest, url = get_latest_version()

	if latest is None:
		return False, None, None

	try:
		return version.parse(latest) > version.parse(current), latest, url
	except version.InvalidVersion:
		return False, latest, url


def update_package() -> bool:
	"""
	Update the package using pip.

	Returns:
	    True if update succeeded, False otherwise

	"""
	try:
		console.print("[yellow]Updating CodeMap using pip...[/]")
		update_result = subprocess.run(  # noqa: S603
			[sys.executable, "-m", "pip", "install", "--upgrade", f"git+https://github.com/{GITHUB_REPO}.git"],
			capture_output=True,
			text=True,
			check=False,
		)

		if update_result.returncode == 0:
			console.print("[green]CodeMap updated successfully![/]")
			return True
		console.print(f"[red]Update failed: {update_result.stderr}[/]")
		return False

	except subprocess.SubprocessError as e:
		console.print(f"[red]Error during update: {e!s}[/]")
		return False


def uninstall_package() -> bool:
	"""
	Uninstall the package using pip.

	Returns:
	    True if uninstallation succeeded, False otherwise

	"""
	try:
		console.print("[yellow]Uninstalling CodeMap using pip...[/]")
		uninstall_result = subprocess.run(  # noqa: S603
			[sys.executable, "-m", "pip", "uninstall", "-y", "codemap"], capture_output=True, text=True, check=False
		)

		if uninstall_result.returncode == 0:
			console.print("[green]CodeMap uninstalled successfully![/]")
			return True
		console.print(f"[red]Uninstallation failed: {uninstall_result.stderr}[/]")
		return False

	except subprocess.SubprocessError as e:
		console.print(f"[red]Error during uninstallation: {e!s}[/]")
		return False


def check_for_updates() -> tuple[bool, str | None]:
	"""
	Check if a newer version of CodeMap is available.

	Returns:
	        tuple: (is_latest, latest_version)
	        - is_latest (bool): True if current version is the latest
	        - latest_version (str | None): Latest version string or None if check failed

	"""
	try:
		current_version = get_current_version()
		pypi_url = "https://pypi.org/pypi/codemap/json"

		# Fetch PyPI data
		response = requests.get(pypi_url, timeout=5)
		response.raise_for_status()

		data = response.json()
		latest_version = data["info"]["version"]

		# Compare versions
		current_parsed = pkg_resources.parse_version(current_version)
		latest_parsed = pkg_resources.parse_version(latest_version)

		is_latest = current_parsed >= latest_parsed
		return is_latest, latest_version
	except (requests.RequestException, ValueError, KeyError, pkg_resources.ResolutionError):
		logger.warning("Failed to check for updates")
		return True, None  # Assume current is latest on failure


def notify_update_available(custom_console: Console | None = None) -> None:
	"""
	Check for updates and notify the user if a newer version is available.

	Args:
	        custom_console: Optional console to print notification to

	"""
	try:
		is_latest, latest_version = check_for_updates()
		# Use provided console or default to the imported one
		output_console = custom_console or console
		if not is_latest and latest_version:
			current_version = get_current_version()
			output_console.print(
				f"[yellow]Update available:[/yellow] {current_version} → {latest_version}\n"
				f"[yellow]Run 'uv pip install --upgrade codemap' to update.[/yellow]\n"
			)
	except (requests.RequestException, ValueError, KeyError, pkg_resources.ResolutionError):  # More specific exceptions
		logger.debug("Failed to check for updates")
		# Silently ignore errors in notification to not disrupt user


def get_system_info() -> dict:
	"""
	Get system information for debugging.

	Returns:
	    Dictionary with system information

	"""
	info = {
		"codemap_version": get_current_version(),
		"python_version": platform.python_version(),
		"platform": platform.platform(),
		"system": platform.system(),
		"python_path": sys.executable,
	}

	# Try to get pip version
	try:
		pip_version = subprocess.run(  # noqa: S603
			[sys.executable, "-m", "pip", "--version"], capture_output=True, text=True, check=False
		)
		if pip_version.returncode == 0:
			match = re.search(r"pip (\d+\.\d+\.\d+)", pip_version.stdout)
			if match:
				info["pip_version"] = match.group(1)
	except subprocess.SubprocessError:
		info["pip_version"] = "unknown"

	return info
