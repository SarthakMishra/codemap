"""Utilities for package management and updates."""

from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
import sys

import requests
from packaging import version

from codemap import __version__
from codemap.utils.cli_utils import console

logger = logging.getLogger(__name__)

GITHUB_REPO = "SarthakMishra/codemap"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
HTTP_OK = 200  # Status code for successful HTTP request


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


def check_for_updates_and_notify() -> None:
	"""Check for updates and notify the user if an update is available."""
	try:
		update_available, latest_version, release_url = is_update_available()
		if update_available and latest_version:
			console.print(f"[yellow]Update available: v{latest_version} (current: v{get_current_version()})[/]")
			console.print("[yellow]Run 'codemap update' to update to the latest version.[/]")
			if release_url:
				console.print(f"[blue]Release notes: {release_url}[/]")
	except Exception as e:  # noqa: BLE001
		# Don't let update checking interfere with normal operation
		# This is intentionally catching all exceptions to avoid disrupting the main program flow
		logger.debug("Error checking for updates: %s", str(e))


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
