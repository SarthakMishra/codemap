"""Utilities for handling paths and directories."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".codemap_cache"


def get_workspace_root() -> Path:
	"""
	Determine the workspace root directory.

	Searches upwards from the current working directory for a common
	project marker file (e.g., pyproject.toml or .git).

	Returns:
	    Path: The absolute path to the workspace root.

	Raises:
	    FileNotFoundError: If the workspace root cannot be determined.
	"""
	current_dir = Path.cwd()
	for parent in [current_dir, *current_dir.parents]:
		if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
			logger.debug(f"Workspace root found at: {parent}")
			return parent
	msg = "Could not determine workspace root. No 'pyproject.toml' or '.git' found."
	raise FileNotFoundError(msg)


def get_cache_path(component_name: str, workspace_root: Path | None = None) -> Path:
	"""
	Get the cache path for a specific component (e.g., graph, vector).

	Args:
	    component_name (str): The name of the component requiring a cache directory.
	    workspace_root (Path | None, optional): The workspace root path.
	                                            If None, it will be determined automatically.
	                                            Defaults to None.

	Returns:
	    Path: The absolute path to the component's cache directory.
	"""
	if workspace_root is None:
		workspace_root = get_workspace_root()
	cache_root = workspace_root / CACHE_DIR_NAME
	component_cache_path = cache_root / component_name
	logger.debug(f"Cache path for component '{component_name}': {component_cache_path}")
	return component_cache_path
