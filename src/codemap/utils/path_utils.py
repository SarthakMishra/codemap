"""Utilities for handling paths and file system operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codemap.config import DEFAULT_CONFIG
from codemap.utils.config_loader import ConfigLoader

if TYPE_CHECKING:
	from collections.abc import Sequence

logger = logging.getLogger(__name__)


def filter_paths_by_gitignore(paths: Sequence[Path], repo_root: Path) -> list[Path]:
	"""
	Filter paths based on .gitignore patterns.

	This function filters a list of paths to exclude those that match
	patterns in a .gitignore file, while preserving the directory structure.

	Args:
	    paths: Sequence of paths to filter
	    repo_root: Root directory of the repository

	Returns:
	    List of paths that don't match any gitignore patterns

	"""
	try:
		import pathspec
		from pathspec.patterns.gitwildmatch import GitWildMatchPattern
	except ImportError:
		logger.warning("pathspec package not installed, gitignore filtering disabled")
		return list(paths)

	# Read .gitignore if it exists
	gitignore_path = repo_root / ".gitignore"
	gitignore_patterns = []

	if gitignore_path.exists():
		# Parse gitignore patterns
		with gitignore_path.open("r", encoding="utf-8") as f:
			gitignore_content = f.read()
		gitignore_patterns = gitignore_content.splitlines()

	# Add default patterns for common directories that should be ignored
	default_ignore_patterns = [
		"__pycache__/",
		"*.py[cod]",
		"*$py.class",
		".git/",
		".pytest_cache/",
		".coverage",
		"htmlcov/",
		".tox/",
		".nox/",
		".hypothesis/",
		".mypy_cache/",
		".ruff_cache/",
		"dist/",
		"build/",
		"*.so",
		"*.egg",
		"*.egg-info/",
		".env/",
		"venv/",
		".venv/",
		"env/",
		"ENV/",
		"node_modules/",
	]

	# Combine patterns with existing ones, avoiding duplicates
	all_patterns = gitignore_patterns + [p for p in default_ignore_patterns if p not in gitignore_patterns]

	# Create path spec with direct import
	spec = pathspec.PathSpec.from_lines(GitWildMatchPattern, all_patterns)

	# Filter paths
	filtered_paths = []

	# Process files first
	file_paths = [p for p in paths if p.is_file()]
	for path in file_paths:
		try:
			rel_path = path.relative_to(repo_root)
			if not spec.match_file(str(rel_path)):
				filtered_paths.append(path)
		except ValueError:
			# Path is not relative to repo_root
			filtered_paths.append(path)

	# Process directories
	dir_paths = [p for p in paths if p.is_dir()]

	# First check which directories are included according to gitignore patterns
	included_dirs = []
	for dir_path in dir_paths:
		try:
			rel_path = dir_path.relative_to(repo_root)
			rel_path_str = str(rel_path) + "/"  # Add trailing slash for directory patterns

			# Skip the directory if it matches a gitignore pattern
			if spec.match_file(rel_path_str):
				logger.debug(f"Skipping ignored directory: {rel_path}")
				continue

			# Check if any parent directory is already ignored
			parent_ignored = False
			for parent in rel_path.parents:
				parent_str = str(parent) + "/"
				if spec.match_file(parent_str):
					parent_ignored = True
					logger.debug(f"Skipping directory with ignored parent: {parent}")
					break

			if not parent_ignored:
				included_dirs.append(dir_path)

		except ValueError:
			# Path is not relative to repo_root
			included_dirs.append(dir_path)

	# Include all directories at all levels to preserve hierarchy
	# Directories with no content might still be needed for the tree visualization
	filtered_paths.extend(included_dirs)

	logger.debug(f"Filtered {len(paths)} paths down to {len(filtered_paths)} after applying gitignore patterns")
	return filtered_paths


def normalize_path(path: str | Path) -> Path:
	"""
	Normalize a path to an absolute Path object.

	Args:
	    path: Path string or object

	Returns:
	    Normalized absolute Path

	"""
	if isinstance(path, str):
		path = Path(path)
	return path.expanduser().resolve()


def get_relative_path(path: Path, base_path: Path) -> Path:
	"""
	Get path relative to base_path if possible, otherwise return absolute path.

	Args:
	    path: The path to make relative
	    base_path: The base path to make it relative to

	Returns:
	    Relative path if possible, otherwise absolute path

	"""
	try:
		return path.relative_to(base_path)
	except ValueError:
		return path.absolute()


def find_project_root(start_path: Path | None = None) -> Path:
	"""
	Determine the project root directory (typically the Git repository root).

	Searches upwards from the starting path (or current working directory if
	start_path is None) for the '.git' directory.

	Args:
	    start_path (Path | None, optional): The path to start searching from.
	                                        Defaults to the current working directory.

	Returns:
	    Path: The absolute path to the project root (containing .git).

	Raises:
	    FileNotFoundError: If the project root cannot be determined based on
	                       the presence of the '.git' directory.

	"""
	if start_path is None:
		current_dir = Path.cwd()
		search_origin_display = "current working directory"
	else:
		# Ensure start_path is absolute for consistent parent traversal
		current_dir = start_path.resolve()
		search_origin_display = f"'{start_path}'"

	# Check the starting directory itself and its parents
	for parent in [current_dir, *current_dir.parents]:
		# Check for .git directory
		is_git_root = (parent / ".git").is_dir()

		if is_git_root:
			logger.debug(f"Project root (Git repository root) found at: {parent}")
			return parent

	# If loop completes without finding a root
	msg = f"Could not determine project root searching upwards from {search_origin_display}. No '.git' directory found."
	raise FileNotFoundError(msg)


def get_cache_path(component_name: str | None = None, workspace_root: Path | None = None) -> Path:
	"""
	Get the cache path for a specific component or the root cache directory.

	Args:
	    component_name (str | None, optional): The name of the component requiring a
	                                           cache directory (e.g., 'graph', 'vector').
	                                           If None, the root cache directory path
	                                           is returned. Defaults to None.
	    workspace_root (Path | None, optional): The workspace root path.
	                                            If None, it will be determined automatically
	                                            using `find_project_root()`. Defaults to None.

	Returns:
	    Path: The absolute path to the component's cache directory if `component_name`
	          is provided, otherwise the absolute path to the root cache directory.

	Raises:
	    FileNotFoundError: If `workspace_root` is None and `find_project_root()` fails.
	    # ConfigError: Config related errors might also be raised by ConfigLoader

	"""
	if workspace_root is None:
		workspace_root = find_project_root()

	# Get ConfigLoader instance, potentially passing the repo_root
	# Ensure ConfigLoader handles initialization correctly if called multiple times
	config_loader = ConfigLoader.get_instance(repo_root=workspace_root)

	# Get cache directory name from config, falling back to default
	# Ensure DEFAULT_CONFIG is accessible here
	cache_dir_name = config_loader.get("cache_dir", DEFAULT_CONFIG.get("cache_dir", ".codemap_cache"))

	cache_root = workspace_root / cache_dir_name

	if component_name is None:
		# Ensure the root cache directory exists
		cache_root.mkdir(parents=True, exist_ok=True)
		logger.debug(f"Root cache path: {cache_root}")
		return cache_root

	# Calculate the specific component's cache path
	component_cache_path = cache_root / component_name
	# Ensure the component cache directory exists
	component_cache_path.mkdir(parents=True, exist_ok=True)
	logger.debug(f"Cache path for component '{component_name}': {component_cache_path}")
	return component_cache_path
