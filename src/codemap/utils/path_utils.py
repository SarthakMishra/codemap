"""Utilities for handling paths and file system operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Sequence

logger = logging.getLogger(__name__)


def filter_paths_by_gitignore(paths: Sequence[Path], repo_root: Path) -> list[Path]:
	"""
	Filter paths based on .gitignore patterns.

	This function filters a list of paths to exclude those that match
	patterns in a .gitignore file.

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
	if not gitignore_path.exists():
		return list(paths)

	# Parse gitignore patterns
	with gitignore_path.open("r", encoding="utf-8") as f:
		gitignore_content = f.read()

	# Create path spec with direct import
	spec = pathspec.PathSpec.from_lines(GitWildMatchPattern, gitignore_content.splitlines())

	# Filter paths
	filtered_paths = []
	for path in paths:
		try:
			rel_path = path.relative_to(repo_root)
			if not spec.match_file(str(rel_path)):
				filtered_paths.append(path)
		except ValueError:
			# Path is not relative to repo_root
			filtered_paths.append(path)

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


def get_git_root(start_path: Path) -> Path | None:
	"""
	Find the root directory of a git repository.

	Args:
	    start_path: Path to start searching from

	Returns:
	    Path to the git root directory, or None if not found

	"""
	current = start_path.absolute()

	while current != current.parent:
		git_dir = current / ".git"
		if git_dir.exists() and git_dir.is_dir():
			return current
		current = current.parent

	return None
