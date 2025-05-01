"""Tests for path utility functions."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Check if pathspec is installed, skip tests if not
try:
	import pathspec
except ImportError:
	pathspec = None


from codemap.utils.path_utils import (
	filter_paths_by_gitignore,
	get_git_root,
	get_relative_path,
	normalize_path,
)
from tests.base import FileSystemTestBase


@pytest.mark.unit
@pytest.mark.fs
class TestPathUtils(FileSystemTestBase):
	"""Test cases for path utility functions."""

	def test_normalize_path_string(self) -> None:
		"""Test normalize_path with a string input."""
		test_path = "~/some/path/../file.txt"
		expected = Path.home() / "some" / "file.txt"
		result = normalize_path(test_path)
		assert result == expected.resolve()

	def test_normalize_path_path_object(self) -> None:
		"""Test normalize_path with a Path object input."""
		test_path = Path("~/another/./path").expanduser()
		expected = Path.home() / "another" / "path"
		result = normalize_path(test_path)
		assert result == expected.resolve()

	def test_get_relative_path_success(self) -> None:
		"""Test get_relative_path when path is relative."""
		base_path = self.temp_dir
		target_path = self.temp_dir / "subdir" / "file.txt"
		expected = Path("subdir/file.txt")
		result = get_relative_path(target_path, base_path)
		assert result == expected

	def test_get_relative_path_failure(self) -> None:
		"""Test get_relative_path when path is not relative."""
		base_path = self.temp_dir / "base"
		target_path = self.temp_dir / "other" / "file.txt"
		expected = target_path.absolute()
		result = get_relative_path(target_path, base_path)
		assert result == expected

	def test_get_git_root_found(self) -> None:
		"""Test get_git_root when .git directory exists."""
		git_root = self.temp_dir / "my_repo"
		git_dir = git_root / ".git"
		git_dir.mkdir(parents=True)
		start_path = git_root / "src" / "module"
		start_path.mkdir(parents=True)

		result = get_git_root(start_path)
		assert result == git_root.resolve()

	def test_get_git_root_found_in_parent(self) -> None:
		"""Test get_git_root when .git is in a parent directory."""
		git_root = self.temp_dir
		git_dir = git_root / ".git"
		git_dir.mkdir(parents=True)
		start_path = git_root / "deeply" / "nested" / "folder"
		start_path.mkdir(parents=True)

		result = get_git_root(start_path)
		assert result == git_root.resolve()

	def test_get_git_root_not_found(self) -> None:
		"""Test get_git_root when no .git directory is found."""
		start_path = self.temp_dir / "not_a_repo"
		start_path.mkdir(parents=True)
		result = get_git_root(start_path)
		assert result is None

	def test_get_git_root_at_root(self) -> None:
		"""Test get_git_root starting from the filesystem root."""
		# This test might behave differently depending on the OS and permissions
		# It primarily tests the loop termination condition
		root_path = Path("/")
		result = get_git_root(root_path)
		# We expect None unless the filesystem root itself is a git repo (unlikely)
		assert result is None


@pytest.mark.skipif(pathspec is None, reason="pathspec package not installed")
@pytest.mark.unit
@pytest.mark.fs
class TestGitignoreFilter(FileSystemTestBase):
	"""Test cases for filter_paths_by_gitignore."""

	def setup_repo(self, gitignore_content: str | None = None) -> Path:
		"""Helper to set up a dummy repo structure."""
		repo_root = self.temp_dir / "repo"
		repo_root.mkdir()
		if gitignore_content is not None:
			gitignore_path = repo_root / ".gitignore"
			gitignore_path.write_text(gitignore_content, encoding="utf-8")
		return repo_root

	def test_filter_paths_no_gitignore(self) -> None:
		"""Test filtering when no .gitignore file exists."""
		repo_root = self.setup_repo()
		paths = [
			repo_root / "file1.py",
			repo_root / "subdir" / "file2.txt",
		]
		(repo_root / "subdir").mkdir()
		self.create_test_file(str(paths[0].relative_to(self.temp_dir)), "")
		self.create_test_file(str(paths[1].relative_to(self.temp_dir)), "")

		result = filter_paths_by_gitignore(paths, repo_root)
		assert result == paths

	def test_filter_paths_simple_ignore(self) -> None:
		"""Test filtering with a simple .gitignore pattern."""
		repo_root = self.setup_repo("*.log\n__pycache__/\n")
		paths = [
			repo_root / "main.py",
			repo_root / "data.log",
			repo_root / "src" / "__pycache__" / "cache.pyc",
			repo_root / "src" / "module.py",
		]
		(repo_root / "src" / "__pycache__").mkdir(parents=True)
		self.create_test_file(str(paths[0].relative_to(self.temp_dir)), "")
		self.create_test_file(str(paths[1].relative_to(self.temp_dir)), "")
		self.create_test_file(str(paths[2].relative_to(self.temp_dir)), "")
		self.create_test_file(str(paths[3].relative_to(self.temp_dir)), "")

		result = filter_paths_by_gitignore(paths, repo_root)
		expected = [
			repo_root / "main.py",
			repo_root / "src" / "module.py",
		]
		assert sorted(result) == sorted(expected)

	def test_filter_paths_outside_repo(self) -> None:
		"""Test filtering paths outside the repo root (should be kept)."""
		repo_root = self.setup_repo("ignored.txt")
		paths = [
			repo_root / "keep.py",
			repo_root / "ignored.txt",
			self.temp_dir / "outside_repo.txt",  # Path outside the repo
		]
		self.create_test_file(str(paths[0].relative_to(self.temp_dir)), "")
		self.create_test_file(str(paths[1].relative_to(self.temp_dir)), "")
		self.create_test_file("outside_repo.txt", "")

		result = filter_paths_by_gitignore(paths, repo_root)
		expected = [
			repo_root / "keep.py",
			self.temp_dir / "outside_repo.txt",
		]
		assert sorted(result) == sorted(expected)

	def test_filter_paths_pathspec_not_installed(self) -> None:
		"""Test behavior when pathspec is not installed."""
		repo_root = self.setup_repo("*.log")
		paths = [repo_root / "file.py", repo_root / "error.log"]
		self.create_test_file(str(paths[0].relative_to(self.temp_dir)), "")
		self.create_test_file(str(paths[1].relative_to(self.temp_dir)), "")

		with (
			patch.dict(sys.modules, {"pathspec": None}),
			patch("codemap.utils.path_utils.logger.warning") as mock_warning,
		):
			# Reload the module to simulate pathspec not being installed
			# This is a bit tricky, might need adjustments based on import structure
			# For simplicity, we might just mock the import check directly
			result = filter_paths_by_gitignore(paths, repo_root)
			# Expect all paths returned if pathspec fails to import
			assert result == paths
			mock_warning.assert_called_once_with("pathspec package not installed, gitignore filtering disabled")

	def test_filter_paths_directory_ignore(self) -> None:
		"""Test ignoring an entire directory."""
		repo_root = self.setup_repo("ignored_dir/")
		paths = [
			repo_root / "main.py",
			repo_root / "ignored_dir" / "file1.txt",
			repo_root / "ignored_dir" / "subdir" / "file2.txt",
			repo_root / "another_dir" / "file3.txt",
		]
		(repo_root / "ignored_dir" / "subdir").mkdir(parents=True)
		(repo_root / "another_dir").mkdir()
		for p in paths:
			# Create dummy files/dirs if they don't exist
			if not p.parent.exists():
				p.parent.mkdir(parents=True, exist_ok=True)
			if not p.exists():
				if "/" in p.name or "." not in p.name:  # Basic check if it looks like a dir
					if not p.exists():
						p.mkdir(exist_ok=True)
				else:
					# Use relative path strings
					self.create_test_file(str(p.relative_to(self.temp_dir)), "")

		filter_paths_by_gitignore(paths, repo_root)
		expected = [
			repo_root / "main.py",
			repo_root / "another_dir" / "file3.txt",
		]
		# Need to filter out the directory paths themselves from the input if testing file filtering
		file_paths_only = [p for p in paths if p.is_file()]
		result_files = filter_paths_by_gitignore(file_paths_only, repo_root)

		assert sorted(result_files) == sorted(expected)

	def test_filter_paths_negation_pattern(self) -> None:
		"""Test filtering with negation patterns in .gitignore."""
		repo_root = self.setup_repo("*.*\n!important.txt\nlogs/\n!logs/audit.log")
		paths = [
			repo_root / "file.py",
			repo_root / "important.txt",
			repo_root / "logs" / "debug.log",
			repo_root / "logs" / "audit.log",
			repo_root / "config.json",
		]
		(repo_root / "logs").mkdir()
		self.create_test_file(str(paths[0].relative_to(self.temp_dir)), "")
		self.create_test_file(str(paths[1].relative_to(self.temp_dir)), "")
		self.create_test_file(str(paths[2].relative_to(self.temp_dir)), "")
		self.create_test_file(str(paths[3].relative_to(self.temp_dir)), "")
		self.create_test_file(str(paths[4].relative_to(self.temp_dir)), "")

		result = filter_paths_by_gitignore(paths, repo_root)
		expected = [
			repo_root / "important.txt",
			repo_root / "logs" / "audit.log",
		]
		assert sorted(result) == sorted(expected)
