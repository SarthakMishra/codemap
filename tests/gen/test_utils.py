"""Tests for gen utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codemap.gen.utils import (
	determine_output_path,
	generate_tree,
	write_documentation,
)
from tests.base import FileSystemTestBase


@pytest.mark.unit
@pytest.mark.fs
class TestGenerateTree(FileSystemTestBase):
	"""Test cases for generate_tree."""

	def test_generate_tree_simple(self) -> None:
		"""Test basic tree generation."""
		target_path = self.temp_dir / "project"
		target_path.mkdir()
		paths = [
			self.create_test_file("project/file1.txt", ""),
			self.create_test_file("project/subdir/file2.py", ""),
			target_path / "emptydir",  # Create an empty directory
		]
		paths[2].mkdir()

		# Get absolute paths for the function
		abs_paths = [p.resolve() for p in paths]

		expected_tree = "├── emptydir\n├── file1.txt\n└── subdir/\n    └── file2.py"
		result = generate_tree(target_path.resolve(), abs_paths)
		assert result == expected_tree

	def test_generate_tree_empty(self) -> None:
		"""Test tree generation with empty input."""
		target_path = self.temp_dir / "empty_project"
		target_path.mkdir()
		result = generate_tree(target_path.resolve(), [])
		assert result == ""

	def test_generate_tree_paths_outside_target(self) -> None:
		"""Test that paths outside the target are ignored."""
		target_path = self.temp_dir / "project"
		target_path.mkdir()
		paths = [
			self.create_test_file("project/file_inside.txt", ""),
			self.create_test_file("outside/file_outside.txt", ""),
		]
		abs_paths = [p.resolve() for p in paths]

		expected_tree = "└── file_inside.txt"
		result = generate_tree(target_path.resolve(), abs_paths)
		assert result == expected_tree

	def test_generate_tree_nested(self) -> None:
		"""Test deeply nested tree generation."""
		target_path = self.temp_dir / "nested_project"
		target_path.mkdir()
		paths = [
			self.create_test_file("nested_project/a/b/c/file.txt", ""),
			self.create_test_file("nested_project/a/d/another.py", ""),
			self.create_test_file("nested_project/root_file.md", ""),
			target_path / "a" / "b" / "empty_leaf",  # Empty leaf directory
			target_path / "e",  # Empty top-level directory
		]
		paths[3].mkdir(parents=True)
		paths[4].mkdir()
		abs_paths = [p.resolve() for p in paths]

		expected_tree = (
			"├── a/\n"
			"│   ├── b/\n"
			"│   │   ├── c/\n"
			"│   │   │   └── file.txt\n"
			"│   │   └── empty_leaf\n"
			"│   └── d/\n"
			"│       └── another.py\n"
			"├── e\n"
			"└── root_file.md"
		)
		result = generate_tree(target_path.resolve(), abs_paths)
		# Normalize line endings just in case
		assert "\n".join(result.splitlines()) == "\n".join(expected_tree.splitlines())

	def test_generate_tree_file_and_dir_conflict_name(self) -> None:
		"""Test scenario where a file might have the same name as a directory part (should stop processing)."""
		target_path = self.temp_dir / "conflict"
		target_path.mkdir()
		paths = [
			self.create_test_file("conflict/a", "content of file named a"),
			# Construct the conflicting path object correctly relative to target_path
			target_path / "a" / "b" / "c.txt",  # This path should be ignored
		]

		# Pass the resolved file path and the conceptual conflicting Path object
		abs_paths = [paths[0].resolve(), paths[1]]

		expected_tree = "└── a"  # Only the file 'a' should appear
		result = generate_tree(target_path.resolve(), abs_paths)
		assert result == expected_tree


@pytest.mark.unit
@pytest.mark.fs
class TestDetermineOutputPath(FileSystemTestBase):
	"""Test cases for determine_output_path."""

	@patch("datetime.datetime")
	def test_determine_output_path_no_args_no_config(self, mock_datetime_module: MagicMock) -> None:
		"""Test default behavior: creates timestamped file in ./documentation."""
		mock_dt_instance = MagicMock()
		mock_datetime_module.now.return_value = mock_dt_instance
		mock_datetime_module.now.return_value.strftime.return_value = "20240101_120000"
		mock_datetime_module.now.return_value.astimezone.return_value.tzinfo = None
		project_root = self.temp_dir / "my_proj"
		project_root.mkdir()

		expected_path = project_root / "documentation" / "documentation_20240101_120000.md"
		result = determine_output_path(project_root, None, {})

		assert result == expected_path
		assert expected_path.parent.exists()
		mock_datetime_module.now.assert_called_once()
		mock_dt_instance.strftime.assert_called_once_with("%Y%m%d_%H%M%S")

	def test_determine_output_path_cli_arg(self) -> None:
		"""Test when CLI output path is provided."""
		project_root = self.temp_dir / "my_proj"
		project_root.mkdir()
		output_arg = self.temp_dir / "output" / "cli_doc.md"
		# output_arg parent should NOT be created by the function

		result = determine_output_path(project_root, output_arg, {"output_dir": "config_docs"})
		assert result == output_arg.resolve()
		assert not output_arg.parent.exists()  # Function shouldn't create parent for explicit path

	def test_determine_output_path_config_file_relative(self) -> None:
		"""Test when config specifies a relative output_file."""
		project_root = self.temp_dir / "my_proj"
		project_root.mkdir()
		config = {"output_file": "docs/config_file.md"}

		expected_path = project_root / "docs" / "config_file.md"
		result = determine_output_path(project_root, None, config)
		assert result == expected_path

	def test_determine_output_path_config_file_absolute(self) -> None:
		"""Test when config specifies an absolute output_file."""
		project_root = self.temp_dir / "my_proj"
		project_root.mkdir()
		# Ensure parent directory exists for the test assertion later if needed
		abs_output_file_dir = self.temp_dir / "absolute_docs"
		abs_output_file_dir.mkdir(exist_ok=True)  # Create the dir for the test
		abs_output_file = abs_output_file_dir / "abs_config_file.md"
		config = {"output_file": str(abs_output_file)}

		result = determine_output_path(project_root, None, config)
		assert result == abs_output_file
		assert abs_output_file.parent.exists()

	@patch("datetime.datetime")
	def test_determine_output_path_config_dir_relative(self, mock_datetime: MagicMock) -> None:
		"""Test when config specifies a relative output_dir."""
		mock_now = MagicMock()
		mock_datetime.now.return_value = mock_now
		mock_now.strftime.return_value = "20240101_130000"
		mock_datetime.now.return_value.astimezone.return_value.tzinfo = None
		project_root = self.temp_dir / "my_proj"
		project_root.mkdir()
		config = {"output_dir": "custom_docs"}

		expected_path = project_root / "custom_docs" / "documentation_20240101_130000.md"
		result = determine_output_path(project_root, None, config)

		assert result == expected_path
		assert expected_path.parent.exists()

	@patch("datetime.datetime")
	def test_determine_output_path_config_dir_absolute(self, mock_datetime: MagicMock) -> None:
		"""Test when config specifies an absolute output_dir."""
		mock_now = MagicMock()
		mock_datetime.now.return_value = mock_now
		mock_now.strftime.return_value = "20240101_140000"
		mock_datetime.now.return_value.astimezone.return_value.tzinfo = None
		project_root = self.temp_dir / "my_proj"
		project_root.mkdir()
		abs_output_dir = self.temp_dir / "absolute_output_dir"
		# Directory should be created by the function
		config = {"output_dir": str(abs_output_dir)}

		expected_path = abs_output_dir / "documentation_20240101_140000.md"
		result = determine_output_path(project_root, None, config)

		assert result == expected_path
		assert abs_output_dir.exists()

	def test_determine_output_path_cli_overrides_config_file(self) -> None:
		"""Test that CLI output overrides config output_file."""
		project_root = self.temp_dir / "my_proj"
		project_root.mkdir()
		output_arg = self.temp_dir / "cli_wins.md"
		config = {"output_file": "config_loses.md"}

		result = determine_output_path(project_root, output_arg, config)
		assert result == output_arg.resolve()

	def test_determine_output_path_config_file_overrides_config_dir(self) -> None:
		"""Test that config output_file overrides config output_dir."""
		project_root = self.temp_dir / "my_proj"
		project_root.mkdir()
		config = {"output_file": "file_wins.md", "output_dir": "dir_loses"}

		expected_path = project_root / "file_wins.md"
		result = determine_output_path(project_root, None, config)
		assert result == expected_path


@pytest.mark.unit
@pytest.mark.fs
class TestWriteDocumentation(FileSystemTestBase):
	"""Test cases for write_documentation."""

	@patch("codemap.utils.cli_utils.console")
	@patch("codemap.utils.cli_utils.ensure_directory_exists")
	@patch("codemap.utils.cli_utils.show_error")
	def test_write_documentation_success(
		self, mock_show_error: MagicMock, mock_ensure_dir: MagicMock, mock_console: MagicMock
	) -> None:
		"""Test writing documentation successfully."""
		output_path = self.temp_dir / "docs" / "output.md"
		documentation = "# My Awesome Docs\\n\\nThis is the content."
		# Manually create the parent directory because ensure_directory_exists is mocked
		output_path.parent.mkdir(parents=True, exist_ok=True)

		write_documentation(output_path, documentation)

		mock_ensure_dir.assert_called_once_with(output_path.parent)
		assert output_path.read_text() == documentation
		mock_console.print.assert_called_once_with(f"[green]Documentation written to {output_path}")
		mock_show_error.assert_not_called()

	@patch("codemap.utils.cli_utils.console")
	@patch("codemap.utils.cli_utils.ensure_directory_exists")
	@patch("codemap.utils.cli_utils.show_error")
	@patch("pathlib.Path.write_text", side_effect=PermissionError("Cannot write"))
	def test_write_documentation_permission_error(
		self, mock_write: MagicMock, mock_show_error: MagicMock, mock_ensure_dir: MagicMock, mock_console: MagicMock
	) -> None:
		"""Test handling PermissionError during writing."""
		output_path = self.temp_dir / "no_access" / "output.md"
		documentation = "Some docs"

		with pytest.raises(PermissionError):
			write_documentation(output_path, documentation)

		mock_ensure_dir.assert_called_once_with(output_path.parent)
		mock_write.assert_called_once_with(documentation)
		mock_show_error.assert_called_once_with(f"Error writing documentation to {output_path}: Cannot write")
		mock_console.print.assert_not_called()

	@patch("codemap.utils.cli_utils.console")
	@patch("codemap.utils.cli_utils.ensure_directory_exists", side_effect=OSError("Cannot create dir"))
	@patch("codemap.utils.cli_utils.show_error")
	def test_write_documentation_ensure_dir_error(
		self, mock_show_error: MagicMock, mock_ensure_dir: MagicMock, mock_console: MagicMock
	) -> None:
		"""Test handling OSError during directory creation."""
		output_path = self.temp_dir / "bad_dir" / "output.md"
		documentation = "Some docs"

		with patch("pathlib.Path.write_text") as mock_write, pytest.raises(OSError, match="Cannot create dir"):
			write_documentation(output_path, documentation)

		mock_ensure_dir.assert_called_once_with(output_path.parent)
		mock_write.assert_not_called()
		mock_show_error.assert_called_once_with(f"Error writing documentation to {output_path}: Cannot create dir")
		mock_console.print.assert_not_called()
