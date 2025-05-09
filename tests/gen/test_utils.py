"""Tests for gen utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest

from codemap.config import ConfigLoader
from codemap.config.config_schema import GenSchema
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

		# The function sorts directories first (emptydir, subdir), then files alphabetically
		expected_tree = "project\n├── emptydir/\n├── subdir/\n│   └── file2.py\n└── file1.txt"
		result = generate_tree(target_path.resolve(), abs_paths)
		assert result == expected_tree

	def test_generate_tree_empty(self) -> None:
		"""Test tree generation with empty input."""
		target_path = self.temp_dir / "empty_project"
		target_path.mkdir()
		result = generate_tree(target_path.resolve(), [])
		assert result == "empty_project/"

	def test_generate_tree_paths_outside_target(self) -> None:
		"""Test that paths outside the target are ignored."""
		target_path = self.temp_dir / "project"
		target_path.mkdir()
		paths = [
			self.create_test_file("project/file_inside.txt", ""),
			self.create_test_file("outside/file_outside.txt", ""),
		]
		abs_paths = [p.resolve() for p in paths]

		expected_tree = "project\n└── file_inside.txt"
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
			"nested_project\n"
			"├── a/\n"
			"│   ├── b/\n"
			"│   │   ├── c/\n"
			"│   │   │   └── file.txt\n"
			"│   │   └── empty_leaf/\n"
			"│   └── d/\n"
			"│       └── another.py\n"
			"├── e/\n"
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

		expected_tree = "conflict\n└── a"  # Only the file 'a' should appear
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

		# Create mock ConfigLoader with default settings
		mock_config = Mock(spec=ConfigLoader)
		mock_gen_schema = Mock(spec=GenSchema)
		mock_gen_schema.output_dir = "documentation"
		type(mock_config).get = PropertyMock(return_value=MagicMock(gen=mock_gen_schema))

		expected_path = project_root / "documentation" / "documentation_20240101_120000.md"
		result = determine_output_path(project_root, mock_config, None)

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

		# Create mock ConfigLoader with custom output_dir setting
		mock_config = Mock(spec=ConfigLoader)
		mock_gen_schema = Mock(spec=GenSchema)
		mock_gen_schema.output_dir = "config_docs"
		type(mock_config).get = PropertyMock(return_value=MagicMock(gen=mock_gen_schema))

		result = determine_output_path(project_root, mock_config, output_arg)
		assert result == output_arg.resolve()
		assert not output_arg.parent.exists()  # Function shouldn't create parent for explicit path

	@patch("pathlib.Path.mkdir")
	def test_determine_output_path_config_dir_relative(self, mock_mkdir: MagicMock) -> None:
		"""Test when config specifies a relative output_dir."""
		project_root = self.temp_dir / "my_proj"
		project_root.mkdir()

		# Create mock ConfigLoader with custom output_dir
		mock_config = Mock(spec=ConfigLoader)
		mock_gen_schema = Mock(spec=GenSchema)
		mock_gen_schema.output_dir = "custom_docs"
		type(mock_config).get = PropertyMock(return_value=MagicMock(gen=mock_gen_schema))

		with patch("datetime.datetime") as mock_datetime:
			mock_now = MagicMock()
			mock_datetime.now.return_value = mock_now
			mock_now.strftime.return_value = "20240101_130000"

			expected_path = project_root / "custom_docs" / "documentation_20240101_130000.md"
			result = determine_output_path(project_root, mock_config, None)

			assert result == expected_path
			mock_mkdir.assert_called()

	@patch("pathlib.Path.mkdir")
	def test_determine_output_path_config_dir_absolute(self, mock_mkdir: MagicMock) -> None:
		"""Test when config specifies an absolute output_dir."""
		project_root = self.temp_dir / "my_proj"
		project_root.mkdir()
		abs_output_dir = self.temp_dir / "absolute_output_dir"

		# Create mock ConfigLoader with absolute output_dir
		mock_config = Mock(spec=ConfigLoader)
		mock_gen_schema = Mock(spec=GenSchema)
		mock_gen_schema.output_dir = str(abs_output_dir)
		type(mock_config).get = PropertyMock(return_value=MagicMock(gen=mock_gen_schema))

		with patch("datetime.datetime") as mock_datetime:
			mock_now = MagicMock()
			mock_datetime.now.return_value = mock_now
			mock_now.strftime.return_value = "20240101_140000"

			expected_path = abs_output_dir / "documentation_20240101_140000.md"
			result = determine_output_path(project_root, mock_config, None)

			assert result == expected_path
			mock_mkdir.assert_called()

	def test_determine_output_path_cli_overrides_config(self) -> None:
		"""Test that CLI output overrides config output_dir."""
		project_root = self.temp_dir / "my_proj"
		project_root.mkdir()
		output_arg = self.temp_dir / "cli_wins.md"

		# Create mock ConfigLoader with output_dir
		mock_config = Mock(spec=ConfigLoader)
		mock_gen_schema = Mock(spec=GenSchema)
		mock_gen_schema.output_dir = "config_dir"
		type(mock_config).get = PropertyMock(return_value=MagicMock(gen=mock_gen_schema))

		result = determine_output_path(project_root, mock_config, output_arg)
		assert result == output_arg.resolve()


@pytest.mark.unit
@pytest.mark.fs
class TestWriteDocumentation(FileSystemTestBase):
	"""Test cases for write_documentation."""

	@patch("codemap.utils.cli_utils.console")
	@patch("codemap.utils.file_utils.ensure_directory_exists")
	def test_write_documentation_success(self, mock_ensure_dir: MagicMock, mock_console: MagicMock) -> None:
		"""Test writing documentation successfully."""
		output_path = self.temp_dir / "docs" / "output.md"
		documentation = "# My Awesome Docs\\n\\nThis is the content."
		# Manually create the parent directory because ensure_directory_exists is mocked
		output_path.parent.mkdir(parents=True, exist_ok=True)

		write_documentation(output_path, documentation)

		mock_ensure_dir.assert_called_once_with(output_path.parent)
		assert output_path.read_text() == documentation
		mock_console.print.assert_called_once_with(f"[green]Documentation written to {output_path}")

	@patch("codemap.utils.cli_utils.console")
	@patch("codemap.utils.file_utils.ensure_directory_exists")
	@patch("pathlib.Path.write_text", side_effect=PermissionError("Cannot write"))
	def test_write_documentation_permission_error(
		self, mock_write: MagicMock, mock_ensure_dir: MagicMock, mock_console: MagicMock
	) -> None:
		"""Test handling PermissionError during writing."""
		output_path = self.temp_dir / "no_access" / "output.md"
		documentation = "Some docs"

		with pytest.raises(PermissionError):
			write_documentation(output_path, documentation)

		mock_ensure_dir.assert_called_once_with(output_path.parent)
		mock_write.assert_called_once_with(documentation)
		mock_console.print.assert_not_called()

	@patch("codemap.utils.cli_utils.console")
	@patch("codemap.utils.file_utils.ensure_directory_exists", side_effect=OSError("Cannot create dir"))
	def test_write_documentation_ensure_dir_error(self, mock_ensure_dir: MagicMock, mock_console: MagicMock) -> None:
		"""Test handling OSError during directory creation."""
		output_path = self.temp_dir / "bad_dir" / "output.md"
		documentation = "Some docs"

		with patch("pathlib.Path.write_text") as mock_write, pytest.raises(OSError, match="Cannot create dir"):
			write_documentation(output_path, documentation)

		mock_ensure_dir.assert_called_once_with(output_path.parent)
		mock_write.assert_not_called()
		mock_console.print.assert_not_called()
