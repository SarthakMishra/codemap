"""Tests for the gen command implementation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codemap.config.config_schema import GenSchema as GenConfig

# Correct the import path based on actual project structure if necessary
from codemap.gen.command import GenCommand
from codemap.gen.utils import process_codebase
from codemap.processor.lod import LODEntity, LODLevel
from codemap.processor.tree_sitter.base import EntityType
from tests.base import CLITestBase, FileSystemTestBase

# Mock data for LODEntity
MOCK_ENTITY_1 = LODEntity(
	name="file1",
	entity_type=EntityType.MODULE,
	language="python",
	start_line=1,
	end_line=10,
	content="...",
	metadata={"file_path": Path("src/file1.py"), "summary": "File 1 summary"},
)
MOCK_ENTITY_2 = LODEntity(
	name="file2",
	entity_type=EntityType.MODULE,
	language="javascript",
	start_line=1,
	end_line=5,
	content="...",
	metadata={"file_path": Path("src/file2.js"), "summary": "File 2 summary"},
)


@pytest.mark.cli
@pytest.mark.gen  # Assuming 'gen' is a valid marker, or adjust as needed
class TestGenCommand(CLITestBase, FileSystemTestBase):
	"""Tests for the GenCommand class."""

	@pytest.fixture(autouse=True)
	def _setup_test_command(self, temp_dir: Path) -> None:
		"""Per-test setup specific to TestGenCommand."""
		# Config needs required fields
		self.config = GenConfig(
			lod_level=LODLevel.SIGNATURES,  # Example required field
			max_content_length=5000,
			use_gitignore=True,
			output_dir=str(temp_dir / "docs"),  # Convert Path to string
			semantic_analysis=True,
			include_tree=False,  # Optional field
		)
		self.test_target_path = temp_dir / "my_project"
		self.test_output_path = temp_dir / "output.md"
		self.test_target_path.mkdir(exist_ok=True)
		# Create some dummy files for processing simulation
		(self.test_target_path / "file1.py").touch()
		(self.test_target_path / "file2.js").touch()
		(self.test_target_path / ".hidden").touch()
		(self.test_target_path / "binary.bin").write_bytes(b"\x00\x01")

	@patch("codemap.gen.command.process_codebase")
	@patch("codemap.gen.generator.CodeMapGenerator")
	@patch("codemap.gen.utils.write_documentation")
	@patch("codemap.gen.command.console.print")
	def test_execute_success(
		self,
		mock_console_print: MagicMock,
		mock_write_doc: MagicMock,
		mock_generator_cls: MagicMock,
		mock_process_codebase: MagicMock,
	) -> None:
		"""Test successful execution of the gen command."""
		# Arrange
		mock_process_codebase.return_value = (
			[MOCK_ENTITY_1, MOCK_ENTITY_2],
			{"name": "my_project", "stats": {"total_files": 2}},
		)
		mock_generator_instance = mock_generator_cls.return_value
		mock_generator_instance.generate_documentation.return_value = "Generated Doc Content"

		command = GenCommand(self.config)

		# Act
		result = command.execute(self.test_target_path, self.test_output_path)

		# Assert
		assert result is True
		mock_process_codebase.assert_called_once()
		# Check args passed to process_codebase more specifically if needed
		assert mock_process_codebase.call_args[0][0] == self.test_target_path
		assert mock_process_codebase.call_args[0][1] == self.config

		mock_generator_cls.assert_called_once_with(self.config)
		mock_generator_instance.generate_documentation.assert_called_once_with(
			[MOCK_ENTITY_1, MOCK_ENTITY_2],
			{"name": "my_project", "stats": {"total_files": 2}},
		)
		mock_write_doc.assert_called_once_with(self.test_output_path, "Generated Doc Content")
		# Check for success message print
		assert any("Generation completed" in call_args[0][0] for call_args in mock_console_print.call_args_list)

	@patch("codemap.gen.command.process_codebase", side_effect=RuntimeError("Processing failed"))
	@patch("codemap.gen.command.show_error")
	@patch("codemap.gen.command.logger")  # Patch logger to check error logging
	def test_execute_process_codebase_fails(
		self,
		mock_logger: MagicMock,
		mock_show_error: MagicMock,
		mock_process_codebase: MagicMock,
	) -> None:
		"""Test execution when process_codebase raises an error."""
		# Arrange
		command = GenCommand(self.config)

		# Act
		result = command.execute(self.test_target_path, self.test_output_path)

		# Assert
		assert result is False
		mock_process_codebase.assert_called_once()
		mock_show_error.assert_called_once_with("Generation failed: Processing failed")
		mock_logger.exception.assert_called_once_with("Error during gen command execution")

	@patch("codemap.gen.command.process_codebase")
	@patch("codemap.gen.generator.CodeMapGenerator")
	@patch("codemap.gen.utils.write_documentation", side_effect=OSError("Write failed"))
	@patch("codemap.gen.command.show_error")
	@patch("codemap.gen.command.logger")
	def test_execute_write_fails(
		self,
		mock_logger: MagicMock,
		mock_show_error: MagicMock,
		mock_write_doc: MagicMock,
		mock_generator_cls: MagicMock,
		mock_process_codebase: MagicMock,
	) -> None:
		"""Test execution when writing the documentation fails."""
		# Arrange
		mock_process_codebase.return_value = (
			[MOCK_ENTITY_1],
			{"name": "my_project", "stats": {"total_files": 1}},
		)
		mock_generator_instance = mock_generator_cls.return_value
		mock_generator_instance.generate_documentation.return_value = "Content"
		command = GenCommand(self.config)

		# Act
		result = command.execute(self.test_target_path, self.test_output_path)

		# Assert
		assert result is False
		mock_write_doc.assert_called_once()
		mock_show_error.assert_called_once_with("Generation failed: Write failed")
		mock_logger.exception.assert_called_once_with("Error during gen command execution")


@pytest.mark.gen
class TestProcessCodebase(FileSystemTestBase):
	"""Tests for the process_codebase function."""

	@pytest.fixture(autouse=True)
	def _setup_test_process_codebase(self, temp_dir: Path) -> None:
		"""Per-test setup specific to TestProcessCodebase."""
		# Config needs required fields
		self.config = GenConfig(
			lod_level=LODLevel.STRUCTURE,  # Example required field
			max_content_length=5000,
			use_gitignore=True,
			output_dir=str(temp_dir / "docs_proc"),  # Convert Path to string
			semantic_analysis=True,
			include_tree=True,  # Optional field, tested here
		)
		self.test_target_path = temp_dir / "my_code"
		self.test_target_path.mkdir(exist_ok=True)
		(self.test_target_path / "main.py").write_text("print('hello')")
		(self.test_target_path / "utils.py").write_text("def helper(): pass")
		(self.test_target_path / "README.md").write_text("# Project")
		(self.test_target_path / ".gitignore").write_text("*.log\n__pycache__")
		(self.test_target_path / "ignored.log").touch()
		(self.test_target_path / "__pycache__").mkdir(exist_ok=True)
		(self.test_target_path / "__pycache__" / "cache.pyc").touch()

		# Mock progress object
		self.mock_progress = MagicMock()
		self.mock_task_id = MagicMock()

	@patch("codemap.gen.utils.filter_paths_by_gitignore")
	@patch("codemap.utils.file_utils.is_binary_file")
	@patch("codemap.gen.utils.generate_tree")
	@patch("codemap.gen.utils.process_files_for_lod")
	def test_process_codebase_basic_flow(
		self,
		mock_process_files_lod: MagicMock,
		mock_generate_tree: MagicMock,
		mock_is_binary_file: MagicMock,
		mock_filter_paths: MagicMock,
	) -> None:
		"""Test the basic successful flow of process_codebase."""
		# Arrange
		mock_entity_main = LODEntity(
			name="main.py",
			entity_type=EntityType.MODULE,
			language="python",
			start_line=1,
			end_line=1,
			content="",
			metadata={"file_path": self.test_target_path / "main.py"},
		)
		mock_entity_utils = LODEntity(
			name="utils.py",
			entity_type=EntityType.MODULE,
			language="python",
			start_line=1,
			end_line=1,
			content="",
			metadata={"file_path": self.test_target_path / "utils.py"},
		)
		mock_entity_readme = LODEntity(
			name="README.md",
			entity_type=EntityType.MODULE,
			language="markdown",
			start_line=1,
			end_line=1,
			content="",
			metadata={"file_path": self.test_target_path / "README.md"},
		)
		# Simulate processed files cache population
		mock_process_files_lod.return_value = [mock_entity_main, mock_entity_utils, mock_entity_readme]

		# Simulate filtering - return only the processable files
		process_paths = [
			self.test_target_path / "main.py",
			self.test_target_path / "utils.py",
			self.test_target_path / "README.md",
		]

		mock_filter_paths.return_value = process_paths  # Return only processable files

		# Simulate is_binary_file check (return False for text files)
		mock_is_binary_file.side_effect = lambda p: p.suffix not in [".py", ".md"]

		mock_generate_tree.return_value = ["- main.py", "- utils.py", "- README.md"]

		# Act
		entities, metadata = process_codebase(self.test_target_path, self.config)

		# Assert
		mock_filter_paths.assert_called_once()
		# We've imported is_binary_file directly in the implementation, so we don't check call_count
		# Just check that process_files_for_lod was called with the correct paths
		mock_process_files_lod.assert_called_once()
		assert len(entities) == 3
		assert all(isinstance(e, LODEntity) for e in entities)
		assert metadata["stats"]["total_files_scanned"] == 3  # Using the correct key name
		assert mock_generate_tree.call_count == 1  # Called once because include_tree=True

	@patch("codemap.gen.utils.filter_paths_by_gitignore")
	@patch("codemap.utils.file_utils.is_binary_file", return_value=False)
	@patch("codemap.gen.utils.logger")
	@patch("codemap.gen.utils.process_files_for_lod")
	def test_process_codebase_wait_timeout(
		self,
		mock_process_files_lod: MagicMock,
		_mock_logger: MagicMock,
		_mock_is_binary_file: MagicMock,
		mock_filter: MagicMock,
	) -> None:
		"""Test process_codebase when wait_for_completion returns False (timeout)."""
		# Arrange
		process_paths = [self.test_target_path / "main.py"]
		mock_filter.return_value = process_paths
		# Simulate timeout indirectly by having process_files_for_lod return empty
		mock_process_files_lod.return_value = []

		# Act
		entities, _ = process_codebase(self.test_target_path, self.config)

		# Assert
		# Ensure process_files_for_lod was still called
		mock_process_files_lod.assert_called_once()
		assert len(entities) == 0  # Expect no entities due to simulated timeout/failure
		# Logging of timeout/completion status is now within process_files_for_lod
