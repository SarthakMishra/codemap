"""Tests for the gen command implementation."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Correct the import path based on actual project structure if necessary
from codemap.gen.command import GenCommand, process_codebase
from codemap.gen.models import GenConfig
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
			output_dir=temp_dir / "docs",  # Use temp_dir directly
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

		mock_generator_cls.assert_called_once_with(self.config, self.test_output_path)
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
			output_dir=temp_dir / "docs_proc",  # Use temp_dir directly
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

	@patch("codemap.gen.command.create_processor")
	@patch("codemap.gen.command.filter_paths_by_gitignore")
	@patch("codemap.gen.command.is_text_file")
	@patch("codemap.gen.command.generate_tree")
	def test_process_codebase_basic_flow(
		self,
		mock_generate_tree: MagicMock,
		mock_is_text_file: MagicMock,
		mock_filter_paths: MagicMock,
		mock_create_processor: MagicMock,
	) -> None:
		"""Test the basic successful flow of process_codebase."""
		# Arrange
		mock_pipeline = MagicMock()
		mock_pipeline.wait_for_completion.return_value = True
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
		mock_pipeline.processed_files = {
			self.test_target_path / "main.py": mock_entity_main,
			self.test_target_path / "utils.py": mock_entity_utils,
			self.test_target_path / "README.md": mock_entity_readme,
		}
		mock_create_processor.return_value = mock_pipeline

		# Simulate filtering - return only the processable files
		process_paths = [
			self.test_target_path / "main.py",
			self.test_target_path / "utils.py",
			self.test_target_path / "README.md",
		]

		mock_filter_paths.return_value = process_paths  # Return only processable files

		# Simulate is_text_file check
		mock_is_text_file.side_effect = lambda p: p.suffix in [".py", ".md"]

		mock_generate_tree.return_value = ["- main.py", "- utils.py", "- README.md"]

		# Act
		entities, metadata = process_codebase(self.test_target_path, self.config, self.mock_progress, self.mock_task_id)

		# Assert
		mock_create_processor.assert_called_once_with(
			repo_path=self.test_target_path, default_lod_level=self.config.lod_level
		)
		mock_filter_paths.assert_called_once()  # Check filter called
		assert mock_pipeline.process_file.call_count == 3  # main.py, utils.py, README.md
		mock_pipeline.process_file.assert_has_calls(
			[
				call(self.test_target_path / "main.py", self.config.lod_level),
				call(self.test_target_path / "utils.py", self.config.lod_level),
				call(self.test_target_path / "README.md", self.config.lod_level),
			],
			any_order=True,
		)

		mock_pipeline.wait_for_completion.assert_called_once()
		mock_pipeline.stop.assert_called_once()

		assert len(entities) == 3
		assert mock_entity_main in entities
		assert mock_entity_utils in entities
		assert mock_entity_readme in entities

		assert metadata["name"] == "my_code"
		assert metadata["stats"]["total_files"] == 3  # Based on is_text_file mock
		assert set(metadata["stats"]["languages"]) == {"python", "markdown"}
		assert "tree" in metadata  # Because include_tree=True
		mock_generate_tree.assert_called_once_with(self.test_target_path, process_paths)
		assert metadata["tree"] == ["- main.py", "- utils.py", "- README.md"]

		# Check progress updates
		self.mock_progress.update.assert_called()

	@patch("codemap.gen.command.create_processor", side_effect=RuntimeError("Init failed"))
	@patch("codemap.gen.command.show_error")
	def test_process_codebase_init_fails(
		self,
		mock_show_error: MagicMock,
		mock_create_processor: MagicMock,
	) -> None:
		"""Test process_codebase when processor initialization fails."""
		# Act & Assert
		with pytest.raises(RuntimeError, match="Processor initialization failed: Init failed"):
			process_codebase(self.test_target_path, self.config, self.mock_progress, self.mock_task_id)
		mock_create_processor.assert_called_once()
		mock_show_error.assert_called_once_with("Processor initialization failed: Init failed")
		self.mock_progress.update.assert_called_with(self.mock_task_id, description="Initializing processor...")

	@patch("codemap.gen.command.create_processor")
	@patch("codemap.gen.command.filter_paths_by_gitignore")
	@patch("codemap.gen.command.is_text_file", return_value=True)  # Assume all are text
	@patch("codemap.gen.command.logger")
	def test_process_codebase_wait_timeout(
		self,
		mock_logger: MagicMock,
		_mock_is_text: MagicMock,
		mock_filter: MagicMock,
		mock_create_processor: MagicMock,
	) -> None:
		"""Test process_codebase when pipeline wait times out (continues)."""
		# Arrange
		mock_pipeline = MagicMock()
		mock_pipeline.wait_for_completion.return_value = False  # Simulate timeout
		mock_pipeline.processed_files = {}  # No files processed in time
		mock_create_processor.return_value = mock_pipeline
		mock_filter.return_value = [self.test_target_path / "main.py"]  # Simulate one file

		# Act
		entities, metadata = process_codebase(self.test_target_path, self.config, self.mock_progress, self.mock_task_id)

		# Assert
		mock_pipeline.wait_for_completion.assert_called_once()
		mock_logger.warning.assert_called_once_with("Processing tasks did not complete within the expected time.")
		assert entities == []  # Should return empty list if timeout and no results yet
		assert metadata["stats"]["total_files"] == 1
		assert metadata["stats"]["total_lines"] == 0
		assert metadata["stats"]["languages"] == []
		mock_pipeline.stop.assert_called_once()
