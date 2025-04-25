"""Tests for the processing pipeline."""

from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from pytest_mock import MockerFixture

from codemap.processor.analysis.git.analyzer import GitMetadataAnalyzer
from codemap.processor.analysis.lsp.analyzer import LSPAnalyzer
from codemap.processor.chunking.base import Chunk, ChunkMetadata, EntityType, Location
from codemap.processor.chunking.tree_sitter import TreeSitterChunker
from codemap.processor.embedding.generator import EmbeddingGenerator
from codemap.processor.embedding.models import EmbeddingResult
from codemap.processor.pipeline import ProcessingJob, ProcessingPipeline
from codemap.processor.watcher import FileWatcher


@pytest.mark.processor
class TestPipeline:
	"""Tests for the processing pipeline."""

	@pytest.fixture
	def temp_repo_dir(self) -> Generator[Path, None, None]:
		"""Create a temporary directory for the test repo."""
		with TemporaryDirectory() as temp_dir:
			yield Path(temp_dir)

	@pytest.fixture
	def mock_git_analyzer(self) -> MagicMock:
		"""Create a mock GitMetadataAnalyzer."""
		mock_git = MagicMock(spec=GitMetadataAnalyzer)
		mock_git.get_current_commit.return_value = "mock-commit-id"
		mock_git.get_current_branch.return_value = "mock-branch"
		return mock_git

	@pytest.fixture
	def mock_lsp_analyzer(self) -> MagicMock:
		"""Create a mock LSP analyzer."""
		mock_lsp = MagicMock(spec=LSPAnalyzer)
		mock_lsp.enrich_chunks.return_value = {"chunk1": MagicMock()}
		return mock_lsp

	@pytest.fixture
	def mock_chunker(self) -> MagicMock:
		"""Create a mock chunker."""
		mock_chunker = MagicMock(spec=TreeSitterChunker)

		# Create a sample chunk for testing
		metadata = ChunkMetadata(
			entity_type=EntityType.CLASS,
			name="TestClass",
			location=Location(
				file_path=Path("test.py"),
				start_line=1,
				end_line=10,
			),
			language="python",
		)
		test_chunk = Chunk(content="class TestClass:", metadata=metadata)

		mock_chunker.chunk.return_value = [test_chunk]
		return mock_chunker

	@pytest.fixture
	def mock_embedding_generator(self) -> MagicMock:
		"""Create a mock embedding generator."""
		mock_generator = MagicMock(spec=EmbeddingGenerator)

		# Mock an embedding result with numpy array to satisfy type check
		embedding_result = EmbeddingResult(
			chunk_id="test_chunk_id",
			embedding=np.array([0.1, 0.2, 0.3]),  # Use numpy array instead of list
			model="test-model",
			content="test content",
			tokens=10,
		)

		mock_generator.generate_embeddings.return_value = [embedding_result]
		mock_generator.generate_embedding.return_value = embedding_result
		return mock_generator

	@pytest.fixture
	def mock_storage(self) -> MagicMock:
		"""Create a mock storage backend."""
		mock_storage = MagicMock()
		mock_storage.search_by_vector.return_value = [(MagicMock(), 0.9)]
		mock_storage.search_by_text.return_value = [(MagicMock(), 0.8)]
		return mock_storage

	@pytest.fixture
	def mock_file_watcher(self) -> MagicMock:
		"""Create a mock file watcher."""
		return MagicMock(spec=FileWatcher)

	@pytest.fixture
	def test_file(self, temp_repo_dir: Path) -> Path:
		"""Create a test file in the temporary directory."""
		test_file = temp_repo_dir / "test.py"
		with Path.open(test_file, "w") as f:
			f.write("# Test file\nclass TestClass:\n    pass\n")
		return test_file

	@pytest.fixture
	def pipeline(
		self,
		temp_repo_dir: Path,
		mock_git_analyzer: MagicMock,
		mock_lsp_analyzer: MagicMock,
		mock_chunker: MagicMock,
		mock_embedding_generator: MagicMock,
		mock_storage: MagicMock,
		mock_file_watcher: MagicMock,
	) -> Generator[ProcessingPipeline, None, None]:
		"""Create a pipeline with mocked components for testing."""
		with (
			patch("codemap.processor.pipeline.GitMetadataAnalyzer", return_value=mock_git_analyzer),
			patch("codemap.processor.pipeline.LSPAnalyzer", return_value=mock_lsp_analyzer),
			patch("codemap.processor.pipeline.TreeSitterChunker", return_value=mock_chunker),
			patch("codemap.processor.pipeline.EmbeddingGenerator", return_value=mock_embedding_generator),
			patch("codemap.processor.pipeline.LanceDBStorage", return_value=mock_storage),
			patch("codemap.processor.pipeline.FileWatcher", return_value=mock_file_watcher),
		):
			pipeline = ProcessingPipeline(repo_path=temp_repo_dir)

			# Directly set mocks for verification in tests
			pipeline.chunker = mock_chunker
			pipeline.git_analyzer = mock_git_analyzer
			pipeline.lsp_analyzer = mock_lsp_analyzer
			pipeline.embedding_generator = mock_embedding_generator
			pipeline.storage = mock_storage
			pipeline.watcher = mock_file_watcher

			yield pipeline

			# Cleanup
			pipeline.stop()

	@pytest.mark.asynchronous
	def test_start_stop_pipeline(self, pipeline: ProcessingPipeline, mock_file_watcher: MagicMock) -> None:
		"""Test starting and stopping the pipeline."""
		# Start the pipeline
		pipeline.start()

		# Verify the watcher was started
		mock_file_watcher.start.assert_called_once()

		# Stop the pipeline
		pipeline.stop()

		# Verify cleanup was called
		mock_file_watcher.stop.assert_called_once()
		assert pipeline.executor._shutdown is True

	@pytest.mark.watcher
	def test_watcher_event_handlers(self, pipeline: ProcessingPipeline) -> None:
		"""Test the file watcher event handlers."""
		# Mock process_file method to avoid actual processing
		with patch.object(pipeline, "process_file") as mock_process, patch("pathlib.Path.exists", new=lambda _: True):
			# Test file creation handler
			pipeline._handle_file_created("test_file.py")
			mock_process.assert_called_with("test_file.py")

			# Test file modification handler
			pipeline._handle_file_modified("test_file.py")
			mock_process.assert_called_with("test_file.py")

		# Test file deletion handler
		with (
			patch.object(pipeline.storage, "delete_file") as mock_delete,
			patch("pathlib.Path.exists", new=lambda _: False),
		):
			pipeline._handle_file_deleted("test_file.py")
			mock_delete.assert_called_with("test_file.py")

			# Verify job status
			job = pipeline.active_jobs.get(Path("test_file.py"))
			assert job is not None
			assert job.is_deletion is True
			assert job.completed_at is not None

	def test_process_file_and_job_status(self, pipeline: ProcessingPipeline, test_file: Path) -> None:
		"""Test file processing and job status tracking."""
		# Process the test file
		with (
			patch.object(pipeline, "_process_file_worker") as mock_worker,
			patch("pathlib.Path.exists", new=lambda _: True),
		):
			pipeline.process_file(test_file)

			# Verify job was created
			job = pipeline.get_job_status(test_file)
			assert job is not None
			assert job.file_path == test_file
			assert job.is_deletion is False

			# Verify worker was called
			mock_worker.assert_called_with(test_file)

	@pytest.mark.asynchronous
	def test_batch_processing(self, pipeline: ProcessingPipeline, temp_repo_dir: Path) -> None:
		"""Test batch processing of multiple files."""
		# Create multiple test files
		test_files = [
			temp_repo_dir / "test1.py",
			temp_repo_dir / "test2.py",
			temp_repo_dir / "test3.py",
		]

		for file_path in test_files:
			with Path.open(file_path, "w") as f:
				f.write(f"# Test file {file_path.name}\n")

		# Mock process_file to avoid actual processing
		with patch.object(pipeline, "process_file") as mock_process, patch("pathlib.Path.exists", new=lambda _: True):
			# Batch process the files - cast to compatible type
			pipeline.batch_process(cast("list[str | Path]", test_files))

			# Verify each file was processed
			assert mock_process.call_count == 3
			mock_process.assert_has_calls([call(file) for file in test_files])

	@pytest.mark.storage
	def test_search_functionality(self, pipeline: ProcessingPipeline, mocker: MockerFixture) -> None:
		"""Test the search functionality of the pipeline."""
		# Create method mocks explicitly
		gen_embed_mock = mocker.patch.object(
			pipeline.embedding_generator,
			"generate_embedding",
			return_value=EmbeddingResult(
				chunk_id="test_chunk_id",
				embedding=np.array([0.1, 0.2, 0.3]),
				model="test-model",
				content="test content",
				tokens=10,
			),
		)
		search_vector_mock = mocker.patch.object(
			pipeline.storage, "search_by_vector", return_value=[(MagicMock(), 0.9)]
		)
		search_text_mock = mocker.patch.object(pipeline.storage, "search_by_text", return_value=[(MagicMock(), 0.8)])

		# Test vector search (default)
		with patch("pathlib.Path.exists", new=lambda _: True):
			pipeline.search("test query")

			# Verify vector search was used
			gen_embed_mock.assert_called_once()
			search_vector_mock.assert_called_once()

			# Test text search
			gen_embed_mock.reset_mock()
			search_vector_mock.reset_mock()

			pipeline.search("test query", use_vector=False)

			# Verify text search was used instead
			gen_embed_mock.assert_not_called()
			search_vector_mock.assert_not_called()
			search_text_mock.assert_called_once()

	@pytest.mark.asynchronous
	def test_cleanup_job(self, pipeline: ProcessingPipeline) -> None:
		"""Test the job cleanup functionality."""
		# Create a test job
		test_file = Path("test_cleanup.py")
		job = ProcessingJob(file_path=test_file)
		pipeline.active_jobs[test_file] = job

		# Run cleanup with no delay
		pipeline._cleanup_job(test_file, delay=0)

		# Verify job was removed
		assert test_file not in pipeline.active_jobs

	@pytest.mark.error_handling
	def test_error_handling_in_process_file(
		self, pipeline: ProcessingPipeline, test_file: Path, mocker: MockerFixture
	) -> None:
		"""Test error handling in the file processing workflow."""
		# Setup chunker to raise an exception
		mocker.patch.object(pipeline.chunker, "chunk", side_effect=ValueError("Test error"))

		# Create job explicitly before processing to ensure it exists
		job = ProcessingJob(file_path=test_file)
		pipeline.active_jobs[test_file] = job

		# Patch the cleanup to prevent it from removing the job
		with patch.object(pipeline, "_cleanup_job"), patch("pathlib.Path.exists", new=lambda _: True):
			# Process file directly (not through thread)
			pipeline._process_file_worker(test_file)

			# Verify job status reflects the error
			job = pipeline.get_job_status(test_file)
			assert job is not None
			assert job.error is not None
			assert isinstance(job.error, ValueError)
			assert str(job.error) == "Test error"
			assert job.completed_at is not None

	@pytest.mark.asynchronous
	def test_callback_execution(self, pipeline: ProcessingPipeline, test_file: Path, mocker: MockerFixture) -> None:
		"""Test that callbacks are executed properly."""
		# Setup callbacks
		on_chunks_processed = MagicMock()
		on_file_deleted = MagicMock()

		pipeline.on_chunks_processed = on_chunks_processed
		pipeline.on_file_deleted = on_file_deleted

		# Create job explicitly before processing to ensure it exists
		job = ProcessingJob(file_path=test_file)
		pipeline.active_jobs[test_file] = job

		# Mock successful chunking
		test_chunk = MagicMock(spec=Chunk)
		mocker.patch.object(pipeline.chunker, "chunk", return_value=[test_chunk])

		# Force the embedding generation to return a valid result
		mocker.patch.object(pipeline.embedding_generator, "generate_embeddings", return_value=[MagicMock()])

		# Process a file
		with (
			patch.object(pipeline, "_cleanup_job"),
			patch("pathlib.Path.exists", new=lambda _: True),
		):  # Prevent actual cleanup
			pipeline._process_file_worker(test_file)

			# Verify chunks processed callback was called with correct arguments
			on_chunks_processed.assert_called_once_with([test_chunk], test_file)

			# Test file deletion callback
			with patch("pathlib.Path.exists", new=lambda _: False):
				pipeline._handle_file_deleted(str(test_file))
				on_file_deleted.assert_called_once()

	@pytest.mark.error_handling
	@pytest.mark.storage
	def test_file_deletion_error_handling(self, pipeline: ProcessingPipeline, mocker: MockerFixture) -> None:
		"""Test error handling during file deletion."""
		# Setup storage to raise an exception during deletion
		mocker.patch.object(pipeline.storage, "delete_file", side_effect=RuntimeError("Deletion error"))

		# Handle file deletion
		with patch("pathlib.Path.exists", new=lambda _: False):
			pipeline._handle_file_deleted("test_error.py")

			# Verify error was captured in job
			job = pipeline.active_jobs.get(Path("test_error.py"))
			assert job is not None
			assert job.error is not None
			assert isinstance(job.error, RuntimeError)
			assert str(job.error) == "Deletion error"

	@pytest.mark.error_handling
	@pytest.mark.storage
	def test_search_error_handling(self, pipeline: ProcessingPipeline, mocker: MockerFixture) -> None:
		"""Test error handling in the search functionality."""
		# Setup embedding generator to raise an exception
		mocker.patch.object(
			pipeline.embedding_generator, "generate_embedding", side_effect=ValueError("Embedding error")
		)
		search_text_mock = mocker.patch.object(pipeline.storage, "search_by_text", return_value=[(MagicMock(), 0.8)])

		# Search should fall back to text search when vector search fails
		with patch("pathlib.Path.exists", new=lambda _: True):
			pipeline.search("test query")

			# Verify fallback to text search
			search_text_mock.assert_called_once()
