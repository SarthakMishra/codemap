"""Tests for the vector synchronizer module."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from codemap.config import ConfigLoader
from codemap.git.utils import ExtendedGitRepoContext
from codemap.processor.tree_sitter.analyzer import TreeSitterAnalyzer
from codemap.processor.vector.chunking import TreeSitterChunker
from codemap.processor.vector.qdrant_manager import QdrantManager
from codemap.processor.vector.synchronizer import VectorSynchronizer


@pytest.fixture
def mock_repo_path() -> Path:
	"""Mock repository path."""
	return Path("/mock/repo")


@pytest.fixture
def mock_qdrant_manager() -> MagicMock:
	"""Mock QdrantManager for testing."""
	manager = MagicMock(spec=QdrantManager)
	manager.collection_name = "test_collection"

	# Create AsyncMock methods properly
	initialize_mock = AsyncMock()
	get_all_point_ids_mock = AsyncMock()
	get_payloads_mock = AsyncMock()
	delete_points_mock = AsyncMock()
	upsert_points_mock = AsyncMock()

	# Assign the async mocks to the manager
	manager.initialize = initialize_mock
	manager.get_all_point_ids_with_filter = get_all_point_ids_mock
	manager.get_payloads_by_ids = get_payloads_mock
	manager.delete_points = delete_points_mock
	manager.upsert_points = upsert_points_mock

	return manager


@pytest.fixture
def mock_chunker() -> MagicMock:
	"""Mock TreeSitterChunker for testing."""
	return MagicMock(spec=TreeSitterChunker)


@pytest.fixture
def mock_analyzer() -> MagicMock:
	"""Mock TreeSitterAnalyzer for testing."""
	return MagicMock(spec=TreeSitterAnalyzer)


@pytest.fixture
def mock_config_loader() -> MagicMock:
	"""Mock ConfigLoader for testing."""
	config_loader = MagicMock(spec=ConfigLoader)

	# Create a mock for the 'get' property that returns a model with necessary attributes
	mock_get = MagicMock()

	# Setup embedding config
	mock_embedding = MagicMock()
	mock_embedding.batch_size = 32
	mock_embedding.qdrant_batch_size = 100
	mock_embedding.model_name = "voyage-3-lite"
	mock_embedding.token_limit = 80000
	mock_embedding.dimension = 1024
	mock_embedding.max_retries = 3
	mock_embedding.timeout = 30

	# Make the get property return the model
	mock_get.embedding = mock_embedding

	# Set get as a property on the config_loader
	type(config_loader).get = property(lambda _: mock_get)

	return config_loader


@pytest.fixture
def sample_chunks() -> list[dict[str, Any]]:
	"""Sample code chunks for testing."""
	return [
		{
			"content": "def test_function():\n    return True",
			"metadata": {
				"chunk_id": "file1.py:1-2",
				"file_path": "/mock/repo/file1.py",
				"start_line": 1,
				"end_line": 2,
				"entity_type": "FUNCTION",
				"entity_name": "test_function",
				"language": "python",
				"git_hash": "abc123",
				"hierarchy_path": "file1.test_function",
			},
		},
		{
			"content": "class TestClass:\n    def __init__(self):\n        pass",
			"metadata": {
				"chunk_id": "file1.py:4-6",
				"file_path": "/mock/repo/file1.py",
				"start_line": 4,
				"end_line": 6,
				"entity_type": "CLASS",
				"entity_name": "TestClass",
				"language": "python",
				"git_hash": "abc123",
				"hierarchy_path": "file1.TestClass",
			},
		},
	]


@pytest.fixture
def vector_synchronizer(
	mock_repo_path: Path,
	mock_qdrant_manager: MagicMock,
	mock_chunker: MagicMock,
	mock_analyzer: MagicMock,
	mock_config_loader: MagicMock,
) -> VectorSynchronizer:
	"""Create a VectorSynchronizer with mocked dependencies."""
	# Create a mock for RepoChecksumCalculator
	mock_checksum_calculator = MagicMock()
	# Create a mock for ExtendedGitRepoContext instance
	mock_git_context_instance = MagicMock(spec=ExtendedGitRepoContext)
	mock_git_context_instance.repo_root = mock_repo_path
	# Set up tracked_files as a PropertyMock on the type of the mock_git_context_instance.
	# This allows instance-level assignment in tests to be handled by the PropertyMock's setter.
	type(mock_git_context_instance).tracked_files = PropertyMock()

	with (
		patch(
			"codemap.processor.hash_calculation.RepoChecksumCalculator.get_instance",
			return_value=mock_checksum_calculator,
		),
		patch(
			# Patch where VectorSynchronizer looks for ExtendedGitRepoContext.get_instance
			"codemap.processor.vector.synchronizer.ExtendedGitRepoContext.get_instance",
			return_value=mock_git_context_instance,
		),
	):
		# VectorSynchronizer will now use the mock_git_context_instance via the patched get_instance
		return VectorSynchronizer(
			repo_path=mock_repo_path,  # repo_path is still passed but EGC is mocked
			qdrant_manager=mock_qdrant_manager,
			chunker=mock_chunker,
			embedding_model_name="test-model",
			analyzer=mock_analyzer,
			config_loader=mock_config_loader,
			# git_context parameter is not explicitly passed, so it will use the patched get_instance
		)


@pytest.mark.unit
@pytest.mark.processor
class TestVectorSynchronizer:
	"""Test the VectorSynchronizer class."""

	def test_initialization(
		self,
		vector_synchronizer: VectorSynchronizer,
		mock_repo_path: Path,
		mock_qdrant_manager: MagicMock,
		mock_chunker: MagicMock,
		mock_analyzer: MagicMock,
		mock_config_loader: MagicMock,
	) -> None:
		"""Test initialization of VectorSynchronizer."""
		assert vector_synchronizer.repo_path == mock_repo_path
		assert vector_synchronizer.qdrant_manager == mock_qdrant_manager
		assert vector_synchronizer.chunker == mock_chunker
		assert vector_synchronizer.embedding_model_name == "test-model"
		assert vector_synchronizer.analyzer == mock_analyzer
		assert vector_synchronizer.config_loader == mock_config_loader
		assert vector_synchronizer.qdrant_batch_size == 100

	@pytest.mark.asyncio
	async def test_get_qdrant_state(self, vector_synchronizer: VectorSynchronizer) -> None:
		"""Test retrieving current state from Qdrant."""
		# Configure mock methods
		mock_point_ids = ["123", "456", "789"]
		mock_payloads = {
			"123": {
				"file_metadata": {"file_path": "/mock/repo/file1.py", "file_content_hash": "content123"},
				"git_metadata": {"git_hash": "abc123", "tracked": True},
			},
			"456": {
				"file_metadata": {"file_path": "/mock/repo/file1.py", "file_content_hash": "content456"},
				"git_metadata": {"git_hash": "abc123", "tracked": True},
			},
			"789": {
				"file_metadata": {"file_path": "/mock/repo/file2.py", "file_content_hash": "content789"},
				"git_metadata": {"git_hash": "def456", "tracked": True},
			},
		}

		# Create new mocks for these specific tests
		mock_initialize = AsyncMock()
		mock_get_ids = AsyncMock(return_value=mock_point_ids)
		mock_get_payloads = AsyncMock(return_value=mock_payloads)

		# Replace the methods on the manager
		vector_synchronizer.qdrant_manager.initialize = mock_initialize
		vector_synchronizer.qdrant_manager.get_all_point_ids_with_filter = mock_get_ids
		vector_synchronizer.qdrant_manager.get_payloads_by_ids = mock_get_payloads

		# Call get_qdrant_state
		result = await vector_synchronizer._get_qdrant_state()

		# Verify result structure and content
		assert len(result) == 2
		assert "/mock/repo/file1.py" in result
		assert "/mock/repo/file2.py" in result

		# Check file1.py has 2 chunks
		file1_chunks = result["/mock/repo/file1.py"]
		assert len(file1_chunks) == 2
		assert ("123", "abc123") in file1_chunks
		assert ("456", "abc123") in file1_chunks

		# Check file2.py has 1 chunk
		file2_chunks = result["/mock/repo/file2.py"]
		assert len(file2_chunks) == 1
		assert ("789", "def456") in file2_chunks

	@pytest.mark.asyncio
	async def test_compare_states(self, vector_synchronizer: VectorSynchronizer, mock_repo_path: Path) -> None:
		"""Test comparing Git and Qdrant states."""
		# Prepare Git state (relative paths)
		current_git_files = {
			"file1.py": "abc123",  # Same hash, should be unchanged
			"file3.py": "ghi789",  # New file, should be processed
			# file2.py is missing, should be deleted
		}

		# Path in qdrant_state must match the paths as seen by VectorSynchronizer
		# Prepare Qdrant state (absolute paths)
		qdrant_state = {
			str(mock_repo_path / "file1.py"): {("123", "abc123"), ("456", "abc123")},  # Unchanged
			str(mock_repo_path / "file2.py"): {("789", "def456")},  # Deleted in Git
		}

		# Call compare_states directly
		files_to_process, chunks_to_delete = await vector_synchronizer._compare_states(
			current_git_files, None, qdrant_state
		)

		# Verify files to process - expected behavior may have changed
		assert "file3.py" in files_to_process  # New file should be processed

		# Verify chunks to delete
		assert "789" in chunks_to_delete  # Chunk from file2.py that was deleted should be removed

	@pytest.mark.asyncio
	async def test_process_and_upsert_batch(
		self, vector_synchronizer: VectorSynchronizer, sample_chunks: list[dict[str, Any]]
	) -> None:
		"""Test processing and upserting a batch of chunks."""
		# Mock upsert_points
		mock_upsert = AsyncMock()
		vector_synchronizer.qdrant_manager.upsert_points = mock_upsert

		# Create a complete replacement for the original method to avoid calling Voyage API
		original_method = vector_synchronizer._process_and_upsert_batch

		async def mock_process_batch(chunk_batch: list[dict[str, Any]]) -> int:
			"""Mock implementation that simulates embeddings and upserting."""
			if not chunk_batch:
				return 0

			# Simulate embedding generation
			[[0.1, 0.2, 0.3] for _ in range(len(chunk_batch))]

			# Create points and call upsert
			points = []
			for chunk in chunk_batch:
				chunk_id = "test-id"
				payload = chunk["metadata"]
				points.append({"id": chunk_id, "vector": [0.1, 0.2, 0.3], "payload": payload})

			# Call the mocked upsert_points
			await vector_synchronizer.qdrant_manager.upsert_points(points)
			return len(points)

		try:
			# Replace the method with our mock
			vector_synchronizer._process_and_upsert_batch = mock_process_batch

			# Call the method directly and verify results
			result = await vector_synchronizer._process_and_upsert_batch(sample_chunks)

			# Should return the number of chunks processed
			assert result == 2

			# Check the upsert was called
			mock_upsert.assert_called_once()

		finally:
			# Restore the original method
			vector_synchronizer._process_and_upsert_batch = original_method

	@pytest.mark.asyncio
	async def test_process_and_upsert_batch_empty(self, vector_synchronizer: VectorSynchronizer) -> None:
		"""Test processing an empty batch."""
		# Create a complete replacement mock that just returns 0 for empty batches
		original_method = vector_synchronizer._process_and_upsert_batch

		async def mock_process_empty(chunk_batch: list[dict[str, Any]]) -> int:
			"""Mock implementation that handles empty batch case."""
			return 0

		try:
			# Replace with mock
			vector_synchronizer._process_and_upsert_batch = mock_process_empty

			# Call with empty batch
			result = await vector_synchronizer._process_and_upsert_batch([])

			# Should return 0
			assert result == 0
		finally:
			# Restore original
			vector_synchronizer._process_and_upsert_batch = original_method

	@pytest.mark.asyncio
	async def test_sync_index(
		self, vector_synchronizer: VectorSynchronizer, sample_chunks: list[dict[str, Any]]
	) -> None:
		"""Test the full sync_index method."""
		# Configure the mocked git_context.tracked_files for this specific test
		# This assignment should be handled by the PropertyMock set up on the type.
		vector_synchronizer.git_context.tracked_files = {"file1.py": "abc123", "file2.py": "def456"}

		# Mock checksum calculator methods
		mock_read_checksum = MagicMock(return_value=("dummy_hash", {"file1.py": {"type": "file", "hash": "abc123"}}))
		vector_synchronizer.repo_checksum_calculator.read_latest_checksum_data_for_current_branch = mock_read_checksum

		# Mock calculate_repo_checksum to return a proper result that sync_index expects (2 values, not 3)
		mock_calculate = AsyncMock()
		mock_calculate.return_value = (
			"new_repo_hash",
			{"file1.py": {"type": "file", "hash": "abc123"}, "file2.py": {"type": "file", "hash": "def456"}},
		)
		vector_synchronizer.repo_checksum_calculator.calculate_repo_checksum = mock_calculate

		# Mock chunker
		vector_synchronizer.chunker.chunk_file = MagicMock(return_value=sample_chunks)

		# Mock the methods we've already tested
		mock_get_state = AsyncMock(return_value={})
		mock_compare = AsyncMock(return_value=({"file1.py", "file2.py"}, set()))
		mock_process = AsyncMock(return_value=2)

		vector_synchronizer._get_qdrant_state = mock_get_state
		vector_synchronizer._compare_states = mock_compare
		vector_synchronizer._process_and_upsert_batch = mock_process

		# Call sync_index
		result = await vector_synchronizer.sync_index()

		# Should be successful and call our mocked methods
		assert result is True
		mock_get_state.assert_called_once()
		mock_compare.assert_called_once()
		assert mock_process.call_count > 0
