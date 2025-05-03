"""Tests for the vector database manager/synchronizer."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from pymilvus import MilvusClient

from codemap.processor.vector import config, manager

# --- Fixtures ---


@pytest.fixture
def mock_milvus_client():
	"""Mocks the MilvusClient."""
	client = MagicMock(spec=MilvusClient)
	client.has_collection.return_value = True  # Assume collection exists by default
	client.query_iterator.return_value = iter([])  # Simulate empty DB initially
	client.delete.return_value = MagicMock()  # Simulate successful delete
	client.insert.return_value = MagicMock()  # Simulate successful insert
	return client


@pytest.fixture
def mock_config_loader():
	"""Mocks the ConfigLoader."""
	loader = MagicMock()
	# Provide necessary config values for vector processing
	loader.config = {
		"embedding": {
			"model_name": "mock-model",
			"dimensions": 3,  # Example dimension
			"batch_size": 16,
		},
		"chunking": {"strategy": "simple", "size": 100, "overlap": 10},
		# Add other required config sections if needed
	}
	return loader


@pytest.fixture
def temp_repo(tmp_path):
	"""Creates a temporary directory simulating a repo root."""
	repo = tmp_path / "test_repo"
	repo.mkdir()
	(repo / ".git").mkdir()
	(repo / "file1.py").write_text("import os\ndef func():\n  print(os.name)")
	(repo / "file2.py").write_text("class B:\n  pass")
	return repo


# --- Patch Dependencies ---


# Patch functions used within the manager module
@pytest.fixture(autouse=True)
def patch_dependencies(mock_milvus_client, mock_config_loader):
	"""Automatically patch dependencies for all tests in this module."""
	# Add a mock for KuzuManager
	mock_kuzu_manager = MagicMock(spec=manager.KuzuManager)
	mock_kuzu_manager.execute_query.return_value = []  # Default to no Kuzu results

	with (
		patch("codemap.processor.vector.manager.get_milvus_client", return_value=mock_milvus_client),
		patch("codemap.processor.vector.manager.ConfigLoader", return_value=mock_config_loader),
		patch("codemap.processor.vector.manager.get_git_tracked_files") as mock_get_git,
		patch("codemap.processor.vector.manager.chunker.chunk_file") as mock_chunk_code,
		patch("codemap.processor.vector.manager.generate_embeddings") as mock_generate_embeddings,
		patch("codemap.processor.vector.manager.find_project_root") as mock_find_root,
		# No need to patch KuzuManager itself if we just pass the mock instance
	):
		# Provide default return values for mocks
		mock_get_git.return_value = {}  # Default to empty git state
		# Return a list of dictionaries matching the Chunk schema
		mock_chunk_code.return_value = [
			{
				config.FIELD_CHUNK_TEXT: "chunk1",
				config.FIELD_START_LINE: 1,
				config.FIELD_END_LINE: 5,
				config.FIELD_CHUNK_TYPE: config.CHUNK_TYPE_FALLBACK,
				config.FIELD_ENTITY_NAME: "",
			},
			{
				config.FIELD_CHUNK_TEXT: "chunk2",
				config.FIELD_START_LINE: 6,
				config.FIELD_END_LINE: 10,
				config.FIELD_CHUNK_TYPE: config.CHUNK_TYPE_FALLBACK,
				config.FIELD_ENTITY_NAME: "",
			},
		]
		# Return a list of NumPy arrays
		mock_generate_embeddings.return_value = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
		mock_find_root.return_value = Path("/mock/repo")  # Default repo path if not provided

		# Yield a dictionary of mocks for potential use in tests
		yield {
			"get_milvus": manager.get_milvus_client,
			"config_loader": manager.ConfigLoader,
			"get_git": mock_get_git,
			"chunk_code": mock_chunk_code,
			"gen_embed": mock_generate_embeddings,
			"find_root": mock_find_root,
			"milvus_client": mock_milvus_client,
			"kuzu_manager": mock_kuzu_manager,  # Add kuzu mock to yielded dict
		}


# --- Test Cases ---


def test_get_milvus_file_hashes_empty(patch_dependencies):
	"""Test fetching state from an empty or non-existent collection."""
	mock_milvus = patch_dependencies["milvus_client"]
	mock_milvus.has_collection.return_value = False  # Simulate non-existent collection

	result = manager._get_milvus_file_hashes(mock_milvus)

	assert result == {}
	mock_milvus.has_collection.assert_called_once_with(config.COLLECTION_NAME)
	mock_milvus.query_iterator.assert_not_called()


def test_get_milvus_file_hashes_with_data(patch_dependencies):
	"""Test fetching state from Milvus with existing data."""
	mock_milvus = patch_dependencies["milvus_client"]
	# Simulate data returned by query_iterator
	mock_batch = [
		{config.FIELD_FILE_PATH: "file1.py", config.FIELD_GIT_HASH: "hash1"},
		{config.FIELD_FILE_PATH: "file2.py", config.FIELD_GIT_HASH: "hash2"},
		{config.FIELD_FILE_PATH: "file1.py", config.FIELD_GIT_HASH: "hash1_dup"},  # Dup path, diff hash
		{config.FIELD_FILE_PATH: "file3.py", config.FIELD_GIT_HASH: "hash2"},  # Dup hash, diff path
	]
	mock_milvus.query_iterator.return_value = iter([mock_batch])  # Single batch iterator

	result = manager._get_milvus_file_hashes(mock_milvus)

	expected = {
		"file1.py": {"hash1", "hash1_dup"},
		"file2.py": {"hash2"},
		"file3.py": {"hash2"},
	}
	assert result == expected
	mock_milvus.has_collection.assert_called_once_with(config.COLLECTION_NAME)
	mock_milvus.query_iterator.assert_called_once()


def test_get_milvus_file_hashes_query_error(patch_dependencies):
	"""Test handling of Milvus exceptions during query."""
	mock_milvus = patch_dependencies["milvus_client"]
	mock_milvus.query_iterator.side_effect = manager.exceptions.MilvusException(1, "Query failed")

	result = manager._get_milvus_file_hashes(mock_milvus)

	assert result is None  # Should return None on error


def test_delete_files_from_milvus(patch_dependencies):
	"""Test deleting files based on paths."""
	mock_milvus = patch_dependencies["milvus_client"]
	files_to_delete = {"file1.py", "subdir/file with space.txt"}

	# Reset mock for clean check
	mock_milvus.delete.reset_mock()

	manager._delete_files_from_milvus(mock_milvus, files_to_delete)

	# Check that delete was called
	mock_milvus.delete.assert_called_once()

	# Check the arguments passed to delete
	call_args = mock_milvus.delete.call_args
	assert call_args is not None
	_, kwargs = call_args  # Unpack, ignore args

	assert kwargs.get("collection_name") == config.COLLECTION_NAME
	actual_expr = kwargs.get("filter", "")

	# Check that delete was called with the correct filter expression
	expected_expr1 = f"{config.FIELD_FILE_PATH} == 'file1.py'"
	expected_expr2 = f"{config.FIELD_FILE_PATH} == 'subdir/file with space.txt'"
	assert expected_expr1 in actual_expr
	assert expected_expr2 in actual_expr
	assert " or " in actual_expr


def test_delete_files_from_milvus_empty_set(patch_dependencies):
	"""Test that delete is not called if the delete set is empty."""
	mock_milvus = patch_dependencies["milvus_client"]
	manager._delete_files_from_milvus(mock_milvus, set())
	mock_milvus.delete.assert_not_called()


def test_delete_files_from_milvus_error(patch_dependencies):
	"""Test handling of errors during deletion."""
	mock_milvus = patch_dependencies["milvus_client"]
	mock_milvus.delete.side_effect = manager.exceptions.MilvusException(1, "Delete failed")
	files_to_delete = {"file1.py"}

	# Should not raise an exception, just log the error
	manager._delete_files_from_milvus(mock_milvus, files_to_delete)
	mock_milvus.delete.assert_called_once()


def test_process_files(patch_dependencies, temp_repo):
	"""Test the main file processing loop (chunk, embed, insert)."""
	mock_milvus = patch_dependencies["milvus_client"]
	mock_chunk = patch_dependencies["chunk_code"]
	mock_embed = patch_dependencies["gen_embed"]
	mock_cfg_loader = patch_dependencies["config_loader"]()
	mock_kuzu = patch_dependencies["kuzu_manager"]  # Get kuzu mock

	files_to_process = {"file1.py", "file2.py"}
	current_git_files = {"file1.py": "hash1", "file2.py": "hash2"}
	files_to_update = {"file1.py"}  # Mark file1 as needing prior deletion

	# Mock file reading
	with patch("codemap.processor.vector.manager.read_file_content") as mock_read:

		def read_side_effect(path) -> str | None:
			if path.name == "file1.py":
				return "content1"
			if path.name == "file2.py":
				return "content2"
			return None

		mock_read.side_effect = read_side_effect

		# Also mock the kuzu lookup function within _process_files
		with patch("codemap.processor.vector.manager._get_kuzu_ids_for_chunks", return_value={}) as mock_kuzu_lookup:
			manager._process_files(
				client=mock_milvus,
				repo_path=temp_repo,
				files_to_process=files_to_process,
				current_git_files=current_git_files,
				files_to_update=files_to_update,
				app_config=mock_cfg_loader.config,
				kuzu_manager=mock_kuzu,  # Pass the mock kuzu manager
			)

	# 1. Check deletion was called for the updated file
	# Assert based on keyword arguments used in the call
	mock_milvus.delete.assert_called_once()
	call_args = mock_milvus.delete.call_args
	assert call_args is not None
	assert call_args.kwargs.get("collection_name") == config.COLLECTION_NAME
	filter_expr = call_args.kwargs.get("filter", "")
	assert f"{config.FIELD_FILE_PATH} == 'file1.py'" in filter_expr

	# 2. Check file reading
	read_calls = [call(temp_repo / "file1.py"), call(temp_repo / "file2.py")]
	mock_read.assert_has_calls(read_calls, any_order=True)

	# 3. Check chunking
	mock_chunk.assert_has_calls(
		[
			call(Path("file1.py"), "content1", "hash1", mock_cfg_loader.config),
			call(Path("file2.py"), "content2", "hash2", mock_cfg_loader.config),
		],
		any_order=True,  # Order depends on set iteration
	)

	# 4. Check embedding
	# Called once per file with the chunks for that file
	assert mock_embed.call_count == 2
	# Check args passed to embedder (example for one call)
	embed_args, _ = mock_embed.call_args_list[0]  # Example: first call
	assert embed_args[0] == ["chunk1", "chunk2"]  # Chunks for the file

	# 5. Check kuzu lookup was called (at least once per file)
	assert mock_kuzu_lookup.call_count >= len(files_to_process)

	# 6. Check insertion
	mock_milvus.insert.assert_called_once()
	insert_call_args = mock_milvus.insert.call_args
	assert insert_call_args is not None
	assert insert_call_args.kwargs.get("collection_name") == config.COLLECTION_NAME
	inserted_data = insert_call_args.kwargs.get("data", [])
	assert len(inserted_data) == len(files_to_process) * len(mock_chunk.return_value)  # Chunks per file
	# Check structure of one inserted item (assuming mock_chunk returns 2 chunks)
	assert config.FIELD_ID in inserted_data[0]
	assert config.FIELD_EMBEDDING in inserted_data[0]
	assert config.FIELD_FILE_PATH in inserted_data[0]
	assert config.FIELD_GIT_HASH in inserted_data[0]


def test_process_files_read_fail(patch_dependencies, temp_repo):
	"""Test that processing skips a file if reading fails."""
	mock_milvus = patch_dependencies["milvus_client"]
	mock_chunk = patch_dependencies["chunk_code"]
	mock_kuzu = patch_dependencies["kuzu_manager"]  # Get kuzu mock
	mock_cfg_loader = patch_dependencies["config_loader"]()

	files_to_process = {"bad_file.txt"}
	current_git_files = {"bad_file.txt": "hash_bad"}

	with patch("codemap.processor.vector.manager.read_file_content", return_value=None) as mock_read:
		manager._process_files(
			client=mock_milvus,
			repo_path=temp_repo,
			files_to_process=files_to_process,
			current_git_files=current_git_files,
			files_to_update=set(),
			app_config=mock_cfg_loader.config,
			kuzu_manager=mock_kuzu,  # Pass mock kuzu
		)

	mock_read.assert_called_once_with(temp_repo / "bad_file.txt")
	mock_chunk.assert_not_called()  # Should not chunk if read fails
	mock_milvus.insert.assert_not_called()  # Should not insert


def test_process_files_embedding_fail(patch_dependencies, temp_repo):
	"""Test that processing skips a file if embedding fails."""
	mock_milvus = patch_dependencies["milvus_client"]
	mock_embed = patch_dependencies["gen_embed"]
	mock_embed.side_effect = ValueError("Embedding failed")  # Simulate failure
	mock_kuzu = patch_dependencies["kuzu_manager"]  # Get kuzu mock
	mock_cfg_loader = patch_dependencies["config_loader"]()

	files_to_process = {"file1.py"}
	current_git_files = {"file1.py": "hash1"}

	with patch("codemap.processor.vector.manager.read_file_content", return_value="content1"):
		manager._process_files(
			client=mock_milvus,
			repo_path=temp_repo,
			files_to_process=files_to_process,
			current_git_files=current_git_files,
			files_to_update=set(),
			app_config=mock_cfg_loader.config,
			kuzu_manager=mock_kuzu,  # Pass mock kuzu
		)

	mock_embed.assert_called_once()  # Should attempt embedding
	mock_milvus.insert.assert_not_called()  # Should not insert if embedding fails


# Test the main synchronize_vectors function (integration-like)
def test_synchronize_vectors_integration(patch_dependencies, temp_repo):
	"""Test the main synchronization function orchestrates correctly."""
	mock_get_git = patch_dependencies["get_git"]
	mock_milvus = patch_dependencies["milvus_client"]
	mock_kuzu = patch_dependencies["kuzu_manager"]  # Get kuzu mock

	# Simulate Git state and DB state needing sync
	git_files = {"file1.py": "hash_new", "file_added.py": "hash_add"}
	milvus_files = {"file1.py": {"hash_old"}, "file_deleted.py": {"hash_del"}}

	mock_get_git.return_value = git_files
	with (
		patch("codemap.processor.vector.manager._get_milvus_file_hashes", return_value=milvus_files) as mock_get_hashes,
		patch("codemap.processor.vector.manager._delete_files_from_milvus") as mock_delete,
		patch("codemap.processor.vector.manager._process_files") as mock_process,
	):
		manager.synchronize_vectors(repo_path=temp_repo, kuzu_manager=mock_kuzu)

		# Verify correct components were called
		mock_get_git.assert_called_once_with(temp_repo)
		mock_get_hashes.assert_called_once_with(mock_milvus)

		# Check deletions
		mock_delete.assert_called_once()
		delete_args, _ = mock_delete.call_args
		assert delete_args[1] == {"file_deleted.py"}  # files_to_delete

		# Check additions/updates processing
		mock_process.assert_called_once()
		process_args, _ = mock_process.call_args
		assert process_args[2] == {"file1.py", "file_added.py"}  # files_to_process
		assert process_args[3] == git_files  # current_git_files
		assert process_args[4] == {"file1.py"}  # files_to_update
		assert process_args[5] == mock_kuzu  # Check kuzu passed down
