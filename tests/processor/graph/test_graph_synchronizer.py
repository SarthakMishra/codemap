"""Tests for the GraphSynchronizer."""

from unittest.mock import MagicMock, call, patch

import pytest

from codemap.processor.graph.graph_builder import GraphBuilder
from codemap.processor.graph.kuzu_manager import KuzuManager
from codemap.processor.graph.synchronizer import GraphSynchronizer

# --- Fixtures ---


@pytest.fixture
def mock_kuzu_manager():
	"""Mocks KuzuManager."""
	manager = MagicMock(spec=KuzuManager)
	# Simulate an empty DB initially
	manager.get_all_file_hashes.return_value = {}
	manager.delete_file_data.return_value = None
	return manager


@pytest.fixture
def mock_graph_builder():
	"""Mocks GraphBuilder."""
	builder = MagicMock(spec=GraphBuilder)
	builder.process_file.return_value = True  # Simulate success by default
	return builder


@pytest.fixture
def temp_repo(tmp_path):
	"""Creates a temporary directory simulating a repo root."""
	repo = tmp_path / "test_repo"
	repo.mkdir()
	(repo / ".git").mkdir()  # Make it look like a git repo
	# Add some dummy files
	(repo / "file1.py").write_text("print('hello')")
	(repo / "subdir").mkdir()
	(repo / "subdir/file2.txt").write_text("data")
	return repo


@pytest.fixture
def graph_synchronizer(temp_repo, mock_kuzu_manager, mock_graph_builder):
	"""Provides a GraphSynchronizer instance with mocks."""
	return GraphSynchronizer(repo_path=temp_repo, kuzu_manager=mock_kuzu_manager, graph_builder=mock_graph_builder)


# --- Test Cases ---


def test_synchronizer_init(temp_repo, mock_kuzu_manager, mock_graph_builder):
	"""Test successful initialization."""
	synchronizer = GraphSynchronizer(temp_repo, mock_kuzu_manager, mock_graph_builder)
	assert synchronizer.repo_path == temp_repo
	assert synchronizer.kuzu_manager == mock_kuzu_manager
	assert synchronizer.graph_builder == mock_graph_builder


@patch("codemap.processor.graph.synchronizer.find_project_root")
def test_synchronizer_init_finds_root(mock_find_root, mock_kuzu_manager, mock_graph_builder, tmp_path):
	"""Test initialization finds repo root if not provided."""
	mock_find_root.return_value = tmp_path
	synchronizer = GraphSynchronizer(None, mock_kuzu_manager, mock_graph_builder)
	assert synchronizer.repo_path == tmp_path
	mock_find_root.assert_called_once()


@patch("codemap.processor.graph.synchronizer.get_git_tracked_files")
def test_sync_graph_no_changes(mock_get_git, graph_synchronizer, mock_kuzu_manager, mock_graph_builder):
	"""Test sync when Git and DB state are identical."""
	git_state = {"file1.py": "hash1", "subdir/file2.txt": "hash2"}
	db_state = {"file1.py": {"hash1"}, "subdir/file2.txt": {"hash2"}}  # Kuzu returns set of hashes

	mock_get_git.return_value = git_state
	mock_kuzu_manager.get_all_file_hashes.return_value = db_state

	result = graph_synchronizer.sync_graph()

	assert result is True
	mock_get_git.assert_called_once_with(graph_synchronizer.repo_path)
	mock_kuzu_manager.get_all_file_hashes.assert_called_once()
	mock_kuzu_manager.delete_file_data.assert_not_called()
	mock_graph_builder.process_file.assert_not_called()


@patch("codemap.processor.graph.synchronizer.get_git_tracked_files")
def test_sync_graph_add_files(mock_get_git, graph_synchronizer, mock_kuzu_manager, mock_graph_builder, temp_repo):
	"""Test sync when new files are added in Git."""
	git_state = {"file1.py": "hash1", "subdir/file2.txt": "hash2", "new_file.py": "hash_new"}
	db_state = {"file1.py": {"hash1"}, "subdir/file2.txt": {"hash2"}}

	mock_get_git.return_value = git_state
	mock_kuzu_manager.get_all_file_hashes.return_value = db_state

	# Create the new file physically for process_file to read
	(temp_repo / "new_file.py").write_text("new content")

	result = graph_synchronizer.sync_graph()

	assert result is True
	mock_kuzu_manager.delete_file_data.assert_not_called()
	# Check that process_file was called ONLY for the new file
	mock_graph_builder.process_file.assert_called_once_with(temp_repo / "new_file.py", "hash_new")


@patch("codemap.processor.graph.synchronizer.get_git_tracked_files")
def test_sync_graph_update_files(mock_get_git, graph_synchronizer, mock_kuzu_manager, mock_graph_builder, temp_repo):
	"""Test sync when existing files are modified in Git."""
	git_state = {"file1.py": "hash1_new", "subdir/file2.txt": "hash2"}
	db_state = {"file1.py": {"hash1"}, "subdir/file2.txt": {"hash2"}}

	mock_get_git.return_value = git_state
	mock_kuzu_manager.get_all_file_hashes.return_value = db_state

	result = graph_synchronizer.sync_graph()

	assert result is True
	mock_kuzu_manager.delete_file_data.assert_not_called()  # Update doesn't delete first
	# Check that process_file was called ONLY for the updated file
	mock_graph_builder.process_file.assert_called_once_with(temp_repo / "file1.py", "hash1_new")


@patch("codemap.processor.graph.synchronizer.get_git_tracked_files")
def test_sync_graph_delete_files(mock_get_git, graph_synchronizer, mock_kuzu_manager, mock_graph_builder):
	"""Test sync when files are deleted in Git."""
	git_state = {"file1.py": "hash1"}
	db_state = {"file1.py": {"hash1"}, "subdir/file2.txt": {"hash2"}, "deleted.py": {"hash_del"}}

	mock_get_git.return_value = git_state
	mock_kuzu_manager.get_all_file_hashes.return_value = db_state

	result = graph_synchronizer.sync_graph()

	assert result is True
	# Check that delete_file_data was called ONLY for the deleted files
	calls = [call("subdir/file2.txt"), call("deleted.py")]
	mock_kuzu_manager.delete_file_data.assert_has_calls(calls, any_order=True)
	assert mock_kuzu_manager.delete_file_data.call_count == 2
	mock_graph_builder.process_file.assert_not_called()  # No additions/updates


@patch("codemap.processor.graph.synchronizer.get_git_tracked_files")
def test_sync_graph_mixed_changes(mock_get_git, graph_synchronizer, mock_kuzu_manager, mock_graph_builder, temp_repo):
	"""Test sync with additions, updates, and deletions."""
	git_state = {"file1.py": "hash1_updated", "new_file.py": "hash_new"}
	db_state = {"file1.py": {"hash1_old"}, "deleted.py": {"hash_del"}}

	mock_get_git.return_value = git_state
	mock_kuzu_manager.get_all_file_hashes.return_value = db_state

	# Create the new file physically
	(temp_repo / "new_file.py").write_text("new content")

	result = graph_synchronizer.sync_graph()

	assert result is True
	# Deletion
	mock_kuzu_manager.delete_file_data.assert_called_once_with("deleted.py")
	# Addition and Update
	calls = [
		call(temp_repo / "file1.py", "hash1_updated"),
		call(temp_repo / "new_file.py", "hash_new"),
	]
	mock_graph_builder.process_file.assert_has_calls(calls, any_order=True)
	assert mock_graph_builder.process_file.call_count == 2


@patch("codemap.processor.graph.synchronizer.get_git_tracked_files")
def test_sync_graph_git_fetch_fails(mock_get_git, graph_synchronizer, mock_kuzu_manager):
	"""Test sync fails if Git state cannot be fetched."""
	mock_get_git.return_value = None  # Simulate failure

	result = graph_synchronizer.sync_graph()

	assert result is False
	mock_get_git.assert_called_once()
	mock_kuzu_manager.get_all_file_hashes.assert_not_called()  # Should bail early


@patch("codemap.processor.graph.synchronizer.get_git_tracked_files")
def test_sync_graph_kuzu_fetch_fails(mock_get_git, graph_synchronizer, mock_kuzu_manager):
	"""Test sync fails if Kuzu state cannot be fetched."""
	git_state = {"file1.py": "hash1"}
	mock_get_git.return_value = git_state
	mock_kuzu_manager.get_all_file_hashes.side_effect = Exception("DB connection error")

	result = graph_synchronizer.sync_graph()

	assert result is False
	mock_get_git.assert_called_once()
	mock_kuzu_manager.get_all_file_hashes.assert_called_once()


@patch("codemap.processor.graph.synchronizer.get_git_tracked_files")
def test_sync_graph_deletion_fails(mock_get_git, graph_synchronizer, mock_kuzu_manager, mock_graph_builder):
	"""Test sync continues but returns False if deletion fails."""
	git_state = {"file1.py": "hash1"}
	db_state = {"file1.py": {"hash1"}, "deleted.py": {"hash_del"}}

	mock_get_git.return_value = git_state
	mock_kuzu_manager.get_all_file_hashes.return_value = db_state
	mock_kuzu_manager.delete_file_data.side_effect = Exception("Cannot delete")

	result = graph_synchronizer.sync_graph()

	assert result is True  # Deletion failure doesn't stop processing, just logs
	mock_kuzu_manager.delete_file_data.assert_called_once_with("deleted.py")
	mock_graph_builder.process_file.assert_not_called()


@patch("codemap.processor.graph.synchronizer.get_git_tracked_files")
def test_sync_graph_processing_fails(
	mock_get_git, graph_synchronizer, mock_kuzu_manager, mock_graph_builder, temp_repo
):
	"""Test sync fails if processing a file fails."""
	git_state = {"file1.py": "hash1_new"}
	db_state = {}

	mock_get_git.return_value = git_state
	mock_kuzu_manager.get_all_file_hashes.return_value = db_state
	mock_graph_builder.process_file.return_value = False  # Simulate failure

	result = graph_synchronizer.sync_graph()

	assert result is False  # Should return False on processing error
	mock_graph_builder.process_file.assert_called_once_with(temp_repo / "file1.py", "hash1_new")
