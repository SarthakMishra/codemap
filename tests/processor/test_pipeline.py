"""Tests for the main ProcessingPipeline."""

from unittest.mock import MagicMock, patch

import pytest

from codemap.processor.pipeline import ProcessingPipeline

# --- Fixtures ---


@pytest.fixture
def mock_repo_path(tmp_path):
	"""Creates a temporary directory simulating a repo root."""
	repo = tmp_path / "test_repo"
	repo.mkdir()
	(repo / ".git").mkdir()  # Mark as git repo
	# Add sample files if needed
	(repo / "main.py").write_text("def main():\n    print('hello')\n")
	return repo


@pytest.fixture
def mock_config_loader(mocker):
	"""Mocks the ConfigLoader class and its instance."""
	mock_cls = mocker.patch("codemap.processor.pipeline.ConfigLoader")
	mock_instance = mock_cls.return_value
	mock_instance.config = {
		"embedding": {},
		"graph_db": {"path": "mock_kuzu.db"},  # Example path
		# Add other necessary mock config values
	}
	return mock_instance  # Return the configured instance


@pytest.fixture
def mock_analyzer(mocker):
	"""Mocks the TreeSitterAnalyzer class."""
	mock_cls = mocker.patch("codemap.processor.pipeline.TreeSitterAnalyzer", autospec=True)
	return mock_cls.return_value  # Return the instance created by the mock class


@pytest.fixture
def mock_kuzu_manager(mocker):
	"""Mocks the KuzuManager class."""
	mock_cls = mocker.patch("codemap.processor.pipeline.KuzuManager", autospec=True)
	mock_instance = mock_cls.return_value
	mock_instance.execute_query.return_value = []
	mock_instance.get_all_file_hashes.return_value = {}
	mock_instance.query_vector_index.return_value = []
	mock_instance.close = mocker.MagicMock()
	return mock_instance


@pytest.fixture
def mock_graph_builder(mocker):
	"""Mocks the GraphBuilder class."""
	mock_cls = mocker.patch("codemap.processor.pipeline.GraphBuilder", autospec=True)
	mock_instance = mock_cls.return_value
	mock_instance.process_file.return_value = True
	return mock_instance


@pytest.fixture
def mock_graph_synchronizer(mocker):
	"""Mocks the GraphSynchronizer class."""
	mock_cls = mocker.patch("codemap.processor.pipeline.GraphSynchronizer", autospec=True)
	mock_instance = mock_cls.return_value
	mock_instance.sync_graph.return_value = True
	return mock_instance


@pytest.fixture
def mock_generate_embeddings(mocker):
	"""Mocks the generate_embeddings function."""
	mock = mocker.patch("codemap.processor.pipeline.generate_embedding")
	mock.return_value = [0.1, 0.2]
	return mock


@pytest.fixture
def processing_pipeline(
	mock_repo_path,  # Use the specific fixture
	mock_config_loader,  # Use the specific fixture
	mock_analyzer,  # Use the specific fixture
	mock_kuzu_manager,  # Use the specific fixture
	mock_graph_builder,  # Use the specific fixture
	mock_graph_synchronizer,  # Use the specific fixture
	mock_generate_embeddings,  # Use the specific fixture
	mocker,  # Inject mocker for patching find_project_root
):
	"""Provides a fully mocked ProcessingPipeline instance using dedicated fixtures."""
	# Patch find_project_root specifically for this fixture
	mocker.patch("codemap.processor.pipeline.find_project_root", return_value=mock_repo_path)

	# Create pipeline instance (sync_on_init=False to avoid auto-sync during test setup)
	# Dependencies are mocked by the fixtures injected above
	pipeline = ProcessingPipeline(repo_path=mock_repo_path, config_loader=mock_config_loader, sync_on_init=False)

	# Verify mocks were used correctly during initialization
	assert pipeline.analyzer is mock_analyzer
	assert pipeline.kuzu_manager is mock_kuzu_manager
	assert pipeline.graph_builder is mock_graph_builder
	assert pipeline.graph_synchronizer is mock_graph_synchronizer
	assert pipeline.has_vector_db is True

	# Store mocks in a dictionary for convenience in tests
	mocks = {
		"config_loader": mock_config_loader,
		"analyzer": mock_analyzer,
		"kuzu_manager": mock_kuzu_manager,
		"graph_builder": mock_graph_builder,
		"graph_synchronizer": mock_graph_synchronizer,
		"generate_embeddings": mock_generate_embeddings,
	}

	return pipeline, mocks


# --- Test Cases ---


def test_pipeline_initialization(processing_pipeline, mock_repo_path) -> None:
	"""Test successful initialization of the pipeline."""
	pipeline, mocks = processing_pipeline

	assert pipeline.repo_path == mock_repo_path
	assert pipeline.config is not None
	assert pipeline.analyzer is mocks["analyzer"]
	assert pipeline.kuzu_manager is mocks["kuzu_manager"]
	assert pipeline.graph_builder is mocks["graph_builder"]
	assert pipeline.graph_synchronizer is mocks["graph_synchronizer"]
	assert pipeline.has_vector_db is True

	# Check mocks called during init
	# Access the class mock if needed, e.g., mocks["analyzer_cls"].assert_called_once()
	# We primarily care that the instances were assigned
	# Check sync functions not called due to sync_on_init=False
	mocks["graph_synchronizer"].sync_graph.assert_not_called()


@patch("codemap.processor.pipeline.find_project_root")
def test_pipeline_init_finds_root(mock_find_root, mock_repo_path, mocker) -> None:
	"""Test pipeline finds repo root if not provided."""
	mock_find_root.return_value = mock_repo_path

	# Use dummy mocks for dependencies required by __init__
	with (
		patch("codemap.processor.pipeline.ConfigLoader"),
		patch("codemap.processor.pipeline.TreeSitterAnalyzer"),
		patch("codemap.processor.pipeline.KuzuManager"),
		patch("codemap.processor.pipeline.GraphBuilder"),
		patch("codemap.processor.pipeline.GraphSynchronizer"),
		# No need to patch sync_vectors etc. here as sync_on_init=False
	):
		# We need a config loader instance for this test path
		mock_cfg_loader_inst = MagicMock()
		mock_cfg_loader_inst.config = {}
		# Pass the instance directly
		_ = ProcessingPipeline(repo_path=None, config_loader=mock_cfg_loader_inst, sync_on_init=False)
		mock_find_root.assert_called_once()


def test_pipeline_sync_databases(processing_pipeline, mocker) -> None:
	"""Test the sync_databases method coordination."""
	pipeline, mocks = processing_pipeline

	# Setup mock git state
	git_state = {"file1.py": "hash1", "file2.py": "hash2"}
	mock_get_git = mocker.patch("codemap.processor.pipeline.get_git_tracked_files", return_value=git_state)

	# Reset the specific mocks we will assert on
	mocks["graph_synchronizer"].sync_graph.reset_mock()

	# Call the method under test
	pipeline.sync_databases()

	# Assertions
	mock_get_git.assert_called_once_with(pipeline.repo_path)
	# Assert graph sync was called on the mock instance
	mocks["graph_synchronizer"].sync_graph.assert_called_once_with(git_state)


def test_pipeline_sync_databases_git_fail(processing_pipeline, mocker) -> None:
	"""Test sync_databases handles failure to get git state."""
	pipeline, mocks = processing_pipeline

	# Patch get_git_tracked_files to simulate failure
	mock_get_git = mocker.patch("codemap.processor.pipeline.get_git_tracked_files", return_value=None)

	# Reset mocks
	mocks["graph_synchronizer"].sync_graph.reset_mock()

	# Call the method
	pipeline.sync_databases()

	# Verify failure handling
	mock_get_git.assert_called_once_with(pipeline.repo_path)
	# Sync methods should NOT have been called if git fails
	mocks["graph_synchronizer"].sync_graph.assert_not_called()


def test_pipeline_semantic_search(processing_pipeline) -> None:
	"""Test semantic search calls embedder and kuzu manager."""
	# Unpack the tuple returned by the fixture
	pipeline, mocks = processing_pipeline

	query = "find function"
	k = 3
	mock_query_embedding = [0.5, 0.5, 0.5]
	mock_vector_results = [{"entity_id": "entity1", "distance": 0.9}, {"entity_id": "entity2", "distance": 0.8}]
	mock_metadata_results = [
		["entity1", "func_a", "FUNCTION", 10, 20, "def func_a():", "doc a", "summary a", "file1.py"],
		["entity2", "func_b", "FUNCTION", 30, 40, "def func_b():", "doc b", "summary b", "file2.py"],
	]

	# Reset mocks to ensure clean test state
	mocks["generate_embeddings"].reset_mock()
	mocks["kuzu_manager"].query_vector_index.reset_mock()
	mocks["kuzu_manager"].execute_query.reset_mock()

	# Set return values
	mocks["generate_embeddings"].return_value = mock_query_embedding
	mocks["kuzu_manager"].query_vector_index.return_value = mock_vector_results
	mocks["kuzu_manager"].execute_query.return_value = mock_metadata_results

	# Call the method under test
	results = pipeline.semantic_search(query, k=k)

	assert results is not None
	assert len(results) == 2
	assert results[0]["id"] == "entity1"
	assert results[0]["distance"] == 0.9
	assert results[0]["metadata"]["name"] == "func_a"
	assert results[0]["metadata"]["file_path"] == "file1.py"
	assert results[1]["id"] == "entity2"
	assert results[1]["metadata"]["name"] == "func_b"

	# Check mocks were called
	mocks["generate_embeddings"].assert_called_once_with(query)
	mocks["kuzu_manager"].query_vector_index.assert_called_once_with(mock_query_embedding, k)
	# Check execute_query was called (for metadata)
	mocks["kuzu_manager"].execute_query.assert_called_once()
	# Optionally, check the constructed metadata query string if needed
	metadata_query_call = mocks["kuzu_manager"].execute_query.call_args[0][0]
	assert "MATCH (e:CodeEntity {entity_id: 'entity1'})" in metadata_query_call
	assert "UNION MATCH (e:CodeEntity {entity_id: 'entity2'})" in metadata_query_call
	assert "RETURN e.entity_id, e.name" in metadata_query_call  # Check some return fields


def test_pipeline_graph_query(processing_pipeline) -> None:
	"""Test graph query delegates to kuzu manager."""
	# Unpack the tuple returned by the fixture
	pipeline, mocks = processing_pipeline

	cypher = "MATCH (n) RETURN n"
	params = {"limit": 10}
	expected_results = [["node1"], ["node2"]]
	mocks["kuzu_manager"].execute_query.return_value = expected_results

	# Reset mock *after* pipeline init and *before* the call under test
	mocks["kuzu_manager"].execute_query.reset_mock()

	results = pipeline.graph_query(cypher, params)

	assert results == expected_results
	mocks["kuzu_manager"].execute_query.assert_called_once_with(cypher, params)


def test_graph_enhanced_search(processing_pipeline) -> None:
	"""Test the graph enhanced search method combining Kuzu vector search and graph traversal."""
	# Unpack the tuple returned by the fixture
	pipeline, mocks = processing_pipeline

	# Setup mock data
	query = "find function calling another function"
	k = 5

	# Mock semantic search results (new format)
	semantic_results = [
		{
			"id": "entity1",
			"distance": 0.9,
			"metadata": {"entity_id": "entity1", "name": "func_a", "file_path": "file1.py"},
		},
		{
			"id": "entity2",
			"distance": 0.8,
			"metadata": {"entity_id": "entity2", "name": "func_b", "file_path": "file2.py"},
		},
	]

	# Mock graph query results for additional context
	graph_context_results = [
		[
			"entity1",
			"entity3",
			["CodeEntity"],
			{"entity_id": "entity3", "name": "func_c", "file_path": "file1.py", "start_line": 5, "end_line": 8},
		],
		[
			"entity2",
			"entity4",
			["CodeEntity"],
			{"entity_id": "entity4", "name": "class_d", "file_path": "file2.py", "start_line": 15, "end_line": 25},
		],
		[
			"entity1",
			"entity1",
			["CodeEntity"],
			{"entity_id": "entity1", "name": "func_a", "file_path": "file1.py"},
		],
		[
			"entity2",
			"entity2",
			["CodeEntity"],
			{"entity_id": "entity2", "name": "func_b", "file_path": "file2.py"},
		],
	]

	# Setup mocks
	with patch.object(pipeline, "semantic_search", return_value=semantic_results) as mock_semantic_search:
		mocks["kuzu_manager"].execute_query.return_value = graph_context_results
		mocks["kuzu_manager"].execute_query.reset_mock()

		# Call the method under test
		results = pipeline.graph_enhanced_search(query, k_vector=k)

		# Verify correct calls were made
		mock_semantic_search.assert_called_once_with(query, k=k)
		mocks["kuzu_manager"].execute_query.assert_called_once()

		# Verify the results have been enhanced with graph context
		assert results is not None
		assert len(results) == 2
		# Access data within the 'vector_hit' key
		assert results[0]["vector_hit"]["id"] == "entity1"
		assert results[1]["vector_hit"]["id"] == "entity2"
		# Check that graph context was added (structure might vary)
		assert "graph_context" in results[0]
		assert len(results[0]["graph_context"]) > 0  # Assuming context was found
		# Check one of the graph context nodes
		context_node_ids = {node["id"] for node in results[0]["graph_context"]}
		assert "entity1" in context_node_ids  # Check initial node is included
		assert "entity3" in context_node_ids  # Check related node is included


def test_get_repository_structure(processing_pipeline) -> None:
	"""Test getting repository structure from the graph database."""
	# Unpack the tuple returned by the fixture
	pipeline, mocks = processing_pipeline

	# Mock query result representing repo structure
	# Updated format: id, labels, name, path, parent_dir_id, file_parent_dir_id
	mock_query_result = [
		["dir::.", ["CodeCommunity"], ".", ".", None, None],  # Root node
		["dir::src", ["CodeCommunity"], "src", "src", "dir::.", None],
		["dir::src/util", ["CodeCommunity"], "util", "src/util", "dir::src", None],
		["file::src/main.py", ["CodeFile"], "main.py", "src/main.py", None, "dir::src"],
		["file::src/util/helper.py", ["CodeFile"], "helper.py", "src/util/helper.py", None, "dir::src/util"],
	]

	mocks["kuzu_manager"].execute_query.return_value = mock_query_result

	# Call the method
	structure = pipeline.get_repository_structure()

	# If the method returns None (possible implementation change), skip detailed assertions
	if structure is None:
		pytest.fail("get_repository_structure returned None unexpectedly")

	# Verify the structure
	assert "name" in structure
	assert structure["name"] == "."
	assert "path" in structure
	assert structure["path"] == "."
	assert "type" in structure
	assert structure["type"] == "directory"

	# Check children
	assert "children" in structure
	assert len(structure["children"]) == 1  # Only src directory at root level

	# Check src directory
	src_dir = structure["children"][0]
	assert src_dir["name"] == "src"
	assert src_dir["path"] == "src"
	assert len(src_dir["children"]) == 2  # main.py and util directory

	# Find main.py file
	main_file = next((child for child in src_dir["children"] if child["name"] == "main.py"), None)
	assert main_file is not None
	assert main_file["type"] == "file"
	assert main_file["path"] == "src/main.py"

	# Find util directory
	util_dir = next((child for child in src_dir["children"] if child["name"] == "util"), None)
	assert util_dir is not None
	assert util_dir["type"] == "directory"
	assert len(util_dir["children"]) == 1  # helper.py

	# Check helper.py
	helper_file = util_dir["children"][0]
	assert helper_file["name"] == "helper.py"
	assert helper_file["type"] == "file"
	assert helper_file["path"] == "src/util/helper.py"


def test_stop_method(processing_pipeline) -> None:
	"""Test the stop method closes connections properly."""
	# Unpack the tuple returned by the fixture
	pipeline, mocks = processing_pipeline

	# Reset the mocks to ensure clean state
	mocks["kuzu_manager"].close.reset_mock()

	# Call the stop method
	pipeline.stop()

	# Verify Kuzu connection is closed
	mocks["kuzu_manager"].close.assert_called_once()
