"""Tests for the KuzuManager."""

import contextlib

import pytest

from codemap.processor.graph.kuzu_manager import DEFAULT_DB_FILE_NAME, KuzuManager
from codemap.processor.tree_sitter.base import EntityType

# --- Fixtures ---


@pytest.fixture
def temp_kuzu_db_path(tmp_path):
	"""Provides a path for a temporary Kuzu database."""
	return tmp_path / DEFAULT_DB_FILE_NAME


@pytest.fixture
def kuzu_manager(temp_kuzu_db_path):
	"""Provides a KuzuManager instance with a temporary database."""
	manager = KuzuManager(db_path=temp_kuzu_db_path)
	yield manager  # Use yield to ensure cleanup
	manager.close()  # Close connection after test
	# Optionally remove the db files, though tmp_path handles dir cleanup
	# for f in temp_kuzu_db_path.parent.glob(f"{KUZU_DB_NAME}*"):
	#     f.unlink(missing_ok=True)


# --- Test Cases ---


def test_kuzu_manager_init_and_connect(temp_kuzu_db_path) -> None:
	"""Test initialization creates DB files and connects."""
	assert not temp_kuzu_db_path.exists()
	manager = KuzuManager(db_path=temp_kuzu_db_path)
	assert temp_kuzu_db_path.exists()  # Directory should be created
	assert manager.db is not None
	assert manager.conn is not None
	manager.close()


def test_kuzu_manager_close(kuzu_manager) -> None:
	"""Test closing the connection."""
	assert kuzu_manager.conn is not None
	kuzu_manager.close()
	assert kuzu_manager.conn is None
	# Try closing again (should be safe)
	kuzu_manager.close()
	assert kuzu_manager.conn is None


def test_kuzu_manager_ensure_schema(tmp_path) -> None:
	"""Test that schema tables are created during initialization."""
	# Create a temporary database path
	db_path = str(tmp_path / "test_kuzu.db")

	try:
		# Create a new KuzuManager with the temporary path
		kuzu_manager = KuzuManager(db_path=db_path)

		# Check if some of the expected tables exist by running a query
		# This doesn't rely on Connection.get_table_names() which might not exist
		node_tables = ["CodeFile", "CodeEntity", "Community"]

		# Simple check - try to query each table and see if the query executes
		# Even if there's no data, a valid query shouldn't throw an error if the table exists
		for table in node_tables:
			query = f"MATCH (n:{table}) RETURN count(*) LIMIT 1"
			result = kuzu_manager.execute_query(query)
			assert result is not None, f"Table {table} query failed"

		# Success if we get here without exceptions
		assert True

	except (RuntimeError, OSError, ValueError) as e:
		# Specific exceptions that might occur during KuzuDB operations
		pytest.fail(f"Schema creation failed: {e}")

	finally:
		# Clean up
		import shutil

		with contextlib.suppress(Exception):
			shutil.rmtree(db_path, ignore_errors=True)


# Basic CRUD and Query Tests (using a real temp DB)


def test_kuzu_add_and_get_file(kuzu_manager) -> None:
	"""Test adding and retrieving a CodeFile node."""
	file_path = "src/main.py"
	git_hash = "abcdef12345"

	# Use add_code_file method instead of add_node
	kuzu_manager.add_code_file(file_path, git_hash, "python")

	# Query for the node (implementation would depend on KuzuManager's query capabilities)
	query = f"MATCH (f:CodeFile {{file_path: '{file_path}'}}) RETURN f.file_path, f.git_hash"
	results = kuzu_manager.execute_query(query)

	# Check the results match what was added
	assert results is not None
	assert len(results) == 1
	assert results[0][0] == file_path
	assert results[0][1] == git_hash


def test_kuzu_add_and_get_entity(kuzu_manager) -> None:
	"""Test adding and retrieving a CodeEntity node."""
	entity_id = "function::main::src/main.py::10::20"
	file_path = "src/main.py"
	name = "main"
	entity_type = EntityType.FUNCTION
	start_line = 10
	end_line = 20
	signature = "def main()"
	docstring = "Main function."
	content = "Function content summary"

	# Add a code entity using the appropriate method
	kuzu_manager.add_code_entity(
		entity_id=entity_id,
		file_path=file_path,
		name=name,
		entity_type=entity_type,
		start_line=start_line,
		end_line=end_line,
		signature=signature,
		docstring=docstring,
		content_summary=content,
	)

	# Query the entity
	query = f"MATCH (e:CodeEntity {{entity_id: '{entity_id}'}}) RETURN e.name, e.entity_type, e.start_line"
	results = kuzu_manager.execute_query(query)

	# Verify results
	assert results is not None
	assert len(results) == 1
	assert results[0][0] == name
	assert results[0][1] == entity_type.name
	assert results[0][2] == start_line


def test_kuzu_add_relationship(kuzu_manager) -> None:
	"""Test adding a relationship between nodes."""
	# First, add two entities to connect
	from_entity_id = "function::caller::src/main.py::1::10"
	to_entity_id = "function::callee::src/util.py::5::15"
	file_path1 = "src/main.py"
	file_path2 = "src/util.py"

	# Add the file nodes first
	kuzu_manager.add_code_file(file_path1, "hash1", "python")
	kuzu_manager.add_code_file(file_path2, "hash2", "python")

	# Add the entity nodes
	kuzu_manager.add_code_entity(
		entity_id=from_entity_id,
		file_path=file_path1,
		name="caller",
		entity_type=EntityType.FUNCTION,
		start_line=1,
		end_line=10,
		signature="def caller()",
		docstring=None,
		content_summary=None,
	)

	kuzu_manager.add_code_entity(
		entity_id=to_entity_id,
		file_path=file_path2,
		name="callee",
		entity_type=EntityType.FUNCTION,
		start_line=5,
		end_line=15,
		signature="def callee()",
		docstring=None,
		content_summary=None,
	)

	# Now add the relationship
	rel_type = "CALLS"
	properties = {"call_site_line": 5}
	kuzu_manager.add_relationship(from_entity_id, to_entity_id, rel_type, properties)

	# Query for the relationship
	query = f"""
		MATCH (caller:CodeEntity {{entity_id: '{from_entity_id}'}})
		      -[r:CALLS]->
		      (callee:CodeEntity {{entity_id: '{to_entity_id}'}})
		RETURN r.call_site_line
	"""
	results = kuzu_manager.execute_query(query)

	# Verify results
	assert results is not None
	assert len(results) == 1
	assert results[0][0] == 5


def test_get_all_file_hashes_empty(kuzu_manager) -> None:
	"""Test getting file hashes when DB is empty."""
	assert kuzu_manager.get_all_file_hashes() == {}


def test_delete_file_data(kuzu_manager) -> None:
	"""Test delete_file_data removes a file and its entities."""
	file_path = "src/to_delete.py"
	git_hash = "hash123"

	# Add a file and an entity
	kuzu_manager.add_code_file(file_path, git_hash, "python")
	kuzu_manager.add_code_entity(
		entity_id="function::test::src/to_delete.py::1::10",
		file_path=file_path,
		name="test",
		entity_type=EntityType.FUNCTION,
		start_line=1,
		end_line=10,
		signature="def test()",
		docstring=None,
		content_summary=None,
	)

	# Verify they exist
	file_query = f"MATCH (f:CodeFile {{file_path: '{file_path}'}}) RETURN count(*)"
	entity_query = f"MATCH (e:CodeEntity)-[:CONTAINS_ENTITY]-(f:CodeFile {{file_path: '{file_path}'}}) RETURN count(*)"

	file_result = kuzu_manager.execute_query(file_query)
	kuzu_manager.execute_query(entity_query)

	assert file_result is not None
	assert len(file_result) == 1
	assert file_result[0][0] >= 1  # At least one file node exists

	# Now delete the file data
	kuzu_manager.delete_file_data(file_path)

	# Verify nodes are gone
	file_result_after = kuzu_manager.execute_query(file_query)
	entity_result_after = kuzu_manager.execute_query(entity_query)

	assert file_result_after is not None
	assert file_result_after[0][0] == 0  # No file nodes
	assert entity_result_after is not None
	assert entity_result_after[0][0] == 0  # No entity nodes


def test_get_all_file_hashes_multiple(kuzu_manager) -> None:
	"""Test get_all_file_hashes with multiple files."""
	# Add two files
	kuzu_manager.add_code_file("src/file1.py", "hash1", "python")
	kuzu_manager.add_code_file("src/file2.py", "hash2", "python")

	# Get all file hashes
	file_hashes = kuzu_manager.get_all_file_hashes()

	# Verify results
	assert isinstance(file_hashes, dict)
	assert len(file_hashes) >= 2
	assert "src/file1.py" in file_hashes
	assert file_hashes.get("src/file1.py") == "hash1"
	assert "src/file2.py" in file_hashes
	assert file_hashes.get("src/file2.py") == "hash2"
