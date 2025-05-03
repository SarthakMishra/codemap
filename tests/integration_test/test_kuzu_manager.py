"""Integration tests for KuzuManager."""

import math
from collections.abc import Generator

import pytest

from codemap.processor.graph.kuzu_manager import KuzuManager
from codemap.processor.tree_sitter.base import EntityType

# Define Mock Data Structure (will be used in tests)
MOCK_FILE_PATH = "mock/sample_module.py"
MOCK_GIT_HASH = "abcdef1234567890"
MOCK_LANGUAGE = "python"

MOCK_ENTITY_CLASS_ID = f"{MOCK_FILE_PATH}:1:CLASS:SampleClass"
MOCK_ENTITY_METHOD_ID = f"{MOCK_FILE_PATH}:2:METHOD:__init__"
MOCK_ENTITY_FUNC_ID = f"{MOCK_FILE_PATH}:5:FUNCTION:sample_function"

MOCK_COMMUNITY_FILE_ID = f"file:{MOCK_FILE_PATH}"
MOCK_COMMUNITY_DIR_ID = "dir:mock"

# Constants for vector tests
VECTOR_DIMENSION = 384  # Must match KuzuManager


@pytest.fixture(scope="session")
def kuzu_manager(tmp_path_factory) -> Generator[KuzuManager, None, None]:
	"""Fixture to create a KuzuManager instance with a temporary DB path for the session."""
	# Use tmp_path_factory for session-scoped fixtures
	db_path = tmp_path_factory.mktemp("kuzu_session_db") / "test_graph.kuzu"
	manager = KuzuManager(str(db_path))
	yield manager
	manager.close()  # Ensure connection is closed at the end of the session


def test_initialization_and_schema(kuzu_manager: KuzuManager) -> None:
	"""Test that the database is initialized and schema is created."""
	assert kuzu_manager.db is not None
	assert kuzu_manager.conn is not None

	# Check if tables exist by trying to query them (simple count)
	# If schema didn't create, these might fail or return errors KuzuManager handles
	# We expect 0 rows, but no execution error
	assert kuzu_manager.execute_query("MATCH (n:CodeFile) RETURN count(n)") == [[0]]
	assert kuzu_manager.execute_query("MATCH (n:CodeEntity) RETURN count(n)") == [[0]]
	assert kuzu_manager.execute_query("MATCH (n:Community) RETURN count(n)") == [[0]]
	# Check a relationship table indirectly
	# This query won't match anything, but shouldn't fail if table exists
	assert kuzu_manager.execute_query("MATCH (:CodeFile)-[:CONTAINS_ENTITY]->(:CodeEntity) RETURN count(*)") == [[0]]


def test_add_and_get_code_file(kuzu_manager: KuzuManager) -> None:
	"""Test adding a CodeFile node and retrieving it."""
	kuzu_manager.add_code_file(MOCK_FILE_PATH, MOCK_GIT_HASH, MOCK_LANGUAGE)

	# Check using the specific getter
	file_hashes = kuzu_manager.get_all_file_hashes()
	assert file_hashes == {MOCK_FILE_PATH: MOCK_GIT_HASH}

	# Verify with a direct query
	result = kuzu_manager.execute_query(
		"MATCH (f:CodeFile {file_path: $path}) RETURN f.git_hash, f.language", {"path": MOCK_FILE_PATH}
	)
	assert result is not None
	assert len(result) == 1
	assert result[0] == [MOCK_GIT_HASH, MOCK_LANGUAGE]


def test_add_communities(kuzu_manager: KuzuManager) -> None:
	"""Test adding community nodes and their parent relationship."""
	kuzu_manager.add_community(MOCK_COMMUNITY_DIR_ID, "DIRECTORY", "mock")
	kuzu_manager.add_community(
		MOCK_COMMUNITY_FILE_ID, "FILE", "sample_module.py", parent_community_id=MOCK_COMMUNITY_DIR_ID
	)

	# Verify directory community
	dir_result = kuzu_manager.execute_query(
		"MATCH (c:Community {community_id: $cid}) RETURN c.level, c.name", {"cid": MOCK_COMMUNITY_DIR_ID}
	)
	assert dir_result == [["DIRECTORY", "mock"]]

	# Verify file community
	file_result = kuzu_manager.execute_query(
		"MATCH (c:Community {community_id: $cid}) RETURN c.level, c.name", {"cid": MOCK_COMMUNITY_FILE_ID}
	)
	assert file_result == [["FILE", "sample_module.py"]]

	# Verify PARENT_COMMUNITY relationship
	parent_rel_result = kuzu_manager.execute_query(
		"MATCH (p:Community)-[:PARENT_COMMUNITY]->(c:Community {community_id: $cid}) RETURN p.community_id",
		{"cid": MOCK_COMMUNITY_FILE_ID},
	)
	assert parent_rel_result == [[MOCK_COMMUNITY_DIR_ID]]


def test_add_code_entities_and_relationships(kuzu_manager: KuzuManager) -> None:
	"""Test adding CodeEntity nodes and CONTAINS, DECLARES, BELONGS_TO relationships."""
	# Pre-requisites: Add File and Communities
	kuzu_manager.add_code_file(MOCK_FILE_PATH, MOCK_GIT_HASH, MOCK_LANGUAGE)
	kuzu_manager.add_community(MOCK_COMMUNITY_DIR_ID, "DIRECTORY", "mock")
	kuzu_manager.add_community(
		MOCK_COMMUNITY_FILE_ID, "FILE", "sample_module.py", parent_community_id=MOCK_COMMUNITY_DIR_ID
	)

	# Add a top-level class entity
	kuzu_manager.add_code_entity(
		entity_id=MOCK_ENTITY_CLASS_ID,
		file_path=MOCK_FILE_PATH,  # Needed for CONTAINS_ENTITY
		name="SampleClass",
		entity_type=EntityType.CLASS,
		start_line=1,
		end_line=4,
		signature=None,
		docstring="A sample class.",
		content_summary="class SampleClass:...",
		parent_entity_id=None,  # Top-level entity
		community_id=MOCK_COMMUNITY_FILE_ID,  # Belongs to file community
	)

	# Add a method entity declared by the class
	kuzu_manager.add_code_entity(
		entity_id=MOCK_ENTITY_METHOD_ID,
		file_path=MOCK_FILE_PATH,  # File path not strictly needed here, but passed for consistency
		name="__init__",
		entity_type=EntityType.METHOD,
		start_line=2,
		end_line=3,
		signature="(self, value)",
		docstring=None,
		content_summary="def __init__(self, value):...",
		parent_entity_id=MOCK_ENTITY_CLASS_ID,  # Declared by the class
		community_id=MOCK_COMMUNITY_FILE_ID,  # Also belongs to file community
	)

	# Verify Class Entity
	class_res = kuzu_manager.execute_query(
		"MATCH (e:CodeEntity {entity_id: $eid}) RETURN e.name, e.entity_type, e.start_line, e.docstring",
		{"eid": MOCK_ENTITY_CLASS_ID},
	)
	assert class_res == [["SampleClass", "CLASS", 1, "A sample class."]]

	# Verify Method Entity
	method_res = kuzu_manager.execute_query(
		"MATCH (e:CodeEntity {entity_id: $eid}) RETURN e.name, e.entity_type, e.start_line, e.signature",
		{"eid": MOCK_ENTITY_METHOD_ID},
	)
	assert method_res == [["__init__", "METHOD", 2, "(self, value)"]]

	# Verify CONTAINS_ENTITY (File -> Class)
	contains_res = kuzu_manager.execute_query(
		"MATCH (f:CodeFile {file_path: $fpath})-[:CONTAINS_ENTITY]->(e:CodeEntity {entity_id: $eid}) RETURN count(*)",
		{"fpath": MOCK_FILE_PATH, "eid": MOCK_ENTITY_CLASS_ID},
	)
	assert contains_res == [[1]]

	# Verify DECLARES (Class -> Method)
	declares_res = kuzu_manager.execute_query(
		"MATCH (p:CodeEntity {entity_id: $pid})-[:DECLARES]->(c:CodeEntity {entity_id: $cid}) RETURN count(*)",
		{"pid": MOCK_ENTITY_CLASS_ID, "cid": MOCK_ENTITY_METHOD_ID},
	)
	assert declares_res == [[1]]

	# Verify BELONGS_TO_COMMUNITY (Class -> File Community)
	belongs_class_res = kuzu_manager.execute_query(
		"MATCH (e:CodeEntity {entity_id: $eid})-[:BELONGS_TO_COMMUNITY]->(c:Community {community_id: $cid}) RETURN count(*)",
		{"eid": MOCK_ENTITY_CLASS_ID, "cid": MOCK_COMMUNITY_FILE_ID},
	)
	assert belongs_class_res == [[1]]

	# Verify BELONGS_TO_COMMUNITY (Method -> File Community)
	belongs_method_res = kuzu_manager.execute_query(
		"MATCH (e:CodeEntity {entity_id: $eid})-[:BELONGS_TO_COMMUNITY]->(c:Community {community_id: $cid}) RETURN count(*)",
		{"eid": MOCK_ENTITY_METHOD_ID, "cid": MOCK_COMMUNITY_FILE_ID},
	)
	assert belongs_method_res == [[1]]


def test_add_generic_relationship(kuzu_manager: KuzuManager) -> None:
	"""Test adding a generic relationship like CALLS."""
	# Prerequisites: Add entities involved in the relationship
	kuzu_manager.add_code_file(MOCK_FILE_PATH, MOCK_GIT_HASH, MOCK_LANGUAGE)
	kuzu_manager.add_community(MOCK_COMMUNITY_DIR_ID, "DIRECTORY", "mock")
	kuzu_manager.add_community(
		MOCK_COMMUNITY_FILE_ID, "FILE", "sample_module.py", parent_community_id=MOCK_COMMUNITY_DIR_ID
	)
	kuzu_manager.add_code_entity(
		entity_id=MOCK_ENTITY_CLASS_ID,
		file_path=MOCK_FILE_PATH,
		name="SampleClass",
		entity_type=EntityType.CLASS,
		start_line=1,
		end_line=4,
		signature=None,
		docstring=None,
		content_summary=None,
		community_id=MOCK_COMMUNITY_FILE_ID,
	)
	kuzu_manager.add_code_entity(
		entity_id=MOCK_ENTITY_FUNC_ID,
		file_path=MOCK_FILE_PATH,
		name="sample_function",
		entity_type=EntityType.FUNCTION,
		start_line=5,
		end_line=6,
		signature="()",
		docstring=None,
		content_summary=None,
		community_id=MOCK_COMMUNITY_FILE_ID,
	)

	# Add a CALLS relationship from Class (simplistic example) to Function
	kuzu_manager.add_relationship(
		from_entity_id=MOCK_ENTITY_CLASS_ID,
		to_entity_id=MOCK_ENTITY_FUNC_ID,
		rel_type="CALLS",
		properties={"call_site_line": 10},  # Example property
	)

	# Verify the CALLS relationship
	calls_res = kuzu_manager.execute_query(
		"MATCH (a:CodeEntity {entity_id: $from_id})-[r:CALLS]->(b:CodeEntity {entity_id: $to_id}) RETURN r.call_site_line",
		{"from_id": MOCK_ENTITY_CLASS_ID, "to_id": MOCK_ENTITY_FUNC_ID},
	)
	assert calls_res == [[10]]


def test_add_embedding(kuzu_manager: KuzuManager) -> None:
	"""Test adding an embedding to a CodeEntity."""
	# Setup: Add an entity
	kuzu_manager.add_code_file(MOCK_FILE_PATH, MOCK_GIT_HASH, MOCK_LANGUAGE)
	kuzu_manager.add_community(MOCK_COMMUNITY_DIR_ID, "DIRECTORY", "mock")
	kuzu_manager.add_community(
		MOCK_COMMUNITY_FILE_ID, "FILE", "sample_module.py", parent_community_id=MOCK_COMMUNITY_DIR_ID
	)
	kuzu_manager.add_code_entity(
		entity_id=MOCK_ENTITY_FUNC_ID,
		file_path=MOCK_FILE_PATH,
		name="sample_function",
		entity_type=EntityType.FUNCTION,
		start_line=5,
		end_line=6,
		signature="()",
		docstring=None,
		content_summary=None,
		community_id=MOCK_COMMUNITY_FILE_ID,
	)

	# Action: Add embedding
	mock_embedding = [float(i) / 100.0 for i in range(VECTOR_DIMENSION)]
	kuzu_manager.add_embedding(MOCK_ENTITY_FUNC_ID, mock_embedding)

	# Verification: Query the embedding property directly
	result = kuzu_manager.execute_query(
		"MATCH (e:CodeEntity {entity_id: $eid}) RETURN e.embedding", {"eid": MOCK_ENTITY_FUNC_ID}
	)
	assert result is not None
	assert len(result) == 1
	# Kuzu might return list of lists, access inner list
	retrieved_embedding = result[0][0]
	assert isinstance(retrieved_embedding, list)
	assert len(retrieved_embedding) == VECTOR_DIMENSION
	# Use a slightly larger relative tolerance (e.g., 1e-7) for float comparison
	assert math.isclose(retrieved_embedding[0], 0.0, rel_tol=1e-7)
	assert math.isclose(retrieved_embedding[100], 1.0, rel_tol=1e-7)
	assert math.isclose(retrieved_embedding[-1], (VECTOR_DIMENSION - 1) / 100.0, rel_tol=1e-7)


def test_create_and_drop_vector_index(kuzu_manager: KuzuManager) -> None:
	"""Test creating and dropping the vector index."""
	# Ensure the CodeEntity table exists (usually done by other tests/setup)
	kuzu_manager.ensure_schema()  # Ensure schema including CodeEntity exists

	# Action 1: Create Index
	# We expect this not to raise an exception
	try:
		kuzu_manager.create_vector_index()
	except Exception as e:
		pytest.fail(f"create_vector_index raised an unexpected exception: {e}")

	# Verification 1 (Optional): Add an entity and try a query - it shouldn't fail structurally
	# This is implicitly tested in test_query_vector_index

	# Action 2: Drop Index
	# We expect this not to raise an exception
	try:
		kuzu_manager.drop_vector_index()
	except Exception as e:
		pytest.fail(f"drop_vector_index raised an unexpected exception: {e}")

	# Verification 2 (Optional): Try creating it again - should work
	try:
		kuzu_manager.create_vector_index()
	except Exception as e:
		pytest.fail(f"Re-creating vector_index after drop raised an unexpected exception: {e}")


def test_query_vector_index(kuzu_manager: KuzuManager) -> None:
	"""Test querying the vector index for nearest neighbors."""
	# Setup: Add entities with embeddings
	kuzu_manager.add_code_file(MOCK_FILE_PATH, MOCK_GIT_HASH, MOCK_LANGUAGE)
	kuzu_manager.add_community(MOCK_COMMUNITY_DIR_ID, "DIRECTORY", "mock")
	kuzu_manager.add_community(
		MOCK_COMMUNITY_FILE_ID, "FILE", "sample_module.py", parent_community_id=MOCK_COMMUNITY_DIR_ID
	)

	# Entity 1: Vector of zeros
	entity_id_1 = f"{MOCK_FILE_PATH}:1:FUNCTION:func_zero"
	kuzu_manager.add_code_entity(
		entity_id_1,
		MOCK_FILE_PATH,
		"func_zero",
		EntityType.FUNCTION,
		1,
		1,
		"()",
		None,
		None,
		community_id=MOCK_COMMUNITY_FILE_ID,
	)
	embedding_1 = [0.0] * VECTOR_DIMENSION
	kuzu_manager.add_embedding(entity_id_1, embedding_1)

	# Entity 2: Vector of tens (more distinct)
	entity_id_2 = f"{MOCK_FILE_PATH}:2:FUNCTION:func_ten"
	kuzu_manager.add_code_entity(
		entity_id_2,
		MOCK_FILE_PATH,
		"func_ten",
		EntityType.FUNCTION,
		2,
		2,
		"()",
		None,
		None,
		community_id=MOCK_COMMUNITY_FILE_ID,
	)
	embedding_2 = [10.0] * VECTOR_DIMENSION
	kuzu_manager.add_embedding(entity_id_2, embedding_2)

	# Entity 3: Vector close to zeros
	entity_id_3 = f"{MOCK_FILE_PATH}:3:FUNCTION:func_near_zero"
	kuzu_manager.add_code_entity(
		entity_id_3,
		MOCK_FILE_PATH,
		"func_near_zero",
		EntityType.FUNCTION,
		3,
		3,
		"()",
		None,
		None,
		community_id=MOCK_COMMUNITY_FILE_ID,
	)
	embedding_3 = [0.1] * VECTOR_DIMENSION
	kuzu_manager.add_embedding(entity_id_3, embedding_3)

	# Action: Create index (must be done AFTER data is added for Kuzu HNSW)
	kuzu_manager.drop_vector_index()  # Ensure clean state before creating
	kuzu_manager.create_vector_index()  # Should now default to L2 metric

	# Action: Query for vector close to zeros
	query_vector = [0.05] * VECTOR_DIMENSION
	k = 2
	results = kuzu_manager.query_vector_index(query_vector, k)

	# Verification
	assert results is not None
	assert len(results) == k

	# Expect entity 2 (tens) and entity 3 (near zeros) to be closest based on default COSINE distance.
	# Cosine distance = 1 - similarity. Vectors pointing in the same direction have similarity 1, distance 0.
	# The zero vector (entity 1) case is handled differently and likely not returned as closest.
	result_ids = {res["entity_id"] for res in results}
	distances = [res["distance"] for res in results]

	# Assertions updated for COSINE metric
	assert entity_id_2 in result_ids  # Vector [10.0]*D -> Distance 0
	assert entity_id_3 in result_ids  # Vector [0.1]*D -> Distance 0
	assert entity_id_1 not in result_ids  # Vector [0.0]*D -> Distance > 0 (or undefined/excluded)

	assert all(d >= 0 for d in distances)  # Distances should be non-negative
	# Check if sorted by distance (query_vector_index should handle this)
	assert distances == sorted(distances)

	# Optional: More specific distance checks depending on metric knowledge
	# With COSINE distance (1 - similarity):
	# dist(query, vec2) = 1 - cos(angle([0.05]*D, [10.0]*D)) = 1 - 1 = 0
	# dist(query, vec3) = 1 - cos(angle([0.05]*D, [0.1]*D)) = 1 - 1 = 0
	# dist(query, vec1) depends on Kuzu's zero vector handling, but likely > 0
	# So vec2 and vec3 should be returned.


def test_delete_file_data(kuzu_manager: KuzuManager) -> None:
	"""Test deleting a file's data, including entities and community cleanup."""
	# --- Setup: Add data for two files ---
	MOCK_FILE_PATH_2 = "mock/another_module.py"
	MOCK_GIT_HASH_2 = "fedcba0987654321"
	MOCK_COMMUNITY_FILE_ID_2 = f"file:{MOCK_FILE_PATH_2}"
	MOCK_ENTITY_FUNC_ID_2 = f"{MOCK_FILE_PATH_2}:1:FUNCTION:another_function"

	# Add communities (shared dir)
	kuzu_manager.add_community(MOCK_COMMUNITY_DIR_ID, "DIRECTORY", "mock")
	kuzu_manager.add_community(
		MOCK_COMMUNITY_FILE_ID, "FILE", "sample_module.py", parent_community_id=MOCK_COMMUNITY_DIR_ID
	)
	kuzu_manager.add_community(
		MOCK_COMMUNITY_FILE_ID_2, "FILE", "another_module.py", parent_community_id=MOCK_COMMUNITY_DIR_ID
	)

	# Add file 1 data
	kuzu_manager.add_code_file(MOCK_FILE_PATH, MOCK_GIT_HASH, MOCK_LANGUAGE)
	kuzu_manager.add_code_entity(
		entity_id=MOCK_ENTITY_CLASS_ID,
		file_path=MOCK_FILE_PATH,
		name="SampleClass",
		entity_type=EntityType.CLASS,
		start_line=1,
		end_line=4,
		signature=None,
		docstring=None,
		content_summary=None,
		community_id=MOCK_COMMUNITY_FILE_ID,
	)
	kuzu_manager.add_code_entity(
		entity_id=MOCK_ENTITY_METHOD_ID,
		file_path=MOCK_FILE_PATH,
		name="__init__",
		entity_type=EntityType.METHOD,
		start_line=2,
		end_line=3,
		signature=None,
		docstring=None,
		content_summary=None,
		parent_entity_id=MOCK_ENTITY_CLASS_ID,
		community_id=MOCK_COMMUNITY_FILE_ID,
	)

	# Add file 2 data
	kuzu_manager.add_code_file(MOCK_FILE_PATH_2, MOCK_GIT_HASH_2, MOCK_LANGUAGE)
	kuzu_manager.add_code_entity(
		entity_id=MOCK_ENTITY_FUNC_ID_2,
		file_path=MOCK_FILE_PATH_2,
		name="another_function",
		entity_type=EntityType.FUNCTION,
		start_line=1,
		end_line=2,
		signature=None,
		docstring=None,
		content_summary=None,
		community_id=MOCK_COMMUNITY_FILE_ID_2,
	)

	# Add a relationship between entities in different files (for testing deletion side effects)
	kuzu_manager.add_relationship(
		from_entity_id=MOCK_ENTITY_METHOD_ID,  # From file 1
		to_entity_id=MOCK_ENTITY_FUNC_ID_2,  # To file 2
		rel_type="CALLS",
	)

	# --- Initial Check ---
	assert kuzu_manager.get_all_file_hashes() == {MOCK_FILE_PATH: MOCK_GIT_HASH, MOCK_FILE_PATH_2: MOCK_GIT_HASH_2}

	# Get actual entity count first
	entity_count_result = kuzu_manager.execute_query("MATCH (e:CodeEntity) RETURN count(e)")
	actual_entity_count = entity_count_result[0][0] if entity_count_result else 0
	assert actual_entity_count >= 3, f"Expected at least 3 entities, got {actual_entity_count}"

	# Get actual relationship count
	rel_count_result = kuzu_manager.execute_query("MATCH ()-[r:CALLS]->() RETURN count(r)")
	actual_rel_count = rel_count_result[0][0] if rel_count_result else 0
	assert actual_rel_count >= 1, f"Expected at least 1 CALLS relationship, got {actual_rel_count}"

	# Get actual community count
	community_count_result = kuzu_manager.execute_query("MATCH (c:Community) RETURN count(c)")
	actual_community_count = community_count_result[0][0] if community_count_result else 0
	assert actual_community_count == 3, f"Expected 3 communities, got {actual_community_count}"

	# --- Delete File 1 ---
	result = kuzu_manager.delete_file_data(MOCK_FILE_PATH)
	assert result is True, "File deletion should succeed"

	# --- Verification ---
	# File 1 node should be gone
	assert kuzu_manager.get_all_file_hashes() == {MOCK_FILE_PATH_2: MOCK_GIT_HASH_2}

	# Entities from File 1 should be gone (Class, Method)
	assert kuzu_manager.execute_query(
		"MATCH (e:CodeEntity {entity_id: $eid}) RETURN count(e)", {"eid": MOCK_ENTITY_CLASS_ID}
	) == [[0]]
	assert kuzu_manager.execute_query(
		"MATCH (e:CodeEntity {entity_id: $eid}) RETURN count(e)", {"eid": MOCK_ENTITY_METHOD_ID}
	) == [[0]]

	# Entity from File 2 should remain
	assert kuzu_manager.execute_query(
		"MATCH (e:CodeEntity {entity_id: $eid}) RETURN count(e)", {"eid": MOCK_ENTITY_FUNC_ID_2}
	) == [[1]]

	# Verify total entity count is now 1
	entity_count_after = kuzu_manager.execute_query("MATCH (e:CodeEntity) RETURN count(e)")
	assert entity_count_after == [[1]], f"Expected 1 entity after deletion, got {entity_count_after}"

	# Relationship originating from File 1 entity should be gone
	rel_count_after = kuzu_manager.execute_query("MATCH ()-[r:CALLS]->() RETURN count(r)")
	assert rel_count_after == [[0]], f"Expected 0 relationships after deletion, got {rel_count_after}"

	# File 1 community should be gone due to cleanup
	assert kuzu_manager.execute_query(
		"MATCH (c:Community {community_id: $cid}) RETURN count(c)", {"cid": MOCK_COMMUNITY_FILE_ID}
	) == [[0]]

	# File 2 community and Dir community should remain
	assert kuzu_manager.execute_query(
		"MATCH (c:Community {community_id: $cid}) RETURN count(c)", {"cid": MOCK_COMMUNITY_FILE_ID_2}
	) == [[1]]
	assert kuzu_manager.execute_query(
		"MATCH (c:Community {community_id: $cid}) RETURN count(c)", {"cid": MOCK_COMMUNITY_DIR_ID}
	) == [[1]]

	# Verify total community count is now 2
	community_count_after = kuzu_manager.execute_query("MATCH (c:Community) RETURN count(c)")
	assert community_count_after == [[2]], f"Expected 2 communities after deletion, got {community_count_after}"


# --- End of Tests ---
