"""Graph database management for KuzuDB."""

import logging
from pathlib import Path
from typing import Any

import kuzu
from kuzu import Connection, Database, QueryResult

from codemap.processor.tree_sitter.base import EntityType
from codemap.utils.file_utils import ensure_directory_exists
from codemap.utils.path_utils import get_cache_path

logger = logging.getLogger(__name__)

# Define a default filename within the component's cache directory
DEFAULT_DB_FILE_NAME = "graph.db"

# Constants for vector index
VECTOR_INDEX_NAME = "entity_embedding_index"
VECTOR_TABLE_NAME = "CodeEntity"
VECTOR_COLUMN_NAME = "embedding"
VECTOR_DIMENSION = 384  # Dimension for sarthak1/Qodo-Embed-M-1-1.5B-M2V-Distilled


class KuzuManager:
	"""Manages the KuzuDB instance, schema, and data operations."""

	def __init__(self, db_path: str | None = None) -> None:
		"""
		Initialize the KuzuManager with a database path.

		If db_path is None, it defaults to a standard location within the
		.codemap_cache directory.

		Args:
		        db_path (Optional[str]): Path to the KuzuDB database file.
		                                                         Defaults to None.

		"""
		if db_path is None:
			# Use utility to get the graph cache directory
			graph_cache_dir = get_cache_path()
			# Ensure the directory exists
			ensure_directory_exists(graph_cache_dir)
			# Construct the full path
			self.db_path = str(graph_cache_dir / DEFAULT_DB_FILE_NAME)
			logger.info(f"No DB path provided, using default: {self.db_path}")
		else:
			# If a path is provided, ensure its directory exists
			db_file_path = Path(db_path)
			ensure_directory_exists(db_file_path.parent)
			self.db_path = db_path
			logger.info(f"Using provided DB path: {self.db_path}")

		self.db: Database | None = None
		self.conn: Connection | None = None
		try:
			self._connect()
			self.ensure_schema()
			self.install_vector_extension()
		except Exception:
			logger.exception(f"Failed to initialize KuzuManager at {self.db_path}")
			# Ensure partial connections are cleaned up
			self.close()
			raise  # Re-raise the exception

	def _connect(self) -> None:
		"""Establish connection to the KuzuDB database."""
		try:
			logger.info(f"Connecting to KuzuDB at: {self.db_path}")
			# Consider configuring buffer pool size and threads based on system/config
			self.db = kuzu.Database(self.db_path)
			self.conn = kuzu.Connection(self.db)
			logger.info("Successfully connected to KuzuDB.")
		except Exception:
			logger.exception(f"Failed to connect to KuzuDB at {self.db_path}")
			self.db = None
			self.conn = None

	def close(self) -> None:
		"""Close the KuzuDB connection."""
		# Kuzu Python API doesn't explicitly have close methods for Connection/Database
		# They are managed by Python's garbage collector.
		# Setting to None helps release references.
		logger.info("Closing KuzuDB connection references.")
		self.conn = None
		self.db = None

	def install_vector_extension(self) -> None:
		"""Installs and loads the Kuzu vector extension."""
		if not self.conn:
			logger.error("Cannot install vector extension, KuzuDB connection not established.")
			return
		try:
			logger.info("Installing and loading Kuzu VECTOR extension...")
			self.conn.execute("INSTALL VECTOR;")
			self.conn.execute("LOAD VECTOR;")
			logger.info("Kuzu VECTOR extension installed and loaded successfully.")
		except Exception as e:
			# Check if extension is already installed/loaded (common scenario)
			if "already exists" in str(e).lower() or "already loaded" in str(e).lower():
				logger.debug(f"Kuzu VECTOR extension likely already installed/loaded: {e}")
			else:
				logger.exception("Failed to install or load Kuzu VECTOR extension")
				# Depending on requirements, you might want to raise the error here

	def execute_query(self, query: str, params: dict[str, Any] | None = None) -> list[list[Any]] | None:
		"""Execute a Cypher query with parameters."""
		if not self.conn:
			logger.error("Cannot execute query, KuzuDB connection not established.")
			return None
		try:
			logger.debug(f"Executing KuzuDB query: {query} with params: {params}")
			prepared_statement = self.conn.prepare(query)
			# Kuzu requires parameters passed as positional args in order
			# Let's assume params dict keys match placeholder names $key
			# We need to extract values in the order placeholders appear in the query.
			# This is tricky without parsing the query string robustly.
			# For now, assume params are passed correctly if needed, or use simple queries.
			# If params is None, execute without args
			query_result_raw = self.conn.execute(prepared_statement, parameters=params if params else {})

			# Check if the result is the expected QueryResult type before iterating
			if not isinstance(query_result_raw, QueryResult):
				logger.error(f"Kuzu execute returned unexpected type: {type(query_result_raw)}. Expected QueryResult.")
				return None

			# Now we know it's a QueryResult
			query_result: QueryResult = query_result_raw
			results = []
			while query_result.has_next():
				results.append(query_result.get_next())
			logger.debug(f"Query returned {len(results)} rows.")
			return results
		except Exception as e:
			# Check if it's the expected "already exists" error
			error_msg = str(e).lower()
			if "already exists in catalog" in error_msg or "binder exception" in error_msg:
				# Log expected schema creation conflicts at DEBUG level
				logger.debug(f"Kuzu query failed (likely schema already exists): {query} - {e}")
				# Re-raise the specific error for helpers to catch
				raise
			# Log unexpected query errors as exceptions
			logger.exception(f"Failed to execute KuzuDB query: {query}")
			return None

	def _create_table_if_not_exists(self, create_statement: str, table_name: str) -> None:
		"""Helper to create a table only if it doesn't exist."""
		# Kuzu doesn't have explicit "IF NOT EXISTS" for CREATE TABLE
		# Attempt to create the table directly. If it fails because it already
		# exists (or for other reasons), log the error but continue.
		try:
			self.execute_query(create_statement)
			logger.info(f"Successfully created or verified table {table_name}.")
		except (RuntimeError, ValueError) as e:
			# Kuzu might throw different errors. Check if the error indicates it already exists.
			# This check is brittle and depends on Kuzu's error messages.
			# Example checks (adapt based on actual Kuzu error messages):
			if "already exists" in str(e).lower() or "Catalog exception" in str(e):
				logger.debug(f"Table {table_name} likely already exists: {e}")
			else:
				# Log other creation errors more visibly but don't necessarily stop schema creation
				logger.warning(f"Could not create or verify table {table_name}. Error: {e}")
				# Depending on severity, you might choose to raise e here

	def _create_rel_table_if_not_exists(self, create_statement: str, table_name: str) -> None:
		"""Helper to create a relationship table only if it doesn't exist."""
		try:
			self.execute_query(create_statement)
			logger.info(f"Successfully created or verified relationship table {table_name}.")
		except (RuntimeError, ValueError) as e:
			if "already exists" in str(e).lower() or "Catalog exception" in str(e):
				logger.debug(f"Relationship table {table_name} likely already exists: {e}")
			else:
				logger.warning(f"Could not create or verify relationship table {table_name}. Error: {e}")

	def ensure_schema(self) -> None:
		"""Define and ensure the graph schema exists in KuzuDB."""
		if not self.conn:
			logger.error("Cannot ensure schema, KuzuDB connection not established.")
			return

		logger.info("Ensuring KuzuDB schema...")

		node_tables = {
			"CodeFile": "CREATE NODE TABLE CodeFile(file_path STRING, git_hash STRING, language STRING, "
			"PRIMARY KEY (file_path))",
			"CodeEntity": "CREATE NODE TABLE CodeEntity(entity_id STRING, name STRING, entity_type STRING, "
			"start_line INT64, end_line INT64, signature STRING, docstring STRING, "
			"content_summary STRING, embedding FLOAT[384], PRIMARY KEY (entity_id))",
			"Community": "CREATE NODE TABLE Community(community_id STRING, level STRING, name STRING, "
			"PRIMARY KEY (community_id))",
		}

		rel_tables = {
			# REL Name: (FROM_Node, TO_Node, Properties_Def, Create_Statement)
			"CONTAINS_ENTITY": (
				"CodeFile",
				"CodeEntity",
				"",
				"CREATE REL TABLE CONTAINS_ENTITY(FROM CodeFile TO CodeEntity)",
			),
			"DECLARES": ("CodeEntity", "CodeEntity", "", "CREATE REL TABLE DECLARES(FROM CodeEntity TO CodeEntity)"),
			"IMPORTS": (
				"CodeEntity",
				"CodeEntity",
				"import_path STRING",
				"CREATE REL TABLE IMPORTS(FROM CodeEntity TO CodeEntity, import_path STRING)",
			),
			"CALLS": (
				"CodeEntity",
				"CodeEntity",
				"call_site_line INT64",
				"CREATE REL TABLE CALLS(FROM CodeEntity TO CodeEntity, call_site_line INT64)",
			),
			"INHERITS_FROM": (
				"CodeEntity",
				"CodeEntity",
				"",
				"CREATE REL TABLE INHERITS_FROM(FROM CodeEntity TO CodeEntity)",
			),
			"BELONGS_TO_COMMUNITY": (
				"CodeEntity",
				"Community",
				"",
				"CREATE REL TABLE BELONGS_TO_COMMUNITY(FROM CodeEntity TO Community)",
			),
			"PARENT_COMMUNITY": (
				"Community",
				"Community",
				"",
				"CREATE REL TABLE PARENT_COMMUNITY(FROM Community TO Community)",
			),
		}

		# Create Node Tables
		for name, stmt in node_tables.items():
			self._create_table_if_not_exists(stmt, name)

		# Create Relationship Tables
		for name, (_, _, _, stmt) in rel_tables.items():
			self._create_rel_table_if_not_exists(stmt, name)

		logger.info("KuzuDB schema ensured.")

	# --- Data Manipulation Methods ---

	def add_code_file(self, file_path: str, git_hash: str, language: str) -> None:
		"""Add or update a CodeFile node."""
		# MERGE creates if not exists, updates if exists (based on PK)
		query = "MERGE (f:CodeFile {file_path: $file_path}) SET f.git_hash = $git_hash, f.language = $language"
		self.execute_query(query, {"file_path": file_path, "git_hash": git_hash, "language": language})

	def add_code_entity(
		self,
		entity_id: str,
		file_path: str,  # Need file_path to link CONTAINS_ENTITY
		name: str | None,
		entity_type: EntityType,
		start_line: int,
		end_line: int,
		signature: str | None,
		docstring: str | None,
		content_summary: str | None,
		parent_entity_id: str | None = None,
		community_id: str | None = None,
	) -> None:
		"""Add or update a CodeEntity node and its core relationships."""
		# Add/Update the CodeEntity node
		# Use MERGE...ON CREATE SET...ON MATCH SET if fine-grained control needed
		# For simplicity, MERGE...SET updates all specified properties
		entity_props = {
			"entity_id": entity_id,
			"name": name if name else "",
			"entity_type": entity_type.name,
			"start_line": start_line,
			"end_line": end_line,
			"signature": signature if signature else "",
			"docstring": docstring if docstring else "",
			"content_summary": content_summary if content_summary else "",
		}
		# Correct SET clause for Kuzu - list properties individually
		set_clauses = []
		query_params = {"entity_id": entity_id}
		for key, value in entity_props.items():
			if key != "entity_id":  # Don't set the primary key
				set_clauses.append(f"e.{key} = ${key}")
				query_params[key] = value

		set_string = ", ".join(set_clauses)
		query_node = f"MERGE (e:CodeEntity {{entity_id: $entity_id}}) SET {set_string}"
		self.execute_query(query_node, query_params)

		# Add CONTAINS_ENTITY relationship (File -> Entity)
		# Only for top-level entities (where parent_entity_id is None)
		if parent_entity_id is None:
			query_contains = (
				"MATCH (f:CodeFile {file_path: $file_path}), (e:CodeEntity {entity_id: $entity_id}) "
				"MERGE (f)-[:CONTAINS_ENTITY]->(e)"
			)
			self.execute_query(query_contains, {"file_path": file_path, "entity_id": entity_id})

		# Add DECLARES relationship (Parent Entity -> Entity)
		if parent_entity_id:
			query_declares = (
				"MATCH (p:CodeEntity {entity_id: $parent_id}), (c:CodeEntity {entity_id: $child_id}) "
				"MERGE (p)-[:DECLARES]->(c)"
			)
			self.execute_query(query_declares, {"parent_id": parent_entity_id, "child_id": entity_id})

		# Add BELONGS_TO_COMMUNITY relationship (Entity -> Community)
		if community_id:
			query_community = (
				"MATCH (e:CodeEntity {entity_id: $entity_id}), (c:Community {community_id: $community_id}) "
				"MERGE (e)-[:BELONGS_TO_COMMUNITY]->(c)"
			)
			self.execute_query(query_community, {"entity_id": entity_id, "community_id": community_id})

	def add_community(self, community_id: str, level: str, name: str, parent_community_id: str | None = None) -> None:
		"""Add or update a Community node and its parent relationship."""
		# Add/Update the Community node
		query_node = "MERGE (c:Community {community_id: $community_id}) SET c.level = $level, c.name = $name"
		self.execute_query(query_node, {"community_id": community_id, "level": level, "name": name})

		# Add PARENT_COMMUNITY relationship (Parent Community -> Community)
		if parent_community_id:
			query_parent = (
				"MATCH (p:Community {community_id: $parent_id}), (c:Community {community_id: $child_id}) "
				"MERGE (p)-[:PARENT_COMMUNITY]->(c)"
			)
			self.execute_query(query_parent, {"parent_id": parent_community_id, "child_id": community_id})

	def add_relationship(
		self, from_entity_id: str, to_entity_id: str, rel_type: str, properties: dict[str, Any] | None = None
	) -> None:
		"""Add a generic relationship between two entities."""
		# Basic MERGE - assumes relationship doesn't have its own unique properties to match on
		# If relationships need updates based on properties, logic needs refinement
		prop_set_clauses = []
		params = {"from_id": from_entity_id, "to_id": to_entity_id}
		if properties:
			for k, v in properties.items():
				prop_set_clauses.append(f"r.{k} = ${k}")
				params[k] = v

		prop_set_string = (", ".join(prop_set_clauses)) if prop_set_clauses else ""  # Join properties

		# Start with the base query
		query = (
			f"MATCH (a:CodeEntity {{entity_id: $from_id}}), (b:CodeEntity {{entity_id: $to_id}}) "
			f"MERGE (a)-[r:{rel_type}]->(b)"
		)
		# Append SET clause only if there are properties to set
		if prop_set_string:
			query += f" SET {prop_set_string}"

		self.execute_query(query, params)

	def delete_file_data(self, file_path: str) -> bool:
		"""
		Delete a CodeFile node and ALL its associated entities and relationships.

		Args:
		        file_path (str): Path of the file to delete from the database

		Returns:
		        bool: True if deletion was successful, False otherwise

		"""
		if not self.conn:
			logger.error("Cannot delete file data, KuzuDB connection not established.")
			return False

		logger.info(f"Attempting to delete all data for file: {file_path}")

		try:
			# Step 0: Drop vector index to avoid hidden relationship constraints
			logger.info("Dropping vector index to prepare for entity deletion...")
			self.drop_vector_index()

			# Step 1: Find all entity IDs associated with the file, including nested entities
			entity_query = """
			MATCH (f:CodeFile {file_path: $file_path})-[:CONTAINS_ENTITY]->(e1:CodeEntity)
			OPTIONAL MATCH (e1)-[:DECLARES*0..]->(e2:CodeEntity)
			RETURN DISTINCT e2.entity_id
			"""
			entity_results = self.execute_query(entity_query, {"file_path": file_path})
			entity_ids = [row[0] for row in entity_results if row[0] is not None] if entity_results else []
			logger.info(f"Found {len(entity_ids)} entities to delete for file {file_path}")

			# CRITICAL: First delete all CONTAINS_ENTITY relationships
			# Delete outgoing CONTAINS_ENTITY relationships from the file node
			file_out_contains_query = """
			MATCH (f:CodeFile {file_path: $file_path})-[r:CONTAINS_ENTITY]->()
			DELETE r
			"""
			self.execute_query(file_out_contains_query, {"file_path": file_path})

			# Delete incoming CONTAINS_ENTITY relationships to entities (should be none but be thorough)
			file_in_contains_query = """
			MATCH ()-[r:CONTAINS_ENTITY]->(e:CodeEntity)
			WHERE e.entity_id IN $entity_ids
			DELETE r
			"""
			self.execute_query(file_in_contains_query, {"entity_ids": entity_ids})

			logger.info("Deleted all CONTAINS_ENTITY relationships")

			# Step 2: Delete entity relationships for each specific type and direction
			if entity_ids:
				# Handle outgoing BELONGS_TO_COMMUNITY relationships
				out_belongs_query = """
				MATCH (e:CodeEntity)-[r:BELONGS_TO_COMMUNITY]->()
				WHERE e.entity_id IN $entity_ids
				DELETE r
				"""
				self.execute_query(out_belongs_query, {"entity_ids": entity_ids})

				# Handle outgoing DECLARES relationships
				out_declares_query = """
				MATCH (e:CodeEntity)-[r:DECLARES]->()
				WHERE e.entity_id IN $entity_ids
				DELETE r
				"""
				self.execute_query(out_declares_query, {"entity_ids": entity_ids})

				# Handle incoming DECLARES relationships
				in_declares_query = """
				MATCH ()-[r:DECLARES]->(e:CodeEntity)
				WHERE e.entity_id IN $entity_ids
				DELETE r
				"""
				self.execute_query(in_declares_query, {"entity_ids": entity_ids})

				# Handle outgoing CALLS relationships
				out_calls_query = """
				MATCH (e:CodeEntity)-[r:CALLS]->()
				WHERE e.entity_id IN $entity_ids
				DELETE r
				"""
				self.execute_query(out_calls_query, {"entity_ids": entity_ids})

				# Handle incoming CALLS relationships
				in_calls_query = """
				MATCH ()-[r:CALLS]->(e:CodeEntity)
				WHERE e.entity_id IN $entity_ids
				DELETE r
				"""
				self.execute_query(in_calls_query, {"entity_ids": entity_ids})

				logger.info("Deleted relationship for entities")

				# Step 3: Delete the entities
				entity_delete_query = """
				MATCH (e:CodeEntity)
				WHERE e.entity_id IN $entity_ids
				DELETE e
				"""
				self.execute_query(entity_delete_query, {"entity_ids": entity_ids})
				logger.info(f"Deleted {len(entity_ids)} entities")

			# Step 4: Delete the file node
			file_delete_query = """
			MATCH (f:CodeFile {file_path: $file_path})
			DELETE f
			"""
			self.execute_query(file_delete_query, {"file_path": file_path})
			logger.info(f"Deleted file node: {file_path}")

			# Step 5: Community cleanup
			file_community_id = f"file:{file_path}"
			self._cleanup_empty_communities(file_community_id)

			# Step 6: Rebuild vector index with remaining entities
			logger.info("Rebuilding vector index...")
			self.create_vector_index()

			return True

		except Exception:
			logger.exception(f"Error deleting data for file: {file_path}")
			# Try to rebuild the index if it was dropped
			try:
				self.create_vector_index()
			except Exception:
				logger.exception("Failed to rebuild vector index after delete failure")
			return False

	def _cleanup_empty_communities(self, community_id_to_check: str | None) -> bool:
		"""
		Recursively check and delete empty communities starting from a given ID.

		Args:
		        community_id_to_check (str | None): The community ID to check and potentially delete

		Returns:
		        bool: True if all cleanup operations succeeded, False if errors occurred

		"""
		if not community_id_to_check:
			return True

		if not self.conn:
			logger.error("Cannot cleanup communities, KuzuDB connection not established.")
			return False

		try:
			logger.debug(f"Checking community for cleanup: {community_id_to_check}")

			# Check if any entities still belong to this community
			check_entities_query = (
				"MATCH (e:CodeEntity)-[:BELONGS_TO_COMMUNITY]->(c:Community {community_id: $cid}) RETURN count(e)"
			)
			entity_count_result = self.execute_query(check_entities_query, {"cid": community_id_to_check})
			entity_count = entity_count_result[0][0] if entity_count_result else 0

			# Check if any child communities still belong to this community
			check_children_query = (
				"MATCH (parent:Community {community_id: $cid})-[r:PARENT_COMMUNITY]->(child:Community) "
				"RETURN count(child)"
			)
			child_count_result = self.execute_query(check_children_query, {"cid": community_id_to_check})
			child_community_count = child_count_result[0][0] if child_count_result else 0

			logger.debug(
				f"Cleanup check for {community_id_to_check}: "
				f"Entity count = {entity_count}, "
				f"Child community count = {child_community_count}"
			)

			# If no entities AND no child communities link to it, delete it and check its parent
			if entity_count == 0 and child_community_count == 0:
				logger.info(f"Community {community_id_to_check} is empty. Deleting...")

				# Find parent community ID *before* deleting the current one
				find_parent_query = (
					"MATCH (p:Community)-[:PARENT_COMMUNITY]->(c:Community {community_id: $cid}) "
					"RETURN p.community_id LIMIT 1"
				)
				parent_result = self.execute_query(find_parent_query, {"cid": community_id_to_check})
				parent_community_id = parent_result[0][0] if parent_result else None

				# Delete specific relationship types with directions
				# Outgoing PARENT_COMMUNITY relationships
				out_parent_rel_query = "MATCH (c:Community {community_id: $cid})-[r:PARENT_COMMUNITY]->() DELETE r"
				self.execute_query(out_parent_rel_query, {"cid": community_id_to_check})

				# Incoming PARENT_COMMUNITY relationships
				in_parent_rel_query = "MATCH ()-[r:PARENT_COMMUNITY]->(c:Community {community_id: $cid}) DELETE r"
				self.execute_query(in_parent_rel_query, {"cid": community_id_to_check})

				# Incoming BELONGS_TO_COMMUNITY relationships (from entities to this community)
				in_belongs_rel_query = "MATCH ()-[r:BELONGS_TO_COMMUNITY]->(c:Community {community_id: $cid}) DELETE r"
				self.execute_query(in_belongs_rel_query, {"cid": community_id_to_check})

				# Then delete the community node
				delete_community_query = "MATCH (c:Community {community_id: $cid}) DELETE c"
				self.execute_query(delete_community_query, {"cid": community_id_to_check})
				logger.info(f"Deleted empty community: {community_id_to_check}")

				# Recursively check the parent only if it exists
				if parent_community_id:
					return self._cleanup_empty_communities(parent_community_id)
				return True

			logger.debug(
				f"Community {community_id_to_check} is not empty (entities: {entity_count}, "
				f"children: {child_community_count}). Skipping deletion."
			)
			return True
		except Exception:
			logger.exception(f"Error during community cleanup for {community_id_to_check}")
			return False

	def get_all_file_hashes(self) -> dict[str, str]:
		"""Retrieve all file paths and their git hashes from the database."""
		query = "MATCH (f:CodeFile) RETURN f.file_path, f.git_hash"
		results = self.execute_query(query)
		return {row[0]: row[1] for row in results} if results else {}

	def add_embedding(self, entity_id: str, embedding: list[float]) -> None:
		"""Add or update the embedding for a CodeEntity."""
		if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
			logger.error(f"Invalid embedding format for entity {entity_id}. Must be list[float].")
			return
		if len(embedding) != VECTOR_DIMENSION:
			logger.error(
				f"Invalid embedding dimension for entity {entity_id}. "
				f"Expected {VECTOR_DIMENSION}, got {len(embedding)}."
			)
			return

		query = "MATCH (e:CodeEntity {entity_id: $entity_id}) SET e.embedding = $embedding"
		self.execute_query(query, {"entity_id": entity_id, "embedding": embedding})

	# --- Vector Index Methods ---

	def create_vector_index(
		self,
		table_name: str = VECTOR_TABLE_NAME,
		index_name: str = VECTOR_INDEX_NAME,
		column_name: str = VECTOR_COLUMN_NAME,
	) -> bool:
		"""
		Creates the HNSW vector index using the CALL procedure.

		Args:
		        table_name (str): The name of the table to index.
		        index_name (str): The name to give the vector index.
		        column_name (str): The name of the column containing embeddings.

		Returns:
		        bool: True if the index was created successfully, False otherwise.

		"""
		if not self.conn:
			logger.error("Cannot create vector index, KuzuDB connection not established.")
			return False

		# Clean, simple CALL syntax
		query = f"CALL CREATE_VECTOR_INDEX('{table_name}', '{index_name}', '{column_name}')"

		try:
			logger.info(f"Creating vector index '{index_name}' on {table_name}.{column_name}...")
			self.conn.execute(query)
			logger.info(f"Vector index '{index_name}' created successfully.")
			return True
		except Exception:
			logger.exception(f"Failed to create vector index '{index_name}'")
			return False

	def drop_vector_index(
		self,
		table_name: str = VECTOR_TABLE_NAME,
		index_name: str = VECTOR_INDEX_NAME,
	) -> bool:
		"""
		Drops the vector index from the database.

		Args:
		        table_name (str): The name of the table containing the index.
		        index_name (str): The name of the vector index.

		Returns:
		        bool: True if the index was dropped or didn't exist, False on error.

		"""
		if not self.conn:
			logger.error("Cannot drop vector index, KuzuDB connection not established.")
			return False

		# Clean, simple CALL syntax
		query = f"CALL DROP_VECTOR_INDEX('{table_name}', '{index_name}')"

		try:
			self.conn.execute(query)
			logger.info(f"Successfully dropped vector index '{index_name}' from table '{table_name}'.")
			return True
		except RuntimeError as e:
			# Gracefully handle the case when the index doesn't exist
			if "doesn't have an index with name" in str(e):
				logger.info(f"Index '{index_name}' on table '{table_name}' doesn't exist, nothing to drop.")
				return True
			logger.exception(f"Failed to drop vector index '{index_name}' via CALL")
			return False

	def query_vector_index(
		self,
		query_vector: list[float],
		k: int = 5,
		table_name: str = VECTOR_TABLE_NAME,
		index_name: str = VECTOR_INDEX_NAME,
	) -> list[dict[str, Any]] | None:
		"""Queries the vector index for the k nearest neighbors."""
		if not self.conn:
			logger.error("Cannot query vector index, KuzuDB connection not established.")
			return None

		if len(query_vector) != VECTOR_DIMENSION:
			logger.error(f"Query vector dimension mismatch. Expected {VECTOR_DIMENSION}, got {len(query_vector)}.")
			return None

		try:
			# Simplified query format with no comments and proper parameter syntax
			query = f"CALL QUERY_VECTOR_INDEX('{table_name}', '{index_name}', $query_vector, {k}) RETURN node.entity_id AS entity_id, distance ORDER BY distance;"  # noqa: E501
			params = {"query_vector": query_vector}

			logger.debug(f"Executing vector query for top {k} neighbors...")
			results_raw = self.execute_query(query, params)

			if results_raw is None:
				logger.error("Vector query execution failed.")
				return None

			# execute_query returns list[list[Any]], convert to list[dict]
			formatted_results = []
			for row in results_raw:
				if len(row) == 2:  # noqa: PLR2004
					formatted_results.append({"entity_id": row[0], "distance": row[1]})
				else:
					logger.warning(f"Unexpected row format in vector query result: {row}")

			logger.debug(f"Vector query returned {len(formatted_results)} results.")
			return formatted_results

		except Exception:
			logger.exception("Failed to execute vector index query")
			return None
