"""Graph database management for KuzuDB."""

import logging
from pathlib import Path
from typing import Any

import kuzu
from kuzu import QueryResult

from codemap.processor.tree_sitter.base import EntityType
from codemap.utils.file_utils import ensure_directory_exists
from codemap.utils.path_utils import get_cache_path

logger = logging.getLogger(__name__)

# Define a default filename within the component's cache directory
DEFAULT_DB_FILE_NAME = "codemap_graph.db"


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
			graph_cache_dir = get_cache_path("graph")
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

		self.db: kuzu.Database | None = None
		self.conn: kuzu.Connection | None = None
		try:
			self._connect()
			self.ensure_schema()
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
		except Exception:
			logger.exception(f"Failed to execute KuzuDB query: {query}")
			return None

	def _create_table_if_not_exists(self, create_statement: str, table_name: str) -> None:
		"""Helper to create a table only if it doesn't exist."""
		# Kuzu doesn't have explicit "IF NOT EXISTS" for CREATE TABLE
		# We need to check if the table exists by querying system tables or trying to query the table
		# Simple approach: Try to query, if it fails with a specific error, create it.
		# More robust: Query SHOW TABLES; (check Kuzu documentation for exact system query)
		try:
			# Check if table exists by trying a simple query
			# Adjust this query based on actual table primary key or structure
			check_query = f"MATCH (n:{table_name}) RETURN n LIMIT 1;"
			self.execute_query(check_query)
			logger.debug(f"Table {table_name} already exists.")
		except Exception as e:
			# Check if the error indicates the table doesn't exist
			# This error message might change between Kuzu versions!
			if f"Table {table_name} does not exist." in str(e):
				logger.info(f"Table {table_name} does not exist. Creating table...")
				self.execute_query(create_statement)
				logger.info(f"Successfully created table {table_name}.")
			else:
				# Re-raise other unexpected errors
				logger.exception(f"Unexpected error checking table {table_name}")
				raise

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
			"content_summary STRING, PRIMARY KEY (entity_id))",
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
			try:
				self._create_table_if_not_exists(stmt, name)
			except Exception:
				logger.exception(f"Failed during schema creation for node table {name}")
				return  # Stop if schema creation fails

		# Create Relationship Tables
		for name, (_, _, _, stmt) in rel_tables.items():
			try:
				self._create_table_if_not_exists(stmt, name)
			except Exception:
				logger.exception(f"Failed during schema creation for rel table {name}")
				return  # Stop if schema creation fails

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
		query_node = "MERGE (e:CodeEntity {entity_id: $entity_id}) SET e += $props"
		self.execute_query(query_node, {"entity_id": entity_id, "props": entity_props})

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
		prop_str = "" if not properties else " { " + ", ".join([f"{k}: ${k}" for k in properties]) + " }"
		query = (
			f"MATCH (a:CodeEntity {{entity_id: $from_id}}), (b:CodeEntity {{entity_id: $to_id}}) "
			f"MERGE (a)-[r:{rel_type}{prop_str}]->(b)"
		)

		params = {"from_id": from_entity_id, "to_id": to_entity_id}
		if properties:
			params.update(properties)

		self.execute_query(query, params)

	def delete_file_data(self, file_path: str) -> None:
		"""Delete a CodeFile node and all its directly contained entities and relationships."""
		# This is complex: needs to delete file, its entities, and potentially cleanup communities
		# Step 1: Delete relationships originating *from* entities within the file (like CALLS, IMPORTS)
		# This requires finding all entities first.
		# Step 2: Delete the file node and its contained entities using DETACH DELETE
		# Step 3: Potentially delete communities if they become empty

		logger.warning(f"delete_file_data for {file_path} - partial implementation.")

		# Find entities belonging to the file
		find_entities_query = """
        MATCH (f:CodeFile {file_path: $file_path})-[:CONTAINS_ENTITY]->(e:CodeEntity)
        RETURN e.entity_id
        """
		entity_results = self.execute_query(find_entities_query, {"file_path": file_path})
		[row[0] for row in entity_results] if entity_results else []

		# Step 1 (Simplified): DETACH DELETE will handle relationships attached to the nodes being deleted.
		# If more complex cleanup needed (e.g., relationships *to* these entities from elsewhere),
		# more queries are required.

		# Step 2: Delete File and its Entities
		delete_query = """
        MATCH (f:CodeFile {file_path: $file_path})
        OPTIONAL MATCH (f)-[:CONTAINS_ENTITY]->(e:CodeEntity)
        DETACH DELETE f, e
        """
		self.execute_query(delete_query, {"file_path": file_path})
		logger.info(f"Deleted CodeFile and associated entities for: {file_path}")

		# Step 3: Community cleanup
		self._cleanup_empty_communities(f"file:{file_path}")

	def _cleanup_empty_communities(self, community_id_to_check: str) -> None:
		"""Recursively check and delete empty communities starting from a given ID."""
		if not community_id_to_check:
			return

		logger.debug(f"Checking community for cleanup: {community_id_to_check}")

		# Check if any entities still belong to this community
		check_entities_query = (
			"MATCH (e:CodeEntity)-[:BELONGS_TO_COMMUNITY]->(c:Community {community_id: $cid}) RETURN count(e)"
		)
		entity_count_result = self.execute_query(check_entities_query, {"cid": community_id_to_check})
		entity_count = entity_count_result[0][0] if entity_count_result else 0

		# Check if any child communities still belong to this community (only relevant for directories)
		child_community_count = 0
		if community_id_to_check.startswith("dir:"):  # Only check children for directory communities
			check_children_query = (
				"MATCH (parent:Community {community_id: $cid})<-[:PARENT_COMMUNITY]-(child:Community) "
				"RETURN count(child)"
			)
			child_count_result = self.execute_query(check_children_query, {"cid": community_id_to_check})
			child_community_count = child_count_result[0][0] if child_count_result else 0

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

			# Delete the empty community
			delete_community_query = "MATCH (c:Community {community_id: $cid}) DETACH DELETE c"
			self.execute_query(delete_community_query, {"cid": community_id_to_check})
			logger.info(f"Deleted empty community: {community_id_to_check}")

			# Recursively check the parent only if it exists
			if parent_community_id:
				self._cleanup_empty_communities(parent_community_id)
		else:
			logger.debug(
				f"Community {community_id_to_check} is not empty (entities: {entity_count}, "
				f"children: {child_community_count}). Skipping deletion."
			)

	def get_all_file_hashes(self) -> dict[str, str]:
		"""Retrieve all file paths and their git hashes from the database."""
		query = "MATCH (f:CodeFile) RETURN f.file_path, f.git_hash"
		results = self.execute_query(query)
		return {row[0]: row[1] for row in results} if results else {}
