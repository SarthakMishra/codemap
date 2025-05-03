"""
Unified pipeline for CodeMap data processing, synchronization, and retrieval.

This module defines the `ProcessingPipeline`, which acts as the central orchestrator
for managing and interacting with both the Kuzu graph database and the Milvus
vector database. It handles initialization, synchronization with the Git repository,
and provides various retrieval methods like semantic search, graph querying,
and Graph-Enhanced Vector Search.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Import Progress and TaskID for type hinting
from codemap.processor.graph.graph_builder import GraphBuilder
from codemap.processor.graph.kuzu_manager import KuzuManager
from codemap.processor.graph.synchronizer import GraphSynchronizer
from codemap.processor.tree_sitter import TreeSitterAnalyzer
from codemap.processor.utils.embedding_utils import MODEL_NAME as EMBEDDING_MODEL_NAME
from codemap.processor.utils.embedding_utils import generate_embedding
from codemap.processor.utils.git_utils import get_git_tracked_files
from codemap.processor.utils.sync_utils import compare_states
from codemap.utils.config_loader import ConfigLoader
from codemap.utils.file_utils import read_file_content
from codemap.utils.path_utils import find_project_root

if TYPE_CHECKING:
	from collections.abc import Callable

	from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)

EXPECTED_METADATA_COLUMNS = 9  # entity_id, name, type, start, end, sig, doc, summary, file_path


class ProcessingPipeline:
	"""
	Orchestrates data processing, synchronization, and retrieval for CodeMap.

	Manages connections and interactions with KuzuDB (graph) and Milvus
	(vector) databases, ensuring they are synchronized with the Git
	repository state. Provides methods for semantic search, graph querying,
	and combined graph-enhanced vector search.

	"""

	def __init__(
		self,
		repo_path: Path | None = None,
		config_loader: ConfigLoader | None = None,
		sync_on_init: bool = True,
		progress: Progress | None = None,
		task_id: TaskID | None = None,
	) -> None:
		"""
		Initialize the unified processing pipeline.

		Args:
		        repo_path: Path to the repository root. If None, it will be determined.
		        config_loader: Application configuration loader. If None, a default one is created.
		        sync_on_init: If True, run database synchronization during initialization.
		        progress: Optional rich Progress instance for unified status display.
		        task_id: Optional rich TaskID for the main initialization/sync task.

		"""
		self.repo_path = repo_path or find_project_root()
		if not self.repo_path:
			msg = "Repository path could not be determined."
			raise ValueError(msg)

		self.config_loader = config_loader or ConfigLoader(str(self.repo_path))
		self.config = self.config_loader.load_config()  # Load the actual config data

		logger.info("Initializing Processing Pipeline for repository: %s", self.repo_path)

		# --- Initialize Shared Components ---
		self.analyzer = TreeSitterAnalyzer()  # Assumes Analyzer doesn't need config for now
		# Store embedding model name for later use with generate_embeddings
		self.embedding_model_name = self.config.get("embedding", {}).get("model_name", EMBEDDING_MODEL_NAME)

		# --- Initialize KuzuDB Components ---
		kuzu_db_path = self.config.get("graph", {}).get("database_path")  # Allow override from config
		self.kuzu_manager = KuzuManager(db_path=kuzu_db_path)  # Uses default path if None
		self.graph_builder = GraphBuilder(self.repo_path, self.kuzu_manager, self.analyzer)
		self.graph_synchronizer = GraphSynchronizer(self.repo_path, self.kuzu_manager, self.graph_builder)
		logger.info("KuzuDB components initialized.")

		# Milvus components removed, vector search handled by Kuzu
		self.has_vector_db = True  # Keep flag for now, indicates vector capability exists

		# --- Check if initial sync is needed (only if sync_on_init is False) ---
		if not sync_on_init:
			try:
				# Check Kuzu graph emptiness
				kuzu_check_query = "MATCH (f:CodeFile) RETURN count(f)"
				kuzu_result = self.kuzu_manager.execute_query(kuzu_check_query)

				needs_sync = False  # Default to not needing sync unless proven empty
				if kuzu_result and isinstance(kuzu_result, (list, tuple)) and len(kuzu_result) > 0:
					count_val = None
					if isinstance(kuzu_result[0], (list, tuple)) and len(kuzu_result[0]) > 0:
						count_val = kuzu_result[0][0]
					elif isinstance(kuzu_result[0], int):  # Handle direct count result
						count_val = kuzu_result[0]

					if count_val == 0:
						logger.info("Kuzu graph appears empty. Triggering initial synchronization.")
						needs_sync = True
					elif count_val is not None:
						logger.info(
							f"Found {count_val} files in Kuzu. Skipping initial sync triggered by emptiness check."
						)
					else:  # Unexpected result format, log but don't force sync
						logger.warning(
							(
								f"Unexpected Kuzu count result format: {kuzu_result}.",
								"Assuming sync not needed based on this check.",
							)
						)

				else:  # Query failed or returned empty/unexpected result
					logger.warning("Could not determine Kuzu graph state from count query. Assuming sync is needed.")
					needs_sync = True  # Force sync if we can't verify state

				if needs_sync:
					logger.info("Triggering database synchronization due to empty Kuzu graph or failed check.")
					# Pass progress context if available
					self.sync_databases(progress=progress, task_id=task_id)
				else:
					logger.info("Initial synchronization skipped based on Kuzu check.")
					# If not syncing, update progress description if context is provided
					if progress and task_id:
						progress.update(task_id, description="[green]✓[/green] Pipeline initialized (sync skipped).")

			except Exception:
				logger.exception("Error checking Kuzu state for initial sync. Skipping automatic sync.")
				# Update progress if context is provided
				if progress and task_id:
					progress.update(task_id, description="[yellow]⚠[/yellow] Pipeline initialized (sync check failed).")
		elif sync_on_init:  # Explicitly handle sync_on_init=True case
			logger.info("`sync_on_init` is True. Performing database synchronization...")
			# Pass progress context if available
			self.sync_databases(progress=progress, task_id=task_id)
			logger.info("Synchronization triggered by `sync_on_init` complete.")
		# --- End Check ---
		# Update progress if context is provided
		elif progress and task_id:
			progress.update(task_id, description="[green]✓[/green] Pipeline initialized (sync skipped).")

		logger.info(f"ProcessingPipeline initialized for repo: {self.repo_path}")

	def stop(self) -> None:
		"""Stops the pipeline and releases resources."""
		logger.info("Stopping ProcessingPipeline...")
		if self.kuzu_manager:
			try:
				self.kuzu_manager.close()
				logger.info("KuzuDB connection closed.")
			except Exception:
				logger.exception("Error closing KuzuDB connection.")
			self.kuzu_manager = None

		logger.info("ProcessingPipeline stopped.")

	# --- Synchronization ---

	def sync_databases(self, progress: Progress | None = None, task_id: TaskID | None = None) -> None:
		"""
		Synchronize both KuzuDB and Milvus databases with the Git repository state.

		Fetches Git state once and passes it to both synchronizers.

		Args:
		    progress: Optional rich Progress instance for status updates.
		    task_id: Optional rich TaskID for the main sync task.

		"""
		logger.info("Starting database synchronization...")
		if progress and task_id is not None:
			progress.update(task_id, description="Preparing for synchronization...", completed=0)

		# Fetch Git state once
		logger.debug("Fetching current Git state...")
		if progress and task_id:
			progress.update(task_id, description="Fetching Git state...", completed=5)

		current_git_files = get_git_tracked_files(self.repo_path)
		if current_git_files is None:
			logger.error("Synchronization failed: Could not get Git tracked files.")
			if progress and task_id:
				progress.update(task_id, description="[red]Error:[/red] Failed to get Git state.")
			return

		git_files_count = len(current_git_files)
		logger.debug(f"Found {git_files_count} files in Git.")
		if progress and task_id:
			progress.update(task_id, description=f"Found {git_files_count} Git files.", completed=10)

		# Determine files to process *before* potentially updating progress total
		try:
			# Use the public method now
			if progress and task_id:
				progress.update(task_id, description="Querying database state...", completed=15)

			existing_db_files = self.graph_synchronizer.get_db_file_hashes()
			if existing_db_files is None:
				logger.error("Graph sync failed: Could not retrieve existing file hashes from Kuzu.")
				if progress and task_id:
					progress.update(task_id, description="[red]Error:[/red] Failed querying DB state")
				return  # Return None as per function signature

			if progress and task_id:
				progress.update(task_id, description="Comparing Git and DB states...", completed=20)

			files_to_add, files_to_update, files_to_delete = compare_states(current_git_files, existing_db_files)
			files_to_process_count = len(files_to_add) + len(files_to_update)

			# --- DEBUG LOGGING ---
			logger.debug(
				f"Sync comparison: Add={len(files_to_add)}, "
				f"Update={len(files_to_update)}, "
				f"Delete={len(files_to_delete)}, "
				f"Process={files_to_process_count}"
			)
			# --- END DEBUG ---

		except Exception:
			logger.exception("Error comparing Git state with DB state.")
			if progress and task_id:
				progress.update(task_id, description="[red]Error:[/red] Failed comparing states")
			return  # Abort sync

		if progress and task_id:
			description = (
				f"Found {len(files_to_add)} to add, {len(files_to_update)} to update, {len(files_to_delete)} to delete"
			)
			progress.update(task_id, description=description, completed=25)

		sync_errors = False

		# --- Sync Graph Database ---
		try:
			logger.info("Synchronizing Kuzu graph database...")
			if progress and task_id:
				progress.update(task_id, description="Starting graph database sync...", completed=30)

			# Pass progress context down
			sync_success_graph = self.graph_synchronizer.sync_graph(
				current_git_files,
				progress=progress,
				task_id=task_id,
			)

			if sync_success_graph:
				logger.info("Kuzu graph database synchronization completed successfully.")
				if progress and task_id:
					progress.update(task_id, completed=100, description="[green]✓[/green] Synchronization complete.")
			else:
				logger.warning("Kuzu graph database synchronization failed or had issues.")
				sync_errors = True
				if progress and task_id:
					progress.update(
						task_id, completed=100, description="[yellow]⚠[/yellow] Sync completed with warnings."
					)
		except Exception:
			logger.exception("Error during Kuzu graph synchronization.")
			if progress and task_id:
				progress.update(task_id, description="[red]Error:[/red] Graph sync failed.", completed=100)
			sync_errors = True

		# Final status reporting
		if sync_errors:
			logger.error("Database synchronization process finished with errors.")
		else:
			logger.info("Database synchronization process finished successfully.")

	def sync_databases_simple(self, progress_callback: Callable[[int], None] | None = None) -> bool:
		"""
		Synchronize databases with a simple callback-based progress reporting.

		Args:
		        progress_callback: Optional function that accepts an integer percentage (0-100)
		                                                to update progress display

		Returns:
		        True if synchronization was successful, False otherwise

		"""
		logger.info("Starting database synchronization (simple mode)...")
		if progress_callback:
			progress_callback(5)  # Start at 5%

		# Fetch Git state
		logger.debug("Fetching current Git state...")
		current_git_files = get_git_tracked_files(self.repo_path)
		if current_git_files is None:
			logger.error("Synchronization failed: Could not get Git tracked files.")
			return False

		if progress_callback:
			progress_callback(15)

		logger.debug(f"Found {len(current_git_files)} files in Git.")

		# Get DB state and compare
		try:
			existing_db_files = self.graph_synchronizer.get_db_file_hashes()
			if existing_db_files is None:
				logger.error("Graph sync failed: Could not retrieve existing file hashes from Kuzu.")
				return False

			if progress_callback:
				progress_callback(25)

			# Compare states to determine changes
			files_to_add, files_to_update, files_to_delete = compare_states(current_git_files, existing_db_files)
			files_to_process_count = len(files_to_add) + len(files_to_update)

			if progress_callback:
				progress_callback(35)

			# Log change summary
			logger.debug(
				f"Sync comparison: Add={len(files_to_add)}, "
				f"Update={len(files_to_update)}, "
				f"Delete={len(files_to_delete)}, "
				f"Process={files_to_process_count}"
			)

		except Exception:
			logger.exception("Error comparing Git state with DB state.")
			return False

		sync_success = True

		# Process deletions (if any)
		if files_to_delete:
			logger.info(f"Deleting {len(files_to_delete)} files...")
			try:
				success = self.graph_synchronizer.delete_files_from_graph(files_to_delete)
				if not success:
					logger.warning("Some files could not be deleted.")
					sync_success = False
			except Exception:
				logger.exception("Error during file deletion.")
				sync_success = False

			if progress_callback:
				progress_callback(45)

		# Process additions and updates
		files_to_process = files_to_add.union(files_to_update)
		if files_to_process:
			logger.info(f"Processing {len(files_to_process)} files...")

			# Setup progress reporting
			processed = 0
			total = len(files_to_process)
			progress_start = 50
			progress_end = 95
			progress_range = progress_end - progress_start

			# Process each file
			for file_path_str in files_to_process:
				file_path = self.repo_path / file_path_str
				if not file_path.is_file():
					logger.warning(f"File not found: {file_path}")
					processed += 1
					continue

				# Delete existing data if this is an update
				if file_path_str in files_to_update:
					self.graph_synchronizer.delete_file_from_graph(file_path_str)

				# Process the file
				try:
					git_hash = current_git_files.get(file_path_str)
					self.graph_builder.process_file(file_path, git_hash=git_hash)
				except Exception:
					logger.exception(f"Error processing file: {file_path}")
					sync_success = False

				# Update progress
				processed += 1
				if progress_callback:
					percent = int(progress_start + (processed / total) * progress_range)
					progress_callback(percent)

			# Rebuild vector index
			if progress_callback:
				progress_callback(96)

			try:
				logger.info("Rebuilding vector index...")
				if self.kuzu_manager is not None:  # Add None check
					if not self.kuzu_manager.drop_vector_index():
						logger.warning("Failed to drop vector index.")

					if self.kuzu_manager.create_vector_index():
						logger.info("Vector index rebuilt successfully.")
					else:
						logger.error("Failed to rebuild vector index.")
				else:
					logger.error("Cannot rebuild vector index: KuzuManager is None")
					sync_success = False
			except Exception:
				logger.exception("Error rebuilding vector index.")
				sync_success = False

			if progress_callback:
				progress_callback(100)

		logger.info(f"Synchronization finished. Success: {sync_success}")
		return sync_success

	# --- Retrieval Methods ---

	def semantic_search(self, query: str, k: int = 5) -> list[dict[str, Any]] | None:
		"""
		Perform semantic search for code entities similar to the query using Kuzu vector index.

		Args:
		        query: The search query string.
		        k: The number of top similar results to retrieve.
		        # filters: Kuzu filter support via projected graphs might be added later.

		Returns:
		        A list of search result dictionaries (including metadata and distance),
		        or None if an error occurs or KuzuManager is unavailable.

		"""
		if not self.kuzu_manager:
			logger.error("KuzuManager not available for semantic search.")
			return None

		logger.debug("Performing semantic search for query: '%s', k=%d", query, k)

		try:
			# 1. Generate query embedding
			query_embedding = generate_embedding(query)  # Use new embedding function
			if query_embedding is None:
				logger.error("Failed to generate embedding for query.")
				return None

			# 2. Query Kuzu vector index
			vector_results = self.kuzu_manager.query_vector_index(query_embedding, k)
			if vector_results is None:
				logger.error("Kuzu vector index query failed.")
				return None
			if not vector_results:
				logger.debug("Kuzu vector index query returned no results.")
				return []

			# 3. Fetch metadata for each result ID from Kuzu
			entity_ids = [res["entity_id"] for res in vector_results]
			# Important: Kuzu parameter substitution for lists in WHERE IN might require specific formatting
			# or might not be directly supported in all contexts. Using multiple queries or MATCH with OR is safer.
			# Let's construct a MATCH query with OR for simplicity and safety.
			if not entity_ids:
				return []
			match_clauses = [f"(e:CodeEntity {{entity_id: '{eid}'}})" for eid in entity_ids]
			match_query = (
				"MATCH "
				+ "\nUNION MATCH ".join(match_clauses)
				+ "\nRETURN e.entity_id, e.name, e.entity_type, "
				+ "e.start_line, e.end_line, e.signature, "
				+ "e.docstring, e.content_summary, e.file_path"
			)

			metadata_results = self.kuzu_manager.execute_query(match_query)
			if metadata_results is None:
				logger.error("Failed to fetch metadata for vector search results from Kuzu.")
				return None  # Or maybe return results without metadata?

			# Create a lookup map for metadata
			metadata_map = {
				row[0]: {
					"entity_id": row[0],
					"name": row[1],
					"entity_type": row[2],
					"start_line": row[3],
					"end_line": row[4],
					"signature": row[5],
					"docstring": row[6],
					"content_summary": row[7],
					"file_path": row[8],
					# Add other fields as needed, e.g., chunk_text if stored/derivable
				}
				for row in metadata_results
				if len(row) == EXPECTED_METADATA_COLUMNS
			}

			# 4. Format results
			formatted_results = []
			for vector_hit in vector_results:
				entity_id = vector_hit["entity_id"]
				metadata = metadata_map.get(entity_id)
				if metadata:
					formatted_results.append(
						{
							"id": entity_id,  # Use Kuzu entity_id as the primary ID
							"distance": vector_hit["distance"],
							"metadata": metadata,
							# Note: 'kuzu_entity_id' is now just 'id' and also present within metadata
						}
					)
				else:
					# Log warning but don't add metadata if lookup failed
					logger.warning(f"Metadata not found for vector hit with entity_id: {entity_id}")
					# Added pass to fix linter error

			logger.debug("Semantic search found %d results.", len(formatted_results))
			return formatted_results

		except Exception:
			logger.exception("Error during semantic search.")
			return None

	def graph_query(self, cypher_query: str, params: dict[str, Any] | None = None) -> list[list[Any]] | None:
		"""
		Execute a raw Cypher query against the Kuzu graph database.

		Args:
		        cypher_query: The Cypher query string.
		        params: Optional dictionary of parameters to pass to the query.

		Returns:
		        A list of results (each result is a list of values), or None on error.

		"""
		if not self.kuzu_manager:
			logger.error("KuzuManager not available for graph query.")
			return None
		logger.debug("Executing graph query: %s with params: %s", cypher_query, params)
		try:
			return self.kuzu_manager.execute_query(cypher_query, params)
		except Exception:
			logger.exception("Error executing graph query.")
			return None

	def graph_enhanced_search(
		self,
		query: str,
		k_vector: int = 5,
		graph_depth: int = 1,
		include_source_code: bool = False,
	) -> list[dict[str, Any]] | None:
		"""
		Perform graph-enhanced vector search (GraphRAG pattern).

		Combines initial vector search results with context retrieved from
		graph traversal starting from the initial hits.

		Args:
		        query: The user's search query.
		        k_vector: The number of initial similar chunks to retrieve via vector search.
		        graph_depth: The maximum depth for graph traversal from initial hits.
		        include_source_code: If True, fetch and include source code snippets
		                                                 for related graph entities.

		Returns:
		        A list of augmented search results, where each result includes the
		        original vector hit and related graph entities, or None on error.

		"""
		logger.info("Starting graph-enhanced search for query: '%s'", query)

		# --- Step 1: Initial Vector Search ---
		initial_hits = self.semantic_search(query, k=k_vector)
		if not initial_hits:
			logger.warning("Graph-enhanced search: Initial vector search yielded no results.")
			return []

		# --- Step 2: Identify Seed Graph Nodes --- #
		# NOTE: This assumes `kuzu_entity_id` is stored as metadata in Milvus during indexing.
		# If not, this step needs modification to derive the Kuzu ID from other metadata.
		seed_entity_ids = []
		milvus_hit_map = {}  # Store original hits for later merging
		for hit in initial_hits:
			entity_id = hit.get("id")  # Get ID from the top level of the new semantic_search result

			if entity_id:
				seed_entity_ids.append(entity_id)
				milvus_hit_map[entity_id] = hit
			else:
				# If ID is missing from semantic search result, log appropriately
				logger.warning(f"Could not determine entity ID for semantic search hit: {hit}")

		if not seed_entity_ids:
			logger.warning("Graph-enhanced search: No entity IDs found in initial semantic search hits.")
			# Return only the initial vector hits if no graph mapping possible
			return initial_hits

		# --- Step 3: Graph Traversal --- #
		# Query to find related entities within the specified depth.
		# This traverses all relationships; refine if specific connections are needed.
		cypher_query = """
		MATCH (initial_entity)
		WHERE initial_entity.entity_id IN $seed_entity_ids
		CALL {
			WITH initial_entity
			MATCH path = (initial_entity)-[*1..$graph_depth]-(related_entity)
			WHERE related_entity IS NOT NULL
			RETURN initial_entity.entity_id AS origin_id, related_entity AS related_node
			UNION
			WITH initial_entity // Include the initial node itself
			RETURN initial_entity.entity_id AS origin_id, initial_entity AS related_node
		}
		RETURN DISTINCT origin_id,
			   related_node.entity_id AS related_entity_id,
			   labels(related_node) AS related_entity_labels,
			   properties(related_node) AS related_entity_data
		"""
		params = {"seed_entity_ids": seed_entity_ids, "graph_depth": graph_depth}

		graph_results_raw = self.graph_query(cypher_query, params)

		# --- Step 4: Augment Results --- #
		augmented_results_map: dict[str, dict[str, Any]] = {}
		# Initialize with original hits
		for entity_id, hit_data in milvus_hit_map.items():
			augmented_results_map[entity_id] = {
				"vector_hit": hit_data,  # Store the original hit under 'vector_hit'
				"graph_context": [],
			}

		if graph_results_raw:
			for row in graph_results_raw:
				origin_id, related_id, related_labels, related_data = row
				if origin_id in augmented_results_map:
					graph_node_info = {
						"id": related_id,
						"labels": related_labels,
						"properties": related_data,
					}

					# --- Optional: Fetch Source Code --- #
					if (
						include_source_code
						and related_data
						and "file_path" in related_data
						and "start_line" in related_data
						and "end_line" in related_data
					):
						try:
							file_path = self.repo_path / related_data["file_path"]
							# Ensure start/end lines are valid integers
							start = int(related_data["start_line"])
							end = int(related_data["end_line"])
							if file_path.is_file() and start >= 1 and end >= start:
								content = read_file_content(file_path)
								lines = content.splitlines()
								# Adjust for 0-based list indexing vs 1-based lines
								snippet_lines = lines[start - 1 : end]
								graph_node_info["source_code"] = "\n".join(snippet_lines)
							else:
								graph_node_info["source_code"] = None  # File exists but lines invalid
						except (ValueError, IndexError, FileNotFoundError, Exception) as e:
							logger.warning(f"Could not fetch source code for entity {related_id}: {e}")
							graph_node_info["source_code"] = None  # Indicate failure

					# Avoid adding duplicate nodes (e.g., initial node appearing via traversal)
					existing_ids = {node["id"] for node in augmented_results_map[origin_id]["graph_context"]}
					if related_id not in existing_ids:
						augmented_results_map[origin_id]["graph_context"].append(graph_node_info)

		# Convert map back to list in the order of initial hits for consistency
		final_results = [augmented_results_map[eid] for eid in seed_entity_ids if eid in augmented_results_map]

		logger.info("Graph-enhanced search completed, returning %d augmented results.", len(final_results))
		return final_results

	# --- Utility / Refactored Methods ---

	def get_repository_structure(self) -> dict[str, Any] | None:
		"""
		Get a structured representation of the repository by querying the Kuzu graph.

		Returns:
		        A hierarchical dictionary representing the repository structure,
		        or None if an error occurs.

		"""
		if not self.kuzu_manager:
			logger.error("KuzuManager not available for get_repository_structure.")
			return None

		logger.debug("Building repository structure from Kuzu graph...")

		# Query to get all files and directories (communities)
		# Assumes: CodeFile nodes exist for files.
		# Assumes: CodeCommunity nodes exist for directories.
		# Assumes: PARENT_COMMUNITY relates directories.
		# Assumes: BELONGS_TO_COMMUNITY relates files to their parent directory.
		query = """
		MATCH (n)
		WHERE n:CodeFile OR n:CodeCommunity
		OPTIONAL MATCH (n)-[:PARENT_COMMUNITY]->(p:CodeCommunity)
		OPTIONAL MATCH (n)-[:BELONGS_TO_COMMUNITY]->(dir:CodeCommunity)
		RETURN n.entity_id AS id,
			   labels(n) AS labels,
			   n.name AS name,
			   n.path AS path,
			   p.entity_id AS parent_dir_id,
			   dir.entity_id AS file_parent_dir_id
		"""

		try:
			results_raw = self.graph_query(query)
			if results_raw is None:
				logger.error("Failed to execute Kuzu query for repository structure.")
				return None

			# Process flat results into a hierarchical structure
			nodes = {}
			children_map: dict[str, list[str]] = {}

			for row in results_raw:
				(node_id, labels, name, path_str, parent_dir_id, file_parent_dir_id) = row
				if not node_id:
					continue

				node_type = "directory" if "CodeCommunity" in labels else "file"
				parent_id = parent_dir_id if node_type == "directory" else file_parent_dir_id

				nodes[node_id] = {
					"id": node_id,
					"type": node_type,
					"name": name or Path(path_str).name if path_str else "Unknown",
					"path": path_str,
					"children": [],
				}

				if parent_id:
					if parent_id not in children_map:
						children_map[parent_id] = []
					children_map[parent_id].append(node_id)

			# Build the tree structure
			repository_root = None
			root_candidates = []

			for node_id, node_data in nodes.items():
				if node_id in children_map:
					# Sort children by name for consistent output
					children_ids = sorted(children_map[node_id], key=lambda cid: nodes[cid]["name"])
					node_data["children"] = [nodes[child_id] for child_id in children_ids]
				# Identify root candidates (nodes without a parent in the map)
				is_child = any(node_id in children for children in children_map.values())
				if not is_child:
					root_candidates.append(node_data)

			# Determine the true root (usually the one matching repo name or path)
			if len(root_candidates) == 1:
				repository_root = root_candidates[0]
			else:
				# Attempt to find the root based on repo path or name
				for candidate in root_candidates:
					if candidate["type"] == "directory" and (
						candidate["path"] == str(self.repo_path) or candidate["name"] == self.repo_path.name
					):
						repository_root = candidate
						break
				if not repository_root and root_candidates:
					logger.warning(
						"Could not definitively determine repository root node from Kuzu. Using first candidate."
					)
					repository_root = root_candidates[0]
				elif not repository_root:
					logger.error("Failed to build repository structure: No root node found.")
					return None

			logger.debug("Successfully built repository structure from Kuzu graph.")
			return repository_root

		except Exception:
			logger.exception("Error building repository structure from Kuzu.")
			return None
