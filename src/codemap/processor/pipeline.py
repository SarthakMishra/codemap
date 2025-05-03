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
from typing import Any

import numpy as np

from codemap.processor.graph.graph_builder import GraphBuilder
from codemap.processor.graph.kuzu_manager import KuzuManager
from codemap.processor.graph.synchronizer import GraphSynchronizer
from codemap.processor.tree_sitter import TreeSitterAnalyzer
from codemap.processor.utils.git_utils import get_git_tracked_files
from codemap.processor.vector.client import get_milvus_client
from codemap.processor.vector.embedder import generate_embeddings
from codemap.processor.vector.manager import synchronize_vectors
from codemap.processor.vector.schema import config as vector_config
from codemap.utils.config_loader import ConfigLoader
from codemap.utils.file_utils import read_file_content
from codemap.utils.path_utils import find_project_root

logger = logging.getLogger(__name__)


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
	) -> None:
		"""
		Initialize the unified processing pipeline.

		Args:
		        repo_path: Path to the repository root. If None, it will be determined.
		        config_loader: Application configuration loader. If None, a default one is created.
		        sync_on_init: If True, run database synchronization during initialization.

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
		self.embedding_model_name = self.config.get("embedding", {}).get(
			"model_name", vector_config.EMBEDDING_MODEL_NAME
		)

		# --- Initialize KuzuDB Components ---
		kuzu_db_path = self.config.get("graph", {}).get("database_path")  # Allow override from config
		self.kuzu_manager = KuzuManager(db_path=kuzu_db_path)  # Uses default path if None
		self.graph_builder = GraphBuilder(self.repo_path, self.kuzu_manager, self.analyzer)
		self.graph_synchronizer = GraphSynchronizer(self.repo_path, self.kuzu_manager, self.graph_builder)
		logger.info("KuzuDB components initialized.")

		# --- Initialize Milvus Components ---
		self.milvus_client = get_milvus_client()  # Ensures singleton connection and collection existence
		if not self.milvus_client:
			# Handle error: Milvus connection failed
			logger.error("Failed to initialize Milvus client. Vector operations will be unavailable.")
			self.has_vector_db = False
		else:
			logger.info("Milvus components initialized.")
			self.has_vector_db = True

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
					self.sync_databases()  # Run the full sync if needed
				else:
					logger.info("Initial synchronization skipped based on Kuzu check.")

			except Exception:
				logger.exception("Error checking Kuzu state for initial sync. Skipping automatic sync.")
		elif sync_on_init:  # Explicitly handle sync_on_init=True case
			logger.info("`sync_on_init` is True. Performing database synchronization...")
			self.sync_databases()
			logger.info("Synchronization triggered by `sync_on_init` complete.")
		# --- End Check ---

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

		if self.milvus_client:
			try:
				# Use the standard close method if available
				if hasattr(self.milvus_client, "close"):
					self.milvus_client.close()
					logger.info("Milvus client connection closed.")
				else:
					logger.warning("Could not find close method on Milvus client.")
			except Exception:
				logger.exception("Error closing Milvus client connection.")
			self.milvus_client = None

		logger.info("ProcessingPipeline stopped.")

	# --- Synchronization ---

	def sync_databases(self) -> None:
		"""
		Synchronize both KuzuDB and Milvus databases with the Git repository state.

		Fetches Git state once and passes it to both synchronizers.

		"""
		logger.info("Starting database synchronization...")

		# Fetch Git state once
		logger.debug("Fetching current Git state...")
		current_git_files = get_git_tracked_files(self.repo_path)
		if current_git_files is None:
			logger.error("Synchronization failed: Could not get Git tracked files.")
			return
		logger.debug(f"Found {len(current_git_files)} files in Git.")

		sync_errors = False

		# --- Sync Graph Database ---
		try:
			logger.info("Synchronizing Kuzu graph database...")
			sync_success_graph = self.graph_synchronizer.sync_graph(current_git_files)
			if sync_success_graph:
				logger.info("Kuzu graph database synchronization completed successfully.")
			else:
				logger.warning("Kuzu graph database synchronization failed or had issues.")
				sync_errors = True
		except Exception:
			logger.exception("Error during Kuzu graph synchronization.")
			sync_errors = True

		# --- Sync Vector Database ---
		if self.milvus_client:
			try:
				logger.info("Synchronizing Milvus vector database...")
				# Use the standalone function, passing repo path and fetched git state
				synchronize_vectors(
					repo_path=self.repo_path,
					current_git_files=current_git_files,
					kuzu_manager=self.kuzu_manager,
				)
				logger.info("Milvus vector database synchronization completed successfully.")
			except Exception:
				logger.exception("Error during Milvus vector synchronization.")
				sync_errors = True
		else:
			logger.warning("Vector database not initialized, skipping Milvus synchronization.")

		if sync_errors:
			logger.error("Database synchronization process finished with errors.")
		else:
			logger.info("Database synchronization process finished successfully.")

	# --- Retrieval Methods ---

	def semantic_search(self, query: str, k: int = 5, filters: str | None = None) -> list[dict[str, Any]] | None:
		"""
		Perform semantic search for code chunks similar to the query using Milvus.

		Args:
		        query: The search query string.
		        k: The number of top similar results to retrieve.
		        filters: Optional Milvus filter expression string (e.g., "language == 'python'").

		Returns:
		        A list of search result dictionaries (including metadata and distance),
		        or None if an error occurs or Milvus is unavailable.

		"""
		if not self.milvus_client:
			logger.error("Milvus client not available for semantic search.")
			return None

		logger.debug("Performing semantic search for query: '%s', k=%d, filters='%s'", query, k, filters)

		try:
			# 1. Generate query embedding
			query_embedding = generate_embeddings(query)  # Pass only query
			if query_embedding is None:
				logger.error("Failed to generate embedding for query.")
				return None

			# Prepare embedding for Milvus
			# For single query, query_embedding will be a numpy array
			if isinstance(query_embedding, np.ndarray):
				search_vectors = [query_embedding.tolist()]
			else:
				# If it's already a list or another format, use it directly
				search_vectors = [query_embedding]

			# 2. Prepare search parameters for MilvusClient.search
			search_params = {
				"metric_type": vector_config.METRIC_TYPE,
				"params": {"nprobe": 10},  # Example search param, adjust as needed
				"anns_field": vector_config.FIELD_EMBEDDING,  # Moved inside
			}
			output_fields = [
				vector_config.FIELD_ID,
				vector_config.FIELD_FILE_PATH,
				vector_config.FIELD_ENTITY_NAME,
				vector_config.FIELD_CHUNK_TYPE,
				vector_config.FIELD_START_LINE,
				vector_config.FIELD_END_LINE,
				vector_config.FIELD_CHUNK_TEXT,
				"kuzu_entity_id",  # Ensure this is stored in Milvus metadata
			]

			# 3. Execute search using MilvusClient signature
			results = self.milvus_client.search(
				collection_name=vector_config.COLLECTION_NAME,
				data=search_vectors,
				filter=filters or "",
				limit=k,
				search_params=search_params,
				output_fields=output_fields,
			)

			# 4. Format results
			formatted_results = []

			# Results is a list (for multiple query vectors), access the first element
			if results and len(results) > 0:
				for hit in results[0]:  # Search result for the first query vector
					# Create a dictionary with result information
					result_dict = {
						"id": hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None),
						"distance": hit.get("distance") if isinstance(hit, dict) else getattr(hit, "distance", 0.0),
						"metadata": {},
					}

					# Extract entity information based on result format
					if isinstance(hit, dict):
						# If hit is a dictionary
						result_dict["metadata"] = hit.get("entity", {})
					else:
						# If hit is a pymilvus.Hit object
						entity = getattr(hit, "entity", None)
						if entity is not None:
							if hasattr(entity, "to_dict"):
								result_dict["metadata"] = entity.to_dict()
							else:
								result_dict["metadata"] = entity

					formatted_results.append(result_dict)

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
			metadata = hit.get("metadata", {})
			entity_id = metadata.get("kuzu_entity_id")

			if entity_id:
				seed_entity_ids.append(entity_id)
				milvus_hit_map[entity_id] = hit
			else:
				logger.warning(f"Could not determine Kuzu entity ID for Milvus hit: {hit.get('id')}")

		if not seed_entity_ids:
			logger.warning("Graph-enhanced search: No corresponding Kuzu entity IDs found for initial hits.")
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
				"vector_hit": hit_data,
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
