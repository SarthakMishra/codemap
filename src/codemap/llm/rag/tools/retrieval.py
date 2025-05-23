"""Retrieval tool for PydanticAI agents to search and retrieve code context."""

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import aiofiles
from pydantic_ai.tools import Tool

from codemap.config import ConfigLoader
from codemap.gen.generator import CodeMapGenerator
from codemap.gen.models import GenConfig
from codemap.processor.lod import LODEntity, LODGenerator, LODLevel
from codemap.processor.pipeline import ProcessingPipeline
from codemap.processor.tree_sitter.analyzer import TreeSitterAnalyzer
from codemap.processor.tree_sitter.base import EntityType

logger = logging.getLogger(__name__)

MIN_ENTITIES_FOR_OUTLIER_DETECTION = 3


class RetrievalContext(TypedDict):
	"""Context information for the retrieval tool."""

	file_path: str
	start_line: int
	end_line: int
	content: str
	score: float


async def retrieve_code_context(query: str) -> str:
	"""Retrieve relevant code chunks based on the query.

	This tool performs semantic search on the codebase, processes the
	retrieved chunks to find common context, extends it with related
	entities, and returns a single LLM-friendly markdown string.

	Args:
	    query: Search query to find relevant code

	Returns:
	    A string containing the formatted markdown of all merged context.
	"""
	pipeline = await ProcessingPipeline.get_instance()
	config_loader = ConfigLoader.get_instance()

	if not pipeline:
		logger.warning("ProcessingPipeline not available, no context will be retrieved.")
		return "Error: Could not retrieve or process context."

	# Use provided limit or configured default
	actual_limit = config_loader.get.rag.max_context_results

	try:
		from codemap.processor.vector.schema import ChunkMetadataSchema

		logger.info(f"Retrieving context for query: '{query}', limit: {actual_limit}")

		# Perform semantic search
		results = await pipeline.semantic_search(query, k=actual_limit)

		# Format results for the LLM
		formatted_results: list[RetrievalContext] = []

		if not results:
			logger.debug("Semantic search returned no results.")
			return "No relevant code context found."

		for r in results:
			# Extract relevant fields from payload
			payload: ChunkMetadataSchema = r.get("payload", {})

			# Get file metadata
			file_path = payload.file_metadata.file_path
			start_line = payload.start_line
			end_line = payload.end_line
			entity_type = payload.entity_type
			entity_name = payload.entity_name
			language = payload.file_metadata.language

			# Build content representation from metadata
			content_parts = []
			content_parts.append(f"Type: {entity_type}")
			if entity_name:
				content_parts.append(f"Name: {entity_name}")

			# Get file content from repository
			try:
				config = config_loader.get if config_loader else None
				if config and config.repo_root and file_path and file_path != "N/A" and start_line > 0 and end_line > 0:
					repo_file_path = config.repo_root / file_path
					if await asyncio.to_thread(repo_file_path.exists):
						async with aiofiles.open(repo_file_path, encoding="utf-8") as f:
							file_content = await f.read()

						lines = file_content.splitlines()
						if start_line <= len(lines) and end_line <= len(lines) and start_line <= end_line:
							code_content = "\n".join(lines[start_line - 1 : end_line])
							if language:
								content_parts.append(f"```{language}\n{code_content}\n```")
							else:
								content_parts.append(f"```\n{code_content}\n```")
						else:
							logger.warning(
								f"Invalid line numbers for file {file_path}: "
								f"start={start_line}, end={end_line}, total_lines={len(lines)}. "
								"Skipping code content for this chunk."
							)
					else:
						logger.warning(f"File path does not exist: {repo_file_path} for {file_path}")
				elif file_path == "N/A":
					logger.warning("File path is 'N/A' for a chunk, cannot retrieve content.")
			except Exception:
				logger.exception(f"Error reading file content for {file_path}")

			content = "\n\n".join(content_parts)

			formatted_results.append(
				RetrievalContext(
					file_path=file_path,
					start_line=start_line,
					end_line=end_line,
					content=content,
					score=r.get("score", -1.0),
				)
			)

		logger.debug(f"Semantic search returned {len(formatted_results)} raw results.")

		# Accumulate, extend, and format context into markdown
		final_markdown_context = accumulate_chunks_and_extend_context(
			formatted_results, config_loader.get.rag.max_context_length
		)
		logger.debug(f"Final extended context length: {len(final_markdown_context)}")
		return final_markdown_context

	except Exception:
		logger.exception("Error retrieving context")
		return "Error: Could not retrieve or process context."


# --- Helper functions ---


def accumulate_chunks_and_extend_context(chunks: list[RetrievalContext], max_context_length: int = 8000) -> str:
	"""
	Accumulates chunks, finds their position in the AST, and traverses the AST to extend context.

	This implementation:
	1. Groups chunks by file and finds common parents
	2. Filters outlier chunks based on AST distance
	3. Uses import/call graphs to include related entities
	4. Generates LLM-friendly markdown output

	Args:
		chunks: List of RetrievalContext objects containing file paths, line numbers, and content.
		max_context_length: Maximum length for the extended context.

	Returns:
		Extended context string in LLM-friendly markdown format.
	"""
	if not chunks:
		return "No chunks provided."

	# Initialize LODGenerator and TreeSitterAnalyzer
	analyzer = TreeSitterAnalyzer()
	lod_generator = LODGenerator(analyzer)

	# Group chunks by file for efficient processing
	chunks_by_file: dict[Path, list[RetrievalContext]] = defaultdict(list)
	for chunk in chunks:
		file_path = Path(chunk["file_path"])
		if file_path.exists():
			chunks_by_file[file_path].append(chunk)

	# Process each file and collect relevant entities
	collected_entities: list[tuple[LODEntity, float]] = []  # (entity, relevance_score)
	import_graph: dict[str, set[str]] = defaultdict(set)  # entity_id -> imported entities
	call_graph: dict[str, set[str]] = defaultdict(set)  # entity_id -> called entities

	for file_path, file_chunks in chunks_by_file.items():
		# Generate LOD for the file at FULL level to get all details
		lod_entity = lod_generator.generate_lod(file_path, level=LODLevel.FULL)
		if not lod_entity:
			continue

		# Find entities containing chunks and their common ancestors
		chunk_entities: list[tuple[LODEntity, RetrievalContext]] = []
		for chunk in file_chunks:
			entity = find_entity_in_ast(lod_entity, chunk["start_line"], chunk["end_line"])
			if entity:
				chunk_entities.append((entity, chunk))

		if not chunk_entities:
			continue

		# Find common ancestors and calculate distances
		common_ancestors = find_common_ancestors(lod_entity, [e[0] for e in chunk_entities])

		# Filter outliers based on AST distance from main cluster
		main_entities = filter_outlier_entities(chunk_entities, common_ancestors)

		# Collect entities with relevance scores
		main_entity_tuples = []
		for entity, chunk in main_entities:
			relevance_score = chunk.get("score", 0.5)
			main_entity_tuples.append((entity, relevance_score))

			# Build import and call graphs
			entity_id = get_entity_id(entity)

			# Extract imports
			dependencies = entity.metadata.get("dependencies", [])
			import_graph[entity_id].update(dependencies)

			# Extract calls
			calls = entity.metadata.get("calls", [])
			call_graph[entity_id].update(calls)

		collected_entities.extend(main_entity_tuples)

		# Include common ancestors if they're meaningful (classes, modules)
		ancestor_tuples = [
			(ancestor, 0.3)
			for ancestor in common_ancestors
			if ancestor.entity_type in (EntityType.CLASS, EntityType.MODULE, EntityType.FUNCTION)
		]
		collected_entities.extend(ancestor_tuples)

	# Extend context using import/call relationships
	extended_entities = extend_with_relationships(collected_entities, import_graph, call_graph)

	# Sort entities by relevance and file location
	extended_entities.sort(key=lambda x: (-x[1], x[0].metadata.get("file_path", ""), x[0].start_line))

	# Generate LLM-friendly markdown using the gen command's approach
	return generate_llm_friendly_markdown(extended_entities, max_context_length)


def find_common_ancestors(root: LODEntity, entities: list[LODEntity]) -> list[LODEntity]:
	"""
	Find common ancestors of multiple entities in the AST.

	Args:
		root: The root entity of the AST
		entities: List of entities to find common ancestors for

	Returns:
		List of common ancestor entities, ordered from most specific to least
	"""
	# Build parent map for efficient ancestor lookup
	parent_map: dict[int, LODEntity] = {}  # Use id() as key

	def build_parent_map(entity: LODEntity, parent: LODEntity | None = None) -> None:
		if parent:
			parent_map[id(entity)] = parent
		for child in entity.children:
			build_parent_map(child, entity)

	build_parent_map(root)

	# Get ancestor chains for each entity
	ancestor_chains: list[list[LODEntity]] = []

	for entity in entities:
		chain = [entity]
		current = entity
		while id(current) in parent_map:
			parent = parent_map[id(current)]
			chain.append(parent)
			current = parent
		ancestor_chains.append(chain)

	# Find common ancestors
	common_ancestors = []

	if not ancestor_chains:
		return common_ancestors

	# Check ancestors level by level
	min_chain_length = min(len(chain) for chain in ancestor_chains)

	for level in range(min_chain_length):
		# Get ancestors at this level from each chain
		ancestors_at_level = [chain[level] for chain in ancestor_chains]

		# Check if all are the same entity
		if all(id(a) == id(ancestors_at_level[0]) for a in ancestors_at_level):
			common_ancestors.append(ancestors_at_level[0])
		else:
			# Stop when we find divergence
			break

	return common_ancestors


def filter_outlier_entities(
	chunk_entities: list[tuple[LODEntity, RetrievalContext]], common_ancestors: list[LODEntity]
) -> list[tuple[LODEntity, RetrievalContext]]:
	"""
	Filter out outlier entities that are too far from the main cluster in the AST.

	Args:
		chunk_entities: List of (entity, chunk) tuples
		common_ancestors: List of common ancestors

	Returns:
		Filtered list of entities that are part of the main cluster
	"""
	if len(chunk_entities) < MIN_ENTITIES_FOR_OUTLIER_DETECTION:
		# Too few entities to determine outliers
		return chunk_entities

	# Calculate AST distances between entities
	distances: dict[int, dict[int, int]] = defaultdict(dict)

	for i, (entity1, _) in enumerate(chunk_entities):
		for j, (entity2, _) in enumerate(chunk_entities):
			if i != j:
				distance = calculate_ast_distance(entity1, entity2, common_ancestors)
				distances[i][j] = distance

	# Calculate average distance for each entity
	avg_distances = {}
	for i in range(len(chunk_entities)):
		if distances.get(i):
			avg_distances[i] = sum(distances[i].values()) / len(distances[i])
		else:
			avg_distances[i] = 0

	# Filter outliers (entities with average distance > 2 * median)
	if avg_distances:
		sorted_distances = sorted(avg_distances.values())
		median_distance = sorted_distances[len(sorted_distances) // 2]
		threshold = max(2 * median_distance, 5)  # At least 5 to avoid being too strict

		filtered_entities = [chunk_entities[i] for i, avg_dist in avg_distances.items() if avg_dist <= threshold]

		return filtered_entities if filtered_entities else chunk_entities

	return chunk_entities


def calculate_ast_distance(entity1: LODEntity, entity2: LODEntity, common_ancestors: list[LODEntity]) -> int:
	"""
	Calculate the distance between two entities in the AST.

	Args:
		entity1: First entity
		entity2: Second entity
		common_ancestors: List of common ancestors

	Returns:
		Distance between entities (number of edges in AST)
	"""
	# If entities are the same, distance is 0
	if entity1 == entity2:
		return 0

	# Use line distance as a factor
	line_distance = abs(entity1.start_line - entity2.start_line)

	# Simple heuristic: if they share a close common ancestor, distance is small
	if common_ancestors:
		# If they share a class or function as ancestor, they're close
		for ancestor in common_ancestors:
			if ancestor.entity_type in (EntityType.CLASS, EntityType.FUNCTION):
				return 2 + min(line_distance // 10, 3)  # Add line distance factor
		# If they only share a module, they're farther
		return 4 + min(line_distance // 20, 6)

	# If no common ancestors, they're very far
	return 10 + min(line_distance // 50, 10)


def get_entity_id(entity: LODEntity) -> str:
	"""Generate a unique ID for an entity."""
	file_path = entity.metadata.get("file_path", "unknown")
	return f"{file_path}:{entity.entity_type.name}:{entity.name}:{entity.start_line}"


def extend_with_relationships(
	entities: list[tuple[LODEntity, float]],
	import_graph: dict[str, set[str]],
	call_graph: dict[str, set[str]],
) -> list[tuple[LODEntity, float]]:
	"""
	Extend the context with related entities based on import and call relationships.

	Args:
		entities: List of (entity, relevance_score) tuples
		import_graph: Import relationships
		call_graph: Call relationships
		lod_generator: LOD generator instance

	Returns:
		Extended list of entities with relevance scores
	"""
	# Create a set of current entity IDs for quick lookup
	current_entity_ids = {get_entity_id(entity) for entity, _ in entities}

	# Track which entities we've already processed to avoid cycles
	processed_ids = set(current_entity_ids)

	# New entities to add
	new_entities: list[tuple[LODEntity, float]] = []

	# Process imports and calls with decreasing relevance
	for entity, relevance in entities:
		entity_id = get_entity_id(entity)

		# Add imported entities with lower relevance
		for imported in import_graph.get(entity_id, []):
			if imported not in processed_ids:
				# Try to find the imported entity in the codebase
				# This is a simplified approach - in practice you'd need better resolution
				processed_ids.add(imported)
				# Give imports 50% of the parent's relevance
				# In a real implementation, you'd resolve the import to an actual entity
				logger.debug(f"Would extend with import: {imported} (relevance: {relevance * 0.5})")

		# Add called entities with lower relevance
		for called in call_graph.get(entity_id, []):
			if called not in processed_ids:
				processed_ids.add(called)
				# Give calls 60% of the parent's relevance
				logger.debug(f"Would extend with call: {called} (relevance: {relevance * 0.6})")

	# Combine original and new entities
	all_entities = entities + new_entities

	# Sort by relevance and deduplicate
	seen_ids = set()
	deduplicated = []
	for entity, score in sorted(all_entities, key=lambda x: -x[1]):
		entity_id = get_entity_id(entity)
		if entity_id not in seen_ids:
			seen_ids.add(entity_id)
			deduplicated.append((entity, score))

	return deduplicated


def generate_llm_friendly_markdown(entities: list[tuple[LODEntity, float]], max_length: int) -> str:
	"""
	Generate LLM-friendly markdown documentation from entities.

	Uses the approach from the gen command to create well-structured markdown.

	Args:
		entities: List of (entity, relevance_score) tuples
		max_length: Maximum length of generated content

	Returns:
		Markdown formatted string
	"""
	# Group entities by file
	entities_by_file: dict[str, list[tuple[LODEntity, float]]] = defaultdict(list)

	for entity, score in entities:
		file_path = entity.metadata.get("file_path", "unknown")
		entities_by_file[file_path].append((entity, score))

	# Create a minimal GenConfig for the generator
	config = GenConfig(
		max_content_length=max_length,
		use_gitignore=False,
		output_dir=Path(),
		semantic_analysis=False,
		lod_level=LODLevel.FULL,
		include_tree=False,
		include_entity_graph=False,
	)

	# Use CodeMapGenerator to format the documentation
	CodeMapGenerator(config, Path("temp.md"))

	# Build content sections
	content_parts = []
	content_parts.append("# Retrieved Code Context\n")

	current_length = len(content_parts[0])

	for file_path, file_entities in sorted(entities_by_file.items()):
		# Sort entities by line number within file
		file_entities.sort(key=lambda x: x[0].start_line)

		file_section = f"\n## {file_path}\n\n"

		for entity, score in file_entities:
			# Skip if adding this entity would exceed max length
			entity_content = format_entity_markdown(entity, score)
			if current_length + len(file_section) + len(entity_content) > max_length:
				content_parts.append("\n... [truncated]")
				return "\n".join(content_parts)

			file_section += entity_content

		content_parts.append(file_section)
		current_length += len(file_section)

	return "\n".join(content_parts)


def format_entity_markdown(entity: LODEntity, relevance_score: float) -> str:
	"""
	Format a single entity as markdown.

	Args:
		entity: The entity to format
		relevance_score: Relevance score (0-1)

	Returns:
		Markdown formatted string
	"""
	lines = []

	# Add relevance indicator
	relevance_marker = "⭐" * min(int(relevance_score * 5), 5)
	lines.append(f"### {entity.entity_type.name}: `{entity.name}` {relevance_marker}\n")

	# Add location info
	lines.append(f"*Lines {entity.start_line}-{entity.end_line}*\n")

	# Add docstring if available
	if entity.docstring:
		lines.append("> " + entity.docstring.replace("\n", "\n> ") + "\n")

	# Add signature if available
	if entity.signature:
		lines.append(f"**Signature:** `{entity.signature}`\n")

	# Add content if available
	if entity.content:
		language = entity.language or ""
		lines.append(f"```{language}")
		lines.append(entity.content.strip())
		lines.append("```\n")

	# Add metadata if relevant
	if entity.metadata.get("dependencies"):
		lines.append(f"**Imports:** {', '.join(entity.metadata['dependencies'])}\n")

	if entity.metadata.get("calls"):
		lines.append(f"**Calls:** {', '.join(entity.metadata['calls'])}\n")

	return "\n".join(lines)


def find_entity_in_ast(entity: LODEntity, start_line: int, end_line: int) -> LODEntity | None:
	"""
	Find the entity in the AST that contains the specified line range.

	Args:
		entity: The LODEntity to search in.
		start_line: The starting line number.
		end_line: The ending line number.

	Returns:
		The LODEntity that contains the specified line range, or None if not found.
	"""
	if entity.start_line <= start_line and entity.end_line >= end_line:
		return entity

	for child in entity.children:
		result = find_entity_in_ast(child, start_line, end_line)
		if result:
			return result

	return None


# --- Create the PydanticAI Tool instance ---

code_retrieval_tool = Tool(
	retrieve_code_context,
	takes_ctx=False,
	name="retrieve_code_context",
	description="Retrieve relevant context from the codebase using semantic search for the given query.",
	prepare=None,
)
