"""Retrieval tool for PydanticAI agents to search and retrieve code context."""

import logging
from pathlib import Path
from typing import TypedDict

import aiofiles
from pydantic_ai.tools import Tool

from codemap.config import ConfigLoader
from codemap.config.config_schema import GenSchema
from codemap.gen.generator import CodeMapGenerator
from codemap.gen.utils import process_codebase
from codemap.processor.pipeline import ProcessingPipeline

logger = logging.getLogger(__name__)

# Constants for validation
MIN_FILE_LINES_FOR_PROCESSING = 3
MAX_METADATA_ONLY_LINES = 2


class RetrievalContext(TypedDict):
	"""Context information for the retrieval tool."""

	file_path: str
	start_line: int
	end_line: int
	content: str
	score: float


def gen_doc(target_path: Path) -> str:
	"""Generate documentation for the given target path."""
	file_tree: bool = False
	include_entity_graph: bool = False

	if target_path.is_dir():
		file_tree = True
		include_entity_graph = True

	config = GenSchema(
		lod_level="skeleton",
		include_entity_graph=include_entity_graph,
		include_tree=file_tree,
		mermaid_show_legend=False,
		mermaid_remove_unconnected=True,
		mermaid_styled=False,
	)

	config_loader = ConfigLoader.get_instance()
	generator = CodeMapGenerator(config)

	entities, metadata = process_codebase(target_path, config, config_loader=config_loader)

	return generator.generate_documentation(entities, metadata)


def consolidate_paths(paths: list[Path], threshold: float = 0.8) -> list[Path]:
	"""Consolidate file paths to directory paths when appropriate.

	If a combination of paths includes all files in a directory or over threshold
	of files in a directory, removes individual file paths and includes
	the directory path instead.

	Args:
	    paths: List of file paths to consolidate
	    threshold: Threshold for consolidation (default 0.8)

	Returns:
	    List of consolidated paths (mix of files and directories)
	"""
	from collections import defaultdict

	if not paths:
		return []

	# Group paths by their immediate parent directory only
	dir_to_files: dict[Path, set[Path]] = defaultdict(set)

	for path in paths:
		if path.is_file():
			parent_dir = path.parent
			dir_to_files[parent_dir].add(path)

	consolidated_paths: list[Path] = []
	files_to_exclude: set[Path] = set()

	# Check each immediate parent directory for consolidation opportunities
	for directory, files_in_dir in dir_to_files.items():
		if not directory.exists() or not directory.is_dir():
			continue

		# Get all files in the immediate directory only (excluding subdirectories)
		try:
			all_files_in_dir = [f for f in directory.iterdir() if f.is_file()]
		except (PermissionError, OSError):
			# Skip directories we can't read
			continue

		if not all_files_in_dir:
			continue

		total_files = len(all_files_in_dir)
		matched_files = len(files_in_dir)
		coverage_ratio = matched_files / total_files

		# If we have all files or over threshold coverage, consolidate to directory
		if coverage_ratio >= threshold:
			consolidated_paths.append(directory)
			files_to_exclude.update(files_in_dir)

	# Add remaining individual files that weren't consolidated
	for path in paths:
		if path not in files_to_exclude:
			if path.is_file():
				consolidated_paths.append(path)
			elif path.is_dir():
				# Keep directory paths as-is
				consolidated_paths.append(path)

	return consolidated_paths


async def retrieve_code_context(query: str) -> str:
	"""Retrieve relevant code chunks based on the query.

	This tool performs semantic search on the codebase and returns
	a formatted markdown string with the retrieved code context.

	Args:
	    query: Search query to find relevant code

	Returns:
	    A string containing the formatted markdown of retrieved context.
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

		paths: list[dict[Path, float]] = []

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
			num_lines = end_line - start_line + 1

			# Calculate coverage as lines covered vs total lines in file
			try:
				async with aiofiles.open(file_path, encoding="utf-8") as f:
					total_lines = sum([1 async for _ in f])
				coverage = num_lines / total_lines if total_lines > 0 else 0
			except (OSError, UnicodeDecodeError):
				# If we can't read the file, assign a default coverage
				coverage = 0.1

			# Check if file_path already exists in paths
			found = False
			for _, path_dict in enumerate(paths):
				if Path(file_path) in path_dict:
					# Update existing coverage with maximum value
					current_coverage = path_dict[Path(file_path)]
					path_dict[Path(file_path)] = max(current_coverage, coverage)
					found = True
					break

			if not found:
				paths.append({Path(file_path): coverage})

		logger.debug(f"Semantic search returned {len(paths)} raw results.")

		# Extract paths with coverage > 0.5 threshold
		filtered_paths: list[Path] = []
		for path_dict in paths:
			for path, coverage in path_dict.items():
				if coverage > 0.5:  # noqa: PLR2004
					filtered_paths.append(path)

		consolidated_paths = consolidate_paths(filtered_paths)
		docs = [gen_doc(path) for path in consolidated_paths]

		# Generate markdown context directly
		return "\n\n---\n\n".join(docs)

	except Exception:
		logger.exception("Error retrieving context")
		return "Error: Could not retrieve or process context."


# --- Create the PydanticAI Tool instance ---

semantic_retrieval_tool = Tool(
	retrieve_code_context,
	takes_ctx=False,
	name="semantic_retrieval",
	description=(
		"Retrieve relevant context from the codebase using semantic search for the given query. "
		"The query should be technical and specific to the codebase. "
		"Mention keywords like functions, classes, modules, etc. if applicable."
	),
	prepare=None,
)
