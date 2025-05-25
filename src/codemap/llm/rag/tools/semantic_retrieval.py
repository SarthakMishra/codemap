"""Retrieval tool for PydanticAI agents to search and retrieve code context."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiofiles
from pydantic_ai.tools import Tool

from codemap.config import ConfigLoader
from codemap.processor.pipeline import ProcessingPipeline
from codemap.utils.git_utils import GitRepoContext

logger = logging.getLogger(__name__)


def merge_overlapping_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Merge overlapping or adjacent chunks from the same file.

	Args:
	    chunks: List of chunk dictionaries sorted by start_line

	Returns:
	    List of merged chunk dictionaries with consolidated line ranges
	"""
	if not chunks:
		return []

	merged = []
	current = chunks[0].copy()

	for next_chunk in chunks[1:]:
		# Check if chunks overlap or are adjacent (allowing 1-line gap)
		if next_chunk["start_line"] <= current["end_line"] + 2:
			# Merge chunks
			current["end_line"] = max(current["end_line"], next_chunk["end_line"])
			current["score"] = max(current["score"], next_chunk["score"])  # Keep best score

			# Combine entity information
			entities = current.setdefault("entities", [])
			entities.append(
				{
					"type": next_chunk["entity_type"],
					"name": next_chunk["entity_name"],
					"hierarchy_path": next_chunk["hierarchy_path"],
					"lines": f"{next_chunk['start_line']}-{next_chunk['end_line']}",
				}
			)

			# Add current chunk's entity info if not already added
			if len(entities) == 1:  # First merge, add original chunk info
				entities.insert(
					0,
					{
						"type": current["entity_type"],
						"name": current["entity_name"],
						"hierarchy_path": current["hierarchy_path"],
						"lines": f"{chunks[0]['start_line']}-{chunks[0]['end_line']}",
					},
				)
		else:
			# No overlap, save current and start new
			merged.append(current)
			current = next_chunk.copy()

	merged.append(current)
	return merged


async def retrieve_code_context(query: str) -> str:
	"""Retrieve relevant code chunks based on the query.

	This tool performs semantic search on the codebase and returns
	a formatted markdown string with the retrieved code context from
	the actual indexed vector chunks, with overlapping chunks merged.

	Args:
	    query: Search query to find relevant code

	Returns:
	    A string containing the formatted markdown of retrieved context.
	"""
	pipeline = await ProcessingPipeline.get_instance()
	config_loader = ConfigLoader.get_instance()

	if not pipeline:
		logger.warning("ProcessingPipeline not available, no context will be retrieved.")
		return (
			"Error: Vector search pipeline not available. "
			"Please ensure the codebase is indexed using 'cm index' command."
		)

	# Use provided limit or configured default
	actual_limit = config_loader.get.rag.max_context_results

	try:
		# Get repository root for resolving relative file paths
		try:
			repo_root = GitRepoContext.get_repo_root()
		except (OSError, ValueError, RuntimeError):
			repo_root = Path.cwd()  # Fallback to current directory

		logger.info(f"Retrieving context for query: '{query}', limit: {actual_limit}")

		# Perform semantic search
		results = await pipeline.semantic_search(query, k=actual_limit)

		if not results:
			logger.warning(f"Semantic search returned no results for query: '{query}'")
			return (
				f"No relevant code context found for query: '{query}'. "
				f"This might indicate that:\n"
				f"1. The codebase index is empty or not properly initialized\n"
				f"2. The search terms don't match indexed content\n"
				f"3. Try using more specific technical terms or function/class names\n"
				f"4. Consider running 'cm index' to rebuild the vector index"
			)

		logger.info(f"Semantic search found {len(results)} raw results")

		# Group results by file path for deduplication
		file_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
		processed_results = 0

		for i, result in enumerate(results, 1):
			try:
				# Extract metadata from payload
				payload = result.get("payload", {})
				if not payload:
					logger.debug(f"Result {i} has no payload, skipping")
					continue

				# Handle both dict and Pydantic model payloads
				if hasattr(payload, "dict"):
					payload = payload.dict()
				elif hasattr(payload, "model_dump"):
					payload = payload.model_dump()

				# Get file metadata
				file_metadata = payload.get("file_metadata", {})
				if not file_metadata:
					logger.debug(f"Result {i} has no file_metadata, skipping")
					continue

				# Handle nested file metadata
				if hasattr(file_metadata, "dict"):
					file_metadata = file_metadata.dict()
				elif hasattr(file_metadata, "model_dump"):
					file_metadata = file_metadata.model_dump()

				file_path_rel = file_metadata.get("file_path", "")
				language = file_metadata.get("language", "text")

				# Get chunk metadata
				start_line = payload.get("start_line", 0)
				end_line = payload.get("end_line", 0)
				entity_type = payload.get("entity_type", "")
				entity_name = payload.get("entity_name", "")
				hierarchy_path = payload.get("hierarchy_path", "")

				# Get the score
				score = result.get("score", 0.0)

				# Validate we have the necessary information
				if not file_path_rel or start_line <= 0 or end_line < start_line:
					logger.debug(
						f"Result {i} has invalid file/line info: file='{file_path_rel}', lines={start_line}-{end_line}"
					)
					continue

				# Store chunk information for later processing
				chunk_info = {
					"file_path_rel": file_path_rel,
					"language": language,
					"start_line": start_line,
					"end_line": end_line,
					"entity_type": entity_type,
					"entity_name": entity_name,
					"hierarchy_path": hierarchy_path,
					"score": score,
					"result_index": i,
				}

				file_chunks[file_path_rel].append(chunk_info)
				processed_results += 1

			except (KeyError, TypeError, ValueError, AttributeError) as e:
				logger.warning(f"Error processing search result {i}: {e}")
				logger.debug(f"Problematic result structure: {result}")
				continue

		logger.info(f"Successfully processed {processed_results} out of {len(results)} search results")

		if not file_chunks:
			msg = (
				f"No valid code chunks could be extracted from {len(results)} search results. "
				f"This suggests the vector index may have structural issues. "
				f"Consider rebuilding the index with 'cm index --no-sync' followed by 'cm index'."
			)
			logger.warning(msg)
			return msg

		markdown_chunks = []
		result_counter = 1

		# Process each file's chunks
		for file_path_rel, chunks in file_chunks.items():
			# Sort chunks by start line
			chunks.sort(key=lambda x: x["start_line"])

			# Merge overlapping/adjacent chunks
			merged_chunks = merge_overlapping_chunks(chunks)

			# Convert relative path to absolute
			file_path_abs = repo_root / file_path_rel

			try:
				# Read the file once for all chunks from this file
				async with aiofiles.open(file_path_abs, encoding="utf-8") as f:  # type: ignore[async]
					file_lines = await f.readlines()

				# Process each merged chunk
				for merged_chunk in merged_chunks:
					start_line = merged_chunk["start_line"]
					end_line = merged_chunk["end_line"]

					# Extract the relevant lines (convert to 0-based indexing)
					if end_line > len(file_lines):
						logger.warning(f"End line {end_line} exceeds file length {len(file_lines)} for {file_path_rel}")
						end_line = len(file_lines)

					chunk_lines = file_lines[start_line - 1 : end_line]
					chunk_content = "".join(chunk_lines).rstrip()

					if not chunk_content.strip():
						logger.debug("Merged chunk has empty content, skipping")
						continue

					# Create markdown section for this merged chunk
					chunk_markdown = f"## Result {result_counter} (Score: {merged_chunk['score']:.3f})\n\n"
					chunk_markdown += f"**File:** `{file_path_rel}`\n"
					chunk_markdown += f"**Lines:** {start_line}-{end_line}\n"

					# Show entity information
					entities = merged_chunk.get("entities", [])
					if entities:
						chunk_markdown += f"**Contains {len(entities)} entities:**\n"
						for entity in entities:
							if entity["name"]:
								chunk_markdown += f"- {entity['type']}: `{entity['name']}` (lines {entity['lines']})\n"
							else:
								chunk_markdown += f"- {entity['type']} (lines {entity['lines']})\n"
					else:
						# Single entity (not merged)
						if merged_chunk["entity_type"]:
							chunk_markdown += f"**Type:** {merged_chunk['entity_type']}\n"
						if merged_chunk["entity_name"]:
							chunk_markdown += f"**Name:** {merged_chunk['entity_name']}\n"
						if merged_chunk["hierarchy_path"]:
							chunk_markdown += f"**Path:** {merged_chunk['hierarchy_path']}\n"

					chunk_markdown += f"\n```{merged_chunk['language']}\n{chunk_content}\n```\n"

					markdown_chunks.append(chunk_markdown)
					logger.debug(f"Successfully processed merged chunk from {file_path_rel}:{start_line}-{end_line}")
					result_counter += 1

			except (OSError, UnicodeDecodeError) as e:
				logger.warning(f"Could not read file {file_path_abs}: {e}")
				# Add a note about the inaccessible file
				error_chunk = f"## Result {result_counter} (File Read Error)\n\n"
				error_chunk += f"**File:** `{file_path_rel}`\n"
				error_chunk += f"**Error:** Could not read file - {e}\n"
				markdown_chunks.append(error_chunk)
				result_counter += 1
				continue

		if not markdown_chunks:
			msg = (
				f"No readable code chunks could be retrieved from {len(file_chunks)} files. "
				f"Files may be inaccessible or have encoding issues."
			)
			logger.warning(msg)
			return msg

		# Join all chunks with separators
		result_markdown = "\n\n---\n\n".join(markdown_chunks)
		logger.info(f"Successfully retrieved {len(markdown_chunks)} merged code chunks from {len(file_chunks)} files")
		return result_markdown

	except Exception as e:
		logger.exception("Error retrieving context")
		# Return a more informative error message instead of raising ModelRetry
		return (
			f"Error occurred while retrieving code context: {e!s}\n\n"
			f"This might indicate:\n"
			f"1. Vector database connection issues\n"
			f"2. Index corruption or initialization problems\n"
			f"3. File system access issues\n\n"
			f"Try running 'cm index' to rebuild the vector index."
		)


# --- Create the PydanticAI Tool instance ---

semantic_retrieval_tool = Tool(
	retrieve_code_context,
	takes_ctx=False,
	name="semantic_retrieval",
	description=(
		"Retrieve relevant code chunks from the codebase using semantic search. "
		"Returns formatted markdown with the actual code snippets, file paths, "
		"line numbers, and metadata from the vector index. Automatically merges "
		"overlapping chunks from the same file to avoid duplication. "
		"Use specific technical terms and keywords for better results."
	),
	prepare=None,
)
