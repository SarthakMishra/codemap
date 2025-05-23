"""Retrieval tool for PydanticAI agents to search and retrieve code context."""

import asyncio
import logging
from typing import TypedDict

import aiofiles
from pydantic_ai.tools import Tool

from codemap.config import ConfigLoader
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
						# Validate and adjust line numbers
						max_lines = len(lines)
						if max_lines == 0:
							logger.warning(f"File {file_path} is empty, skipping code content.")
						elif max_lines < MIN_FILE_LINES_FOR_PROCESSING:
							logger.warning(
								f"File {file_path} has <{MIN_FILE_LINES_FOR_PROCESSING} lines, skipping code content."
							)
						elif start_line <= 0 or end_line <= 0:
							logger.warning(
								f"Invalid line numbers for file {file_path}: "
								f"start={start_line}, end={end_line}. Using line 1."
							)
							code_content = lines[0] if lines else ""
							content_parts.append(f"```\n{code_content}\n```")
						elif start_line > max_lines or end_line > max_lines:
							logger.warning(
								f"Line numbers exceed file length for {file_path}: "
								f"start={start_line}, end={end_line}, total_lines={max_lines}. "
								"Using available lines."
							)
							# Use what we can
							adjusted_start = min(start_line, max_lines)
							adjusted_end = min(end_line, max_lines)
							code_content = "\n".join(lines[adjusted_start - 1 : adjusted_end])
							content_parts.append(f"```\n{code_content}\n```")
						else:
							code_content = "\n".join(lines[start_line - 1 : end_line])
							content_parts.append(f"```\n{code_content}\n```")
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

		# Generate markdown context directly
		return generate_simple_markdown_context(formatted_results, config_loader.get.rag.max_context_length)

	except Exception:
		logger.exception("Error retrieving context")
		return "Error: Could not retrieve or process context."


def generate_simple_markdown_context(results: list[RetrievalContext], max_length: int) -> str:
	"""Generate simple markdown from retrieval results."""
	if not results:
		return "# Retrieved Code Context\n\nNo relevant code found."

	content_parts = ["# Retrieved Code Context\n"]
	current_length = len(content_parts[0])

	# Filter out results that don't have meaningful content
	meaningful_results = []
	for result in results:
		content = result["content"]

		# Skip if content is empty or only contains metadata without code
		if not content or content.strip() == "":
			continue

		# Skip if content only has Type/Name metadata without actual code
		lines = content.split("\n")
		has_code_block = any(line.strip().startswith("```") for line in lines)
		non_metadata_lines = [line for line in lines if not line.startswith(("Type:", "Name:"))]

		# Skip if no code block and only metadata
		if not has_code_block and len(non_metadata_lines) <= MAX_METADATA_ONLY_LINES:
			continue

		# Skip files with problematic line numbers (like __init__.py files)
		if "__init__.py" in result["file_path"] and not has_code_block:
			continue

		meaningful_results.append(result)

	if not meaningful_results:
		return "# Retrieved Code Context\n\nNo relevant code with substantial content found."

	for result in meaningful_results:
		section = f"## {result['file_path']} (Lines {result['start_line']}-{result['end_line']})\n"
		section += f"**Relevance Score:** {result['score']:.2f}\n\n"
		section += result["content"] + "\n\n"

		if current_length + len(section) > max_length:
			content_parts.append("... [content truncated due to length limit]")
			break

		content_parts.append(section)
		current_length += len(section)

	return "\n".join(content_parts)


# --- Create the PydanticAI Tool instance ---

code_retrieval_tool = Tool(
	retrieve_code_context,
	takes_ctx=False,
	name="retrieve_code_context",
	description=(
		"Retrieve relevant context from the codebase using semantic search for the given query. "
		"The query should be technical and specific to the codebase. "
		"Mention keywords like functions, classes, modules, etc. if applicable."
	),
	prepare=None,
)
