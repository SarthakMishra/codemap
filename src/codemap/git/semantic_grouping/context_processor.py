"""
Context processing utilities for LLM prompts.

This module provides functionality to process and format code contexts
for LLM prompts using tree-sitter analysis and Level of Detail (LOD) to
optimize context length while preserving meaningful content.

"""

import logging
from pathlib import Path

from codemap.git.diff_splitter import DiffChunk
from codemap.processor.lod import LODEntity, LODGenerator, LODLevel

logger = logging.getLogger(__name__)

# Token limits for context
DEFAULT_MAX_TOKENS = 4000
CHUNK_TOKEN_ESTIMATE = 500  # Conservative estimate tokens per chunk
MAX_CHUNKS = 6  # Maximum number of chunks to include if no token estimation
MAX_SIMPLE_CHUNKS = 3  # Maximum number of chunks to include if no LOD processing


def process_chunks_with_lod(chunks: list[DiffChunk], max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
	"""
	Process diff chunks using LOD to create optimized context for LLM prompts.

	Args:
	    chunks: List of diff chunks to process
	    max_tokens: Maximum tokens allowed in the formatted context

	Returns:
	    Formatted markdown context optimized for token usage

	"""
	# If chunks list is small, we might not need LOD processing
	if len(chunks) <= MAX_SIMPLE_CHUNKS:
		return format_regular_chunks(chunks[:MAX_CHUNKS])

	# Set up LOD generator and estimate number of chunks we can include
	lod_generator = LODGenerator()
	estimated_chunk_count = min(max_tokens // CHUNK_TOKEN_ESTIMATE, len(chunks))
	prioritized_chunks = prioritize_chunks(chunks, min(estimated_chunk_count, MAX_CHUNKS))

	# Start with highest LOD level and progressively reduce if needed
	lod_levels = [LODLevel.STRUCTURE, LODLevel.SIGNATURES]
	formatted_chunks = []
	current_level_index = 0

	while current_level_index < len(lod_levels):
		current_level = lod_levels[current_level_index]
		formatted_chunks = []

		for chunk in prioritized_chunks:
			# Get file paths from chunk
			file_paths = get_file_paths_from_chunk(chunk)

			if not file_paths:
				# If we can't extract paths, use regular formatting for this chunk
				formatted_chunks.append(format_chunk(chunk))
				continue

			# Process each file in the chunk with LOD
			lod_formatted = []
			for file_path in file_paths:
				path = Path(file_path)
				if not path.exists():
					continue

				# Generate LOD representation
				lod_entity = lod_generator.generate_lod(path, level=current_level)
				if lod_entity:
					lod_formatted.append(format_lod_entity(lod_entity, file_path, current_level))

			if lod_formatted:
				formatted_chunks.append("\n".join(lod_formatted))
			else:
				# Fallback to regular formatting
				formatted_chunks.append(format_chunk(chunk))

		# Estimate if we're within token limit
		total_context = "\n\n".join(formatted_chunks)
		estimated_tokens = estimate_tokens(total_context)

		if estimated_tokens <= max_tokens or current_level_index == len(lod_levels) - 1:
			break

		# Try with lower LOD level
		current_level_index += 1

	# If we still exceed the token limit, truncate
	total_context = "\n\n".join(formatted_chunks)
	if estimate_tokens(total_context) > max_tokens:
		total_context = truncate_context(total_context, max_tokens)

	return total_context


def prioritize_chunks(chunks: list[DiffChunk], max_count: int) -> list[DiffChunk]:
	"""
	Prioritize chunks based on heuristics (file types, changes, etc.).

	This is a simple implementation that could be extended with more
	sophisticated dissimilarity metrics.

	Args:
	    chunks: List of chunks to prioritize
	    max_count: Maximum number of chunks to return

	Returns:
	    Prioritized list of chunks

	"""
	# Simple heuristics for now:
	# 1. Prefer chunks with code files over non-code files
	# 2. Prefer chunks with more files (more central changes)
	# 3. Prefer chunks with more added/changed lines

	def chunk_score(chunk: DiffChunk) -> float:
		"""Calculates a priority score for a diff chunk based on heuristics.

		The score is calculated using three factors:
		1. Presence of code files (60% weight)
		2. Number of files affected (20% weight)
		3. Size of content changes (20% weight)

		Args:
			chunk: The diff chunk to score

		Returns:
			float: A score between 0 and 1 representing the chunk's priority
		"""
		# Check if any files are code files
		code_file_score = 0
		for file in chunk.files:
			if any(file.endswith(ext) for ext in [".py", ".js", ".ts", ".java", ".c", ".cpp", ".go"]):
				code_file_score = 1
				break

		# Score based on number of files
		file_count_score = min(len(chunk.files), 3) / 3

		# Score based on content size (as proxy for changes)
		content_score = min(len(chunk.content), 1000) / 1000

		return code_file_score * 0.6 + file_count_score * 0.2 + content_score * 0.2

	# Sort chunks by score and return top max_count
	return sorted(chunks, key=chunk_score, reverse=True)[:max_count]


def get_file_paths_from_chunk(chunk: DiffChunk) -> list[str]:
	"""
	Extract file paths from a diff chunk.

	Args:
	    chunk: The diff chunk to process

	Returns:
	    List of file paths

	"""
	return [file for file in chunk.files if file]


def format_lod_entity(entity: LODEntity, file_path: str, level: LODLevel) -> str:
	"""
	Format an LOD entity as GitHub-flavored markdown.

	Args:
	    entity: The LOD entity to format
	    file_path: Path to the source file
	    level: LOD level used

	Returns:
	    Formatted markdown string

	"""
	# Start with file header
	result = f"## {file_path}\n\n"

	# Format the entity based on LOD level
	if level == LODLevel.STRUCTURE:
		result += format_entity_structure(entity, 0)
	elif level == LODLevel.SIGNATURES:
		result += format_entity_signatures(entity, 0)

	return result


def format_entity_structure(entity: LODEntity, indent: int) -> str:
	"""Format entity with structure (signatures and hierarchy)."""
	indent_str = "  " * indent
	result = f"{indent_str}- **{entity.entity_type.name}**: `{entity.name}`"

	if entity.signature:
		result += f"\n{indent_str}  ```\n{indent_str}  {entity.signature}\n{indent_str}  ```"

	if entity.children:
		result += "\n"
		for child in entity.children:
			result += format_entity_structure(child, indent + 1)

	return result + "\n"


def format_entity_signatures(entity: LODEntity, indent: int) -> str:
	"""Format entity with just signatures."""
	indent_str = "  " * indent
	result = f"{indent_str}- **{entity.entity_type.name}**: `{entity.name}`"

	if entity.signature:
		result += f" - `{entity.signature}`"

	if entity.children:
		result += "\n"
		for child in entity.children:
			result += format_entity_signatures(child, indent + 1)

	return result + "\n"


def format_regular_chunks(chunks: list[DiffChunk]) -> str:
	"""
	Format chunks using the regular approach when LOD is not necessary.

	Args:
	    chunks: List of chunks to format

	Returns:
	    Formatted markdown string

	"""
	formatted_chunks = [format_chunk(chunk) for chunk in chunks]
	return "\n\n".join(formatted_chunks)


def format_chunk(chunk: DiffChunk) -> str:
	"""
	Format a single diff chunk as markdown.

	Args:
	    chunk: The diff chunk to format

	Returns:
	    Formatted markdown string

	"""
	# Format file paths
	file_section = "## Files\n"
	for file in chunk.files:
		if file:
			file_section += f"- {file}\n"

	# Format content
	content_section = "### Changes\n```diff\n" + chunk.content + "\n```"

	return file_section + "\n" + content_section


def estimate_tokens(text: str) -> int:
	"""
	Estimate the number of tokens in a text.

	This is a simple estimation that can be improved with
	actual tokenizer implementations if needed.

	Args:
	    text: Text to estimate tokens for

	Returns:
	    Estimated token count

	"""
	# Simple estimation: 4 characters per token on average
	return len(text) // 4


def truncate_context(context: str, max_tokens: int) -> str:
	"""
	Truncate context to fit within token limit.

	Args:
	    context: Context to truncate
	    max_tokens: Maximum allowed tokens

	Returns:
	    Truncated context

	"""
	# Simple truncation by estimating tokens
	if estimate_tokens(context) <= max_tokens:
		return context

	# Split into chunks and preserve as many complete chunks as possible
	chunks = context.split("\n\n")
	result_chunks = []
	current_token_count = 0

	for chunk in chunks:
		chunk_tokens = estimate_tokens(chunk)
		if current_token_count + chunk_tokens <= max_tokens - 100:  # Reserve 100 tokens for truncation marker
			result_chunks.append(chunk)
			current_token_count += chunk_tokens
		else:
			# Add truncation marker and stop
			result_chunks.append("\n\n[...TRUNCATED...]\n\n")
			break

	return "\n\n".join(result_chunks)
