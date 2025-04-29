"""Semantic code compression functionality."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import cast

from codemap.processor.analysis.git.models import GitMetadata
from codemap.processor.analysis.tree_sitter.base import EntityType
from codemap.processor.chunking.base import Chunk

from .models import CompressionStrategy, GenerationMode

logger = logging.getLogger(__name__)

# Constants for importance thresholds
HIGH_IMPORTANCE_THRESHOLD = 7.0
MEDIUM_IMPORTANCE_THRESHOLD = 5.0
LOW_IMPORTANCE_THRESHOLD = 3.0


def score_chunk_importance(chunk: Chunk) -> float:
	"""
	Score the semantic importance of a code chunk.

	Uses various heuristics to determine importance, including:
	- Entity type (classes and functions get higher scores)
	- Number of references/dependencies
	- Recency of modifications
	- Presence of docstrings

	Args:
	    chunk: The chunk to score

	Returns:
	    A score between 0.0 and 10.0

	"""
	base_score = 1.0
	metadata = chunk.metadata

	# Higher score for more important entity types
	entity_type_scores = {
		EntityType.MODULE: 6.0,
		EntityType.CLASS: 5.0,
		EntityType.FUNCTION: 4.0,
		EntityType.METHOD: 4.0,
		EntityType.ENUM: 3.5,
		EntityType.INTERFACE: 5.0,
		EntityType.IMPORT: 6.0,
		EntityType.VARIABLE: 2.0,
		EntityType.CONSTANT: 3.0,
		EntityType.PROPERTY: 2.5,
		EntityType.UNKNOWN: 1.0,
	}
	base_score += entity_type_scores.get(metadata.entity_type, 0.0)

	# Higher score for chunks with descriptions/docstrings
	if metadata.description:
		base_score += 1.0

	# Higher score for chunks with dependencies/imports
	if metadata.dependencies:
		base_score += min(len(metadata.dependencies) * 0.2, 1.0)

	# Higher score based on git metadata if available
	if metadata.git:
		# Recently modified files get higher score
		git_meta = cast("GitMetadata", metadata.git)

		# Check for last_modified_at first, fallback to timestamp
		modification_time = git_meta.last_modified_at or git_meta.timestamp
		if modification_time:
			days_since_modified = (datetime.now(UTC) - modification_time).days
			recency_score = max(0, 30 - days_since_modified) / 30.0
			base_score += recency_score * 2.0

		# More frequently modified files get higher score - estimate based on commit info
		# Since we don't have a commit_count attribute, we'll use a simple approximation:
		# if it has a commit_id and has been committed, it gets a score boost
		if git_meta.is_committed and git_meta.commit_id:
			base_score += 0.5

	# Cap score at 10.0
	return min(base_score, 10.0)


class SemanticCompressor:
	"""Compresses code while preserving semantic meaning."""

	def __init__(self, strategy: CompressionStrategy, mode: GenerationMode) -> None:
		"""
		Initialize the semantic compressor.

		Args:
		        strategy: Compression strategy to use
		        mode: Generation mode (LLM or human readable)

		"""
		self.strategy = strategy
		self.mode = mode

	def compress_chunks(self, chunks: list[Chunk], token_limit: int) -> list[Chunk]:
		"""
		Compress chunks to fit within token limit while preserving semantics.

		Args:
		    chunks: List of chunks to compress
		    token_limit: Maximum token limit

		Returns:
		    Compressed list of chunks

		"""
		# If no token limit or compression disabled, return chunks as is
		if token_limit <= 0 or self.strategy == CompressionStrategy.NONE:
			return chunks

		# Score chunks by importance
		scored_chunks = [(chunk, score_chunk_importance(chunk)) for chunk in chunks]
		sorted_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)

		# Select compression method based on strategy
		if self.strategy == CompressionStrategy.SMART:
			return self._smart_compress(sorted_chunks, token_limit)
		if self.strategy == CompressionStrategy.AGGRESSIVE:
			return self._aggressive_compress(sorted_chunks, token_limit)
		if self.strategy == CompressionStrategy.MINIMAL:
			return self._minimal_compress(sorted_chunks, token_limit)

		# Fallback to smart compression
		return self._smart_compress(sorted_chunks, token_limit)

	def _smart_compress(self, scored_chunks: list[tuple[Chunk, float]], token_limit: int) -> list[Chunk]:
		"""
		Smart compression balances chunk importance with content preservation.

		Args:
		    scored_chunks: List of (chunk, score) tuples sorted by importance
		    token_limit: Maximum token limit

		Returns:
		    Compressed list of chunks

		"""
		result = []
		token_count = 0

		for chunk, score in scored_chunks:
			compressed_content = self._compress_chunk_content(chunk, score)
			compressed_chunk = Chunk(
				content=compressed_content,
				metadata=chunk.metadata,
				children=chunk.children,
			)

			# Estimate token count (simple approximation)
			tokens = len(compressed_content.split())

			if token_count + tokens <= token_limit:
				result.append(compressed_chunk)
				token_count += tokens
			# For top importance chunks, try harder to include by aggressive compression
			elif score > HIGH_IMPORTANCE_THRESHOLD:
				aggressive_content = self._compress_chunk_content_aggressive(chunk)
				aggressive_chunk = Chunk(
					content=aggressive_content,
					metadata=chunk.metadata,
					children=chunk.children,
				)
				aggressive_tokens = len(aggressive_content.split())

				if token_count + aggressive_tokens <= token_limit:
					result.append(aggressive_chunk)
					token_count += aggressive_tokens

		return result

	def _aggressive_compress(self, scored_chunks: list[tuple[Chunk, float]], token_limit: int) -> list[Chunk]:
		"""
		Aggressive compression focuses on including as many chunks as possible.

		Args:
		    scored_chunks: List of (chunk, score) tuples sorted by importance
		    token_limit: Maximum token limit

		Returns:
		    Compressed list of chunks

		"""
		result = []
		token_count = 0

		for chunk, _score in scored_chunks:
			compressed_content = self._compress_chunk_content_aggressive(chunk)
			compressed_chunk = Chunk(
				content=compressed_content,
				metadata=chunk.metadata,
				children=chunk.children,
			)

			tokens = len(compressed_content.split())

			if token_count + tokens <= token_limit:
				result.append(compressed_chunk)
				token_count += tokens

		return result

	def _minimal_compress(self, scored_chunks: list[tuple[Chunk, float]], token_limit: int) -> list[Chunk]:
		"""
		Minimal compression only includes the most essential chunks.

		Args:
		    scored_chunks: List of (chunk, score) tuples sorted by importance
		    token_limit: Maximum token limit

		Returns:
		    Compressed list of chunks

		"""
		result = []
		token_count = 0

		# Only keep chunks with scores above medium importance threshold
		high_importance = [(chunk, score) for chunk, score in scored_chunks if score > MEDIUM_IMPORTANCE_THRESHOLD]

		for chunk, _score in high_importance:
			# For minimal compression, use the most aggressive compression
			compressed_content = self._compress_chunk_content_aggressive(chunk)
			compressed_chunk = Chunk(
				content=compressed_content,
				metadata=chunk.metadata,
				children=chunk.children,
			)

			tokens = len(compressed_content.split())

			if token_count + tokens <= token_limit:
				result.append(compressed_chunk)
				token_count += tokens

		return result

	def _compress_chunk_content(self, chunk: Chunk, importance_score: float) -> str:
		"""
		Compress chunk content based on its importance score.

		Args:
		    chunk: The chunk to compress
		    importance_score: Importance score of the chunk

		Returns:
		    Compressed content

		"""
		content = chunk.content
		metadata = chunk.metadata
		entity_type = metadata.entity_type

		# For LLM mode, apply different compression strategies
		if self.mode == GenerationMode.LLM:
			# Remove docstrings for less important chunks in LLM mode
			if importance_score < MEDIUM_IMPORTANCE_THRESHOLD and entity_type not in (
				EntityType.MODULE,
				EntityType.IMPORT,
			):
				content = self._remove_docstrings(content, metadata.language)

			# Remove comments for low importance chunks
			if importance_score < LOW_IMPORTANCE_THRESHOLD:
				content = self._remove_comments(content, metadata.language)

			# For functions and methods, keep signature but summarize body for low importance chunks
			if importance_score < LOW_IMPORTANCE_THRESHOLD and entity_type in (EntityType.FUNCTION, EntityType.METHOD):
				content = self._summarize_function_body(content, metadata.language)
		# For human mode, preserve docstrings and comments but may remove some implementation details
		elif importance_score < LOW_IMPORTANCE_THRESHOLD and entity_type in (EntityType.FUNCTION, EntityType.METHOD):
			content = self._preserve_docstrings_summarize_implementation(content, metadata.language)

		return content

	def _compress_chunk_content_aggressive(self, chunk: Chunk) -> str:
		"""
		Aggressively compress chunk content, keeping only essentials.

		Args:
		    chunk: The chunk to compress

		Returns:
		    Aggressively compressed content

		"""
		content = chunk.content
		metadata = chunk.metadata
		entity_type = metadata.entity_type

		# Remove all comments and docstrings in LLM mode
		if self.mode == GenerationMode.LLM:
			content = self._remove_comments(content, metadata.language)
			content = self._remove_docstrings(content, metadata.language)

			# For classes, keep only class definition and method signatures
			if entity_type == EntityType.CLASS:
				content = self._keep_only_signatures(content, metadata.language)

			# For functions, keep only signature and return statements
			elif entity_type in (EntityType.FUNCTION, EntityType.METHOD):
				content = self._keep_only_signature_and_returns(content, metadata.language)
		else:
			# For human mode, keep docstrings but remove implementation details
			content = self._preserve_docstrings_summarize_implementation(content, metadata.language)

		return content

	def _remove_docstrings(self, content: str, language: str) -> str:
		"""
		Remove docstrings from code content.

		Args:
		    content: Code content
		    language: Programming language

		Returns:
		    Content with docstrings removed

		"""
		# Simple implementation - would be more robust with tree-sitter
		if language == "python":
			# Remove triple-quoted docstrings (simplified approach)
			import re

			content = re.sub(r'"""[\s\S]*?"""', '"""..."""', content)
			content = re.sub(r"'''[\s\S]*?'''", "'''...'''", content)

		return content

	def _remove_comments(self, content: str, language: str) -> str:
		"""
		Remove comments from code content.

		Args:
		    content: Code content
		    language: Programming language

		Returns:
		    Content with comments removed

		"""
		# Simple implementation - would be more robust with tree-sitter
		if language in ("python", "javascript", "typescript"):
			import re

			# Remove single line comments
			content = re.sub(r"#.*$", "", content, flags=re.MULTILINE)
			content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)

		return content

	def _summarize_function_body(self, content: str, language: str) -> str:
		"""
		Keep function signature but summarize body.

		Args:
		    content: Code content
		    language: Programming language

		Returns:
		    Content with function body summarized

		"""
		if language == "python":
			import re

			# Match function definition and first line of body
			match = re.match(r"(def\s+[\w_]+\([^)]*\).*?:)[\s\S]*", content)
			if match:
				return f"{match.group(1)}\n    # ... implementation ...\n"

		return content

	def _keep_only_signatures(self, content: str, language: str) -> str:
		"""
		Keep only class definition and method signatures.

		Args:
		    content: Code content
		    language: Programming language

		Returns:
		    Content with only signatures

		"""
		if language == "python":
			import re

			# Keep class definition
			class_def = re.match(r"(class\s+[\w_]+(\([^)]*\))?.*?:)", content)
			if class_def:
				class_header = class_def.group(1)

				# Find all method signatures
				methods = re.findall(r"(\s+def\s+[\w_]+\([^)]*\).*?:)", content)

				# Reconstruct with class header and method signatures
				if methods:
					return (
						class_header + "\n" + "\n".join(method + "\n    # ... implementation ..." for method in methods)
					)

		return content

	def _keep_only_signature_and_returns(self, content: str, language: str) -> str:
		"""
		Keep only function signature and return statements.

		Args:
		    content: Code content
		    language: Programming language

		Returns:
		    Content with only signature and returns

		"""
		if language == "python":
			import re

			# Match function definition
			sig_match = re.match(r"(def\s+[\w_]+\([^)]*\).*?:)", content)
			if sig_match:
				signature = sig_match.group(1)

				# Find return statements
				returns = re.findall(r"(\s+return\s+.*)", content)

				# Reconstruct with signature and returns
				if returns:
					return signature + "\n    # ... implementation ...\n" + "\n".join(returns)
				return signature + "\n    # ... implementation ...\n"

		return content

	def _preserve_docstrings_summarize_implementation(self, content: str, language: str) -> str:
		"""
		Preserve docstrings but summarize implementation details.

		Args:
		    content: Code content
		    language: Programming language

		Returns:
		    Content with preserved docstrings and summarized implementation

		"""
		if language == "python":
			import re

			# Extract docstring if present
			docstring_match = re.match(r'(def\s+[\w_]+\([^)]*\).*?:)\s*(\n\s+"""[\s\S]*?""")', content)
			if docstring_match:
				signature = docstring_match.group(1)
				docstring = docstring_match.group(2)
				return f"{signature}{docstring}\n    # ... implementation ...\n"

		return content
