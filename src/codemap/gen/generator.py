"""Code documentation generator implementation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codemap.processor.analysis.tree_sitter.base import EntityType
from codemap.processor.chunking.base import Chunk

from .compressor import SemanticCompressor
from .models import DocFormat, GenConfig

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


class CodeMapGenerator:
	"""Generates code documentation for LLM context or human-readable docs."""

	def __init__(self, config: GenConfig, output_path: Path) -> None:
		"""
		Initialize the code map generator.

		Args:
		    config: Generation configuration settings
		    output_path: Path to write the output

		"""
		self.config = config
		self.output_path = output_path
		self.compressor = SemanticCompressor(config.compression_strategy, config.mode)

	def generate_llm_context(self, chunks: list[Chunk], metadata: dict) -> str:
		"""
		Generate LLM-optimized context from the processed chunks.

		Args:
		    chunks: List of code chunks
		    metadata: Repository metadata

		Returns:
		    Generated context as string

		"""
		# Compress chunks to fit within token limit
		compressed_chunks = self.compressor.compress_chunks(chunks, self.config.token_limit)

		# Generate markdown content for LLM consumption
		content = []

		# Add header with repository metadata
		content.append("# Code Repository Documentation")
		content.append(f"Generated on: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")

		# Add repository statistics
		if "stats" in metadata:
			stats = metadata["stats"]
			content.append("\n## Repository Statistics")
			content.append(f"- Total files: {stats.get('total_files', 0)}")
			content.append(f"- Total lines of code: {stats.get('total_lines', 0)}")
			content.append(f"- Languages: {', '.join(stats.get('languages', []))}")

		# Add directory structure if requested
		if self.config.include_tree and "tree" in metadata:
			content.append("\n## Directory Structure")
			content.append("```")
			content.append(metadata["tree"])
			content.append("```")

		# Add code chunks
		content.append("\n## Code Content")

		for chunk in compressed_chunks:
			location = chunk.metadata.location
			content.append(f"\n### {chunk.full_name}")
			content.append(f"File: {location.file_path}")
			content.append(f"Lines: {location.start_line}-{location.end_line}")
			content.append(f"Type: {chunk.metadata.entity_type.name}")

			if chunk.metadata.description:
				content.append(f"\nDescription: {chunk.metadata.description}")

			content.append("\n```" + chunk.metadata.language)
			content.append(chunk.content)
			content.append("```")

		return "\n".join(content)

	def generate_human_docs(self, chunks: list[Chunk], metadata: dict) -> str:
		"""
		Generate human-readable documentation from the processed chunks.

		Args:
		    chunks: List of code chunks
		    metadata: Repository metadata

		Returns:
		    Generated documentation as string

		"""
		# Generate documentation based on selected format
		if self.config.doc_format == DocFormat.MARKDOWN:
			return self._generate_markdown_docs(chunks, metadata)
		if self.config.doc_format == DocFormat.MINTLIFY:
			return self._generate_mintlify_docs()
		if self.config.doc_format == DocFormat.MKDOCS:
			return self._generate_mkdocs_docs()
		if self.config.doc_format == DocFormat.MDX:
			return self._generate_mdx_docs()

		# Default to markdown if format not recognized
		return self._generate_markdown_docs(chunks, metadata)

	def _generate_markdown_docs(self, chunks: list[Chunk], metadata: dict) -> str:
		"""
		Generate markdown documentation.

		Args:
		    chunks: List of code chunks
		    metadata: Repository metadata

		Returns:
		    Markdown documentation as string

		"""
		content = []

		# Add header with repository information
		repo_name = metadata.get("name", "Repository")
		content.append(f"# {repo_name} Documentation")
		content.append("\n" + metadata.get("description", ""))

		# Add table of contents
		content.append("\n## Table of Contents")

		# Group chunks by file
		files = {}
		for chunk in chunks:
			file_path = chunk.metadata.location.file_path
			if file_path not in files:
				files[file_path] = []
			files[file_path].append(chunk)

		# Create TOC entries
		for file_path in sorted(files.keys()):
			rel_path = file_path.name
			content.append(f"- [{rel_path}](#{rel_path.replace('.', '-')})")

		# Add installation instructions if available
		if "installation" in metadata:
			content.append("\n## Installation")
			content.append(metadata["installation"])

		# Add usage instructions if available
		if "usage" in metadata:
			content.append("\n## Usage")
			content.append(metadata["usage"])

		# Add code documentation grouped by file
		content.append("\n## Code Documentation")

		for file_path, file_chunks in sorted(files.items()):
			rel_path = file_path.name
			content.append(f"\n### {rel_path}")

			# Sort chunks by line number
			sorted_chunks = sorted(file_chunks, key=lambda c: c.metadata.location.start_line)

			for chunk in sorted_chunks:
				if chunk.metadata.entity_type in (EntityType.CLASS, EntityType.FUNCTION, EntityType.METHOD):
					content.append(f"\n#### {chunk.metadata.name}")

					# Add description if available
					if chunk.metadata.description:
						content.append(f"\n{chunk.metadata.description}")

					# Add code with syntax highlighting
					content.append("\n```" + chunk.metadata.language)
					content.append(chunk.content)
					content.append("```")

		return "\n".join(content)

	def _generate_mintlify_docs(self) -> str:
		"""
		Generate Mintlify documentation.

		Returns:
		    Mintlify documentation as string

		"""
		# Basic implementation, would be expanded in future
		content = []
		content.append("# Mintlify Documentation")
		content.append("Support for Mintlify format will be implemented in a future release.")
		return "\n".join(content)

	def _generate_mkdocs_docs(self) -> str:
		"""
		Generate MkDocs documentation.

		Returns:
		    MkDocs documentation as string

		"""
		# Basic implementation, would be expanded in future
		content = []
		content.append("# MkDocs Documentation")
		content.append("Support for MkDocs format will be implemented in a future release.")
		return "\n".join(content)

	def _generate_mdx_docs(self) -> str:
		"""
		Generate MDX documentation.

		Returns:
		    MDX documentation as string

		"""
		# Basic implementation, would be expanded in future
		content = []
		content.append("# MDX Documentation")
		content.append("Support for MDX format will be implemented in a future release.")
		return "\n".join(content)
