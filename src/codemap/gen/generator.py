"""Code documentation generator implementation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from codemap.processor.lod import LODEntity, LODLevel
from codemap.processor.tree_sitter.base import EntityType

from .models import GenConfig

logger = logging.getLogger(__name__)


class CodeMapGenerator:
	"""Generates code documentation based on LOD (Level of Detail)."""

	def __init__(self, config: GenConfig, output_path: Path) -> None:
		"""
		Initialize the code map generator.

		Args:
		    config: Generation configuration settings
		    output_path: Path to write the output

		"""
		self.config = config
		self.output_path = output_path

	def generate_documentation(self, entities: list[LODEntity], metadata: dict) -> str:
		"""
		Generate markdown documentation from the processed LOD entities.

		Args:
		    entities: List of LOD entities
		    metadata: Repository metadata

		Returns:
		    Generated documentation as string

		"""
		content = []

		# Add header with repository information
		repo_name = metadata.get("name", "Repository")
		content.append(f"# {repo_name} Documentation")
		content.append(f"\nGenerated on: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")

		if "description" in metadata:
			content.append("\n" + metadata.get("description", ""))

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

		# Add table of contents
		content.append("\n## Table of Contents")

		# Group entities by file
		files: dict[Path, list[LODEntity]] = {}
		for entity in entities:
			file_path = Path(entity.metadata.get("file_path", ""))
			if not file_path.name:
				continue

			if file_path not in files:
				files[file_path] = []
			files[file_path].append(entity)

		# Create TOC entries
		for file_path in sorted(files.keys()):
			rel_path = file_path.name
			content.append(f"- [{rel_path}](#{rel_path.replace('.', '-')})")

		# Add code documentation grouped by file
		content.append("\n## Code Documentation")

		for file_path, file_entities in sorted(files.items()):
			rel_path = file_path.name
			content.append(f"\n### {rel_path}")

			# Sort entities by line number
			sorted_entities = sorted(file_entities, key=lambda e: e.start_line)

			for entity in sorted_entities:
				# Skip entities that aren't substantial code elements
				if entity.entity_type not in (
					EntityType.CLASS,
					EntityType.FUNCTION,
					EntityType.METHOD,
					EntityType.MODULE,
				):
					continue

				content.append(f"\n#### {entity.name}")

				# Add description/docstring if available
				if entity.docstring:
					content.append(f"\n{entity.docstring}")

				# Add signature if available (LOD level 3+)
				if entity.signature:
					content.append("\n```")
					content.append(entity.signature)
					content.append("```")

				# Add implementation details based on LOD level
				if self.config.lod_level == LODLevel.FULL and entity.content:
					lang = entity.language or ""
					content.append(f"\n```{lang}")
					content.append(entity.content)
					content.append("```")

				# Process child entities if any
				if entity.children:
					for child in entity.children:
						if child.entity_type in (EntityType.METHOD, EntityType.FUNCTION, EntityType.CLASS):
							content.append(f"\n##### {child.name}")

							if child.docstring:
								content.append(f"\n{child.docstring}")

							if child.signature:
								content.append("\n```")
								content.append(child.signature)
								content.append("```")

		return "\n".join(content)
