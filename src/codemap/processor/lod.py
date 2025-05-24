"""
Level of Detail (LOD) implementation for code analysis.

This module provides functionality for generating different levels of detail
from source code using tree-sitter analysis. The LOD approach provides a hierarchical
view of code, from high-level entity names to detailed implementations.

LOD levels:
- LOD1: Just entity names and types in files (classes, functions, etc.)
- LOD2: Entity names with docstrings
- LOD3: Entity names, docstrings, and signatures
- LOD4: Complete entity implementations

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from codemap.processor.tree_sitter.analyzer import TreeSitterAnalyzer
from codemap.processor.tree_sitter.base import EntityType

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


class LODLevel(Enum):
	"""Enumeration of Level of Detail levels."""

	SIGNATURES = 1  # Top-level entity names, docstrings, and signatures
	STRUCTURE = 2  # All entity signatures, indented structure
	DOCS = 3  # Level 2 + Docstrings for all entities
	SKELETON = 4  # Level 3 + Implementation skeleton (key patterns, control flow, important assignments)
	FULL = 5  # Level 4 + Full implementation content


@dataclass
class LODEntity:
	"""Represents a code entity at a specific level of detail."""

	name: str
	"""Name of the entity."""

	entity_type: EntityType
	"""Type of entity (class, function, etc.)."""

	start_line: int
	"""Starting line number (1-indexed)."""

	end_line: int
	"""Ending line number (1-indexed)."""

	docstring: str = ""
	"""Entity docstring, if available."""

	signature: str = ""
	"""Entity signature (e.g., function parameters), if available."""

	content: str = ""
	"""Complete entity content/implementation."""

	children: list[LODEntity] = field(default_factory=list)
	"""Child entities contained within this entity."""

	language: str = ""
	"""Programming language of the entity."""

	metadata: dict[str, Any] = field(default_factory=dict)
	"""Additional metadata about the entity."""


class LODGenerator:
	"""Generates different levels of detail from source code."""

	def __init__(self, analyzer: TreeSitterAnalyzer | None = None) -> None:
		"""
		Initialize the LOD generator.

		Args:
			analyzer: Optional shared TreeSitterAnalyzer instance. If None, a new one is created.
		"""
		self.analyzer = analyzer or TreeSitterAnalyzer()

	def generate_lod(self, file_path: Path, level: LODLevel = LODLevel.STRUCTURE) -> LODEntity | None:
		"""
		Generate LOD representation for a file.

		Args:
		    file_path: Path to the file to analyze
		    level: Level of detail to generate (default changed to STRUCTURE)

		Returns:
		    LODEntity representing the file, or None if analysis failed

		"""
		# Analyze file with tree-sitter - analyzer now handles content reading & caching
		analysis_result = self.analyzer.analyze_file(file_path)  # Pass only file_path
		if not analysis_result:
			logger.warning(f"Failed to analyze {file_path}")
			return None

		# Convert analysis result to LOD, passing the file_path
		return self._convert_to_lod(analysis_result, level, file_path)

	def _convert_to_lod(
		self, analysis_result: dict[str, Any], level: LODLevel, file_path: Path | None = None, is_root: bool = True
	) -> LODEntity:
		"""
		Convert tree-sitter analysis to LOD format.

		Args:
		    analysis_result: Tree-sitter analysis result
		    level: Level of detail to generate
		    file_path: Path to the file being analyzed (present for the root entity)
		    is_root: Whether the entity is the root entity for the file

		Returns:
		    LODEntity representation

		"""
		entity_type_str = analysis_result.get("type", "UNKNOWN")
		try:
			entity_type = getattr(EntityType, entity_type_str)
		except AttributeError:
			entity_type = EntityType.UNKNOWN

		location = analysis_result.get("location", {})
		start_line = location.get("start_line", 1)
		end_line = location.get("end_line", 1)

		# Get the name from analysis result
		entity_name = analysis_result.get("name", "")

		# For modules with placeholder names, use the filename instead
		if entity_type == EntityType.MODULE and entity_name.startswith("<anonymous-") and file_path:
			entity_name = file_path.stem  # Get filename without extension

		entity = LODEntity(
			name=entity_name,
			entity_type=entity_type,
			start_line=start_line,
			end_line=end_line,
			language=analysis_result.get("language", ""),
		)

		# Store file_path for all entities for node ID generation, but mark as root only for the top entity
		if file_path:
			entity.metadata["file_path"] = str(file_path)
			if is_root and "full_content_str" in analysis_result:
				# If full_content_str is available from analyzer, store it in root entity metadata
				entity.metadata["full_content_str"] = analysis_result["full_content_str"]

		if level.value >= LODLevel.DOCS.value:
			entity.docstring = analysis_result.get("docstring", "")

		if level.value >= LODLevel.SIGNATURES.value:
			# Extract signature from content if available
			content = analysis_result.get("content", "")
			entity.signature = self._extract_signature(content, entity_type, entity.language)

		if level.value >= LODLevel.SKELETON.value or entity_type in {EntityType.COMMENT, EntityType.CONSTANT}:
			entity.content = analysis_result.get("content", "")

		# Process children recursively (propagate file_path to children but mark as non-root)
		children = analysis_result.get("children", [])
		for child in children:
			child_entity = self._convert_to_lod(child, level, file_path, is_root=False)
			entity.children.append(child_entity)

		# Add any additional metadata
		if "dependencies" in analysis_result:
			entity.metadata["dependencies"] = analysis_result["dependencies"]
		if "calls" in analysis_result:
			entity.metadata["calls"] = analysis_result["calls"]

		return entity

	def _extract_signature(self, content: str, entity_type: EntityType, _language: str) -> str:
		"""
		Extract function/method signature from content.

		This is a simple implementation; ideally, the language-specific handlers
		should provide this functionality.

		Args:
		    content: Full entity content
		    entity_type: Type of entity
		    _language: Programming language (unused currently)

		Returns:
		    Signature string

		"""
		if not content:
			return ""

		# For functions and methods, extract the first line (declaration)
		if entity_type in [EntityType.FUNCTION, EntityType.METHOD, EntityType.CLASS, EntityType.INTERFACE]:
			lines = content.split("\n")
			if lines:
				# Return first line without trailing characters
				return lines[0].rstrip(":{")

		return ""
