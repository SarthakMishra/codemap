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
from codemap.utils.file_utils import read_file_content

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


class LODLevel(Enum):
	"""Enumeration of Level of Detail levels."""

	NAMES = 1  # Just entity names (classes, functions, etc.)
	DOCS = 2  # Names with docstrings
	SIGNATURES = 3  # Names, docstrings, and signatures
	FULL = 4  # Complete implementation


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

	def __init__(self) -> None:
		"""Initialize the LOD generator."""
		self.analyzer = TreeSitterAnalyzer()

	def generate_lod(self, file_path: Path, level: LODLevel = LODLevel.NAMES) -> LODEntity | None:
		"""
		Generate LOD representation for a file.

		Args:
		    file_path: Path to the file to analyze
		    level: Level of detail to generate

		Returns:
		    LODEntity representing the file, or None if analysis failed

		"""
		# Read file content
		content = read_file_content(file_path)
		if not content:
			logger.warning(f"Could not read content from {file_path}")
			return None

		# Analyze file with tree-sitter
		analysis_result = self.analyzer.analyze_file(file_path, content)
		if not analysis_result:
			logger.warning(f"Failed to analyze {file_path}")
			return None

		# Convert analysis result to LOD
		return self._convert_to_lod(analysis_result, level)

	def _convert_to_lod(self, analysis_result: dict[str, Any], level: LODLevel) -> LODEntity:
		"""
		Convert tree-sitter analysis to LOD format.

		Args:
		    analysis_result: Tree-sitter analysis result
		    level: Level of detail to generate

		Returns:
		    LODEntity representation

		"""
		entity_type_str = analysis_result.get("type", "UNKNOWN")
		try:
			entity_type = EntityType[entity_type_str]
		except KeyError:
			entity_type = EntityType.UNKNOWN

		location = analysis_result.get("location", {})
		start_line = location.get("start_line", 1)
		end_line = location.get("end_line", 1)

		# Create the entity with appropriate level of detail
		entity = LODEntity(
			name=analysis_result.get("name", ""),
			entity_type=entity_type,
			start_line=start_line,
			end_line=end_line,
			language=analysis_result.get("language", ""),
		)

		# Add details based on LOD level
		if level.value >= LODLevel.DOCS.value:
			entity.docstring = analysis_result.get("docstring", "")

		if level.value >= LODLevel.SIGNATURES.value:
			# Extract signature from content if available
			content = analysis_result.get("content", "")
			entity.signature = self._extract_signature(content, entity_type, entity.language)

		if level.value >= LODLevel.FULL.value:
			entity.content = analysis_result.get("content", "")

		# Process children recursively
		children = analysis_result.get("children", [])
		for child in children:
			child_entity = self._convert_to_lod(child, level)
			entity.children.append(child_entity)

		# Add any additional metadata
		if "dependencies" in analysis_result:
			entity.metadata["dependencies"] = analysis_result["dependencies"]

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
