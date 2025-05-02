"""Builds the code graph by parsing files and populating KuzuDB."""

import logging
from pathlib import Path
from typing import Any, TypedDict

from codemap.processor.graph.kuzu_manager import KuzuManager
from codemap.processor.tree_sitter.analyzer import TreeSitterAnalyzer, get_language_by_extension
from codemap.processor.tree_sitter.base import EntityType

logger = logging.getLogger(__name__)


class EntityMetadata(TypedDict, total=False):
	"""Type definition for entity metadata."""

	name: str
	entity_type: EntityType
	start_line: int
	end_line: int
	signature: str
	docstring: str
	content_summary: str
	children: list["EntityMetadata"]


class GraphBuilder:
	"""Parses code files and populates the KuzuDB graph."""

	def __init__(self, repo_path: Path, kuzu_manager: KuzuManager, analyzer: TreeSitterAnalyzer) -> None:
		"""
		Initialize the GraphBuilder with repo path, database connection, and code analyzer.

		Args:
		        repo_path: Path to the repository root
		        kuzu_manager: Manager for KuzuDB operations
		        analyzer: TreeSitter code analyzer for parsing files

		"""
		self.repo_path = repo_path
		self.kuzu_manager = kuzu_manager
		self.analyzer = analyzer

	def _generate_entity_id(self, file_path: Path, entity: EntityMetadata) -> str:
		"""Generate a unique ID for a code entity."""
		# Simple approach: file_path + start_line + type + name
		# This might need better ID generation in production
		rel_path = file_path.relative_to(self.repo_path).as_posix()
		# Use get() to safely access TypedDict keys that might not be present
		type_name = entity.get("entity_type", EntityType.UNKNOWN).name
		entity_name = entity.get("name", "unnamed")
		start_line = entity.get("start_line", 0)
		return f"{rel_path}:{start_line}:{type_name}:{entity_name}"

	def _process_entity_recursive(
		self,
		entity: EntityMetadata,
		file_path: Path,
		file_lang: str,
		parent_id: str | None = None,
		community_id: str | None = None,
	) -> None:
		"""Process an entity and all its child entities recursively."""
		try:
			entity_id = self._generate_entity_id(file_path, entity)
			rel_file_path = file_path.relative_to(self.repo_path).as_posix()

			# Create entity in graph DB
			self.kuzu_manager.add_code_entity(
				entity_id=entity_id,
				file_path=rel_file_path,
				name=entity.get("name", ""),
				entity_type=entity.get("entity_type", EntityType.UNKNOWN),
				start_line=entity.get("start_line", 0),
				end_line=entity.get("end_line", 0),
				signature=entity.get("signature", ""),
				docstring=entity.get("docstring", ""),
				content_summary=entity.get("content_summary", ""),
				parent_entity_id=parent_id,
				community_id=community_id,
			)

			# Process child entities
			if children := entity.get("children"):
				for child in children:
					self._process_entity_recursive(
						entity=child,
						file_path=file_path,
						file_lang=file_lang,
						parent_id=entity_id,
						community_id=community_id,
					)
		except Exception:
			logger.exception(f"Failed to process entity {entity.get('name', 'unknown')} in {file_path}")

	def _ensure_communities(self, file_path: Path) -> str:
		"""Ensure communities (file, directories) exist for a given file path and return the file community ID."""
		parts = list(file_path.parts)
		current_path_str = ""
		parent_community_id = None
		# Assume project root community exists or is created elsewhere if needed globally
		# For simplicity, we'll create communities based on the path relative to repo root

		# Ensure directory communities exist
		for _i, part in enumerate(parts[:-1]):  # Iterate through directories
			current_path_str = str(Path(current_path_str) / part) if current_path_str else part
			community_id = f"dir:{current_path_str}"
			self.kuzu_manager.add_community(
				community_id=community_id, level="DIRECTORY", name=part, parent_community_id=parent_community_id
			)
			parent_community_id = community_id

		# Ensure file community exists
		file_community_id = f"file:{file_path!s}"
		self.kuzu_manager.add_community(
			community_id=file_community_id,
			level="FILE",
			name=file_path.name,
			parent_community_id=parent_community_id,  # Link file to its directory
		)
		return file_community_id

	def process_file(self, file_path: Path, git_hash: str | None = None) -> bool:
		"""
		Process a single file and add it to the graph database.

		Args:
		        file_path: Path to the file to process
		        git_hash: Optional git hash of the file content

		Returns:
		        bool: True if successful, False otherwise

		"""
		try:
			# Check if file exists and is readable
			if not file_path.exists() or not file_path.is_file():
				logger.warning(f"File not found or not a regular file: {file_path}")
				return False

			# Determine language based on file extension
			file_ext = file_path.suffix.lower()
			language = get_language_by_extension(file_path)
			if not language:
				logger.warning(f"Unsupported language for file: {file_path}")
				return False

			# Read file content
			try:
				content = file_path.read_text(encoding="utf-8")
			except Exception:
				logger.exception(f"Failed to read file {file_path}")
				return False

			# Parse file with TreeSitter
			rel_file_path = file_path.relative_to(self.repo_path).as_posix()
			analysis_result = self.analyzer.analyze_file(file_path, content, language)
			if not analysis_result.get("success", False) or not analysis_result.get("children"):
				logger.warning(f"No entities found in file: {file_path}")
				# Still add the file to track it, even if no entities found
				self.kuzu_manager.add_code_file(rel_file_path, git_hash or "", file_ext)
				return True

			# Add file to graph DB
			self.kuzu_manager.add_code_file(rel_file_path, git_hash or "", file_ext)

			# Calculate community ID (file level)
			# This uses a simple directory-based approach
			# More sophisticated community detection can be implemented
			file_community_id = f"file:{rel_file_path}"
			dir_path = file_path.parent
			parent_community_id = None

			# Create directory-based community hierarchy
			current_dir = dir_path
			while current_dir not in (self.repo_path, self.repo_path.parent):
				rel_dir_path = current_dir.relative_to(self.repo_path).as_posix()
				dir_community_id = f"dir:{rel_dir_path}"
				dir_name = current_dir.name

				# Add community node
				self.kuzu_manager.add_community(
					community_id=dir_community_id,
					level="directory",
					name=dir_name,
					parent_community_id=parent_community_id,
				)

				# Update for next iteration
				parent_community_id = dir_community_id
				current_dir = current_dir.parent

			# Create file-level community with directory as parent
			self.kuzu_manager.add_community(
				community_id=file_community_id,
				level="file",
				name=file_path.name,
				parent_community_id=parent_community_id,
			)

			# Process all entities in the analysis result
			# Convert to the right format for our _process_entity_recursive method
			entities = analysis_result.get("children", [])
			for entity in entities:
				# Transform the entity into the expected format
				transformed_entity = self._transform_entity(entity)
				self._process_entity_recursive(
					entity=transformed_entity,
					file_path=file_path,
					file_lang=language,
					parent_id=None,  # Top-level entity
					community_id=file_community_id,
				)

			# Process imports and other relationships
			self._process_imports(file_path, entities)
			self._process_calls(file_path, entities)
			self._process_inheritance(file_path, entities)

			logger.info(f"Successfully processed and added to graph: {file_path}")
			return True
		except Exception:
			logger.exception(f"Failed to process file {file_path} for graph")
			return False

	def _transform_entity(self, entity: dict) -> EntityMetadata:
		"""Transform tree-sitter analysis entity format to our internal format."""
		try:
			entity_type_str = entity.get("type", "UNKNOWN")
			entity_type = (
				EntityType[entity_type_str] if entity_type_str in EntityType.__members__ else EntityType.UNKNOWN
			)

			location = entity.get("location", {})

			transformed: EntityMetadata = {
				"name": entity.get("name", ""),
				"entity_type": entity_type,
				"start_line": location.get("start_line", 0),
				"end_line": location.get("end_line", 0),
				"docstring": entity.get("docstring", ""),
				"content_summary": entity.get("content", "")[:100] if entity.get("content") else "",
			}

			# Transform children recursively
			if children := entity.get("children"):
				transformed["children"] = [self._transform_entity(child) for child in children]

			return transformed
		except Exception:
			logger.exception(f"Failed to transform entity {entity.get('name', 'unknown')}")
			# Return at least a minimal valid entity
			return {"name": "error", "entity_type": EntityType.UNKNOWN, "start_line": 0, "end_line": 0}

	def _process_imports(self, file_path: Path, entities: list[Any]) -> None:
		"""Process and record import relationships."""
		# Implementation depends on the entity format from tree-sitter analysis
		# This is a placeholder - actual implementation would need to match the data format

	def _resolve_import_path(self, file_path: Path, import_path: str) -> Path | None:
		"""Resolve an import path to a file path."""
		# This is a simplified implementation
		# Real implementation would handle Python package path resolution,
		# relative imports, etc. based on the language
		try:
			# Handle absolute imports (e.g., 'codemap.processor.graph')
			if not import_path.startswith("."):
				parts = import_path.split(".")
				potential_path = self.repo_path / "src" / "/".join(parts)
				# Try with .py extension
				if (potential_path.with_suffix(".py")).exists():
					return potential_path.with_suffix(".py")
				# Try as directory with __init__.py
				if (potential_path / "__init__.py").exists():
					return potential_path / "__init__.py"
				return None

			# Handle relative imports (e.g., '.utils')
			parts = import_path.split(".")
			current_dir = file_path.parent
			# Handle dots at start of import (./, ../, etc.)
			dot_count = 0
			for part in parts:
				if not part:  # Empty part = one dot
					dot_count += 1
				else:
					break

			# Move up directories based on dot count
			for _ in range(dot_count - 1):  # -1 because one dot = current dir
				current_dir = current_dir.parent

			# Build path from remaining parts
			remaining_parts = [p for p in parts[dot_count:] if p]
			if not remaining_parts:
				# Import like "." or ".." - import the package itself
				return current_dir / "__init__.py"

			potential_path = current_dir / "/".join(remaining_parts)
			# Try with .py extension
			if (potential_path.with_suffix(".py")).exists():
				return potential_path.with_suffix(".py")
			# Try as directory with __init__.py
			if (potential_path / "__init__.py").exists():
				return potential_path / "__init__.py"
			return None
		except Exception:
			logger.exception(f"Error resolving import path: {import_path}")
			return None

	def _process_calls(self, file_path: Path, entities: list[Any]) -> None:
		"""Process and record function/method call relationships."""
		# Walk the entity tree and find call nodes
		# For each call, try to resolve the target function/method
		# TBD: Implementation depends on TreeSitter analyzer capabilities

	def _process_inheritance(self, file_path: Path, entities: list[Any]) -> None:
		"""Process and record class inheritance relationships."""
		# For each class entity, check if it has inheritance information
		# For each base class, try to resolve the target class
		# TBD: Implementation depends on TreeSitter analyzer capabilities
