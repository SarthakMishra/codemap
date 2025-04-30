"""
Tree-sitter based code analysis.

This module provides functionality for analyzing source code using tree-
sitter.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from tree_sitter import Language, Node, Parser
from tree_sitter_language_pack import SupportedLanguage, get_language

from codemap.processor.tree_sitter.base import EntityType
from codemap.processor.tree_sitter.languages import LANGUAGE_CONFIGS, LANGUAGE_HANDLERS, LanguageSyntaxHandler

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)

# Language name mapping for tree-sitter-language-pack
LANGUAGE_NAMES: dict[str, SupportedLanguage] = {
	"python": "python",
	"javascript": "javascript",
	"typescript": "typescript",
	# Add more languages as needed
}


def get_language_by_extension(file_path: Path) -> str | None:
	"""
	Get language name from file extension.

	Args:
	    file_path: Path to the file

	Returns:
	    Language name if supported, None otherwise

	"""
	ext = file_path.suffix
	for lang, config in LANGUAGE_CONFIGS.items():
		if ext in config.file_extensions:
			return lang
	return None


class TreeSitterAnalyzer:
	"""Analyzer for source code using tree-sitter."""

	def __init__(self) -> None:
		"""Initialize the tree-sitter analyzer."""
		self.parsers: dict[str, Parser] = {}
		self._load_parsers()

	def _load_parsers(self) -> None:
		"""
		Load tree-sitter parsers for supported languages.

		This method attempts to load parsers for all configured languages
		using tree-sitter-language-pack. If a language fails to load, it will
		be logged but won't prevent other languages from loading.

		"""
		self.parsers: dict[str, Parser] = {}
		failed_languages: list[tuple[str, str]] = []

		for lang in LANGUAGE_CONFIGS:
			try:
				# Get the language name for tree-sitter-language-pack
				ts_lang_name = LANGUAGE_NAMES.get(lang)
				if not ts_lang_name:
					continue

				# Get the language from tree-sitter-language-pack
				language: Language = get_language(ts_lang_name)

				# Create a new parser and set its language
				parser: Parser = Parser()
				parser.language = language

				self.parsers[lang] = parser
			except (ValueError, RuntimeError, ImportError) as e:
				failed_languages.append((lang, str(e)))
				logger.debug("Failed to load language %s: %s", lang, str(e))

		if failed_languages:
			failed_names = ", ".join(f"{lang} ({err})" for lang, err in failed_languages)
			logger.warning("Failed to load parsers for languages: %s", failed_names)

	def get_parser(self, language: str) -> Parser | None:
		"""
		Get the parser for a language.

		Args:
		    language: The language to get a parser for

		Returns:
		    A tree-sitter parser or None if not supported

		"""
		return self.parsers.get(language)

	def parse_file(self, file_path: Path, content: str, language: str | None = None) -> tuple[Node | None, str]:
		"""
		Parse a file and return its root node and determined language.

		Args:
		    file_path: Path to the file to parse
		    content: Content of the file
		    language: Optional language override

		Returns:
		    A tuple containing the parse tree root node (or None if parsing failed)
		    and the determined language

		"""
		# Determine language if not provided
		if not language:
			language = get_language_by_extension(file_path)
			if not language:
				logger.warning("Could not determine language for file %s", file_path)
				return None, ""

		# Get the parser for this language
		parser = self.get_parser(language)
		if not parser:
			logger.warning("No parser for language %s", language)
			return None, language

		try:
			# Parse the content using tree-sitter
			content_bytes = content.encode("utf-8")
			tree = parser.parse(content_bytes)
			return tree.root_node, language
		except Exception:
			logger.exception("Failed to parse file %s", file_path)
			return None, language

	def get_syntax_handler(self, language: str) -> LanguageSyntaxHandler | None:
		"""
		Get the syntax handler for a language.

		Args:
		    language: The language to get a handler for

		Returns:
		    A syntax handler or None if not supported

		"""
		handler_class = LANGUAGE_HANDLERS.get(language)
		if not handler_class:
			return None
		return handler_class()

	def analyze_node(
		self,
		node: Node,
		content_bytes: bytes,
		file_path: Path,
		language: str,
		handler: LanguageSyntaxHandler,
		parent_node: Node | None = None,
	) -> dict:
		"""
		Analyze a tree-sitter node and return structured information.

		Args:
		    node: The tree-sitter node
		    content_bytes: Source code content as bytes
		    file_path: Path to the source file
		    language: Programming language
		    handler: Language-specific syntax handler
		    parent_node: Parent node if any

		Returns:
		    Dict with node analysis information

		"""
		# Check if we should skip this node
		if handler.should_skip_node(node):
			return {}

		# Get entity type for this node from the handler
		entity_type = handler.get_entity_type(node, parent_node, content_bytes)

		# Skip unknown/uninteresting nodes unless they might contain interesting children
		if entity_type == EntityType.UNKNOWN and not node.named_child_count > 0:
			return {}

		# Get name and other metadata
		name = handler.extract_name(node, content_bytes)
		docstring_text, docstring_node = handler.find_docstring(node, content_bytes)

		# Get node content
		try:
			node_content = content_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
		except (UnicodeDecodeError, IndexError):
			node_content = ""

		# Extract dependencies from import statements
		dependencies = []
		if entity_type == EntityType.IMPORT:
			try:
				dependencies = handler.extract_imports(node, content_bytes)
			except (AttributeError, UnicodeDecodeError, IndexError, ValueError) as e:
				logger.warning("Failed to extract dependencies: %s", e)

		# Build result
		result = {
			"type": entity_type.name if entity_type != EntityType.UNKNOWN else "UNKNOWN",
			"name": name,
			"location": {
				"start_line": node.start_point[0] + 1,  # Convert to 1-based
				"end_line": node.end_point[0] + 1,
				"start_col": node.start_point[1],
				"end_col": node.end_point[1],
			},
			"docstring": docstring_text,
			"content": node_content,
			"children": [],
			"language": language,
		}

		# Add dependencies only if they exist to keep the output clean
		if dependencies:
			result["dependencies"] = dependencies

		# Extract function calls if the entity is a function or method
		calls = []
		if entity_type in (EntityType.FUNCTION, EntityType.METHOD):
			body_node = handler.get_body_node(node)
			if body_node:
				try:
					calls = handler.extract_calls(body_node, content_bytes)
				except (AttributeError, IndexError, UnicodeDecodeError, ValueError) as e:
					logger.warning("Failed to extract calls for %s: %s", name or "<anonymous>", e)

		# Add calls only if they exist
		if calls:
			result["calls"] = calls

		# Process child nodes
		body_node = handler.get_body_node(node)
		children_to_process = handler.get_children_to_process(node, body_node)

		for child in children_to_process:
			if docstring_node and child == docstring_node:
				continue  # Skip docstring node

			child_result = self.analyze_node(child, content_bytes, file_path, language, handler, node)

			if child_result:  # Only add non-empty results
				result["children"].append(child_result)

		return result

	def analyze_file(
		self,
		file_path: Path,
		content: str,
		language: str | None = None,
	) -> dict:
		"""
		Analyze a file and return its structural information.

		Args:
		    file_path: Path to the file to analyze
		    content: Content of the file
		    language: Optional language override
		    git_metadata: Optional Git metadata

		Returns:
		    Dict with file analysis information

		"""
		# Parse the file
		root_node, determined_language = self.parse_file(file_path, content, language)
		if not root_node or not determined_language:
			return {
				"file": str(file_path),
				"language": determined_language or "unknown",
				"success": False,
				"error": "Failed to parse file",
			}

		# Get handler for the language
		handler = self.get_syntax_handler(determined_language)
		if not handler:
			return {
				"file": str(file_path),
				"language": determined_language,
				"success": False,
				"error": f"No handler for language {determined_language}",
			}

		# Analyze the root node
		content_bytes = content.encode("utf-8")
		entity_type = handler.get_entity_type(root_node, None, content_bytes)
		if entity_type == EntityType.UNKNOWN:
			entity_type = EntityType.MODULE  # Default to MODULE type

		# Extract module-level docstring
		module_description, module_docstring_node = handler.find_docstring(root_node, content_bytes)

		# Create result
		result = {
			"file": str(file_path),
			"language": determined_language,
			"success": True,
			"type": entity_type.name,
			"name": file_path.stem,
			"location": {
				"start_line": root_node.start_point[0] + 1,
				"end_line": root_node.end_point[0] + 1,
				"start_col": root_node.start_point[1],
				"end_col": root_node.end_point[1],
			},
			"docstring": module_description,
			"children": [],
		}

		# Process children of the root node
		children_to_process = handler.get_children_to_process(root_node, None)
		for child in children_to_process:
			# Skip the module docstring node if found
			if module_docstring_node and child == module_docstring_node:
				continue

			child_result = self.analyze_node(child, content_bytes, file_path, determined_language, handler)

			if child_result:  # Only add non-empty results
				result["children"].append(child_result)

		return result
