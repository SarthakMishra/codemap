"""
Tree-sitter based code analysis.

This module provides functionality for analyzing source code using tree-
sitter.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
		self.ast_cache: dict[Path, tuple[float, Node, Parser]] = {}

	def get_parser(self, language: str) -> Parser | None:
		"""
		Get the parser for a language, loading it if necessary.

		Args:
		    language: The language to get a parser for

		Returns:
		    A tree-sitter parser or None if not supported
		"""
		if language in self.parsers:
			return self.parsers[language]

		# Lazy load the parser
		ts_lang_name = LANGUAGE_NAMES.get(language)
		if not ts_lang_name:
			return None
		try:
			lang_obj: Language = get_language(ts_lang_name)
			parser: Parser = Parser()
			parser.language = lang_obj
			self.parsers[language] = parser
			return parser
		except (ValueError, RuntimeError, ImportError) as e:
			logger.debug("Failed to load language %s: %s", language, str(e))
			return None

	def parse_file(
		self, file_path: Path, language: str | None = None
	) -> tuple[Node | None, str, Parser | None, bytes | None]:
		"""
		Parse a file and return its root node, determined language, the parser used, and content_bytes.

		Utilizes a cache to avoid re-parsing unchanged files.

		Args:
		    file_path: Path to the file to parse
		    language: Optional language override

		Returns:
		    A tuple containing the parse tree root node (or None if parsing failed),
		    the determined language string, the parser instance, and the file content as bytes (if read).
		"""
		# Determine language if not provided
		determined_language = language
		if not determined_language:
			determined_language = get_language_by_extension(file_path)
			if not determined_language:
				logger.debug("Could not determine language for file %s", file_path)
				return None, "", None, None

		try:
			current_mtime = file_path.stat().st_mtime
			if file_path in self.ast_cache:
				# Assuming cache stores (mtime, ast_root, parser, content_bytes)
				# For now, let's simplify and not cache content_bytes directly with AST to avoid large cache items
				# We will re-read if cache hit for AST, but this is still better than re-parsing.
				# A more advanced cache could handle content_bytes.
				cached_mtime, cached_tree_root, cached_parser = self.ast_cache[file_path]
				if current_mtime == cached_mtime:
					logger.debug("AST cache hit for: %s. Content will be re-read by caller if needed.", file_path)
					# To return content_bytes here, we would need to have cached it or re-read it.
					# For simplicity of this step, parse_file will return None for content_bytes
					# on a pure AST cache hit.
					# The caller (analyze_file) will then read it. The main win (no re-parse) is achieved.
					return cached_tree_root, determined_language, cached_parser, None
		except FileNotFoundError:
			logger.warning("File not found during mtime check: %s", file_path)
			if file_path in self.ast_cache:
				del self.ast_cache[file_path]
			return None, determined_language, None, None

		parser = self.get_parser(determined_language)
		if not parser:
			logger.debug("No parser for language %s", determined_language)
			return None, determined_language, None, None

		content_bytes_read: bytes | None = None  # Initialize here
		try:
			with file_path.open("rb") as f:
				content_bytes_read = f.read()
			tree = parser.parse(content_bytes_read)
			root_node = tree.root_node
			current_mtime_after_read = file_path.stat().st_mtime
			self.ast_cache[file_path] = (
				current_mtime_after_read,
				root_node,
				parser,
			)  # Not caching content_bytes in AST cache for now
			logger.debug("Parsed and cached AST for: %s", file_path)
			return root_node, determined_language, parser, content_bytes_read  # Return read content_bytes
		except FileNotFoundError:
			logger.warning("File not found during parsing: %s", file_path)
			if file_path in self.ast_cache:
				del self.ast_cache[file_path]
			# If file not found, content_bytes_read would not have been assigned (or remains None)
			return None, determined_language, parser, None
		except Exception:
			logger.exception("Failed to parse file %s", file_path)
			# content_bytes_read will be None if open failed, or the read bytes if parse failed
			return None, determined_language, parser, content_bytes_read

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
				logger.debug("Failed to extract dependencies: %s", e)

		# Build result
		result: dict[str, Any] = {
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
					logger.debug("Failed to extract calls for %s: %s", name or "<anonymous>", e)

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
		language: str | None = None,
	) -> dict:
		"""
		Analyze a file and return its structural information.

		Uses cached ASTs.
		The returned dictionary will include a 'full_content_str' key
		if the file content was successfully read.

		Args:
		    file_path: Path to the file
		    language: Optional language override

		Returns:
		    Structured analysis of the file or an empty dict on failure.
		    Includes 'full_content_str' with the file's decoded content if read.
		"""
		root_node, resolved_language, _, read_content_bytes = self.parse_file(file_path, language)

		if not root_node or not resolved_language:
			return {}

		handler = self.get_syntax_handler(resolved_language)
		if not handler:
			logger.debug("No syntax handler for language %s", resolved_language)
			return {}

		content_bytes_for_analysis = read_content_bytes
		decoded_full_content_str = None

		if content_bytes_for_analysis is None:
			try:
				with file_path.open("rb") as f:
					content_bytes_for_analysis = f.read()
			except Exception:
				logger.exception("Failed to re-read file content for analysis %s", file_path)
				return {}

		if content_bytes_for_analysis is not None:
			try:
				decoded_full_content_str = content_bytes_for_analysis.decode("utf-8", errors="ignore")
			except Exception:
				logger.exception("Failed to decode file content for %s", file_path)
				# Continue without decoded_full_content_str if decoding fails

		# Perform node-level analysis (recursive)
		analysis_data = self.analyze_node(root_node, content_bytes_for_analysis, file_path, resolved_language, handler)

		# Add the full decoded content to the top-level result if available
		if decoded_full_content_str is not None:
			analysis_data["full_content_str"] = decoded_full_content_str

		return analysis_data
