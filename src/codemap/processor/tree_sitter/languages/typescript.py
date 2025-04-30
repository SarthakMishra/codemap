"""TypeScript-specific configuration for syntax chunking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codemap.processor.tree_sitter.base import EntityType
from codemap.processor.tree_sitter.languages.base import LanguageConfig
from codemap.processor.tree_sitter.languages.javascript import JavaScriptSyntaxHandler

if TYPE_CHECKING:
	from tree_sitter import Node

logger = logging.getLogger(__name__)


class TypeScriptConfig(LanguageConfig):
	"""TypeScript-specific syntax chunking configuration."""

	# File-level entities
	module: ClassVar[list[str]] = ["program"]
	namespace: ClassVar[list[str]] = ["export_statement", "namespace_declaration"]

	# Type definitions
	class_: ClassVar[list[str]] = ["class_declaration", "class"]
	interface: ClassVar[list[str]] = ["interface_declaration"]
	protocol: ClassVar[list[str]] = []  # TypeScript doesn't have protocols
	struct: ClassVar[list[str]] = []  # TypeScript doesn't have structs
	enum: ClassVar[list[str]] = ["enum_declaration"]
	type_alias: ClassVar[list[str]] = ["type_alias_declaration"]

	# Functions and methods
	function: ClassVar[list[str]] = [
		"function_declaration",
		"function",
		"arrow_function",
		"generator_function_declaration",
	]
	method: ClassVar[list[str]] = ["method_definition", "method_signature"]
	property_def: ClassVar[list[str]] = ["property_identifier", "public_field_definition", "property_signature"]
	test_case: ClassVar[list[str]] = ["call_expression"]  # Special detection for test frameworks
	test_suite: ClassVar[list[str]] = ["call_expression"]  # Special detection for test frameworks

	# Variables and constants
	variable: ClassVar[list[str]] = ["variable_declaration", "lexical_declaration"]
	constant: ClassVar[list[str]] = ["variable_declaration", "lexical_declaration"]  # const declarations
	class_field: ClassVar[list[str]] = ["public_field_definition"]

	# Code organization
	import_: ClassVar[list[str]] = ["import_statement"]
	decorator: ClassVar[list[str]] = ["decorator"]

	# Documentation
	comment: ClassVar[list[str]] = ["comment"]
	docstring: ClassVar[list[str]] = ["comment"]  # TS uses comments for documentation

	file_extensions: ClassVar[list[str]] = [".ts", ".tsx"]
	tree_sitter_name: ClassVar[str] = "typescript"


TYPESCRIPT_CONFIG = TypeScriptConfig()


class TypeScriptSyntaxHandler(JavaScriptSyntaxHandler):
	"""
	TypeScript-specific syntax handling logic.

	Inherits from JavaScript handler to reuse common logic.

	"""

	def __init__(self) -> None:
		"""Initialize with TypeScript configuration."""
		super().__init__()
		# Override the config from JavaScriptSyntaxHandler
		self.config = TYPESCRIPT_CONFIG

	def get_entity_type(self, node: Node, parent: Node | None, content_bytes: bytes) -> EntityType:
		"""
		Determine the EntityType for a TypeScript node.

		Args:
		    node: The tree-sitter node
		    parent: The parent node (if any)
		    content_bytes: Source code content as bytes

		Returns:
		    The entity type

		"""
		node_type = node.type
		logger.debug(
			"Getting entity type for TypeScript node: type=%s, parent_type=%s",
			node_type,
			parent.type if parent else None,
		)

		# TypeScript-specific types
		if node_type == "interface_declaration":
			return EntityType.INTERFACE
		if node_type == "enum_declaration":
			return EntityType.ENUM
		if node_type == "type_alias_declaration":
			return EntityType.TYPE_ALIAS
		if node_type == "namespace_declaration":
			return EntityType.NAMESPACE
		if node_type == "method_signature":
			return EntityType.METHOD
		if node_type == "property_signature":
			return EntityType.PROPERTY

		# Use the JavaScript logic for common types
		return super().get_entity_type(node, parent, content_bytes)

	def extract_name(self, node: Node, content_bytes: bytes) -> str:
		"""
		Extract the name identifier from a definition node.

		Args:
		    node: The tree-sitter node
		    content_bytes: Source code content as bytes

		Returns:
		    The extracted name

		"""
		# Handle TypeScript-specific node types first
		name_node = None

		if node.type in [
			"interface_declaration",
			"enum_declaration",
			"type_alias_declaration",
			"namespace_declaration",
		] or node.type in ["method_signature", "property_signature"]:
			name_node = node.child_by_field_name("name")

		if name_node:
			try:
				return content_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")
			except (UnicodeDecodeError, IndexError, AttributeError) as e:
				logger.warning("Failed to decode TypeScript name: %s", e)
				return f"<decoding-error-{node.type}>"

		# Fall back to JavaScript name extraction
		return super().extract_name(node, content_bytes)

	def get_body_node(self, node: Node) -> Node | None:
		"""
		Get the node representing the 'body' of a definition.

		Args:
		    node: The tree-sitter node

		Returns:
		    The body node if available, None otherwise

		"""
		# TypeScript-specific handling
		if node.type in ["interface_declaration", "namespace_declaration"] or node.type == "enum_declaration":
			return node.child_by_field_name("body")

		# Fall back to JavaScript body extraction
		return super().get_body_node(node)

	def get_children_to_process(self, node: Node, body_node: Node | None) -> list[Node]:
		"""
		Get the list of child nodes that should be recursively processed.

		Args:
		    node: The tree-sitter node
		    body_node: The body node if available

		Returns:
		    List of child nodes to process

		"""
		# TypeScript-specific handling
		if node.type == "type_alias_declaration":
			# Type aliases don't have children to process
			return []

		# Fall back to JavaScript children processing
		return super().get_children_to_process(node, body_node)

	def extract_imports(self, node: Node, content_bytes: bytes) -> list[str]:
		"""
		Extract imported module names from a TypeScript import statement.

		Args:
		    node: The tree-sitter node representing an import statement
		    content_bytes: Source code content as bytes

		Returns:
		    List of imported module names as strings

		"""
		# TypeScript import statements are the same as JavaScript
		return super().extract_imports(node, content_bytes)
