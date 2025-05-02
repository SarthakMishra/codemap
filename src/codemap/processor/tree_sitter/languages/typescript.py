"""TypeScript-specific configuration for syntax chunking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codemap.processor.tree_sitter.base import EntityType
from codemap.processor.tree_sitter.languages.base import LanguageConfig
from codemap.processor.tree_sitter.languages.javascript import JAVASCRIPT_CONFIG, JavaScriptSyntaxHandler

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
	function: ClassVar[list[str]] = [*JAVASCRIPT_CONFIG.function, "function_signature"]
	method: ClassVar[list[str]] = [*JAVASCRIPT_CONFIG.method, "method_signature"]
	property_def: ClassVar[list[str]] = [*JAVASCRIPT_CONFIG.property_def, "public_field_definition"]
	test_case: ClassVar[list[str]] = JAVASCRIPT_CONFIG.test_case
	test_suite: ClassVar[list[str]] = JAVASCRIPT_CONFIG.test_suite

	# Variables and constants
	variable: ClassVar[list[str]] = JAVASCRIPT_CONFIG.variable
	constant: ClassVar[list[str]] = JAVASCRIPT_CONFIG.constant
	class_field: ClassVar[list[str]] = [*JAVASCRIPT_CONFIG.class_field, "public_field_definition"]

	# Code organization
	import_: ClassVar[list[str]] = [*JAVASCRIPT_CONFIG.import_, "import_alias"]
	decorator: ClassVar[list[str]] = JAVASCRIPT_CONFIG.decorator

	# Documentation
	comment: ClassVar[list[str]] = JAVASCRIPT_CONFIG.comment
	docstring: ClassVar[list[str]] = JAVASCRIPT_CONFIG.docstring

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
		# Revert to super() and ignore potential linter false positive
		super().__init__(TYPESCRIPT_CONFIG)  # type: ignore[call-arg] # pylint: disable=too-many-function-args

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

		# Check for TypeScript specific types first
		if node_type in self.config.interface:
			return EntityType.INTERFACE
		if node_type in self.config.type_alias:
			return EntityType.TYPE_ALIAS
		if node_type == "enum_declaration":
			return EntityType.ENUM
		if node_type == "module":  # TS internal modules/namespaces
			return EntityType.NAMESPACE
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
		if node.type in ("interface_declaration", "function_signature", "method_signature"):
			return None  # Interfaces and signatures have no body block
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

	def get_enclosing_node_of_type(self, node: Node, target_type: EntityType) -> Node | None:
		"""
		Find the first ancestor node matching the target TypeScript entity type.

		Handles INTERFACE specifically and falls back to the JavaScript handler
		for other types (CLASS, FUNCTION, METHOD, MODULE).

		Args:
		    node: The starting node.
		    target_type: The EntityType to search for in ancestors.

		Returns:
		    The ancestor node if found, otherwise None.

		"""
		if target_type == EntityType.INTERFACE:
			target_node_types = ["interface_declaration"]
			current = node.parent
			while current:
				if current.type in target_node_types:
					return current
				current = current.parent
			return None
		# Fall back to JS handler for other types (CLASS, FUNCTION, METHOD, MODULE)
		return super().get_enclosing_node_of_type(node, target_type)
