"""
Base configuration for language-specific syntax chunking.

This module provides the base configuration class for defining how
different programming languages map their syntax elements to code
chunks.

"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from codemap.processor.tree_sitter.base import EntityType

if TYPE_CHECKING:
	from tree_sitter import Node


@dataclass(frozen=True)
class LanguageConfig:
	"""
	Configuration for language-specific syntax chunking.

	This class defines how a specific programming language's syntax
	elements map to different types of code chunks. Each field is a list of
	syntax node types that represent that kind of entity in the language's
	AST.

	"""

	# File-level entities
	module: ClassVar[list[str]]
	"""Node types that represent entire modules/files."""

	namespace: ClassVar[list[str]]
	"""Node types for namespace/package declarations."""

	# Type definitions
	class_: ClassVar[list[str]]
	"""Node types for class definitions."""

	interface: ClassVar[list[str]]
	"""Node types for interface definitions."""

	protocol: ClassVar[list[str]]
	"""Node types for protocol/trait definitions."""

	struct: ClassVar[list[str]]
	"""Node types for struct definitions."""

	enum: ClassVar[list[str]]
	"""Node types for enum declarations."""

	type_alias: ClassVar[list[str]]
	"""Node types for type aliases/typedefs."""

	# Functions and methods
	function: ClassVar[list[str]]
	"""Node types for function declarations."""

	method: ClassVar[list[str]]
	"""Node types for method declarations."""

	property_def: ClassVar[list[str]]
	"""Node types for property/getter/setter declarations."""

	test_case: ClassVar[list[str]]
	"""Node types that identify test functions."""

	test_suite: ClassVar[list[str]]
	"""Node types that identify test classes/suites."""

	# Variables and constants
	variable: ClassVar[list[str]]
	"""Node types for variable declarations."""

	constant: ClassVar[list[str]]
	"""Node types for constant declarations."""

	class_field: ClassVar[list[str]]
	"""Node types for class field declarations."""

	# Code organization
	import_: ClassVar[list[str]]
	"""Node types for import statements."""

	decorator: ClassVar[list[str]]
	"""Node types for decorators/annotations."""

	# Documentation
	comment: ClassVar[list[str]]
	"""Node types for general comments."""

	docstring: ClassVar[list[str]]
	"""Node types for documentation strings."""

	# Language-specific metadata
	file_extensions: ClassVar[list[str]]
	"""File extensions associated with this language (e.g., ['.py', '.pyi'])."""

	tree_sitter_name: ClassVar[str] = ""
	"""Tree-sitter language identifier."""

	# Optional node types that might be language-specific
	decorators: ClassVar[list[str] | None] = None
	class_fields: ClassVar[list[str] | None] = None

	@property
	def all_node_types(self) -> set[str]:
		"""
		Get all node types defined in this configuration.

		Returns:
		    A set of all node types from all categories.

		"""
		all_types = set()
		for attr in [
			self.module,
			self.namespace,
			self.class_,
			self.interface,
			self.protocol,
			self.struct,
			self.enum,
			self.type_alias,
			self.function,
			self.method,
			self.property_def,
			self.test_case,
			self.test_suite,
			self.variable,
			self.constant,
			self.class_field,
			self.import_,
			self.decorator,
			self.comment,
			self.docstring,
			self.decorators,
			self.class_fields,
		]:
			if attr:  # Skip None values
				all_types.update(attr)
		return all_types


class LanguageSyntaxHandler(abc.ABC):
	"""Abstract base class for language-specific syntax handling."""

	def __init__(self, config: LanguageConfig) -> None:
		"""
		Initialize with language configuration.

		Args:
		    config: Language-specific configuration

		"""
		self.config = config

	@abc.abstractmethod
	def get_entity_type(self, node: Node, parent: Node | None, content_bytes: bytes) -> EntityType:
		"""
		Determine the EntityType for a given node.

		Args:
		    node: The tree-sitter node
		    parent: The parent node (if any)
		    content_bytes: Source code content as bytes

		Returns:
		    The entity type

		"""

	@abc.abstractmethod
	def find_docstring(self, node: Node, content_bytes: bytes) -> tuple[str | None, Node | None]:
		"""
		Find the docstring associated with a definition node.

		Args:
		    node: The tree-sitter node
		    content_bytes: Source code content as bytes

		Returns:
		    A tuple containing:
		    - The extracted docstring text (or None).
		    - The specific AST node representing the docstring that should be skipped
		      during child processing (or None).

		"""

	@abc.abstractmethod
	def extract_name(self, node: Node, content_bytes: bytes) -> str:
		"""
		Extract the name identifier from a definition node.

		Args:
		    node: The tree-sitter node
		    content_bytes: Source code content as bytes

		Returns:
		    The extracted name

		"""

	@abc.abstractmethod
	def get_body_node(self, node: Node) -> Node | None:
		"""
		Get the main body node for a function, method, or class.

		This should be overridden by subclasses to find the appropriate block node.

		Args:
		    node: The node representing the function, method, or class.

		Returns:
		    The body node, or None if not applicable/found.

		"""
		# Default implementation: returns None or the node itself as a naive fallback
		# Subclasses should find specific body nodes like 'block', 'statement_block' etc.
		return node  # Naive fallback - subclasses MUST override

	@abc.abstractmethod
	def get_children_to_process(self, node: Node, body_node: Node | None) -> list[Node]:
		"""
		Get the list of child nodes that should be recursively processed.

		Args:
		    node: The tree-sitter node
		    body_node: The body node if available

		Returns:
		    List of child nodes to process

		"""

	@abc.abstractmethod
	def should_skip_node(self, node: Node) -> bool:
		"""
		Determine if a node should be skipped entirely during processing.

		Args:
		    node: The tree-sitter node

		Returns:
		    True if the node should be skipped

		"""

	@abc.abstractmethod
	def extract_imports(self, node: Node, content_bytes: bytes) -> list[str]:
		"""
		Extract imported dependency names from an import node.

		Args:
		    node: The tree-sitter node (should be an import type)
		    content_bytes: Source code content as bytes

		Returns:
		    List of imported names

		"""

	@abc.abstractmethod
	def extract_calls(self, node: Node, content_bytes: bytes) -> list[str]:
		"""
		Extract names of functions/methods called within a node's scope.

		Args:
		    node: The tree-sitter node (e.g., function/method body)
		    content_bytes: Source code content as bytes

		Returns:
		    List of called function/method names

		"""

	def extract_signature(self, node: Node, content_bytes: bytes) -> str | None:
		"""
		Extract the signature (definition line without body) for a function, class, etc.

		Args:
		    node: The node to extract the signature from.
		    content_bytes: Source code content as bytes.

		Returns:
		    The signature string, or None if not applicable.

		"""
		# Default implementation: return the first line of the node's text
		try:
			first_line = content_bytes[node.start_byte : node.end_byte].split(b"\n", 1)[0]
			return first_line.decode("utf-8", errors="ignore").strip()
		except (IndexError, UnicodeDecodeError):
			# Catch specific errors related to slicing and decoding
			return None

	def get_enclosing_node_of_type(self, node: Node, target_type: EntityType) -> Node | None:
		"""
		Find the first ancestor node that matches the target entity type.

		Args:
		    node: The starting node.
		    target_type: The EntityType to search for in ancestors.

		Returns:
		    The ancestor node if found, otherwise None.

		"""
		current = node.parent
		while current:
			# We need content_bytes to determine the type accurately, but we don't have it here.
			# This highlights a limitation of doing this purely structurally without context.
			# Subclasses might need a different approach or access to the analyzer/content.
			# For a basic structural check:
			# entity_type = self.get_entity_type(current, current.parent, ???) # Need content_bytes
			# if entity_type == target_type:
			#    return current

			# Simplistic check based on node type name (less reliable)
			target_name = str(target_type.name).lower()  # Extract name explicitly for type checker
			if target_name in current.type.lower():  # Very rough check
				return current
			current = current.parent
		return None


class PythonConfig(LanguageConfig):
	"""Configuration for Python language."""

	module: ClassVar[list[str]] = ["module"]
	class_: ClassVar[list[str]] = ["class_definition"]
	function: ClassVar[list[str]] = ["function_definition"]
	property_def: ClassVar[list[str]] = ["decorated_definition"]
	struct: ClassVar[list[str]] = []
	docstring: ClassVar[list[str]] = ["string"]
	file_extensions: ClassVar[list[str]] = [".py", ".pyi"]


class JavaScriptConfig(LanguageConfig):
	"""Configuration for JavaScript language."""

	module: ClassVar[list[str]] = ["program"]
	class_: ClassVar[list[str]] = ["class_declaration", "class"]
	function: ClassVar[list[str]] = ["function_declaration", "method_definition", "function"]
	property_def: ClassVar[list[str]] = ["property_definition", "property_identifier"]
	struct: ClassVar[list[str]] = []
	docstring: ClassVar[list[str]] = ["comment"]
	file_extensions: ClassVar[list[str]] = [".js", ".jsx"]


class TypeScriptConfig(LanguageConfig):
	"""Configuration for TypeScript language."""

	module: ClassVar[list[str]] = ["program"]
	class_: ClassVar[list[str]] = ["class_declaration", "class"]
	function: ClassVar[list[str]] = ["function_declaration", "method_definition", "function"]
	property_def: ClassVar[list[str]] = ["property_definition", "property_identifier"]
	struct: ClassVar[list[str]] = []
	docstring: ClassVar[list[str]] = ["comment"]
	file_extensions: ClassVar[list[str]] = [".ts", ".tsx"]
