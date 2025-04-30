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
		Get the node representing the 'body' of a definition.

		Args:
		    node: The tree-sitter node

		Returns:
		    The body node if available, None otherwise

		"""

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


class RustConfig(LanguageConfig):
	"""Configuration for Rust language."""

	module: ClassVar[list[str]] = ["source_file"]
	class_: ClassVar[list[str]] = ["impl_item"]
	function: ClassVar[list[str]] = ["function_item"]
	property_def: ClassVar[list[str]] = []
	struct: ClassVar[list[str]] = ["struct_item"]
	docstring: ClassVar[list[str]] = ["line_comment", "block_comment"]
	file_extensions: ClassVar[list[str]] = [".rs"]


class GoConfig(LanguageConfig):
	"""Configuration for Go language."""

	module: ClassVar[list[str]] = ["source_file"]
	class_: ClassVar[list[str]] = ["type_declaration"]
	function: ClassVar[list[str]] = ["function_declaration"]
	property_def: ClassVar[list[str]] = []
	struct: ClassVar[list[str]] = ["struct_type"]
	docstring: ClassVar[list[str]] = ["comment"]
	file_extensions: ClassVar[list[str]] = [".go"]
