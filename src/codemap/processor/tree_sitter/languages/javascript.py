"""JavaScript-specific configuration for syntax chunking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codemap.processor.tree_sitter.base import EntityType
from codemap.processor.tree_sitter.languages.base import LanguageConfig, LanguageSyntaxHandler

if TYPE_CHECKING:
	from tree_sitter import Node

logger = logging.getLogger(__name__)


class JavaScriptConfig(LanguageConfig):
	"""JavaScript-specific syntax chunking configuration."""

	# File-level entities
	module: ClassVar[list[str]] = ["program"]
	namespace: ClassVar[list[str]] = ["export_statement"]  # Using export as namespace indicator

	# Type definitions
	class_: ClassVar[list[str]] = ["class_declaration", "class"]
	interface: ClassVar[list[str]] = []  # Pure JS doesn't have interfaces
	protocol: ClassVar[list[str]] = []  # Pure JS doesn't have protocols
	struct: ClassVar[list[str]] = []  # Pure JS doesn't have structs
	enum: ClassVar[list[str]] = []  # Pure JS doesn't have enums
	type_alias: ClassVar[list[str]] = []  # Pure JS doesn't have type aliases

	# Functions and methods
	function: ClassVar[list[str]] = [
		"function_declaration",
		"function",
		"arrow_function",
		"generator_function_declaration",
	]
	method: ClassVar[list[str]] = ["method_definition"]
	property_def: ClassVar[list[str]] = ["property_identifier", "public_field_definition"]
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
	docstring: ClassVar[list[str]] = ["comment"]  # JS uses comments for documentation

	file_extensions: ClassVar[list[str]] = [".js", ".jsx", ".mjs", ".cjs"]
	tree_sitter_name: ClassVar[str] = "javascript"


JAVASCRIPT_CONFIG = JavaScriptConfig()


class JavaScriptSyntaxHandler(LanguageSyntaxHandler):
	"""JavaScript-specific syntax handling logic."""

	def __init__(self) -> None:
		"""Initialize with JavaScript configuration."""
		super().__init__(JAVASCRIPT_CONFIG)

	def get_entity_type(self, node: Node, parent: Node | None, content_bytes: bytes) -> EntityType:
		"""
		Determine the EntityType for a JavaScript node.

		Args:
		    node: The tree-sitter node
		    parent: The parent node (if any)
		    content_bytes: Source code content as bytes

		Returns:
		    The entity type

		"""
		node_type = node.type
		logger.debug(
			"Getting entity type for JavaScript node: type=%s, parent_type=%s",
			node_type,
			parent.type if parent else None,
		)

		# Module-level
		if node_type in self.config.module:
			return EntityType.MODULE
		if node_type in self.config.namespace:
			return EntityType.NAMESPACE

		# Documentation
		if node_type in self.config.comment:
			# Check if this is a JSDoc comment (starts with /**)
			if self._is_jsdoc_comment(node, content_bytes):
				return EntityType.DOCSTRING
			return EntityType.COMMENT

		# Type definitions
		if node_type in self.config.class_:
			return EntityType.CLASS

		# Functions and methods
		if node_type in self.config.function:
			# Check if this is a test function (for frameworks like Jest, Mocha)
			if self._is_test_function(node, content_bytes):
				return EntityType.TEST_CASE
			return EntityType.FUNCTION

		if node_type in self.config.method:
			return EntityType.METHOD

		# Check for test suite declarations (describe blocks in Jest/Mocha)
		if node_type in self.config.test_suite and self._is_test_suite(node, content_bytes):
			return EntityType.TEST_SUITE

		# Property definitions
		if node_type in self.config.property_def:
			return EntityType.PROPERTY

		# Variables and constants
		if node_type in self.config.variable:
			# Check if it's a const declaration
			if self._is_constant(node, content_bytes):
				return EntityType.CONSTANT
			return EntityType.VARIABLE

		# Class fields
		if node_type in self.config.class_field:
			return EntityType.CLASS_FIELD

		# Code organization
		if node_type in self.config.import_:
			return EntityType.IMPORT

		return EntityType.UNKNOWN

	def _is_jsdoc_comment(self, node: Node, content_bytes: bytes) -> bool:
		"""
		Check if a comment node is a JSDoc comment.

		Args:
		    node: The comment node
		    content_bytes: Source code content as bytes

		Returns:
		    True if the node is a JSDoc comment

		"""
		if node.type != "comment":
			return False

		try:
			comment_text = content_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
			return comment_text.startswith("/**") and comment_text.endswith("*/")
		except (UnicodeDecodeError, IndexError):
			return False

	def _is_constant(self, node: Node, content_bytes: bytes) -> bool:
		"""
		Check if a variable declaration is a constant.

		Args:
		    node: The variable declaration node
		    content_bytes: Source code content as bytes

		Returns:
		    True if the node is a constant declaration

		"""
		if node.type not in ["variable_declaration", "lexical_declaration"]:
			return False

		try:
			decl_text = content_bytes[node.start_byte : node.start_byte + 5].decode("utf-8", errors="ignore")
			return decl_text.startswith("const")
		except (UnicodeDecodeError, IndexError):
			return False

	def _is_test_function(self, node: Node, content_bytes: bytes) -> bool:
		"""
		Check if a function is a test function.

		Args:
		    node: The function node
		    content_bytes: Source code content as bytes

		Returns:
		    True if the node is a test function

		"""
		if node.type == "call_expression":
			callee = node.child_by_field_name("function")
			if callee:
				try:
					callee_text = content_bytes[callee.start_byte : callee.end_byte].decode("utf-8", errors="ignore")
					return callee_text in ["it", "test"]
				except (UnicodeDecodeError, IndexError):
					pass
		return False

	def _is_test_suite(self, node: Node, content_bytes: bytes) -> bool:
		"""
		Check if a node is a test suite declaration.

		Args:
		    node: The node
		    content_bytes: Source code content as bytes

		Returns:
		    True if the node is a test suite declaration

		"""
		if node.type == "call_expression":
			callee = node.child_by_field_name("function")
			if callee:
				try:
					callee_text = content_bytes[callee.start_byte : callee.end_byte].decode("utf-8", errors="ignore")
					return callee_text == "describe"
				except (UnicodeDecodeError, IndexError):
					pass
		return False

	def find_docstring(self, node: Node, content_bytes: bytes) -> tuple[str | None, Node | None]:
		"""
		Find the docstring associated with a definition node.

		Args:
		    node: The tree-sitter node
		    content_bytes: Source code content as bytes

		Returns:
		    A tuple containing:
		    - The extracted docstring text (or None).
		    - The specific AST node representing the docstring (or None).

		"""
		# For functions, classes, and other definition nodes
		parent_node = node.parent

		# Look for JSDoc comments immediately preceding the node
		if parent_node:
			index = None
			for i, child in enumerate(parent_node.children):
				if child == node:
					index = i
					break

			if index is not None and index > 0:
				prev_node = parent_node.children[index - 1]
				if prev_node.type == "comment" and self._is_jsdoc_comment(prev_node, content_bytes):
					try:
						comment_text = content_bytes[prev_node.start_byte : prev_node.end_byte].decode(
							"utf-8", errors="ignore"
						)
						# Clean JSDoc format: remove /** */ and trim
						comment_text = comment_text.strip()
						comment_text = comment_text.removeprefix("/**")
						comment_text = comment_text.removesuffix("*/")
						return comment_text.strip(), prev_node
					except (UnicodeDecodeError, IndexError) as e:
						logger.warning("Failed to decode JavaScript comment: %s", e)

		return None, None

	def extract_name(self, node: Node, content_bytes: bytes) -> str:
		"""
		Extract the name identifier from a definition node.

		Args:
		    node: The tree-sitter node
		    content_bytes: Source code content as bytes

		Returns:
		    The extracted name

		"""
		# Try to find the name field based on node type
		name_node = None

		if node.type in ["function_declaration", "class_declaration", "method_definition"]:
			name_node = node.child_by_field_name("name")
		elif node.type == "property_identifier":
			name_node = node
		elif node.type in ["variable_declaration", "lexical_declaration"]:
			# Get the first declarator and its name
			declarator = node.child_by_field_name("declarations")
			if declarator and declarator.named_child_count > 0:
				first_declarator = declarator.named_children[0]
				name_node = first_declarator.child_by_field_name("name")
		elif node.type == "public_field_definition":
			name_node = node.child_by_field_name("name")

		if name_node:
			try:
				return content_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")
			except (UnicodeDecodeError, IndexError, AttributeError) as e:
				logger.warning("Failed to decode JavaScript name: %s", e)
				return f"<decoding-error-{node.type}>"

		# For call expressions that represent tests or suites
		if node.type == "call_expression":
			callee = node.child_by_field_name("function")
			arguments = node.child_by_field_name("arguments")

			if callee and arguments and arguments.named_child_count > 0:
				# First argument is typically the test/suite name
				first_arg = arguments.named_children[0]
				if first_arg.type == "string":
					try:
						name = content_bytes[first_arg.start_byte : first_arg.end_byte].decode("utf-8", errors="ignore")
						# Remove quotes
						return name.strip("\"'")
					except (UnicodeDecodeError, IndexError):
						pass

		return f"<anonymous-{node.type}>"

	def get_body_node(self, node: Node) -> Node | None:
		"""
		Get the node representing the 'body' of a definition.

		Args:
		    node: The tree-sitter node

		Returns:
		    The body node if available, None otherwise

		"""
		# Different fields based on node type
		if node.type in ["function_declaration", "method_definition", "class_declaration"]:
			return node.child_by_field_name("body")
		if node.type in ["arrow_function"]:
			body = node.child_by_field_name("body")
			# Arrow functions can have expression bodies or block bodies
			if body and body.type != "statement_block":
				return None  # Expression bodies don't have children to process
			return body
		if node.type == "program":
			return node  # Program itself is the body

		return None

	def get_children_to_process(self, node: Node, body_node: Node | None) -> list[Node]:
		"""
		Get the list of child nodes that should be recursively processed.

		Args:
		    node: The tree-sitter node
		    body_node: The body node if available

		Returns:
		    List of child nodes to process

		"""
		# Process children of the body node if it exists, otherwise process direct children
		if body_node:
			return list(body_node.children)

		# Special handling for certain nodes
		if node.type in ["variable_declaration", "lexical_declaration"]:
			# Process the declarations field
			declarations = node.child_by_field_name("declarations")
			return [declarations] if declarations else []

		return list(node.children)

	def should_skip_node(self, node: Node) -> bool:
		"""
		Determine if a node should be skipped entirely during processing.

		Args:
		    node: The tree-sitter node

		Returns:
		    True if the node should be skipped

		"""
		# Skip non-named nodes (like punctuation, operators)
		if not node.is_named:
			return True

		# Skip syntax nodes that don't contribute to code structure
		return node.type in ["(", ")", "{", "}", "[", "]", ";", ".", ",", ":", "=>"]

	def extract_imports(self, node: Node, content_bytes: bytes) -> list[str]:
		"""
		Extract imported module names from a JavaScript import statement.

		Args:
		    node: The tree-sitter node representing an import statement
		    content_bytes: Source code content as bytes

		Returns:
		    List of imported module names as strings

		"""
		if node.type not in self.config.import_:
			return []

		imported_names = []

		try:
			# Find the source (module path) of the import
			source_node = node.child_by_field_name("source")
			if not source_node:
				return []

			# Extract the module path from the string literal
			module_path = content_bytes[source_node.start_byte : source_node.end_byte].decode("utf-8", errors="ignore")
			# Remove quotes
			module_path = module_path.strip("\"'")

			# Check for different import patterns:

			# 1. Default import: "import Name from 'module'"
			default_import = node.child_by_field_name("default")
			if default_import:
				name = content_bytes[default_import.start_byte : default_import.end_byte].decode(
					"utf-8", errors="ignore"
				)
				imported_names.append(f"{module_path}.default")

			# 2. Named imports: "import { foo, bar as baz } from 'module'"
			named_imports = node.child_by_field_name("named_imports")
			if named_imports:
				for child in named_imports.children:
					if child.type == "import_specifier":
						imported_name = child.child_by_field_name("name")
						if imported_name:
							name = content_bytes[imported_name.start_byte : imported_name.end_byte].decode(
								"utf-8", errors="ignore"
							)
							imported_names.append(f"{module_path}.{name}")

			# 3. Namespace import: "import * as Name from 'module'"
			namespace_import = node.child_by_field_name("namespace_import")
			if namespace_import:
				imported_names.append(f"{module_path}.*")

			# If no specific imports found but we have a module, add the whole module
			if not imported_names and module_path:
				imported_names.append(module_path)

		except (UnicodeDecodeError, IndexError, AttributeError) as e:
			logger.warning("Failed to decode JavaScript imports: %s", e)

		return imported_names

	def extract_calls(self, node: Node, content_bytes: bytes) -> list[str]:
		"""
		Extract names of functions/methods called within a JS node's scope.

		Recursively searches for 'call_expression' nodes and extracts the function identifier.

		Args:
		    node: The tree-sitter node (e.g., function/method body)
		    content_bytes: Source code content as bytes

		Returns:
		    List of called function/method names

		"""
		calls = []
		for child in node.children:
			if child.type == "call_expression":
				function_node = child.child_by_field_name("function")
				if function_node:
					# Extract the identifier (e.g., funcName, obj.methodName)
					try:
						call_name = content_bytes[function_node.start_byte : function_node.end_byte].decode(
							"utf-8", errors="ignore"
						)
						calls.append(call_name)
					except UnicodeDecodeError:
						pass  # Ignore decoding errors
			# Recursively search deeper within non-call children
			else:
				calls.extend(self.extract_calls(child, content_bytes))
		return list(set(calls))  # Return unique calls
