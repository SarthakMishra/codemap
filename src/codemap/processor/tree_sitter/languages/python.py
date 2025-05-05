"""Python-specific configuration for syntax chunking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codemap.processor.tree_sitter.base import EntityType
from codemap.processor.tree_sitter.languages.base import LanguageConfig, LanguageSyntaxHandler

if TYPE_CHECKING:
	from tree_sitter import Node

logger = logging.getLogger(__name__)


class PythonConfig(LanguageConfig):
	"""Python-specific syntax chunking configuration."""

	# File-level entities
	module: ClassVar[list[str]] = ["module"]
	namespace: ClassVar[list[str]] = []  # Python doesn't have explicit namespaces

	# Type definitions
	class_: ClassVar[list[str]] = ["class_definition"]
	interface: ClassVar[list[str]] = ["class_definition"]  # Python uses ABC classes
	protocol: ClassVar[list[str]] = ["class_definition"]  # Protocol classes
	struct: ClassVar[list[str]] = ["class_definition"]  # Python uses regular classes
	enum: ClassVar[list[str]] = ["class_definition"]  # Enum classes
	type_alias: ClassVar[list[str]] = ["assignment"]  # Type assignments

	# Functions and methods
	function: ClassVar[list[str]] = ["function_definition"]
	method: ClassVar[list[str]] = ["function_definition"]  # Inside class
	property_def: ClassVar[list[str]] = ["decorated_definition"]  # @property decorated functions
	test_case: ClassVar[list[str]] = ["function_definition"]  # test_* functions
	test_suite: ClassVar[list[str]] = ["class_definition"]  # Test* classes

	# Variables and constants
	variable: ClassVar[list[str]] = ["assignment"]
	constant: ClassVar[list[str]] = ["assignment"]  # Uppercase assignments
	class_field: ClassVar[list[str]] = ["class_variable_definition"]

	# Code organization
	import_: ClassVar[list[str]] = ["import_statement", "import_from_statement"]
	decorator: ClassVar[list[str]] = ["decorator"]

	# Documentation
	comment: ClassVar[list[str]] = ["comment"]
	docstring: ClassVar[list[str]] = ["string"]  # First string in module/class/function

	file_extensions: ClassVar[list[str]] = [".py", ".pyi"]
	tree_sitter_name: ClassVar[str] = "python"


PYTHON_CONFIG = PythonConfig()


class PythonSyntaxHandler(LanguageSyntaxHandler):
	"""Python-specific syntax handling logic."""

	@staticmethod
	def _get_node_text(node: Node, content_bytes: bytes) -> str:
		"""Helper to get node text safely."""
		try:
			return content_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
		except IndexError:
			return ""

	def __init__(self) -> None:
		"""Initialize with Python configuration."""
		super().__init__(PYTHON_CONFIG)

	def get_entity_type(self, node: Node, parent: Node | None, content_bytes: bytes) -> EntityType:
		"""
		Determine the EntityType for a Python node.

		Args:
		    node: The tree-sitter node
		    parent: The parent node (if any)
		    content_bytes: Source code content as bytes

		Returns:
		    The entity type

		"""
		node_type = node.type
		logger.debug(
			"Getting entity type for Python node: type=%s, parent_type=%s", node_type, parent.type if parent else None
		)

		# Print node content for debugging
		try:
			node_content = self._get_node_text(node, content_bytes)
			logger.debug("Node content: %s", node_content)
		except (UnicodeDecodeError, IndexError) as e:
			logger.debug("Failed to decode node content: %s", str(e))

		# Special case: if this is an expression statement containing a constant assignment
		if node_type == "expression_statement":
			# Check if it contains an assignment that is a constant
			for child in node.children:
				if child.type == "assignment":
					name_node = child.child_by_field_name("left")
					if name_node:
						name = self._get_node_text(name_node, content_bytes)

						# Get the right side for type detection
						value_node = child.child_by_field_name("right")
						value_text = ""
						if value_node:
							try:
								value_text = self._get_node_text(value_node, content_bytes)
							except (UnicodeDecodeError, IndexError) as e:
								logger.debug("Failed to decode type value: %s", str(e))

						# Check for type alias - TypeVar or anything referencing typing types like Dict, List, etc.
						if "TypeVar" in value_text or any(
							typing_type in value_text
							for typing_type in ["Dict", "List", "Tuple", "Set", "Union", "Optional", "Callable", "Any"]
						):
							logger.debug("Expression statement with TYPE_ALIAS: %s", name)
							return EntityType.TYPE_ALIAS

						# Check for constant (all uppercase with at least one letter)
						if name.isupper() and any(c.isalpha() for c in name):
							logger.debug("Expression statement with CONSTANT assignment: %s", name)
							return EntityType.CONSTANT
						# Check for regular variable
						if not name.startswith("_") and any(c.isalpha() for c in name):
							logger.debug("Expression statement with VARIABLE assignment: %s", name)
							return EntityType.VARIABLE

		# Module-level
		if node_type in self.config.module:
			return EntityType.MODULE
		if node_type in self.config.namespace:
			return EntityType.NAMESPACE

		# Documentation
		if node_type in self.config.docstring:
			# Check if this is a docstring (first string in a container)
			if self._is_docstring(node, parent):
				return EntityType.DOCSTRING
			return EntityType.UNKNOWN  # Regular string literals
		if node_type in self.config.comment:
			return EntityType.COMMENT

		# Type definitions
		if node_type in self.config.class_:
			return EntityType.CLASS
		if node_type in self.config.interface:
			# Would need to check for ABC inheritance to be precise
			return EntityType.INTERFACE
		if node_type in self.config.protocol:
			# Would need to check for Protocol inheritance to be precise
			return EntityType.PROTOCOL
		if node_type in self.config.type_alias:
			# For assignments, check if it's a constant (all uppercase) first
			if node_type == "assignment":
				name_node = node.child_by_field_name("left")
				if name_node:
					name = self._get_node_text(name_node, content_bytes)
					logger.debug("Checking potential constant in type_alias: %s (is_upper: %s)", name, name.isupper())
					# Improved check for constants: name is uppercase and contains at least one letter
					if name.isupper() and any(c.isalpha() for c in name):
						logger.debug("Identified as CONSTANT: %s", name)
						return EntityType.CONSTANT

			# Otherwise, treat as a type alias
			return EntityType.TYPE_ALIAS

		# Functions and methods
		if node_type in self.config.function:
			# Check if this is a test function
			name = self.extract_name(node, content_bytes)
			if name.startswith("test_"):
				return EntityType.TEST_CASE

			# Check if this is a method by looking for class ancestry
			if self._is_within_class_context(node):
				return EntityType.METHOD
			return EntityType.FUNCTION

		# Check for properties - decorated definitions
		if node_type in self.config.property_def:
			for child in node.children:
				if child.type == "decorator":
					decorator_text = self._get_node_text(child, content_bytes)
					if "@property" in decorator_text:
						return EntityType.PROPERTY
			# If no @property decorator, treat as method if in class, otherwise function
			if self._is_within_class_context(node):
				return EntityType.METHOD
			return EntityType.FUNCTION

		# Variables and constants
		if node_type in self.config.variable:
			# Check if it looks like a constant (uppercase name)
			name_node = node.child_by_field_name("left")
			if name_node:
				name = self._get_node_text(name_node, content_bytes)
				logger.debug("Checking potential constant: %s (is_upper: %s)", name, name.isupper())
				# Improved check for constants: name is uppercase and contains at least one letter
				if name.isupper() and any(c.isalpha() for c in name):
					logger.debug("Identified as CONSTANT: %s", name)
					return EntityType.CONSTANT
			logger.debug("Identified as VARIABLE: node_type=%s", node_type)
			return EntityType.VARIABLE

		# Class fields
		if node_type in self.config.class_field:
			return EntityType.CLASS_FIELD

		# Code organization
		if node_type in self.config.import_:
			return EntityType.IMPORT
		if node_type in self.config.decorator:
			return EntityType.DECORATOR

		return EntityType.UNKNOWN

	def _is_within_class_context(self, node: Node) -> bool:
		"""
		Check if the node is defined within a class definition.

		Args:
		    node: The tree-sitter node

		Returns:
		    True if the node is within a class context

		"""
		ancestor = node.parent
		while ancestor:
			if ancestor.type in self.config.class_:
				return True
			# Stop search if we hit module or another function definition
			if ancestor.type in self.config.module or ancestor.type in self.config.function:
				break
			ancestor = ancestor.parent
		return False

	def _is_docstring(self, node: Node, parent: Node | None) -> bool:
		"""
		Check if a string node is a docstring.

		Args:
		    node: The string node
		    parent: The parent node

		Returns:
		    True if the node is a docstring

		"""
		if not parent:
			return False

		# For module docstrings, check if it's the first string in the module
		if parent.type == "module":
			non_comment_children = [c for c in parent.children if c.type not in ["comment"]]
			return bool(non_comment_children and node == non_comment_children[0])

		# For expression statements containing string literals
		if parent.type == "expression_statement":
			# Check if this is the first child of a function or class body
			grandparent = parent.parent
			if grandparent and grandparent.type == "block":
				great_grandparent = grandparent.parent
				if great_grandparent and great_grandparent.type in (self.config.function + self.config.class_):
					# Check if it's the first item in the block
					non_comment_children = [c for c in grandparent.children if c.type not in ["comment"]]
					return bool(non_comment_children and parent == non_comment_children[0])
			return False

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
		body_node = self.get_body_node(node)
		if not body_node:
			# Handle module docstring case (no explicit body node)
			if node.type == "module":
				body_node = node  # Treat module itself as the body context
			else:
				return None, None

		if body_node.named_child_count == 0:
			return None, None

		# Look for the first child that might be a docstring
		first_body_child = None
		for child in body_node.children:
			if child.is_named:
				first_body_child = child
				break

		if not first_body_child:
			return None, None

		actual_string_node = None
		docstring_container_node = None  # The node to skip during processing

		if first_body_child.type == "expression_statement":
			# For expression statements containing string literals
			for child in first_body_child.children:
				if child.type in self.config.docstring:
					actual_string_node = child
					docstring_container_node = first_body_child
					break
		elif first_body_child.type in self.config.docstring:
			# Direct string literal
			actual_string_node = first_body_child
			docstring_container_node = first_body_child

		if actual_string_node:
			try:
				docstring_text = self._get_node_text(actual_string_node, content_bytes).strip("\"' \n")
				return docstring_text, docstring_container_node
			except (UnicodeDecodeError, IndexError, AttributeError) as e:
				logger.warning("Failed to decode/extract Python docstring: %s", e)

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
		# Try to find the name field
		name_node = node.child_by_field_name("name")

		# Handle assignments
		if not name_node and node.type == "assignment":
			name_node = node.child_by_field_name("left")

		# Handle expression statements with assignments
		if not name_node and node.type == "expression_statement":
			for child in node.children:
				if child.type == "assignment":
					name_node = child.child_by_field_name("left")
					if name_node:
						break

		# Handle decorated definitions
		if not name_node and node.type == "decorated_definition":
			func_def = node.child_by_field_name("definition")
			if func_def:
				name_node = func_def.child_by_field_name("name")

		if name_node:
			try:
				return self._get_node_text(name_node, content_bytes)
			except (UnicodeDecodeError, IndexError, AttributeError) as e:
				logger.warning("Failed to decode Python name: %s", e)
				return f"<decoding-error-{node.type}>"

		return f"<anonymous-{node.type}>"

	def get_body_node(self, node: Node) -> Node | None:
		"""Get the block node for function/class definition body."""
		if node.type in ("function_definition", "class_definition", "decorated_definition"):
			# Handle decorated definitions properly
			actual_def_node = node
			if node.type == "decorated_definition":
				# Find the actual function/class definition node within the decoration
				for child in node.children:
					if child.type in ("function_definition", "class_definition"):
						actual_def_node = child
						break
				else:
					return None  # Could not find definition within decorator

			# Find the 'block' node which contains the body statements
			for child in actual_def_node.children:
				if child.type == "block":
					return child
		return None  # Not a function/class definition or no block found

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
		return list(body_node.children) if body_node else list(node.children)

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
		return node.type in ["(", ")", "{", "}", "[", "]", ";", ".", ","]

	def extract_imports(self, node: Node, content_bytes: bytes) -> list[str]:
		"""
		Extract imported module names from a Python import statement.

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
			# Handle regular import statements: "import foo, bar"
			if node.type == "import_statement":
				for child in node.children:
					if child.type == "dotted_name":
						module_name = self._get_node_text(child, content_bytes)
						imported_names.append(module_name)

			# Handle import from statements: "from foo.bar import baz, qux"
			elif node.type == "import_from_statement":
				# Get the module being imported from
				module_node = None
				for child in node.children:
					if child.type == "dotted_name":
						module_node = child
						break

				if module_node:
					module_name = self._get_node_text(module_node, content_bytes)

					# Get the imported names
					import_node = node.child_by_field_name("import")
					if import_node:
						# Check for the wildcard import case: "from foo import *"
						for child in import_node.children:
							if child.type == "wildcard_import":
								imported_names.append(f"{module_name}.*")
								return imported_names

						# Regular named imports
						for child in import_node.children:
							if child.type == "import_list":
								for item in child.children:
									if item.type in {"dotted_name", "identifier"}:
										name = self._get_node_text(item, content_bytes)
										imported_names.append(f"{module_name}.{name}")
		except (UnicodeDecodeError, IndexError, AttributeError) as e:
			logger.warning("Failed to decode Python imports: %s", e)

		return imported_names

	def extract_calls(self, node: Node, content_bytes: bytes) -> list[str]:
		"""
		Extract names of functions/methods called within a Python node's scope.

		Recursively searches for 'call' nodes and extracts the function identifier.

		Args:
		    node: The tree-sitter node (e.g., function/method body)
		    content_bytes: Source code content as bytes

		Returns:
		    List of called function/method names

		"""
		calls = []
		for child in node.children:
			if child.type == "call":
				function_node = child.child_by_field_name("function")
				if function_node:
					# Extract the identifier (could be simple name or attribute access like obj.method)
					# For simplicity, we take the full text of the function node
					try:
						call_name = self._get_node_text(function_node, content_bytes)
						calls.append(call_name)
					except UnicodeDecodeError:
						pass  # Ignore decoding errors
			# Recursively search within the arguments or children of the call if needed, but often not necessary
			# for call details, just the name.
			# Else, recursively search deeper within non-call children
			else:
				calls.extend(self.extract_calls(child, content_bytes))
		return list(set(calls))  # Return unique calls

	def extract_signature(self, node: Node, content_bytes: bytes) -> str | None:
		"""Extract the signature up to the colon ':' for Python functions/classes."""
		signature_node = node
		# If it's a decorated definition, find the actual definition node for the signature start
		if node.type == "decorated_definition":
			for child in node.children:
				if child.type in ("function_definition", "class_definition"):
					signature_node = child
					break
			else:
				return self._get_node_text(node, content_bytes).splitlines()[0]  # Fallback to first line of decorator

		# Find the colon that ends the signature part
		colon_node = None
		for child in signature_node.children:
			if child.type == ":":
				colon_node = child
				break
			# Handle async functions where 'def' is preceded by 'async'
			if child.type == "async":
				continue  # skip 'async' keyword itself
			if child.type in {"def", "class"}:
				continue  # skip 'def'/'class' keywords
			# Stop if we hit the body block before finding a colon (shouldn't happen in valid code)
			if child.type == "block":
				break

		if colon_node:
			# Extract text from the start of the definition node up to the end of the colon
			start_byte = signature_node.start_byte
			end_byte = colon_node.end_byte
			try:
				return content_bytes[start_byte:end_byte].decode("utf-8", errors="ignore").strip()
			except IndexError:
				return None
		else:
			# Fallback: if no colon found (e.g., malformed code?), return the first line
			return self._get_node_text(signature_node, content_bytes).splitlines()[0]

	def get_enclosing_node_of_type(self, node: Node, target_type: EntityType) -> Node | None:
		"""Find the first ancestor node matching the target Python entity type."""
		target_node_types = []
		if target_type == EntityType.CLASS:
			target_node_types = ["class_definition", "decorated_definition"]  # Include decorated
		elif target_type == EntityType.FUNCTION:
			target_node_types = ["function_definition", "decorated_definition"]  # Include decorated
		elif target_type == EntityType.MODULE:
			# Module is typically the root node or identified by file, not easily findable as ancestor type
			return None  # Or return root node? Depends on desired behavior.
		# Add other types if needed

		if not target_node_types:
			return None

		current = node.parent
		while current:
			# Check if the current node is the target type or a decorator containing it
			node_to_check = current
			actual_node_type = current.type

			if current.type == "decorated_definition":
				# Check the *content* of the decorated definition
				found_target_in_decorator = False
				for child in current.children:
					if child.type in target_node_types and child.type != "decorated_definition":
						# We found the actual class/func def inside the decorator
						node_to_check = child
						actual_node_type = child.type
						found_target_in_decorator = True
						break
				if not found_target_in_decorator:
					actual_node_type = "decorated_definition"  # Treat decorator itself if no target found within

			# Now check if the node (or the one found inside decorator) matches
			if actual_node_type in target_node_types and actual_node_type != "decorated_definition":
				return node_to_check  # Return the actual definition node

			current = current.parent
		return None

	def _find_decorated_definition(self, node: Node) -> Node | None:
		"""Helper to get the actual definition node from a decorated_definition."""
		if node.type == "decorated_definition":
			for child in node.children:
				if child.type in ("function_definition", "class_definition"):
					return child
		return None
