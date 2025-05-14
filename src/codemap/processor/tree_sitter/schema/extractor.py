"""Extract entities from Tree-sitter."""

import logging
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import xxhash  # Add import for xxhash
from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from codemap.processor.hash_calculation import RepoChecksumCalculator
from codemap.utils.git_utils import GitError, GitRepoContext

from .entity import EntitySchema, LocationSchema, MetadataSchema, ScopeSchema
from .entity.types import EntityType, ScopeType
from .language_map import LANGUAGE_NODE_MAPPING
from .languages import LANGUAGES, NodeTypes, SupportedLanguages

logger = logging.getLogger(__name__)


def _find_repo_root(file_path: str) -> Path:
	"""Find the repository root directory for a given file path.

	Args:
		file_path: Path to a file

	Returns:
		Path to the repository root
	"""
	path = Path(file_path).resolve()

	# Go up directories until we find a .git directory or reach the filesystem root
	current_dir = path.parent if path.is_file() else path
	while current_dir != current_dir.parent:  # Stop at filesystem root
		if (current_dir / ".git").exists():
			return current_dir
		current_dir = current_dir.parent

	# If no .git directory found, use the file's directory as fallback
	return path.parent if path.is_file() else path


def _hash_string(text: str) -> str:
	"""Generate a hash for a string using xxhash.

	TEMPORARY SOLUTION: This function duplicates RepoChecksumCalculator._hash_string
	to avoid accessing private methods. It will be removed once the tree-sitter
	structure is fully integrated.

	Args:
		text: The text to hash

	Returns:
		A hexadecimal hash string
	"""
	hasher = xxhash.xxh3_128()
	hasher.update(text.encode("utf-8"))
	return hasher.hexdigest()


def _generate_entity_id(
	entity_type: str, name: str | None, file_path: str, location: LocationSchema, content: bytes
) -> str:
	"""Generate a stable, content-based hash ID for an entity using RepoChecksumCalculator.

	Args:
		entity_type: The entity type (e.g., FUNCTION, CLASS)
		name: The name of the entity, if available
		file_path: Path to the file containing the entity
		location: Location of the entity in the file
		content: The raw content of the entity

	Returns:
		An integer hash ID
	"""
	# Create an identifier string that uniquely identifies this entity
	identifier = (
		f"{entity_type}:{name or 'anonymous'}:{file_path}:"
		f"{location.start_line}:{location.start_col}:"
		f"{location.end_line}:{location.end_col}"
	)

	# Find the repository root
	repo_root = _find_repo_root(file_path)

	# Create a GitRepoContext for the repository
	git_context = None
	# Using suppress is more idiomatic than try-except-pass
	with suppress(GitError):
		git_context = GitRepoContext.get_instance()

	# Get a RepoChecksumCalculator instance
	RepoChecksumCalculator.get_instance(repo_root, git_context)

	# Use our local hashing function
	content_text = content.decode("utf-8", errors="ignore")
	content_hash = _hash_string(content_text)
	identifier += f":{content_hash}"

	# Generate final hash
	return _hash_string(identifier)


def _determine_scope_type(entity_type: str) -> ScopeType:
	"""Determine the scope type based on the entity type.

	Args:
		entity_type: The mapped entity type

	Returns:
		A ScopeType value
	"""
	# Map common entity types to scope types
	scope_type_map = {
		"FUNCTION": "FUNCTION",
		"METHOD": "METHOD",
		"CLASS": "CLASS",
		"MODULE": "GLOBAL",
		"BLOCK": "BLOCK",
		"LOOP": "LOOP",
		"IF": "BLOCK",
		"SWITCH": "SWITCH",
		"TRY_BLOCK": "TRY_BLOCK",
		"COMPREHENSION": "COMPREHENSION",
		"LAMBDA": "LAMBDA",
		"PATTERN": "PATTERN",
		"TEMPLATE": "TEMPLATE",
	}

	# Try to map based on entity type first
	for type_prefix, scope_type in scope_type_map.items():
		if entity_type.startswith(type_prefix):
			return cast("ScopeType", scope_type)

	# Default to BLOCK if we can't determine
	return "BLOCK"


def _is_true_scope_creator(node_type: str, entity_type: str) -> bool:
	"""Determine if a node actually creates a new lexical scope.

	This is a more restrictive version that only returns True for nodes
	that truly create a new lexical scope in most programming languages.

	Args:
		node_type: The type of the tree-sitter node
		entity_type: The mapped entity type

	Returns:
		True if the node creates a new lexical scope
	"""
	# Skip trivial nodes like punctuation
	if node_type in ["(", ")", "{", "}", "[", "]", ";", ":", ",", "."]:
		return False

	# Skip string literals and components
	if "string" in node_type.lower() or node_type in ["string_start", "string_content", "string_end"]:
		return False

	# Skip identifiers and simple expressions
	if node_type in ["identifier", "variable", "name", "expression"]:
		return False

	# Primary scope creators - these almost always create a lexical scope
	if entity_type in ["CLASS", "FUNCTION", "METHOD", "MODULE"]:
		return True

	# Control flow constructs that often create scopes
	if entity_type in ["BLOCK", "LOOP", "IF", "SWITCH", "TRY_BLOCK", "COMPREHENSION", "LAMBDA"]:
		# For these types, check the node_type to avoid creating scopes for every node
		# within these constructs
		node_type_lower = node_type.lower()

		# Block-level constructs
		if "block" in node_type_lower:
			return True

		# Function-like
		if (
			"function" in node_type_lower
			or "method" in node_type_lower
			or "lambda" in node_type_lower
			or "arrow_function" in node_type_lower
		):
			return True

		# Class-like
		if "class" in node_type_lower:
			return True

		# Control flow blocks
		if node_type_lower in [
			"for_statement",
			"while_statement",
			"do_statement",
			"if_statement",
			"else_clause",
			"switch_statement",
			"try_statement",
			"catch_clause",
			"finally_clause",
		]:
			return True

	# Specialized constructs - these are important scope-creating nodes
	specialized_node_types = [
		"module",
		"namespace",
		"program",
		"file",
		"compilation_unit",
		"class_declaration",
		"interface_declaration",
		"enum_declaration",
		"function_definition",
		"method_definition",
		"constructor_declaration",
		"for_statement",
		"while_statement",
		"if_statement",
		"switch_statement",
		"try_statement",
		"catch_clause",
		"with_statement",
	]

	return any(pattern in node_type.lower() for pattern in specialized_node_types)


def extract_entities(source: bytes, file_path: str) -> list[EntitySchema]:
	"""Parses source code and extracts entities using Tree-sitter.

	Uses tree_sitter_language_pack to parse the source code based on the
	file extension, maps the Abstract Syntax Tree (AST) nodes to Pydantic
	schemas defined in `.entity_schema`, and returns a flat list of
	these entities.

	Args:
		source: The source code content as bytes.
		file_path: The path to the file being parsed. Used to determine
			the language from the file extension.

	Returns:
		A flat list of extracted entities, represented by Pydantic schemas.

	Raises:
		ValueError: If the file extension from `file_path` is not supported
			or mapped in `EXT_TO_LANG`.
	"""
	ext = Path(file_path).suffix.lstrip(".").lower()
	lang_info = next((lang_def for lang_def in LANGUAGES if ext in lang_def["extensions"]), None)
	if not lang_info:
		msg = f"Unsupported extension '.{ext}'"
		raise ValueError(msg)

	lang: SupportedLanguages = lang_info["name"]

	parser = get_parser(lang)
	tree = parser.parse(source)

	entities: list[EntitySchema] = []
	# Track scopes with their IDs for building the hierarchy
	scope_map: dict[str, ScopeSchema] = {}

	# The current scope stack tracks (node, scope_id) tuples
	scope_stack: list[tuple[Node, str]] = []

	# Root scope for the module
	module_node = tree.root_node
	module_name = Path(file_path).stem

	sl, sc = module_node.start_point
	el, ec = module_node.end_point

	module_loc = LocationSchema(
		start_line=sl + 1,
		start_col=sc + 1,
		end_line=el + 1,
		end_col=ec + 1,
	)

	module_meta = MetadataSchema(
		file_path=file_path,
		language=lang,
		node_kind="module",
	)

	# Generate a global scope for the module
	module_id = _generate_entity_id("MODULE", module_name, file_path, module_loc, source)

	root_scope = ScopeSchema(
		scope_id=module_id,
		parent_scope_id=None,  # No parent for the root scope
		scope_type="GLOBAL",
		declarations=[],
		imports=[],
	)

	scope_map[module_id] = root_scope

	# Module entity
	module_entity = EntitySchema(
		id=module_id,
		type="MODULE",
		name=module_name,
		parent_id=None,
		children=[],
		location=module_loc,
		metadata=module_meta,
		scope=root_scope,
	)

	entities.append(module_entity)

	def visit(node: Node, parent_scope_id: str) -> EntitySchema | None:
		tag = node.type

		language_specific_mapping = LANGUAGE_NODE_MAPPING.get(lang)
		if language_specific_mapping is None:
			msg = f"Language '{lang}' not found in LANGUAGE_NODE_MAPPING."
			raise ValueError(msg)

		tag_for_lookup = cast("NodeTypes", tag)
		etype_str = language_specific_mapping.get(tag_for_lookup)
		if not etype_str:
			# Skip unmapped nodes
			for child_node in node.children:
				visit(child_node, parent_scope_id)
			return None

		etype = cast("EntityType", etype_str)

		# Extract name if available
		name_node = node.child_by_field_name("name") or node.child_by_field_name("identifier")
		name = None
		if name_node:
			raw = source[name_node.start_byte : name_node.end_byte]
			name = raw.decode("utf-8", errors="ignore")

		# Create location
		sl, sc = node.start_point
		el, ec = node.end_point

		loc = LocationSchema(
			start_line=sl + 1,
			start_col=sc + 1,
			end_line=el + 1,
			end_col=ec + 1,
		)

		meta = MetadataSchema(
			file_path=file_path,
			language=lang,
			node_kind=node.type,
		)

		# Extract node content for ID generation
		node_content = source[node.start_byte : node.end_byte]

		# Generate a stable entity ID
		entity_id = _generate_entity_id(etype, name, file_path, loc, node_content)

		# Determine if this node creates a new scope - using more restrictive check
		is_scope_creator = _is_true_scope_creator(node.type, etype)

		# Create scope for this entity if it's a scope creator
		scope_id = parent_scope_id
		scope_type = "BLOCK"  # Default

		if is_scope_creator:
			# Use entity type to determine scope type
			scope_type = _determine_scope_type(etype)

			scope_id = entity_id  # Use entity ID as scope ID for scope creators

			# Create a new scope
			scope = ScopeSchema(
				scope_id=scope_id,
				parent_scope_id=parent_scope_id,
				scope_type=scope_type,
				declarations=[],
				imports=[],
			)

			scope_map[scope_id] = scope

			# Update parent scope's declarations
			if parent_scope_id in scope_map:
				scope_map[parent_scope_id].declarations.append(entity_id)
		else:
			# Non-scope creators inherit parent scope
			scope_id = parent_scope_id

			# Still register this entity in the parent's declarations
			if parent_scope_id in scope_map:
				scope_map[parent_scope_id].declarations.append(entity_id)

		# Get the scope (either new or parent's)
		scope = scope_map[scope_id]

		# Check for imports
		if node.type in ["import_statement", "import_declaration", "use_declaration"]:
			# Extract import information
			import_name = None
			for child in node.children:
				if child.type in ["string", "identifier", "dotted_name"]:
					import_bytes = source[child.start_byte : child.end_byte]
					import_name = import_bytes.decode("utf-8", errors="ignore")
					break

			if import_name and scope_id in scope_map:
				scope_map[scope_id].imports.append(import_name)

		# Create the entity schema
		data: dict[str, Any] = {
			"id": entity_id,
			"type": etype,
			"name": name,
			"parent_id": None,  # Will be set when added to parent's children
			"scope": scope,
			"location": loc,
			"metadata": meta,
		}

		entity_schema_fields = EntitySchema.model_fields
		if "children" in entity_schema_fields.keys():  # noqa: SIM118
			data["children"] = []

		entity = EntitySchema(**data)
		entities.append(entity)

		# Update scope stack for nested processing
		if is_scope_creator:
			scope_stack.append((node, scope_id))

		# Process children
		supports_children = "children" in entity_schema_fields.keys()  # noqa: SIM118
		if supports_children and hasattr(entity, "children") and isinstance(entity.children, list):
			for child_node in node.children:
				child_ent = visit(child_node, scope_id)
				if child_ent:
					child_ent.parent_id = entity_id
					entity.children.append(child_ent)
		else:
			for child_node in node.children:
				visit(child_node, scope_id)

		# Pop from scope stack when done with this scope
		if is_scope_creator:
			scope_stack.pop()

		return entity

	# Visit all children of the root node with the module scope as parent
	for child_node in tree.root_node.children:
		child_entity = visit(child_node, module_id)
		if child_entity:
			child_entity.parent_id = module_id
			module_entity.children.append(child_entity)

	return entities
