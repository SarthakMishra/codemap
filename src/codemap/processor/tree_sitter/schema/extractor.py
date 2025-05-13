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
from .entity.types import EntityType, ScopeType, VisibilityModifier
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
) -> int:
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
	hash_hex = _hash_string(identifier)

	# Convert hex hash to integer (taking first 8 chars for reasonable integer size)
	return int(hash_hex[:8], 16) % (2**31)


def _determine_scope_type(node: Node, entity_type: str) -> ScopeType:
	"""Determine the scope type based on the node and entity type.

	Args:
		node: The tree-sitter Node
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

	# If we can't determine from entity type, check node type
	node_type = node.type.lower()

	# Common node type patterns
	if "function" in node_type or "method" in node_type:
		return "FUNCTION"
	if "class" in node_type:
		return "CLASS"
	if "block" in node_type:
		return "BLOCK"
	if "for" in node_type or "while" in node_type:
		return "LOOP"
	if "if" in node_type or "else" in node_type:
		return "BLOCK"
	if "switch" in node_type or "case" in node_type:
		return "SWITCH"
	if "try" in node_type or "catch" in node_type or "finally" in node_type:
		return "TRY_BLOCK"
	if "lambda" in node_type or "arrow_function" in node_type:
		return "LAMBDA"

	# Default to BLOCK if we can't determine
	return "BLOCK"


def _determine_visibility(node: Node, source: bytes, lang: str) -> VisibilityModifier:
	"""Determine the visibility modifier for an entity.

	Args:
		node: The tree-sitter Node
		source: The source code as bytes
		lang: The programming language

	Returns:
		A VisibilityModifier value
	"""
	# Check for explicit visibility modifiers in node or its children
	for child in [node, *node.children]:
		if child.type in ["public", "private", "protected", "internal"]:
			return cast("VisibilityModifier", child.type.upper())

	# Check based on language-specific patterns
	if lang == "python":
		# Python convention: leading underscore for "protected", double for "private"
		name_node = node.child_by_field_name("name") or node.child_by_field_name("identifier")
		if name_node:
			name = source[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")
			if name.startswith("__") and not name.endswith("__"):
				return "PRIVATE"
			if name.startswith("_"):
				return "PROTECTED"

	elif lang in ["java", "csharp", "typescript"]:
		# Look for modifiers in nearby nodes
		modifier_node = node.prev_sibling
		if modifier_node and modifier_node.type in ["public", "private", "protected", "internal"]:
			return cast("VisibilityModifier", modifier_node.type.upper())

	# Default to PUBLIC for top-level entities, PRIVATE for others
	parent = node.parent
	if not parent or parent.type in ["source_file", "program", "module"]:
		return "PUBLIC"

	return "PRIVATE"


def _extract_namespace(node: Node, source: bytes, lang: str, file_path: str) -> str | None:
	"""Extract namespace information for the entity.

	Args:
		node: The tree-sitter Node
		source: The source code as bytes
		lang: The programming language
		file_path: Path to the file

	Returns:
		A namespace string or None
	"""
	# Language-specific namespace extraction
	if lang == "python":
		# Use package structure for Python
		parts = Path(file_path).parts
		if "src" in parts:
			pkg_parts = parts[parts.index("src") + 1 :]
			return ".".join(pkg_parts)
	elif lang == "java":
		# Try to find package declaration
		root = node
		while root.parent:
			root = root.parent

		for child in root.children:
			if child.type == "package_declaration":
				pkg_node = child.child_by_field_name("name")
				if pkg_node:
					return source[pkg_node.start_byte : pkg_node.end_byte].decode("utf-8", errors="ignore")
	elif lang in ["javascript", "typescript"]:
		# For JS/TS, could use the file path relative to project root
		return Path(file_path).stem

	return None


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
	scope_map: dict[int, ScopeSchema] = {}

	# The current scope stack (node, scope_id) tuples
	scope_stack: list[tuple[Node, int]] = []

	# Track declarations for populating scope.declarations
	declarations_by_scope_id: dict[int, list[int]] = {}

	# Track imports for populating scope.imports
	imports_by_scope_id: dict[int, list[str]] = {}

	entity_schema_fields = EntitySchema.model_fields

	def visit(node: Node, parent_scope_id: int | None = None) -> EntitySchema | None:
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

		name_node = node.child_by_field_name("name") or node.child_by_field_name("identifier")
		name = None
		if name_node:
			raw = source[name_node.start_byte : name_node.end_byte]
			name = raw.decode("utf-8", errors="ignore")

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

		# Determine if this node creates a new scope
		is_scope_creator = etype in [
			"CLASS",
			"FUNCTION",
			"METHOD",
			"MODULE",
			"BLOCK",
			"LOOP",
			"IF",
			"SWITCH",
			"TRY_BLOCK",
			"COMPREHENSION",
			"LAMBDA",
		] or any(
			scope_type in etype
			for scope_type in ["CLASS", "FUNCTION", "METHOD", "MODULE", "BLOCK", "LOOP", "SCOPE", "PATTERN"]
		)

		# Create scope for this entity
		scope_type = _determine_scope_type(node, etype)
		scope_id = entity_id  # Use the entity ID as the scope ID for scope creators

		if not is_scope_creator and parent_scope_id is not None:
			# Non-scope creators inherit parent scope ID
			scope_id = parent_scope_id

		visibility = _determine_visibility(node, source, lang)
		namespace = _extract_namespace(node, source, lang, file_path)

		# Initialize declarations and imports lists if this is a new scope
		if is_scope_creator:
			declarations_by_scope_id[scope_id] = []
			imports_by_scope_id[scope_id] = []

		# Add this entity's ID to parent scope's declarations
		if parent_scope_id is not None and parent_scope_id in declarations_by_scope_id:
			declarations_by_scope_id[parent_scope_id].append(entity_id)

		# Check for imports
		if node.type in ["import_statement", "import_declaration", "use_declaration"]:
			# Extract import information
			import_name = None
			for child in node.children:
				if child.type in ["string", "identifier", "dotted_name"]:
					import_bytes = source[child.start_byte : child.end_byte]
					import_name = import_bytes.decode("utf-8", errors="ignore")
					break

			if import_name and parent_scope_id is not None and parent_scope_id in imports_by_scope_id:
				imports_by_scope_id[parent_scope_id].append(import_name)

		# Create the scope schema
		scope = ScopeSchema(
			scope_id=scope_id,
			parent_scope_id=parent_scope_id,
			scope_type=scope_type,
			visibility=visibility,
			namespace=namespace,
			# Temporal attributes could be enhanced with language-specific logic
			temporal_attributes={
				"lifetime": "automatic",  # Default to automatic memory management
				"hoisting": "none",  # Default to no hoisting
			},
			# Declarations and imports will be populated after all nodes are processed
			declarations=[],
			imports=[],
			location=loc,
			# Language-specific metadata could be enhanced
			language_metadata={},
			metadata=meta,
		)

		# Add scope to map for lookup
		scope_map[scope_id] = scope

		# Create the entity schema
		data: dict[str, Any] = {
			"id": entity_id,
			"type": etype,
			"name": name,
			"scope": scope,
		}

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
				child_ent = visit(child_node, scope_id if is_scope_creator else parent_scope_id)
				if child_ent:
					entity.children.append(child_ent)
		else:
			for child_node in node.children:
				visit(child_node, scope_id if is_scope_creator else parent_scope_id)

		# Pop from scope stack when done with this scope
		if is_scope_creator:
			scope_stack.pop()

		return entity

	# Start with the root node and no parent scope
	visit(tree.root_node)

	# Populate declarations and imports in all scopes
	for entity in entities:
		scope_id = entity.scope.scope_id
		if scope_id in declarations_by_scope_id:
			entity.scope.declarations = declarations_by_scope_id[scope_id]
		if scope_id in imports_by_scope_id:
			entity.scope.imports = imports_by_scope_id[scope_id]

	return entities
