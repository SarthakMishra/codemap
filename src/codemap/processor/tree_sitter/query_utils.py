"""Utility functions for working with Tree-sitter queries."""

import contextlib
import logging
from collections import defaultdict
from pathlib import Path

from tree_sitter import Language, Node, Query

from codemap.processor.tree_sitter.base import EntityType

logger = logging.getLogger(__name__)

# Map Tree-sitter tag captures to EntityType
TAG_TO_ENTITY_TYPE = {
	"definition.class": EntityType.CLASS,
	"definition.function": EntityType.FUNCTION,
	"definition.interface": EntityType.INTERFACE,
	"definition.method": EntityType.METHOD,
	"definition.module": EntityType.MODULE,
	"definition.struct": EntityType.STRUCT,
	"definition.enum": EntityType.ENUM,
	"definition.type_alias": EntityType.TYPE_ALIAS,
	"definition.variable": EntityType.VARIABLE,
	"definition.constant": EntityType.CONSTANT,
	"definition.class_field": EntityType.CLASS_FIELD,
	"definition.property": EntityType.PROPERTY,
	"definition.protocol": EntityType.PROTOCOL,
	"import": EntityType.IMPORT,
	"decorator": EntityType.DECORATOR,
	"comment": EntityType.COMMENT,
	"docstring": EntityType.DOCSTRING,
	"test_case": EntityType.TEST_CASE,
	"test_suite": EntityType.TEST_SUITE,
}


def load_tags_query(language: Language, language_name: str) -> Query:
	"""Load the tags.scm query for a given language."""
	scm_path = Path(__file__).parent / "scm" / f"{language_name}.scm"
	logger.debug("Attempting to load tags.scm for language %s from %s", language_name, scm_path)
	if not scm_path.exists():
		logger.warning("No tags.scm file found for language %s at %s", language_name, scm_path)
		msg = f"No tags.scm file for language {language_name} at {scm_path}"
		raise FileNotFoundError(msg)

	with scm_path.open("r", encoding="utf-8") as f:
		query_text = f.read()

	logger.debug("Successfully loaded tags.scm for %s (%d chars)", language_name, len(query_text))
	return Query(language, query_text)


def extract_entities_from_tree(language: Language, root_node: Node, language_name: str) -> list[dict[str, object]]:
	"""
	Run tags.scm query and yield dicts with EntityType, name, node, and docstring.

	Args:
		language: The tree-sitter Language object
		root_node: The root node of the parsed syntax tree
		language_name: The language name (for loading the correct tags.scm)

	Returns:
		List of dicts with keys: type (EntityType), name (str), node (Node), doc (str|None)
	"""
	try:
		logger.debug("Extracting entities for language %s", language_name)
		query = load_tags_query(language, language_name)
		if not query:
			return []

		raw_captures = query.captures(root_node)
		logger.debug("Initial query.captures() type: %s", type(raw_captures))

		captures = []  # This will be List[Tuple[Node, str]]
		if isinstance(raw_captures, dict):
			logger.debug("Raw captures is a dict. Normalizing to list of (Node, str) tuples.")
			for capture_name_key, nodes_list in raw_captures.items():
				if isinstance(nodes_list, list):
					for node_item in nodes_list:
						if isinstance(node_item, Node) and isinstance(capture_name_key, str):
							captures.append((node_item, capture_name_key))
						else:
							logger.warning(
								"Skipping item in dict-based captures: Node or key type mismatch. Node: %s, Key: %s",
								type(node_item),
								type(capture_name_key),
							)
				else:
					logger.warning(
						"Value for capture key '%s' in dict is not a list (type: %s). Skipping.",
						capture_name_key,
						type(nodes_list),
					)
		elif isinstance(raw_captures, list):
			logger.debug("Raw captures is a list. Validating and using as is.")
			for i, item in enumerate(raw_captures):
				if (
					isinstance(item, tuple)
					and len(item) == 2  # noqa: PLR2004
					and isinstance(item[0], Node)
					and isinstance(item[1], str)
				):
					captures.append(item)
				else:
					logger.warning(
						"Skipping item in list-based captures: Item %d not (Node, str) tuple. Type: %s, Repr: %s",
						i,
						type(item),
						repr(item)[:100],
					)
		else:
			logger.error("Unexpected type for raw_captures: %s. Cannot process captures.", type(raw_captures))
			return []  # Cannot proceed

		logger.debug("Normalized captures list contains %d items.", len(captures))

		entities = []
		node_to_name = {}
		node_to_doc = {}
		node_captures = defaultdict(list)
		for node, capture_name in captures:
			node_captures[node].append(capture_name)

		logger.debug("Processing captures to extract names and docs")
		for node, capture_name_list in node_captures.items():
			if "name" in capture_name_list and node.parent is not None and node.text is not None:
				try:
					name_text = node.text.decode("utf-8")
					node_to_name[node.parent] = name_text
					logger.debug(
						"Found name '%s' for node at line %d (parent: %s)",
						name_text,
						node.start_point[0] + 1,
						node.parent.type,
					)
				except (AttributeError, UnicodeDecodeError):
					logger.warning("Failed to decode name text for node %s", node)

			if "doc" in capture_name_list and node.parent is not None and node.text is not None:
				try:
					doc_text = node.text.decode("utf-8")
					node_to_doc[node.parent] = doc_text
					logger.debug(
						"Found docstring (%d chars) for node at line %d (parent: %s)",
						len(doc_text),
						node.start_point[0] + 1,
						node.parent.type,
					)
				except (AttributeError, UnicodeDecodeError):
					logger.warning("Failed to decode doc text for node %s", node)

		logger.debug("Processing captures to extract entities")
		processed_nodes = set()
		for node, capture_name in captures:
			if node in processed_nodes:
				continue

			if capture_name in {"name", "doc", "decorator"}:
				continue

			if capture_name in TAG_TO_ENTITY_TYPE:
				entity_type = TAG_TO_ENTITY_TYPE[capture_name]
				name = node_to_name.get(node)
				if not name:
					child_names = []
					for child in node.children:
						if child in node_captures and "name" in node_captures[child]:
							with contextlib.suppress(AttributeError, UnicodeDecodeError):
								child_name_text = child.text.decode("utf-8")
								child_names.append(child_name_text)

					if child_names:
						name = child_names[0]
						logger.debug("Found name '%s' via child @name search for %s", name, entity_type.name)

				if not name:
					with contextlib.suppress(AttributeError, UnicodeDecodeError):
						if node.text is not None:
							raw_text = node.text.decode("utf-8").strip()
							name = (
								raw_text.split()[0] if raw_text and (" " in raw_text or "\n" in raw_text) else raw_text
							)
							if name and len(name) > 70:  # noqa: PLR2004
								name = name[:67] + "..."
							logger.debug(
								"Used fallback name '%s' for entity type %s at line %d",
								name,
								entity_type.name,
								node.start_point[0] + 1,
							)

				doc = node_to_doc.get(node)
				decorators = []
				if node.parent and node.parent.type == "decorated_definition":
					decorated_def_node = node.parent
					for child_node in decorated_def_node.children:
						if child_node.type == "decorator":
							decorator_name = None
							for potential_name_node in child_node.children:
								if (
									potential_name_node in node_captures
									and "name" in node_captures[potential_name_node]
								):
									with contextlib.suppress(AttributeError, UnicodeDecodeError):
										decorator_name = potential_name_node.text.decode("utf-8")
										break
							if not decorator_name and child_node.text:
								with contextlib.suppress(AttributeError, UnicodeDecodeError):
									decorator_name = child_node.text.decode("utf-8").strip().lstrip("@")
							if decorator_name:
								decorators.append(decorator_name)
							processed_nodes.add(node)

				logger.debug(
					"Found entity: type=%s, name=%s, line=%d, has_doc=%s, decorators=%s",
					entity_type.name,
					name or "<unnamed>",
					node.start_point[0] + 1,
					"Yes" if doc else "No",
					decorators or "[]",
				)

				entities.append({"type": entity_type, "name": name, "node": node, "doc": doc, "decorators": decorators})

		logger.debug("Extracted %d entities from %s", len(entities), language_name)
		return entities
	except Exception:
		logger.exception("Error extracting entities from %s", language_name)
		return []
