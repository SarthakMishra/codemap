"""Extract entities from Tree-sitter."""

from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from .entity.base import AnyEntitySchema
from .entity.scope import LocationSchema, MetadataSchema
from .language_map import LANGUAGE_NODE_MAPPING
from .languages import LANGUAGES, SupportedLanguages


def extract_entities(source: bytes, file_path: str) -> list[AnyEntitySchema]:
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
	lang: SupportedLanguages | None = next((lang["name"] for lang in LANGUAGES if ext in lang["extensions"]), None)
	if not lang:
		msg = f"Unsupported extension '.{ext}'"
		raise ValueError(msg)

	parser = get_parser(lang)
	tree = parser.parse(source)

	entities: list[AnyEntitySchema] = []
	id_counter = 1

	def visit(node: Node, parent_id: int | None = None) -> AnyEntitySchema | None:
		nonlocal id_counter

		tag = node.type

		if tag not in LANGUAGE_NODE_MAPPING.get(lang, {}):
			msg = f"No mapping found for node type '{tag}'"
			raise ValueError(msg)

		# 1) exact mapping or error
		mapping = LANGUAGE_NODE_MAPPING.get(lang, {}).get(tag)
		if not mapping:
			msg = f"No mapping found for node type '{node.type}'"
			raise ValueError(msg)
		etype, schema_cls = mapping

		# 2) extract name if any
		name_node = node.child_by_field_name("name") or node.child_by_field_name("identifier")
		name = None
		if name_node:
			raw = source[name_node.start_byte : name_node.end_byte]
			name = raw.decode("utf-8", errors="ignore")

		# 3) build location & metadata
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

		# 4) prepare kwargs for Pydantic schema
		data: dict = {
			"id": id_counter,
			"type": etype,
			"name": name,
			"location": loc,
			"metadata": meta,
		}
		if "parent_id" in schema_cls.model_fields:
			data["parent_id"] = parent_id
		if "children" in schema_cls.model_fields:
			# initialize an empty list for children
			data["children"] = []

		# 5) instantiate & record
		entity = schema_cls(**data)
		my_id = id_counter
		id_counter += 1
		entities.append(entity)

		# 6) recurse
		# we'll only append into `entity.children` if the schema truly
		# declares a `children` field
		supports_children = "children" in schema_cls.model_fields
		for child in node.children:
			child_ent = visit(child, parent_id=my_id)
			if child_ent and supports_children:
				entity.children.append(child_ent)

		return entity

	# start traversal
	visit(tree.root_node, parent_id=None)
	return entities
