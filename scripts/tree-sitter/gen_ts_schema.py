"""Generate schema.py with a schema for each supported language, including name, extensions, and nodes."""

import json
from pathlib import Path
from typing import Any

from rich.console import Console

# Paths
LITERALS_PY_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/literals.py")
EXT_MAP_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/json/extension-map.json")
NODE_TYPES_DIR = Path("src/codemap/processor/tree_sitter/schema/languages/json/languages")
SCHEMA_PY_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/schema.py")

console = Console()


def load_literals() -> dict[str, set[str]]:
	"""Load SupportedLanguages, SupportedExtensions, and NodeTypes from literals.py as sets."""
	literals = {}
	with LITERALS_PY_PATH.open() as f:
		code = f.read()
	for key in ("SupportedLanguages", "SupportedExtensions", "NodeTypes"):
		start = code.find(f"{key} = Literal [")
		if start == -1:
			continue
		start = code.find("[", start) + 1
		end = code.find("]", start)
		items = set()
		for line in code[start:end].splitlines():
			stripped_line = line.strip().strip(",")
			if not stripped_line or stripped_line.startswith("#"):
				continue
			# Remove quotes
			if (stripped_line.startswith('"') and stripped_line.endswith('"')) or (
				stripped_line.startswith("'") and stripped_line.endswith("'")
			):
				items.add(stripped_line[1:-1])
		literals[key] = items
	return literals


def load_extension_map() -> dict[str, list[str]]:
	"""Load language to extensions mapping from extension-map.json."""
	with EXT_MAP_PATH.open() as f:
		ext_map = json.load(f)
	lang_to_ext = {}
	for entry in ext_map:
		if entry.get("type") != "programming":
			continue
		name = entry.get("name", "").lower().replace("-", "").replace(" ", "")
		extensions = [ext.lstrip(".") for ext in entry.get("extensions", [])]
		lang_to_ext[name] = extensions
	return lang_to_ext


def normalize_language_name(name: str) -> str:
	"""Normalize language name to a lowercase string with no spaces or dashes."""
	return name.lower().replace("-", "").replace(" ", "")


def extract_nodes_for_language(json_path: Path) -> list[str]:
	"""Extract all node types from a node-types.json file."""
	with json_path.open() as f:
		data = json.load(f)
	nodes = set()

	def collect_types(obj: Any):  # noqa: ANN401
		if isinstance(obj, dict):
			if "type" in obj and isinstance(obj["type"], str):
				nodes.add(obj["type"])
			for v in obj.values():
				collect_types(v)
		elif isinstance(obj, list):
			for item in obj:
				collect_types(item)

	collect_types(data)
	return sorted(nodes)


def escape_literal(val: str) -> str:
	"""Escape special characters in a string to be used in a Python literal."""
	val = val.replace("\\", r"\\")
	val = val.replace("\n", r"\\n")
	val = val.replace("\r", r"\\r")
	return val.replace("\t", r"\\t")


def generate_schema_py():
	"""Generate schema.py with a schema for each supported language, including name, extensions, and nodes."""
	literals = load_literals()
	supported_languages = literals["SupportedLanguages"]
	supported_extensions = literals["SupportedExtensions"]
	literals["NodeTypes"]
	lang_to_ext = load_extension_map()
	schema = []
	for node_file in NODE_TYPES_DIR.glob("*.json"):
		lang_name = node_file.stem
		norm_lang = normalize_language_name(lang_name)
		if norm_lang not in supported_languages:
			continue
		extensions = [ext for ext in lang_to_ext.get(norm_lang, []) if ext in supported_extensions]
		nodes = extract_nodes_for_language(node_file)
		schema.append(
			{
				"name": norm_lang,
				"extensions": extensions,
				"nodes": nodes,
			}
		)
	# Write schema.py
	with SCHEMA_PY_PATH.open("w", encoding="utf-8") as f:
		f.write('"""Auto-generated language schema. DO NOT EDIT MANUALLY."""\n\n')
		f.write("# ruff: noqa: E501, RUF001\n")
		f.write("from typing import TypedDict, List\n")
		f.write("from .literals import SupportedLanguages, SupportedExtensions, NodeTypes\n\n")
		f.write("class LanguageSchema(TypedDict):\n")
		f.write('    """Schema for a language."""\n')
		f.write("    name: SupportedLanguages\n")
		f.write("    extensions: List[SupportedExtensions]\n")
		f.write("    nodes: List[NodeTypes]\n\n")
		f.write("LANGUAGE_SCHEMAS: List[LanguageSchema] = [\n")
		for entry in schema:
			f.write("    {\n")
			f.write(f'        "name": "{entry["name"]}",\n')
			# Escape extensions
			ext_list = []
			for ext in entry["extensions"]:
				ext_escaped = escape_literal(ext)
				if '"' in ext_escaped:
					ext_list.append(f"'{ext_escaped}'")
				elif "'" in ext_escaped:
					ext_list.append(f'"{ext_escaped}"')
				else:
					ext_list.append(f'"{ext_escaped}"')
			f.write(f'        "extensions": [{", ".join(ext_list)}],\n')
			# Escape nodes
			node_list = []
			for n in entry["nodes"]:
				n_escaped = escape_literal(n)
				if '"' in n_escaped:
					node_list.append(f"'{n_escaped}'")
				elif "'" in n_escaped:
					node_list.append(f'"{n_escaped}"')
				else:
					node_list.append(f'"{n_escaped}"')
			f.write(f'        "nodes": [{", ".join(node_list)}],\n')
			f.write("    },\n")
		f.write("]\n")
	console.print(f"[green]Generated {SCHEMA_PY_PATH} with {len(schema)} language schemas.")


if __name__ == "__main__":
	generate_schema_py()
