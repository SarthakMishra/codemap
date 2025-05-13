"""
Script to auto-generate language_map.py by mapping Tree-sitter node types to entity schemas using semantic similarity.

- Embeds node types (from literals.py) and entity schema names/descriptions (from entity/base.py)
- Uses model2vec (embedding_utils.py) for embeddings
- Uses sklearn cosine similarity for matching (as in clusterer.py)
- Uses node_annotations.json for node descriptions
- Outputs LANGUAGE_NODE_MAPPING in the format of language_map.py
"""

import importlib
import json
import re  # For parsing literals
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # Add project root to sys.path

# Import StaticModel at the module level for type hinting and single load
from model2vec import StaticModel

# Paths
LITERALS_PY_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/literals.py")
ENTITY_TYPES_PY_PATH = Path(
	"src/codemap/processor/tree_sitter/schema/entity/types.py"
)  # Path to get EntityType literals
LANG_SCHEMA_PY_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/schema.py")
LANG_MAP_PY_PATH = Path("src/codemap/processor/tree_sitter/schema/language_map.py")
NODE_ANNOTATIONS_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/json/node_annotations.json")
TYPE_ANNOTATIONS_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/json/type_annotations.json")

MODEL_NAME = "sarthak1/Qodo-Embed-M-1-1.5B-M2V-Distilled"
MIN_CONFIDENCE = 0.40

console = Console()


def load_node_annotations() -> dict[str, dict[str, str]]:
	"""Load node descriptions from node_annotations.json file."""
	try:
		if not NODE_ANNOTATIONS_PATH.exists():
			console.print(
				f"[yellow]Node annotations file not found at {NODE_ANNOTATIONS_PATH}. Continuing without descriptions.[/yellow]"
			)
			return {}

		with NODE_ANNOTATIONS_PATH.open("r", encoding="utf-8") as f:
			annotations = json.load(f)

		console.print(f"[green]Loaded node descriptions for {len(annotations)} languages.[/green]")
		return annotations
	except Exception as e:
		console.print(f"[yellow]Failed to load node annotations: {e}. Continuing without descriptions.[/yellow]")
		return {}


def load_type_annotations() -> dict[str, str]:
	"""Load entity type descriptions from type_annotations.json file."""
	try:
		if not TYPE_ANNOTATIONS_PATH.exists():
			console.print(
				f"[yellow]Type annotations file not found at {TYPE_ANNOTATIONS_PATH}. Using default extraction from types.py.[/yellow]"
			)
			return {}

		with TYPE_ANNOTATIONS_PATH.open("r", encoding="utf-8") as f:
			annotations = json.load(f)

		console.print(f"[green]Loaded descriptions for {len(annotations)} entity types.[/green]")
		return annotations
	except Exception as e:
		console.print(f"[yellow]Failed to load type annotations: {e}. Using default extraction from types.py.[/yellow]")
		return {}


def generate_embedding(text: str, model: StaticModel) -> list[float] | None:
	"""
	Generate embedding for a single text string using a pre-loaded model.

	Args:
		text: The text string to embed.
		model: The pre-loaded StaticModel instance.

	Returns:
		Embedding (list of floats) or None if an error occurs.
	"""
	try:
		# model.encode typically expects a list of strings
		embedding_array = model.encode([text])
		# Assuming model.encode([text]) returns a 2D array like [[0.1, 0.2, ...]] for a single text
		if embedding_array is not None and embedding_array.ndim == 2 and embedding_array.shape[0] == 1:
			return embedding_array[0].tolist()
		if (
			embedding_array is not None and embedding_array.ndim == 1
		):  # If it directly returns 1D for single string list
			return embedding_array.tolist()
		console.print(f"[yellow]Unexpected embedding shape for text '{text}'. Skipping.[/yellow]")
		return None
	except Exception as e:
		console.print(f"[red]Error generating embedding for '{text}': {e}[/red]")
		return None


def extract_entity_types() -> list[str]:
	"""Extract entity type names from types.py without parsing comments."""
	try:
		with ENTITY_TYPES_PY_PATH.open("r", encoding="utf-8") as f:
			content = f.read()

		# Find the EntityType union definition
		entity_type_match = re.search(r"EntityType\s*=\s*Literal\s*\[([^\]]+)\]", content, re.DOTALL)
		if not entity_type_match:
			console.print("[red]Could not find EntityType definition in types.py[/red]")
			return []

		# Extract all constituent type names from the file
		# First, get all type literals defined in the file
		type_literals = []
		for literal_match in re.finditer(
			r"([A-Z_]+EntityType|VisibilityModifier|ScopeType|StatementType|ExpressionType|OperatorEntityType)\s*=\s*Literal\s*\[([^\]]+)\]",
			content,
			re.DOTALL,
		):
			literal_match.group(1)
			type_content = literal_match.group(2)

			# Extract individual literals from each type definition
			type_literals.extend(literal.group(1) for literal in re.finditer(r'["\']([\w_]+)["\']', type_content))

		# Also add the literal "UNKNOWN" which appears directly in EntityType
		if "UNKNOWN" not in type_literals:
			type_literals.append("UNKNOWN")

		return type_literals
	except Exception as e:
		console.print(f"[red]Error extracting entity types: {e}[/red]")
		return []


def load_target_entity_types_with_descriptions() -> list[tuple[str, str | None]]:
	"""Load entity type names and descriptions from type_annotations.json or extract from types.py."""
	type_annotations = load_type_annotations()

	if type_annotations:
		# If we have the annotations file, use it
		return [(name, desc) for name, desc in type_annotations.items()]
	# Otherwise extract just the names without descriptions
	console.print("[yellow]No type annotations found. Using entity type names without descriptions.[/yellow]")
	entity_types = extract_entity_types()
	return [(name, None) for name in entity_types]


def load_supported_language_literals() -> list[tuple[str, str | None]]:
	"""Load SupportedLanguages from literals.py as (name, None) tuples."""
	try:
		with LITERALS_PY_PATH.open("r", encoding="utf-8") as f:
			content = f.read()

		# Extract language literals
		langs = []
		lang_match = re.search(r"SupportedLanguages\s*=\s*Literal\s*\[([^\]]+)\]", content, re.DOTALL)
		if lang_match:
			lang_content = lang_match.group(1)
			langs.extend(
				(lang_literal.group(1), None) for lang_literal in re.finditer(r'["\']([\w_]+)["\']', lang_content)
			)

		return langs
	except Exception as e:
		console.print(f"[red]Error loading supported languages: {e}[/red]")
		return []


def load_language_schemas() -> dict[str, list[str]]:
	"""Load language to node types mapping from schema.py."""
	module_name = "codemap.processor.tree_sitter.schema.languages.schema"
	spec = importlib.util.find_spec(module_name)
	if spec is None:
		msg = f"Could not find module {module_name}"
		console.print(f"[red]{msg}[/red]")
		raise ImportError(msg)
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)  # type: ignore[attr-defined]
	lang_nodes_map = {}
	if hasattr(module, "LANGUAGES") and isinstance(module.LANGUAGES, list):
		for lang_item in module.LANGUAGES:
			if isinstance(lang_item, dict) and "name" in lang_item and "nodes" in lang_item:
				lang_nodes_map[lang_item["name"]] = lang_item["nodes"]
			else:
				console.print(f"[yellow]Skipping malformed language item in schema.py: {lang_item}[/yellow]")
	else:
		console.print("[red]LANGUAGES constant not found or not a list in schema.py[/red]")
	return lang_nodes_map


def main(
	output_path: Path = LANG_MAP_PY_PATH,
	similarity_threshold: float = MIN_CONFIDENCE,
	config_path: Path | None = None,  # Not used currently
) -> None:
	"""Main function to generate language_map.py."""
	console.print(f"[cyan]Loading embedding model ({MODEL_NAME})...[/cyan]")
	try:
		model = StaticModel.from_pretrained(MODEL_NAME)
		console.print("[green]Embedding model loaded successfully.[/green]")
	except Exception as e:
		console.print(f"[red]Failed to load embedding model: {e}[/red]")
		return

	# Load necessary literals
	supported_languages = load_supported_language_literals()  # List of (name, None)
	language_to_nodes_map = load_language_schemas()  # Maps lang_name to list of its node_names
	target_entity_types_with_desc = load_target_entity_types_with_descriptions()  # List of (name, description)

	# Load node descriptions from node_annotations.json
	node_annotations = load_node_annotations()  # Maps lang_name to dict of node_name -> description

	if not target_entity_types_with_desc:
		console.print(
			"[red]No target entity type literals loaded. Cannot generate mappings. Check entity/types.py and parsing logic.[/red]"
		)
		return

	console.print(f"[cyan]Generating embeddings for {len(target_entity_types_with_desc)} target entity types...[/cyan]")
	target_type_embeddings_list = []
	valid_target_type_names = []  # Store only the name for mapping
	for name, description in target_entity_types_with_desc:
		text_to_embed = f"{name}: {description}" if description else name
		emb = generate_embedding(text_to_embed, model)
		if emb is not None:
			target_type_embeddings_list.append(emb)
			valid_target_type_names.append(name)

	if not target_type_embeddings_list:
		console.print("[red]No target entity type embeddings could be generated. Aborting.[/red]")
		return
	target_type_embeddings = np.array(target_type_embeddings_list)

	lang_map_output: dict[
		str, dict[str, tuple[str, str, str]]
	] = {}  # Output format: {lang: {node_name: (entity_type_str, prefix, comment)}}

	console.print(f"[cyan]Processing {len(supported_languages)} languages...[/cyan]")
	for lang_idx, (lang_name, _) in enumerate(supported_languages):
		lang_tree_sitter_nodes = language_to_nodes_map.get(lang_name)
		if not lang_tree_sitter_nodes:
			console.print(
				f"[yellow]No tree-sitter nodes found for language: {lang_name} in schema.py. Skipping.[/yellow]"
			)
			continue

		console.print(
			f"[magenta]Processing language: {lang_name} ({lang_idx + 1}/{len(supported_languages)}) with {len(lang_tree_sitter_nodes)} nodes...[/magenta]"
		)

		# Get node descriptions for this language
		lang_node_descriptions = node_annotations.get(lang_name, {})
		if lang_node_descriptions:
			console.print(f"[green]Found {len(lang_node_descriptions)} node descriptions for {lang_name}[/green]")
		else:
			console.print(f"[yellow]No node descriptions found for {lang_name}. Using node names only.[/yellow]")

		node_to_entity_type_map: dict[str, tuple[str, str, str]] = {}

		for ts_node_name in lang_tree_sitter_nodes:
			# Preprocess node name for embedding: replace underscores with spaces
			node_name_for_embed = ts_node_name.replace("_", " ")

			# Use node description if available
			node_description = lang_node_descriptions.get(ts_node_name, "")
			text_to_embed = f"{node_name_for_embed}: {node_description}" if node_description else node_name_for_embed

			node_embedding = generate_embedding(text_to_embed, model)
			mapped_entity_type: str = "UNKNOWN"  # Default to UNKNOWN
			prefix_for_file = ""  # Empty prefix by default (no commenting out)
			comment_for_file = ""  # Empty comment by default

			if node_embedding is not None and target_type_embeddings.size > 0:
				similarities = cosine_similarity(np.array(node_embedding).reshape(1, -1), target_type_embeddings)[0]
				best_idx = np.argmax(similarities)
				best_score = similarities[best_idx]

				if best_score >= similarity_threshold:
					mapped_entity_type = valid_target_type_names[best_idx]
				else:
					best_guess_type = (
						valid_target_type_names[best_idx] if best_idx < len(valid_target_type_names) else "N/A"
					)
					# Low confidence mapping: comment out the line and add score info
					prefix_for_file = "# "
					comment_for_file = (
						f"  # LOW CONFIDENCE ({best_score:.2f}) for node {ts_node_name!r} -> {best_guess_type}"
					)
			else:
				# Unable to embed: comment out the line
				prefix_for_file = "# "
				comment_for_file = f"  # UNABLE TO EMBED node {ts_node_name!r}"

			node_to_entity_type_map[ts_node_name] = (mapped_entity_type, prefix_for_file, comment_for_file)

		lang_map_output[lang_name] = node_to_entity_type_map

	# Write language_map.py
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		f.write('"""Auto-generated language to node mapping. DO NOT EDIT MANUALLY."""\n\n')
		f.write("# ruff: noqa: E501, RUF001, F405, F821\n")  # Removed F403 (star import) as it's no longer used
		# No more schema class imports from .entity.base
		f.write("from .entity.types import EntityType\n")  # Only EntityType is needed now from types
		f.write("from .languages.literals import NodeTypes, SupportedLanguages\n\n")
		# Updated NodeMapping definition
		f.write("NodeMapping = dict[NodeTypes, EntityType]\n\n")

		f.write("LANGUAGE_NODE_MAPPING: dict[SupportedLanguages, NodeMapping] = {\n")
		num_langs = len(lang_map_output)
		for i, (lang_key, mapping_dict) in enumerate(lang_map_output.items()):
			lang_key_repr = repr(lang_key)
			f.write(f"    {lang_key_repr}: {{\n")

			for node_name_key, mapped_data in mapping_dict.items():
				mapped_entity_type_val, prefix_val, comment_suffix_val = mapped_data

				node_name_key_repr = repr(node_name_key)
				mapped_entity_type_repr = repr(mapped_entity_type_val)

				line_end = "," if not prefix_val else ""  # No comma for commented lines
				f.write(
					f"        {prefix_val}{node_name_key_repr}: {mapped_entity_type_repr}{line_end}{comment_suffix_val}\n"
				)

			f.write("    }")
			if i < num_langs - 1:
				f.write(",")
			f.write("\n")
		f.write("}\n")

	console.print(f"[green]Successfully generated language map at: {output_path}[/green]")


if __name__ == "__main__":
	main()
