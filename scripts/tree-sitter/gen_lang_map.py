"""
Script to generate language_map.py by mapping Tree-sitter node types to entity types using LLM.

- Uses LLM to map node types to appropriate entity types based on descriptions
- Reads from node_annotations.json and type_annotations.json
- Uses Pydantic models for structured output processing
- Outputs the mapping as language_map.json, which can be converted to language_map.py
- Implements incremental updates by checking what's already processed
"""

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from codemap.config import ConfigLoader
from codemap.llm.api import MessageDict, call_llm_api

# Paths
LITERALS_PY_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/literals.py")
ENTITY_TYPES_PY_PATH = Path("src/codemap/processor/tree_sitter/schema/entity/types.py")
LANG_SCHEMA_PY_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/schema.py")
LANG_MAP_PY_PATH = Path("src/codemap/processor/tree_sitter/schema/language_map.py")
NODE_ANNOTATIONS_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/json/node_annotations.json")
TYPE_ANNOTATIONS_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/json/type_annotations.json")
OUTPUT_JSON_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/json/language_map.json")

MAX_BATCH = 0  # Maximum number of batches to process per run
BATCH_SIZE = 5  # Number of nodes per batch (reduced from 10)

console = Console()


# Pydantic models for structured output
class NodeEntityMapping(BaseModel):
	"""Mapping from a node type to an entity type with confidence."""

	entity_type: str = Field(..., description="The entity type that best matches the node")
	confidence: float = Field(
		..., description="Confidence score from 0.0 to 1.0 where 1.0 is highest confidence", ge=0.0, le=1.0
	)
	reasoning: str = Field(..., description="Brief explanation of why this entity type was chosen")


class BatchMappingResult(BaseModel):
	"""Result of mapping a batch of nodes to entity types."""

	mappings: dict[str, NodeEntityMapping] = Field(
		..., description="Dictionary mapping node names to their entity type mappings"
	)


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


def extract_node_types() -> list[str]:
	"""Extract the node types from literals.py NodeTypes Literal union."""
	try:
		# Import the module with NodeTypes literal
		literals_module_path = "src.codemap.processor.tree_sitter.schema.languages.literals"
		spec = importlib.util.find_spec(literals_module_path)
		if not spec:
			console.print(f"[red]Module not found: {literals_module_path}[/red]")
			return []

		literals_module = importlib.import_module(literals_module_path)

		# Try to get the NodeTypes from the module
		if hasattr(literals_module, "NodeTypes") and hasattr(literals_module.NodeTypes, "__args__"):
			# Extract valid node types from the Literal union
			node_types = [arg for arg in literals_module.NodeTypes.__args__ if isinstance(arg, str)]
			console.print(f"[green]Found {len(node_types)} valid node types in literals.py[/green]")
			return node_types
		console.print("[red]NodeTypes literal not found or does not have __args__ attribute.[/red]")
		return []

	except ImportError as e:
		console.print(f"[red]Failed to import literals module: {e}[/red]")
		return []
	except Exception as e:
		console.print(f"[red]Error extracting node types: {e}[/red]")
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


def load_existing_mappings() -> dict[str, dict[str, str]]:
	"""Load existing language -> node -> entity type mappings if the file exists."""
	if not OUTPUT_JSON_PATH.exists():
		return {}

	try:
		with OUTPUT_JSON_PATH.open("r", encoding="utf-8") as f:
			return json.load(f)
	except (json.JSONDecodeError, OSError) as e:
		console.print(f"[yellow]Warning: Failed to load existing mappings: {e}[/yellow]")
		return {}


def batch(iterable, n=10):
	"""Batch data into lists of length n."""
	length = len(iterable)
	for ndx in range(0, length, n):
		yield iterable[ndx : ndx + n]


def map_nodes_to_entity_types(
	node_names: list[str],
	node_descriptions: dict[str, str],
	entity_types_with_desc: list[tuple[str, str | None]],
	lang: str,
	config_loader: ConfigLoader,
	all_mappings: dict[str, dict[str, str]],
) -> dict[str, str]:
	"""
	Map node names to entity types using LLM.

	Args:
		node_names: List of node names to map
		node_descriptions: Dictionary mapping node names to their descriptions
		entity_types_with_desc: List of tuples (entity_type, description)
		lang: Programming language name
		config_loader: ConfigLoader instance for LLM API
		all_mappings: Current mapping dictionary to update and save after each call

	Returns:
		Dictionary mapping node names to entity types
	"""
	# Create a formatted list of available entity types with descriptions
	entity_types_formatted = []
	entity_type_names = []

	for entity_type, description in entity_types_with_desc:
		entity_type_names.append(entity_type)
		if description:
			entity_types_formatted.append(f"- {entity_type}: {description}")
		else:
			entity_types_formatted.append(f"- {entity_type}")

	entity_types_text = "\n".join(entity_types_formatted)

	# Generate a simpler system prompt for the LLM
	system_prompt = "You are an expert at mapping programming language syntax nodes to semantic entity types."

	# Process nodes in batches
	batch_mappings = {}

	for node_batch in batch(node_names, BATCH_SIZE):
		# Create node descriptions for the batch
		nodes_with_desc = []
		for node in node_batch:
			desc = node_descriptions.get(node, "")
			nodes_with_desc.append(f"- {node}: {desc}" if desc else f"- {node}")

		nodes_text = "\n".join(nodes_with_desc)
		json_schema = BatchMappingResult.model_json_schema()

		# Create a simpler user prompt
		user_prompt = (
			f"Map these {lang} programming language nodes to the most appropriate entity type:\n\n"
			f"{nodes_text}\n\n"
			f"Available entity types:\n{entity_types_text}\n\n"
			f"Only use entity types from the provided list.\n"
			f"IMPORTANT: Must return a JSON object **STRICTLY** following the provided schema:\n"
			f"1. entity_type: The most appropriate entity type from the list\n"
			f"2. confidence: A value from 0.0 to 1.0\n"
			f"3. reasoning: Brief explanation (less than 15 words) for your choice\n\n"
			f"--- **JSON Schema** ---\n\n"
			f"{json_schema}"
		)

		# Prepare messages for LLM API
		messages: list[MessageDict] = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]

		try:
			# Try with Pydantic model validation
			response = call_llm_api(messages, config_loader, BatchMappingResult)

			if isinstance(response, BatchMappingResult):
				# Check and filter valid entity types
				current_batch_mappings = {}
				for node_name, mapping in response.mappings.items():
					# Verify the entity type is in our allowed list
					if mapping.entity_type in entity_type_names:
						# Only add the mapping if confidence is above threshold
						if mapping.confidence >= 0.5:
							current_batch_mappings[node_name] = mapping.entity_type
					else:
						console.print(
							f"[yellow]Warning: Invalid entity type '{mapping.entity_type}' for node '{node_name}'. Skipping.[/yellow]"
						)

				batch_mappings.update(current_batch_mappings)

				# Update all_mappings and save immediately after each successful call
				if lang not in all_mappings:
					all_mappings[lang] = {}
				all_mappings[lang].update(current_batch_mappings)
				save_mappings_to_json(all_mappings)

				console.print(
					f"[green]Processed batch with {len(current_batch_mappings)} valid mappings and saved progress[/green]"
				)
			else:
				console.print(f"[red]Unexpected response type: {type(response)}[/red]")

		except Exception as e:
			console.print(f"[red]Error processing batch: {e}[/red]")

			# Fallback approach - try parsing without structured validation
			try:
				# Call LLM API without structured validation
				response = call_llm_api(messages, config_loader)

				if isinstance(response, str):
					# Try to parse the JSON response manually
					import json
					import re

					# Extract JSON from the string response
					json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
					if not json_match:
						json_match = re.search(r"\{.*\}", response, re.DOTALL)

					if json_match:
						json_str = json_match.group(1) if json_match.group(0).startswith("```") else json_match.group(0)
						try:
							mappings_dict = json.loads(json_str)

							# Process the mappings
							current_batch_mappings = {}
							for node_name, mapping in mappings_dict.items():
								if isinstance(mapping, dict) and "entity_type" in mapping:
									entity_type = mapping["entity_type"]
									if entity_type in entity_type_names:
										current_batch_mappings[node_name] = entity_type
								elif isinstance(mapping, str) and mapping in entity_type_names:
									current_batch_mappings[node_name] = mapping

							batch_mappings.update(current_batch_mappings)

							# Update all_mappings and save immediately after fallback success
							if lang not in all_mappings:
								all_mappings[lang] = {}
							all_mappings[lang].update(current_batch_mappings)
							save_mappings_to_json(all_mappings)

							console.print(
								f"[yellow]Processed batch with fallback method: {len(current_batch_mappings)} mappings and saved progress[/yellow]"
							)
						except json.JSONDecodeError:
							console.print("[red]Failed to parse JSON response in fallback mode[/red]")
			except Exception as fallback_error:
				console.print(f"[red]Fallback processing also failed: {fallback_error}[/red]")

	return batch_mappings


def save_mappings_to_json(mappings: dict[str, dict[str, str]]) -> None:
	"""Save language -> node -> entity type mappings to JSON file."""
	OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
	with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
		json.dump(mappings, f, indent=2, ensure_ascii=False)
	console.print(f"[bold green]Saved language mappings to {OUTPUT_JSON_PATH}[/bold green]")


def filter_valid_node_types(
	mappings: dict[str, dict[str, str]], valid_node_types: list[str]
) -> dict[str, dict[str, str]]:
	"""Filter mappings to only include valid node types."""
	filtered_mappings = {}
	skipped_count = 0
	valid_node_set = set(valid_node_types)

	for lang, lang_mappings in mappings.items():
		filtered_mappings[lang] = {}
		for node_name, entity_type in lang_mappings.items():
			if node_name in valid_node_set:
				filtered_mappings[lang][node_name] = entity_type
			else:
				skipped_count += 1
				console.print(
					f"[yellow]Warning: Node '{node_name}' for language '{lang}' is not in NodeTypes literal. Skipping.[/yellow]"
				)

	if skipped_count > 0:
		console.print(f"[yellow]Skipped {skipped_count} node mappings that were not in NodeTypes literal.[/yellow]")

	return filtered_mappings


def generate_language_map_py() -> None:
	"""Generate language_map.py from the JSON mappings file."""
	if not OUTPUT_JSON_PATH.exists():
		console.print(f"[red]Mappings file not found at {OUTPUT_JSON_PATH}. Cannot generate language_map.py.[/red]")
		return

	try:
		with OUTPUT_JSON_PATH.open("r", encoding="utf-8") as f:
			mappings = json.load(f)

		# Get valid node types for filtering
		valid_node_types = extract_node_types()
		if not valid_node_types:
			console.print(
				"[red]Failed to extract valid node types from literals.py. Cannot generate valid language_map.py.[/red]"
			)
			return

		# Filter to only include valid node types
		filtered_mappings = filter_valid_node_types(mappings, valid_node_types)

		# Save filtered mappings back to JSON
		save_mappings_to_json(filtered_mappings)

		LANG_MAP_PY_PATH.parent.mkdir(parents=True, exist_ok=True)
		with LANG_MAP_PY_PATH.open("w", encoding="utf-8") as f:
			f.write('"""Auto-generated language to node mapping. DO NOT EDIT MANUALLY."""\n\n')
			f.write("# ruff: noqa: E501, RUF001, F405, F821\n")
			f.write("from .entity.types import EntityType\n")
			f.write("from .languages.literals import NodeTypes, SupportedLanguages\n\n")
			f.write("NodeMapping = dict[NodeTypes, EntityType]\n\n")
			f.write("LANGUAGE_NODE_MAPPING: dict[SupportedLanguages, NodeMapping] = {\n")

			num_langs = len(filtered_mappings)
			for i, (lang_key, mapping_dict) in enumerate(filtered_mappings.items()):
				lang_key_repr = repr(lang_key)
				f.write(f"\t{lang_key_repr}: {{\n")

				for node_name_key, entity_type_val in mapping_dict.items():
					node_name_key_repr = repr(node_name_key)
					entity_type_repr = repr(entity_type_val)
					f.write(f"\t\t{node_name_key_repr}: {entity_type_repr},\n")

				f.write("\t}")
				if i < num_langs - 1:
					f.write(",")
				f.write("\n")
			f.write("}\n")

		console.print(f"[green]Successfully generated language_map.py at: {LANG_MAP_PY_PATH}[/green]")
	except Exception as e:
		console.print(f"[red]Error generating language_map.py: {e}[/red]")


def main() -> None:
	"""Main function to generate language mappings using LLM."""
	parser = argparse.ArgumentParser(description="Generate language mappings using LLM.")
	parser.add_argument("--lang", type=str, help="Language to process (matches schema.py name)")
	parser.add_argument(
		"--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size for LLM calls (default: {BATCH_SIZE})"
	)
	parser.add_argument("--force-all", action="store_true", help="Force regeneration of all mappings")
	parser.add_argument("--generate-py", action="store_true", help="Generate language_map.py from JSON")
	parser.add_argument("--filter-only", action="store_true", help="Only filter existing mappings by valid node types")
	parser.add_argument("--max-retries", type=int, default=2, help="Maximum number of retries for failed batches")
	args = parser.parse_args()

	# If only filtering, just load existing mappings, filter, and generate language_map.py
	if args.filter_only:
		console.print("[cyan]Only filtering existing mappings by valid node types...[/cyan]")
		valid_node_types = extract_node_types()
		if not valid_node_types:
			console.print("[red]Failed to extract valid node types. Cannot filter mappings.[/red]")
			return

		existing_mappings = load_existing_mappings()
		filtered_mappings = filter_valid_node_types(existing_mappings, valid_node_types)
		save_mappings_to_json(filtered_mappings)
		generate_language_map_py()
		return

	# If only generating Python file from existing JSON
	if args.generate_py:
		console.print("[cyan]Generating language_map.py from existing JSON mappings...[/cyan]")
		generate_language_map_py()
		return

	config_loader = ConfigLoader()

	# Load valid node types
	valid_node_types = extract_node_types()
	if not valid_node_types:
		console.print("[red]Failed to extract valid node types from literals.py. Cannot generate mappings.[/red]")
		return

	# Load existing mappings if not forcing regeneration
	existing_mappings = {} if args.force_all else load_existing_mappings()

	# Load necessary data
	supported_languages = load_supported_language_literals()
	language_to_nodes_map = load_language_schemas()
	target_entity_types_with_desc = load_target_entity_types_with_descriptions()
	node_annotations = load_node_annotations()

	if not target_entity_types_with_desc:
		console.print(
			"[red]No target entity type literals loaded. Cannot generate mappings. Check entity/types.py and type_annotations.json.[/red]"
		)
		return

	# Only process specified language if provided, otherwise process all supported languages
	langs = (
		[args.lang] if args.lang and args.lang in language_to_nodes_map else [lang for lang, _ in supported_languages]
	)

	console.print(f"[cyan]Processing {len(langs)} languages...[/cyan]")

	all_mappings = existing_mappings.copy()
	batch_size = args.batch_size if args.batch_size > 0 else BATCH_SIZE
	max_retries = args.max_retries

	for lang_idx, lang in enumerate(langs):
		# Initialize language entry if it doesn't exist
		if lang not in all_mappings:
			all_mappings[lang] = {}

		lang_tree_sitter_nodes = language_to_nodes_map.get(lang, [])
		if not lang_tree_sitter_nodes:
			console.print(f"[yellow]No tree-sitter nodes found for language: {lang} in schema.py. Skipping.[/yellow]")
			continue

		# Filter out nodes that are not in the valid node types list before processing
		valid_lang_nodes = [node for node in lang_tree_sitter_nodes if node in valid_node_types]
		if not valid_lang_nodes:
			console.print(
				f"[yellow]No valid tree-sitter nodes found for language: {lang} after filtering. Skipping.[/yellow]"
			)
			continue

		# Create a list of unprocessed nodes (not in existing mappings)
		unprocessed_nodes = [node for node in valid_lang_nodes if node not in all_mappings.get(lang, {})]
		total_unprocessed = len(unprocessed_nodes)

		if total_unprocessed == 0:
			console.print(
				f"[green]All {len(valid_lang_nodes)} nodes for language '{lang}' already processed. Skipping.[/green]"
			)
			continue

		console.print(
			f"[cyan]Processing language '{lang}' ({lang_idx + 1}/{len(langs)}): {total_unprocessed} nodes remaining to map[/cyan]"
		)

		# Process nodes in batches
		with Progress() as progress:
			task = progress.add_task(f"[cyan]Mapping {lang} nodes...", total=total_unprocessed)

			# We'll try to process batches with retries for failed ones
			nodes_to_process = unprocessed_nodes.copy()
			failed_nodes = []
			retry_count = 0

			while nodes_to_process and (MAX_BATCH == 0 or retry_count <= max_retries):
				# Process in batches
				batches = list(batch(nodes_to_process, batch_size))
				if MAX_BATCH > 0:
					batches = batches[:MAX_BATCH]

				# Reset for this attempt
				nodes_to_process = []

				for i, node_batch in enumerate(batches):
					console.print(f"[cyan]Processing batch {i + 1}/{len(batches)} for {lang}...[/cyan]")

					# Get descriptions for the nodes in this batch
					node_descriptions = {node: node_annotations.get(lang, {}).get(node, "") for node in node_batch}

					try:
						# Map nodes to entity types (pass all_mappings to save after each successful call)
						batch_mappings = map_nodes_to_entity_types(
							node_batch,
							node_descriptions,
							target_entity_types_with_desc,
							lang,
							config_loader,
							all_mappings,
						)

						# Add successful mappings (already done within map_nodes_to_entity_types)

						# Track failed nodes for retry
						successful_nodes = set(batch_mappings.keys())
						current_failed = [node for node in node_batch if node not in successful_nodes]
						if current_failed:
							failed_nodes.extend(current_failed)
							console.print(
								f"[yellow]Failed to map {len(current_failed)} nodes in batch {i + 1}[/yellow]"
							)

						# Update progress
						progress.update(task, advance=len(successful_nodes))

					except Exception as e:
						console.print(f"[red]Error processing batch {i + 1}: {e}[/red]")
						failed_nodes.extend(node_batch)

				# If we have failed nodes and still have retries left, try again with smaller batch size
				if failed_nodes and retry_count < max_retries:
					retry_count += 1
					nodes_to_process = failed_nodes
					failed_nodes = []
					batch_size = max(1, batch_size // 2)  # Reduce batch size for retries
					console.print(
						f"[yellow]Retry attempt {retry_count}/{max_retries} with {len(nodes_to_process)} nodes and batch size {batch_size}[/yellow]"
					)
				else:
					# Add any remaining failed nodes to a "failed_nodes" section for reference
					if failed_nodes:
						if "failed_nodes" not in all_mappings:
							all_mappings["failed_nodes"] = {}
						if lang not in all_mappings["failed_nodes"]:
							all_mappings["failed_nodes"][lang] = []
						all_mappings["failed_nodes"][lang].extend(failed_nodes)
						console.print(
							f"[red]{len(failed_nodes)} nodes could not be mapped after {retry_count} retries. Added to failed_nodes section.[/red]"
						)
						# Save the failed nodes too
						save_mappings_to_json(all_mappings)
					break

		# Filter mappings to include only valid node types
		filtered_mappings = filter_valid_node_types(all_mappings, valid_node_types)
		save_mappings_to_json(filtered_mappings)

		# Generate language_map.py
		generate_language_map_py()

	console.print("[bold green]Successfully completed language mapping generation![/bold green]")


if __name__ == "__main__":
	main()
