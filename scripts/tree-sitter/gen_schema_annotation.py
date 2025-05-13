"""
Script to auto-generate node type descriptions for each language in schema.py using an LLM.

- Reads language and node type data from schema.py (or the data used to generate it)
- Batches node names and sends them to an LLM for description/comment generation
- Uses the LLM API pattern from src/codemap/llm/api.py (call_llm_api, ConfigLoader, etc.)
- Outputs a mapping of node name to description for each language as a JSON file
- Does NOT modify schema.py directly; output is for later merging
- Usage: python gen_schema_annotation.py --lang <language> [--batch-size 20] [--output annotations.json]

Requirements: Ensure your environment is set up to use the LLM API (see src/codemap/llm/api.py)
"""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console

from codemap.config import ConfigLoader
from codemap.llm.api import MessageDict, call_llm_api

# Path to the generated schema.py (or the data source for nodes)
SCHEMA_PY_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/schema.py")
ANNOTATION_OUTPUT_PATH = Path("src/codemap/processor/tree_sitter/schema/languages/json/node_annotations.json")
MAX_BATCH = 10

console = Console()


def load_schema_nodes() -> dict[str, list[str]]:
	"""Load language -> node list mapping from schema.py by parsing the LANGUAGES list."""
	# Read the schema.py file
	schema_text = SCHEMA_PY_PATH.read_text(encoding="utf-8")
	lang_to_nodes = {}

	# Find LANGUAGES list definition
	languages_start = schema_text.find("LANGUAGES: list[LanguageSchema] = [")
	if languages_start == -1:
		# Try alternative format
		languages_start = schema_text.find("LANGUAGES = [")
		if languages_start == -1:
			msg = "LANGUAGES list not found in schema.py"
			raise RuntimeError(msg)

	# Manual parsing of the language entries
	bracket_count = 0
	in_language_entry = False
	current_lang = None
	current_nodes = []
	in_nodes_list = False  # Track when we're inside the nodes list

	# Skip to the start of the list content
	list_start = schema_text.find("[", languages_start)

	# Process line by line to find each language entry
	for line_raw in schema_text[list_start:].splitlines():
		line = line_raw.strip()

		# Track when we're inside a language entry
		if line.startswith("{"):
			in_language_entry = True
			bracket_count += 1
			continue

		if in_language_entry:
			# Find language name
			if '"name":' in line:
				name_part = line.split('"name":')[1].strip()
				# Extract the language name from quotes
				if name_part.startswith('"') and '",' in name_part:
					current_lang = name_part.split('",')[0].strip('"')
				else:
					# Handle other formats
					current_lang = name_part.split(",")[0].strip().strip('"')

			# Find the start of nodes list
			if '"nodes":' in line and "[" in line:
				bracket_count += 1
				in_nodes_list = True
				continue

			# End of nodes list
			if in_nodes_list and line.startswith("]"):
				in_nodes_list = False
				continue

			# Node entries - only process when inside the nodes list
			if in_nodes_list and current_lang and (line.startswith('"') or line.strip().startswith('"')):
				# This line is a node entry
				node = line.strip(", \t\"'")
				if node and not node.startswith("name") and not node.startswith("extensions"):
					current_nodes.append(node)

			# End of language entry
			if line.startswith("}"):
				bracket_count -= 1
				if current_lang:
					lang_to_nodes[current_lang] = current_nodes
					current_lang = None
					current_nodes = []
					in_language_entry = False
					in_nodes_list = False

	console.print(f"Loaded nodes for {len(lang_to_nodes)} languages: {', '.join(lang_to_nodes.keys())}")
	console.print(
		f"First language: {next(iter(lang_to_nodes.keys()))} has {len(next(iter(lang_to_nodes.values())))} nodes"
	)

	return lang_to_nodes


def batch(iterable, n=10):
	"""Batch data into lists of length n."""
	length = len(iterable)
	for ndx in range(0, length, n):
		yield iterable[ndx : ndx + n]


class NodeDescription(BaseModel):
	"""Schema for a single node description."""

	description: str = Field(
		..., description="A concise, clear, and accurate description of what the node represents in code"
	)


class OutputSchema(BaseModel):
	"""Schema for the output of the LLM, maps node names to descriptions."""

	nodes: dict[str, str] = Field(..., description="Dictionary mapping node names to their descriptions")


def clean_json_response(json_str: str) -> str:
	"""Clean and sanitize JSON response to handle common issues."""
	import re

	# Remove any leading/trailing non-JSON content
	json_start = json_str.find("{")
	json_end = json_str.rfind("}") + 1

	if json_start == -1 or json_end <= json_start:
		return json_str

	json_str = json_str[json_start:json_end]

	# Fix missing quotes around values
	# Match patterns like: "key": value without quotes
	json_str = re.sub(r'"([^"]+)":\s*([^",\{\}\[\]]+)(?=,|}|$)', r'"\1": "\2"', json_str)

	# Fix common JSON formatting issues
	# Replace any triple quotes with single quotes
	json_str = json_str.replace('"""', '"').replace("'''", "'")

	# Fix missing quotes around keys
	json_str = re.sub(r"(\s*)([a-zA-Z0-9_]+)(\s*):(\s*)", r'\1"\2"\3:\4', json_str)

	# Fix trailing commas
	json_str = re.sub(r",(\s*})", r"\1", json_str)
	json_str = re.sub(r",(\s*])", r"\1", json_str)

	# Fix invalid escape sequences
	json_str = json_str.replace("\\", "\\\\").replace('\\"', '\\\\"')

	return re.sub(r'([^\\])\\([^"\\/bfnrtu])', r"\1\\\\\2", json_str)


def extract_node_descriptions_from_text(text: str) -> dict[str, str]:
	"""Extract node descriptions from text when JSON parsing fails."""
	import re

	node_descriptions = {}

	# Find lines with pattern: "node_name": description text
	# or node_name: description text without quotes
	pattern1 = r'"([^"]+)":\s*"?([^"}\n,]+[^}\n,]*)"?,?'
	pattern2 = r'(?<!")([\w_]+)(?!"):\s*"?([^"}\n,]+[^}\n,]*)"?,?'

	# Combine both patterns for matching
	for pattern in [pattern1, pattern2]:
		matches = re.findall(pattern, text)
		for node_name, description in matches:
			node_name = node_name.strip()  # noqa: PLW2901
			description = description.strip()  # noqa: PLW2901
			if node_name and description:
				node_descriptions[node_name] = description

	return node_descriptions


def annotate_nodes_with_llm(
	node_names: list[str],
	lang: str,
	config_loader: ConfigLoader,
	batch_size: int = 20,
	system_prompt: str = "You are an expert code documentation assistant. For each programming language node type, generate a concise, clear, and accurate description of what it represents in code. Output a JSON object mapping each node name to its description. Do not invent node names. Only describe the provided list. Use one sentence per node.",
) -> dict[str, str]:
	"""Annotate a list of node names using the LLM, batching as needed."""
	all_descriptions = {}
	batch_count = 0

	# For certain languages, use a smaller batch size to avoid JSON parsing issues
	if lang in ["python", "javascript", "php"]:
		batch_size = min(batch_size, 10)
		console.print(f"[yellow]Using smaller batch size ({batch_size}) for {lang} nodes[/yellow]")

	for batch_count, node_batch in enumerate(batch(node_names, batch_size), 1):
		if MAX_BATCH and batch_count >= MAX_BATCH:
			break

		prompt = (
			f"Describe the following programming language node types for {lang} programming language. "
			"Return a JSON object mapping each node name to a one-sentence description.\n"
			f"Your response must be a valid JSON object where the keys are node names and values are descriptions.\n"
			f"Ensure each value (description) is enclosed in double quotes.\n"
			f"Node names: {json.dumps(node_batch)}"
		)

		messages: list[MessageDict] = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": prompt},
		]
		try:
			# Get raw JSON response from LLM
			response_text = call_llm_api(messages, config_loader)

			# Try to parse the response as JSON
			if isinstance(response_text, str):
				try:
					# Try to parse as JSON directly
					node_descriptions = json.loads(response_text)
					if isinstance(node_descriptions, dict):
						all_descriptions.update(node_descriptions)
						console.print(
							f"[green]Successfully processed batch with {len(node_descriptions)} nodes[/green]"
						)
					else:
						msg = "Response is not a dictionary"
						raise TypeError(msg)
				except json.JSONDecodeError:
					console.print("[yellow]Failed to parse response as JSON, trying to extract JSON from text[/yellow]")

					# Try to extract JSON from text
					json_start = response_text.find("{")
					json_end = response_text.rfind("}") + 1

					if json_start >= 0 and json_end > json_start:
						json_str = response_text[json_start:json_end]

						console.print(f"[yellow]Extracted JSON (first 100 chars): {json_str[:100]}...[/yellow]")
						# Apply cleaning for all languages
						json_str = clean_json_response(json_str)
						console.print(f"[yellow]Cleaned JSON (first 100 chars): {json_str[:100]}...[/yellow]")

						try:
							node_descriptions = json.loads(json_str)
							if isinstance(node_descriptions, dict):
								all_descriptions.update(node_descriptions)
								console.print(
									f"[green]Successfully extracted and processed JSON with {len(node_descriptions)} nodes[/green]"
								)
							else:
								msg = "Extracted JSON is not a dictionary"
								raise TypeError(msg)
						except json.JSONDecodeError as e:
							console.print(f"[red]Failed to parse extracted JSON: {e}")
							console.print(f"[red]JSON snippet: {json_str[:200]}...[/red]")

							# Try the regex-based fallback method
							console.print("[yellow]Trying to salvage individual node descriptions...[/yellow]")

							# Method 1: Simple regex extraction
							import re

							pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
							matches = re.findall(pattern, json_str)

							if matches:
								salvaged = dict(matches)
								console.print(f"[green]Salvaged {len(salvaged)} nodes with basic regex[/green]")
								all_descriptions.update(salvaged)
							else:
								# Method 2: More sophisticated text extraction
								salvaged = extract_node_descriptions_from_text(json_str)
								if salvaged:
									console.print(
										f"[green]Salvaged {len(salvaged)} nodes with advanced extraction[/green]"
									)
									all_descriptions.update(salvaged)
								else:
									console.print("[red]Failed to salvage any nodes[/red]")
			else:
				console.print(f"[red]Unexpected response type: {type(response_text)}[/red]")

		except Exception as e:
			console.print(f"[red]Batch failed: {e}. Skipping batch: {node_batch}[/red]")

	return all_descriptions


def load_existing_annotations() -> dict[str, dict[str, str]]:
	"""Load existing node annotations if the file exists."""
	if not ANNOTATION_OUTPUT_PATH.exists():
		return {}

	try:
		with ANNOTATION_OUTPUT_PATH.open("r", encoding="utf-8") as f:
			return json.load(f)
	except (json.JSONDecodeError, OSError) as e:
		console.print(f"[yellow]Warning: Failed to load existing annotations: {e}[/yellow]")
		return {}


def main():
	"""Batch annotate node types in schema.py using LLM."""
	parser = argparse.ArgumentParser(description="Batch annotate node types in schema.py using LLM.")
	parser.add_argument("--lang", type=str, help="Language to annotate (matches schema.py name)")
	parser.add_argument("--batch-size", type=int, default=20, help="Batch size for LLM calls")
	parser.add_argument("--output", type=str, default=str(ANNOTATION_OUTPUT_PATH), help="Output JSON file path")
	parser.add_argument("--force-all", action="store_true", help="Force regeneration of all node descriptions")
	args = parser.parse_args()

	config_loader = ConfigLoader()
	lang_to_nodes = load_schema_nodes()
	existing_annotations = load_existing_annotations() if not args.force_all else {}

	langs = [args.lang] if args.lang else list(lang_to_nodes.keys())

	all_annotations: dict[str, dict[str, str]] = existing_annotations.copy()
	for lang in langs:
		# Initialize language entry if it doesn't exist
		if lang not in all_annotations:
			all_annotations[lang] = {}

		node_names = lang_to_nodes[lang]

		# Filter out nodes that already have descriptions
		nodes_to_annotate = [node for node in node_names if node not in all_annotations.get(lang, {})]

		if not nodes_to_annotate:
			console.print(f"[green]All {len(node_names)} nodes for {lang} already have descriptions.[/green]")
			continue

		console.print(f"[cyan]Annotating {len(nodes_to_annotate)} new nodes for language: {lang}...[/cyan]")
		desc_map = annotate_nodes_with_llm(nodes_to_annotate, lang, config_loader, batch_size=args.batch_size)

		# Update with new descriptions
		all_annotations[lang].update(desc_map)
		console.print(f"[green]Annotated {len(desc_map)} new nodes for {lang}.[/green]")

	# Save output
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(all_annotations, f, indent=2, ensure_ascii=False)
	console.print(f"[bold green]Saved node annotations to {output_path}[/bold green]")


if __name__ == "__main__":
	main()
