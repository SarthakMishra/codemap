"""Code documentation generator implementation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from codemap.processor.lod import LODEntity, LODLevel
from codemap.processor.tree_sitter.base import EntityType

from .models import GenConfig

logger = logging.getLogger(__name__)


# --- Mermaid Helper --- #
def _escape_mermaid_label(label: str) -> str:
	# Basic escaping for Mermaid node labels
	# Replace potentially problematic characters
	label = label.replace("[", "(").replace("]", ")")
	label = label.replace("{", "(").replace("}", ")")
	return label.replace('"', "#quot;")  # Use HTML entity for quotes


class CodeMapGenerator:
	"""Generates code documentation based on LOD (Level of Detail)."""

	def __init__(self, config: GenConfig, output_path: Path) -> None:
		"""
		Initialize the code map generator.

		Args:
		    config: Generation configuration settings
		    output_path: Path to write the output

		"""
		self.config = config
		self.output_path = output_path

	def _generate_mermaid_diagram(self, entities: list[LODEntity]) -> str:
		"""Generate a Mermaid diagram string for entity relationships."""
		mermaid_lines = [
			"graph LR",  # Left-to-Right layout
			# --- Definitions for node styles (Darker with white text) ---
			"  classDef fileNode fill:#555,stroke:#FFF,stroke-width:1px,color:white;",  # Dark Grey
			"  classDef classNode fill:#8B008B,stroke:#FFF,stroke-width:1px,color:white;",  # Dark Magenta
			"  classDef funcNode fill:#00008B,stroke:#FFF,stroke-width:1px,color:white;",  # Dark Blue
			"  classDef constNode fill:#483D8B,stroke:#CCC,stroke-width:1px,color:white;",  # Dark Slate Blue
			"  classDef varNode fill:#808000,stroke:#CCC,stroke-width:1px,color:white;",  # Olive
			"  classDef internalImportNode fill:#008B8B,stroke:#FFF,stroke-width:1px,color:white;",  # Dark Cyan
			"  classDef externalImportNode fill:#B8860B,stroke:#FFF,stroke-width:1px,color:white;",  # Dark Goldenrod
			# --- Legend ---
			"  subgraph Legend",
			"    direction LR",
			'    legend_file["File"]:::fileNode',
			'    legend_class{"Class"}:::classNode',
			'    legend_func("Function/Method"):::funcNode',
			'    legend_const["Constant"]:::constNode',
			'    legend_var["Variable"]:::varNode',
			'    legend_import_int["Internal Import"]:::internalImportNode',
			'    legend_import_ext(("External Import")):::externalImportNode',
			"  end",
			"",  # Add a blank line for separation
		]
		nodes = set()  # Track added node IDs to avoid duplicates
		edges = set()  # Track added edges to avoid duplicates
		entity_map: dict[str, LODEntity] = {}  # Map node ID to entity for lookup
		processed_entity_ids = set()
		internal_paths = {
			str(e.metadata.get("file_path")) for e in entities if e.metadata.get("file_path")
		}  # Set of files being analyzed
		name_to_node_ids: dict[str, list[str]] = {}  # Map name to list of node IDs

		# --- Node ID Generation ---
		def get_node_id(entity: LODEntity) -> str:
			file_path_str = entity.metadata.get("file_path", "unknown_file")
			# Use start line for uniqueness within a file
			base_id = f"{file_path_str}_{entity.start_line}_{entity.name or entity.entity_type.name}"
			# Sanitize for Mermaid ID (alphanumeric, underscore)
			return "".join(c if c.isalnum() else "_" for c in base_id)

		# --- Recursive Node and Edge Processing ---
		def process_entity(entity: LODEntity, parent_node_id: str | None = None) -> None:
			nonlocal nodes, edges, processed_entity_ids

			# Skip UNKNOWN entities entirely for the diagram
			if entity.entity_type == EntityType.UNKNOWN:
				return

			# Use the generated node ID for processing checks
			entity_node_id = get_node_id(entity)

			# Skip if already processed using the node ID
			if entity_node_id in processed_entity_ids:
				return
			processed_entity_ids.add(entity_node_id)

			entity_map[entity_node_id] = entity  # Store for lookup

			# Determine node shape and label
			label = _escape_mermaid_label(entity.name or f"({entity.entity_type.name.lower()})")
			node_definition = ""
			node_class = ""  # CSS class for the node
			if entity.entity_type == EntityType.MODULE:
				file_label = _escape_mermaid_label(Path(entity.metadata.get("file_path", "unknown")).name)
				node_definition = f'  {entity_node_id}["{file_label}"]'
				node_class = "fileNode"
			elif entity.entity_type == EntityType.CLASS:
				node_definition = f'  {entity_node_id}{{"{label}"}}'
				node_class = "classNode"
			elif entity.entity_type in (EntityType.FUNCTION, EntityType.METHOD):
				node_definition = f'  {entity_node_id}("{label}")'
				node_class = "funcNode"
			elif entity.entity_type == EntityType.IMPORT:
				pass  # No node for import statement itself
			elif entity.entity_type == EntityType.CONSTANT:
				node_definition = f'  {entity_node_id}["{label}"]'
				node_class = "constNode"
			elif entity.entity_type == EntityType.VARIABLE:
				node_definition = f'  {entity_node_id}["{label}"]'
				node_class = "varNode"

			# Only add node if it has a definition and hasn't been added
			if node_definition and entity_node_id not in nodes:
				# Append class styling if defined
				if node_class:
					node_definition += f":::{node_class}"
				mermaid_lines.append(node_definition)
				nodes.add(entity_node_id)

			# Add edge from parent (e.g., class -> method)
			if parent_node_id and entity_node_id in nodes:
				# Solid arrow with label for membership/declaration
				edge = f"  {parent_node_id} --->|declares| {entity_node_id}"  # Renamed label
				if edge not in edges:
					mermaid_lines.append(edge)
					edges.add(edge)

			# Add to name map if it's a callable entity
			if entity.entity_type in (EntityType.FUNCTION, EntityType.METHOD):
				name = entity.name
				if name:
					if name not in name_to_node_ids:
						name_to_node_ids[name] = []
					name_to_node_ids[name].append(entity_node_id)

			# Process dependencies (imports)
			dependencies = entity.metadata.get("dependencies", [])
			importing_node_id = parent_node_id if entity.entity_type == EntityType.IMPORT else entity_node_id
			if importing_node_id and importing_node_id in nodes:
				for dep in dependencies:
					# Check if the dependency is internal (part of the analyzed files)
					# This is a basic check; accurately mapping 'dep' string to an internal
					# entity/file ID requires more advanced import resolution.
					# For now, treat all imports as potentially external or unresolved internal

					is_external = not dep.startswith(".") and not any(
						dep.startswith(str(p)) for p in internal_paths if isinstance(p, Path)
					)

					# Represent dependency - use simple string ID for now
					dep_id = "dep_" + "".join(c if c.isalnum() else "_" for c in dep)
					dep_label = _escape_mermaid_label(dep)

					# Add dependency node if new
					if dep_id not in nodes:
						# Assign class based on external/internal
						dep_class = "externalImportNode" if is_external else "internalImportNode"
						node_shape = f'(("{dep_label}"))' if is_external else f'["{dep_label}"]'
						mermaid_lines.append(f"  {dep_id}{node_shape}:::{dep_class}")  # Add class styling
						nodes.add(dep_id)

					# Add edge from importing node to dependency
					# Use dashed arrow with label for imports
					edge = f"  {importing_node_id} -.->|imports| {dep_id}"
					if edge not in edges:
						mermaid_lines.append(edge)
						edges.add(edge)

			# Process children recursively
			current_node_id_for_children = entity_node_id if entity_node_id in nodes else parent_node_id
			for child in entity.children:
				# Skip unknown children
				if child.entity_type != EntityType.UNKNOWN:
					process_entity(child, current_node_id_for_children)

		# --- Main Processing Loop --- #
		# Process top-level entities (usually MODULE type)
		for entity in entities:
			if entity.entity_type == EntityType.MODULE:
				process_entity(entity)

		# --- Add Call Edges --- #
		mermaid_lines.append("\n  %% Calls")  # Add a comment separator
		call_edge_indices = []  # Store indices of call edges
		for caller_node_id, caller_entity in entity_map.items():
			if caller_entity.entity_type in (EntityType.FUNCTION, EntityType.METHOD):
				calls = caller_entity.metadata.get("calls", [])
				for called_name in calls:
					# Basic name matching (might need refinement for complex calls like obj.method)
					simple_called_name = called_name.split(".")[-1]  # Get last part
					if simple_called_name in name_to_node_ids:
						for target_node_id in name_to_node_ids[simple_called_name]:
							# Avoid self-calls in diagram for clarity?
							if caller_node_id == target_node_id:
								continue
							call_edge = f"  {caller_node_id} -->|calls| {target_node_id}"
							if call_edge not in edges:
								mermaid_lines.append(call_edge)
								call_edge_indices.append(
									len(mermaid_lines) - 1
								)  # Store index relative to start of mermaid_lines
								edges.add(call_edge)

		# --- Apply Link Styles --- #
		mermaid_lines.append("\n  %% Link Styles")
		for i, _edge_index in enumerate(call_edge_indices):
			# Note: Mermaid linkStyle indices seem 0-based relative to *all* links defined
			# We need to calculate the correct index if other links exist before calls.
			# However, applying style after definition *might* work based on order. Let's try.
			# A safer way might be to collect all edges first, then generate lines with styles.
			mermaid_lines.append(f"  linkStyle {i} stroke:green,stroke-width:2px")  # Apply to call edges

		return "\n".join(mermaid_lines)

	def generate_documentation(self, entities: list[LODEntity], metadata: dict) -> str:
		"""
		Generate markdown documentation from the processed LOD entities.

		Args:
		    entities: List of LOD entities
		    metadata: Repository metadata

		Returns:
		    Generated documentation as string

		"""
		content = []

		# Add header with repository information
		repo_name = metadata.get("name", "Repository")
		content.append(f"# {repo_name} Documentation")
		content.append(f"\nGenerated on: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")

		if "description" in metadata:
			content.append("\n" + metadata.get("description", ""))

		# Add repository statistics
		if "stats" in metadata:
			stats = metadata["stats"]
			content.append("\n## Repository Statistics")
			content.append(f"- Total files: {stats.get('total_files', 0)}")
			content.append(f"- Total lines of code: {stats.get('total_lines', 0)}")
			content.append(f"- Languages: {', '.join(stats.get('languages', []))}")

		# Add directory structure if requested
		if self.config.include_tree and "tree" in metadata:
			content.append("\n## Directory Structure")
			content.append("```")
			content.append(metadata["tree"])
			content.append("```")

		# Add Mermaid diagram if entities exist and config enables it
		if entities and self.config.include_entity_graph:
			content.append("\n## Entity Relationships")
			content.append("```mermaid")
			mermaid_diagram = self._generate_mermaid_diagram(entities)
			content.append(mermaid_diagram)
			content.append("```")

		# Add table of contents
		content.append("\n## Table of Contents")

		# Group entities by file
		files: dict[Path, list[LODEntity]] = {}
		for entity in entities:
			file_path = Path(entity.metadata.get("file_path", ""))
			if not file_path.name:
				continue

			if file_path not in files:
				files[file_path] = []
			files[file_path].append(entity)

		# Create TOC entries
		for file_path in sorted(files.keys()):
			rel_path = file_path.name
			content.append(f"- [{rel_path}](#{rel_path.replace('.', '-')})")

		# Add code documentation grouped by file
		content.append("\n## Code Documentation")

		# Helper function to format a single entity recursively
		def format_entity_recursive(entity: LODEntity, level: int) -> list[str]:
			entity_content = []
			indent = "  " * level
			list_prefix = f"{indent}- "

			# Basic entry: Type and Name/Signature
			entry_line = f"{list_prefix}**{entity.entity_type.name.capitalize()}**: `{entity.name}`"
			if self.config.lod_level.value >= LODLevel.STRUCTURE.value and entity.signature:
				entry_line = f"{list_prefix}**{entity.entity_type.name.capitalize()}**: `{entity.signature}`"
			# Special handling for comments
			elif entity.entity_type == EntityType.COMMENT and entity.content:
				comment_lines = entity.content.strip().split("\n")
				# Format as italicized blockquote
				entity_content.extend([f"{indent}> *{line.strip()}*" for line in comment_lines])
				entry_line = None  # Don't print the default entry line
			elif not entity.name and entity.entity_type == EntityType.MODULE:
				# Skip module node if it has no name (handled by file heading)
				# Don't add the list item itself
				entry_line = None  # Don't print the default entry line

			# Add the generated entry line if it wasn't skipped
			if entry_line:
				entity_content.append(entry_line)

			# Add Docstring if level is DOCS or FULL (and not a comment)
			if (
				entity.entity_type != EntityType.COMMENT
				and self.config.lod_level.value >= LODLevel.DOCS.value
				and entity.docstring
			):
				docstring_lines = entity.docstring.strip().split("\n")
				# Indent docstring relative to the list item
				entity_content.extend([f"{indent}  > {line}" for line in docstring_lines])

			# Add Content if level is FULL
			if self.config.lod_level.value >= LODLevel.FULL.value and entity.content:
				content_lang = entity.language or ""
				entity_content.append(f"{indent}  ```{content_lang}")
				# Indent content lines as well
				content_lines = entity.content.strip().split("\n")
				entity_content.extend([f"{indent}  {line}" for line in content_lines])
				entity_content.append(f"{indent}  ```")

			# Recursively format children
			for child in sorted(entity.children, key=lambda e: e.start_line):
				# Skip unknown children
				if child.entity_type != EntityType.UNKNOWN:
					entity_content.extend(format_entity_recursive(child, level + 1))

			return entity_content

		first_file = True
		for file_path, file_entities in sorted(files.items()):
			# Add a divider before each file section except the first one
			if not first_file:
				content.append("\n---")  # Horizontal rule
			first_file = False

			rel_path = file_path.name
			content.append(f"\n### {rel_path}")

			# Sort top-level entities by line number
			sorted_entities = sorted(file_entities, key=lambda e: e.start_line)

			if self.config.lod_level == LODLevel.SIGNATURES:
				# Level 1: Only top-level signatures
				for entity in sorted_entities:
					if entity.entity_type in (
						EntityType.CLASS,
						EntityType.FUNCTION,
						EntityType.METHOD,
						EntityType.INTERFACE,
						EntityType.MODULE,
					):
						content.append(f"\n#### {entity.name or '(Module Level)'}")
						if entity.signature:
							sig_lang = entity.language or ""
							content.append(f"\n```{sig_lang}")
							content.append(entity.signature)
							content.append("```")
			else:
				# Levels 2, 3, 4: Use recursive formatting
				for entity in sorted_entities:
					# Process top-level entities (usually MODULE, but could be others if file has only one class/func)
					if entity.entity_type == EntityType.MODULE:
						# If it's the module, start recursion from its children
						for child in sorted(entity.children, key=lambda e: e.start_line):
							# Skip unknown children
							if child.entity_type != EntityType.UNKNOWN:
								content.extend(format_entity_recursive(child, level=0))
					# Handle cases where the top-level entity isn't MODULE (e.g., a file with just one class)
					# Skip if unknown
					elif entity.entity_type != EntityType.UNKNOWN:
						content.extend(format_entity_recursive(entity, level=0))

		return "\n".join(content)
