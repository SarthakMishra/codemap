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
	"""Escapes special characters in Mermaid node labels.

	Replaces potentially problematic characters with safer alternatives:
	- Square brackets with parentheses
	- Curly braces with parentheses
	- Double quotes with HTML entity

	Args:
	    label: The original label text to be escaped.

	Returns:
	    The escaped label text with special characters replaced.
	"""
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
		"""Generate a Mermaid diagram string for entity relationships using subgraphs."""
		# Convert config strings to lower case for case-insensitive comparison
		allowed_entities = {e.lower() for e in self.config.mermaid_entities} if self.config.mermaid_entities else None
		allowed_relationships = (
			{r.lower() for r in self.config.mermaid_relationships} if self.config.mermaid_relationships else None
		)

		# Helper to check if an entity type should be included
		def should_include_entity(entity_type: EntityType) -> bool:
			"""Determines if an entity type should be included in the diagram based on allowed entities.

			If no allowed entities are specified in the config, all entities will be included. Otherwise,
			only entities whose type matches one of the allowed entity types will be included.

			Args:
				entity_type: The type of entity to check for inclusion.

			Returns:
				bool: True if the entity should be included, False otherwise.
			"""
			if not allowed_entities:
				return True  # Include all if not specified
			return entity_type.name.lower() in allowed_entities

		def should_include_relationship(relationship_type: str) -> bool:
			"""Determines if a relationship type should be included in the diagram based on allowed relationships.

			If no allowed relationships are specified in the config, all relationships will be included. Otherwise,
			only relationships whose type matches one of the allowed relationship types will be included.

			Args:
				relationship_type: The type of relationship to check for inclusion.

			Returns:
				bool: True if the relationship should be included, False otherwise.
			"""
			if not allowed_relationships:
				return True  # Include all if not specified
			return relationship_type.lower() in allowed_relationships

		# --- Data Structures --- #
		# node_id -> (definition_line, class_name) for regular nodes
		node_definitions: dict[str, tuple[str, str]] = {}
		# subgraph_id -> (label, type, list_of_contained_node_ids)
		subgraph_definitions: dict[str, tuple[str, str, list[str]]] = {}
		# subgraph_id -> parent_subgraph_id (for nesting)
		subgraph_hierarchy: dict[str, str] = {}
		# Edges (parent_id, child_id, label, type)
		edges: list[tuple[str, str, str, str]] = []
		# Track processed entities/subgraphs to avoid duplicates
		processed_ids = set()
		# Map entity ID to entity object
		entity_map: dict[str, LODEntity] = {}
		# Track which nodes/subgraphs are connected by edges
		connected_ids = set()
		# Map simple function/method names to their full node IDs
		name_to_node_ids: dict[str, list[str]] = {}
		# Keep track of nodes defined outside any subgraph (like external imports)
		global_nodes: set[str] = set()

		internal_paths = {str(e.metadata.get("file_path")) for e in entities if e.metadata.get("file_path")}

		def get_node_id(entity: LODEntity) -> str:
			"""Generates a unique node ID for an entity in Mermaid diagram format.

			The ID is constructed from the entity's file path, start line, and name/type,
			and is sanitized to be Mermaid-compatible (alphanumeric + underscore only).

			Args:
				entity: The entity to generate an ID for.

			Returns:
				str: A Mermaid-compatible node ID string.
			"""
			file_path_str = entity.metadata.get("file_path", "unknown_file")
			base_id = f"{file_path_str}_{entity.start_line}_{entity.name or entity.entity_type.name}"
			# Ensure Mermaid compatibility (alphanumeric + underscore)
			return "".join(c if c.isalnum() else "_" for c in base_id)

		def process_entity_recursive(entity: LODEntity, current_subgraph_id: str | None = None) -> None:
			"""Recursively processes an entity to build Mermaid diagram components.

			This function handles:
			- Creating subgraphs for modules and classes
			- Creating nodes for functions, methods, variables and constants
			- Processing dependencies (imports)
			- Establishing relationships between entities
			- Recursively processing child entities

			Args:
				entity: The entity to process
				current_subgraph_id: The ID of the current subgraph context (if any)

			Returns:
				None: Modifies various data structures in the closure:
					- node_definitions: Dictionary of node definitions
					- subgraph_definitions: Dictionary of subgraph definitions
					- subgraph_hierarchy: Dictionary of subgraph parent-child relationships
					- edges: List of edges between nodes/subgraphs
					- processed_ids: Set of processed entity IDs
					- entity_map: Dictionary mapping node IDs to entities
					- connected_ids: Set of connected node/subgraph IDs
					- name_to_node_ids: Dictionary mapping names to node IDs
					- global_nodes: Set of nodes defined outside subgraphs
			"""
			nonlocal processed_ids, connected_ids, global_nodes

			entity_node_id = get_node_id(entity)

			if entity.entity_type == EntityType.UNKNOWN or entity_node_id in processed_ids:
				return

			processed_ids.add(entity_node_id)
			entity_map[entity_node_id] = entity
			include_this_entity = should_include_entity(entity.entity_type)

			next_subgraph_id = current_subgraph_id

			# --- Handle Subgraphs (Module, Class) --- #
			if entity.entity_type in (EntityType.MODULE, EntityType.CLASS) and include_this_entity:
				subgraph_label = _escape_mermaid_label(
					entity.name or Path(entity.metadata.get("file_path", "unknown")).name
				)
				subgraph_type = "moduleSubgraph" if entity.entity_type == EntityType.MODULE else "classSubgraph"
				if entity.entity_type == EntityType.MODULE and current_subgraph_id:  # Nested module
					subgraph_type = "submoduleSubgraph"

				subgraph_definitions[entity_node_id] = (subgraph_label, subgraph_type, [])
				if current_subgraph_id:
					subgraph_hierarchy[entity_node_id] = current_subgraph_id
				# Mark the container as potentially connected if it has children or dependencies
				connected_ids.add(entity_node_id)
				next_subgraph_id = entity_node_id  # Children belong to this new subgraph

			# --- Handle Nodes (Functions, Methods, Vars, Consts, Imports) --- #
			elif include_this_entity and entity.entity_type != EntityType.IMPORT:  # Imports handled separately
				node_definition = ""
				node_class = ""
				entity_type_name = str(entity.entity_type.name).lower()
				label = _escape_mermaid_label(entity.name or f"({entity_type_name})")

				if entity.entity_type in (EntityType.FUNCTION, EntityType.METHOD):
					node_definition = f'{entity_node_id}("{label}")'
					node_class = "funcNode"
				elif entity.entity_type == EntityType.CONSTANT:
					node_definition = f'{entity_node_id}["{label}"]'
					node_class = "constNode"
				elif entity.entity_type == EntityType.VARIABLE:
					node_definition = f'{entity_node_id}["{label}"]'
					node_class = "varNode"
				# Add other types if needed

				if node_definition and entity_node_id not in node_definitions:
					node_definitions[entity_node_id] = (node_definition, node_class)
					if current_subgraph_id:
						subgraph_definitions[current_subgraph_id][2].append(entity_node_id)
					else:
						# Should not happen often if root is module, but handle just in case
						global_nodes.add(entity_node_id)

			# --- Add to Name Map (for call edges) --- #
			if entity.entity_type in (EntityType.FUNCTION, EntityType.METHOD):
				name = entity.name
				if name:
					if name not in name_to_node_ids:
						name_to_node_ids[name] = []
					name_to_node_ids[name].append(entity_node_id)

			# --- Process Dependencies (Imports) --- #
			dependencies = entity.metadata.get("dependencies", [])
			# Imports are associated with the module/class they are in, or globally if nowhere else

			if dependencies and should_include_relationship("imports"):
				for dep in dependencies:
					is_external = not dep.startswith(".") and not any(
						dep.startswith(str(p).replace("\\", "/"))
						for p in internal_paths
						if p  # Handle path separators
					)
					dep_id = "dep_" + "".join(c if c.isalnum() else "_" for c in dep)
					dep_label = _escape_mermaid_label(dep)

					if dep_id not in node_definitions and dep_id not in processed_ids:
						processed_ids.add(dep_id)  # Mark as processed to avoid duplicate definitions
						dep_class = "externalImportNode" if is_external else "internalImportNode"
						node_shape = f'(("{dep_label}"))' if is_external else f'["{dep_label}"]'
						node_definitions[dep_id] = (f"{dep_id}{node_shape}", dep_class)
						global_nodes.add(dep_id)  # Imports are defined globally

					# Add edge from the importing *container* (subgraph) to the dependency
					# Use the entity_node_id of the *module* or *class* for the source of the import edge
					source_node_id = (
						entity_node_id
						if entity.entity_type in (EntityType.MODULE, EntityType.CLASS)
						else current_subgraph_id
					)
					if source_node_id and source_node_id != dep_id:  # Check source_node_id validity
						edge_tuple = (source_node_id, dep_id, "imports", "import")
						if edge_tuple not in edges:
							edges.append(edge_tuple)
							connected_ids.add(source_node_id)
							connected_ids.add(dep_id)

			# --- Process Children Recursively --- #
			for child in sorted(entity.children, key=lambda e: e.start_line):
				process_entity_recursive(child, next_subgraph_id)

				# --- Define Parent Edge (Declares) --- #
				# Edge from container (subgraph) to child node/subgraph
				child_node_id = get_node_id(child)
				if (
					next_subgraph_id  # Ensure there is a parent subgraph
					and child.entity_type != EntityType.UNKNOWN
					and child_node_id in processed_ids  # Ensure child was processed (not filtered out)
					and should_include_relationship("declares")
				):
					# Only add edge if child is *directly* contained (node or subgraph)
					is_child_node = child_node_id in node_definitions
					is_child_subgraph = child_node_id in subgraph_definitions

					if is_child_node or is_child_subgraph:
						edge_tuple = (next_subgraph_id, child_node_id, "declares", "declare")
						if edge_tuple not in edges:
							edges.append(edge_tuple)
							connected_ids.add(next_subgraph_id)
							connected_ids.add(child_node_id)

		# --- Main Processing Loop --- #
		for entity in entities:
			# Start processing from top-level modules
			if entity.entity_type == EntityType.MODULE and entity.metadata.get("file_path"):
				process_entity_recursive(entity, current_subgraph_id=None)

		# --- Define Call Edges --- #
		if should_include_relationship("calls"):
			for caller_node_id, caller_entity in entity_map.items():
				# Check if the caller is a function/method node that was actually defined
				if caller_node_id in node_definitions and caller_entity.entity_type in (
					EntityType.FUNCTION,
					EntityType.METHOD,
				):
					calls = caller_entity.metadata.get("calls", [])
					for called_name in calls:
						# Try matching full name first, then simple name
						possible_target_ids = []
						if (
							called_name in name_to_node_ids
						):  # Full name match? (e.g., class.method) - Less likely with simple parsing
							possible_target_ids.extend(name_to_node_ids[called_name])
						else:
							simple_called_name = called_name.split(".")[-1]
							if simple_called_name in name_to_node_ids:
								possible_target_ids.extend(name_to_node_ids[simple_called_name])

						for target_node_id in possible_target_ids:
							# Ensure target is also a defined node and not the caller itself
							if target_node_id in node_definitions and caller_node_id != target_node_id:
								edge_tuple = (caller_node_id, target_node_id, "calls", "call")
								if edge_tuple not in edges:
									edges.append(edge_tuple)
									connected_ids.add(caller_node_id)
									connected_ids.add(target_node_id)

		# --- Assemble Final Mermaid String --- #
		mermaid_lines = ["graph LR"]  # Or TD for Top-Down if preferred

		# --- Define Style Strings (Instead of classDef) ---
		style_map = {
			# Node Styles
			"funcNode": "fill:#007bff,stroke:#FFF,stroke-width:1px,color:white",  # Blue
			"constNode": "fill:#6f42c1,stroke:#FFF,stroke-width:1px,color:white",  # Purple
			"varNode": "fill:#fd7e14,stroke:#FFF,stroke-width:1px,color:white",  # Orange
			"internalImportNode": "fill:#20c997,stroke:#FFF,stroke-width:1px,color:white",  # Teal
			"externalImportNode": "fill:#ffc107,stroke:#333,stroke-width:1px,color:#333",  # Yellow
			# Subgraph Styles
			"moduleSubgraph": "fill:#121630,color:#FFF",  # Dark Grey BG
			"submoduleSubgraph": "fill:#2a122e,color:#FFF",  # Lighter Grey BG
			"classSubgraph": "fill:#100f5e,color:#FFF",  # Light Green BG
		}

		# --- Render Logic --- #
		rendered_elements = set()  # Track IDs of things actually rendered
		output_lines = []
		style_lines = []  # Collect style commands separately
		used_style_keys = set()  # Track which styles (funcNode, classSubgraph etc.) are used

		# Function to recursively render subgraphs and their nodes
		def render_subgraph(subgraph_id: str, indent: str = "") -> None:
			"""Recursively renders a Mermaid subgraph and its contents.

			Args:
				subgraph_id: The ID of the subgraph to render
				indent: String used for indentation in the output (default: "")

			Returns:
				None: Output is written to output_lines and style_lines lists

			Side Effects:
				- Adds to rendered_elements set to track rendered items
				- Appends lines to output_lines for Mermaid graph definition
				- Appends style commands to style_lines
				- Updates used_style_keys with any styles actually used
			"""
			if subgraph_id in rendered_elements:
				return
			rendered_elements.add(subgraph_id)

			label, sg_type, contained_node_ids = subgraph_definitions[subgraph_id]
			output_lines.append(f'{indent}subgraph {subgraph_id}["{label}"]')
			output_lines.append(f"{indent}  direction LR")  # Or TD

			# Render nodes inside this subgraph
			for node_id in contained_node_ids:
				if node_id in node_definitions:
					# Apply filtering if enabled
					if self.config.mermaid_remove_unconnected and node_id not in connected_ids:
						continue
					if node_id in rendered_elements:
						continue  # Should not happen, but safeguard
					rendered_elements.add(node_id)

					definition, node_class = node_definitions[node_id]
					output_lines.append(f"{indent}  {definition}")
					if node_class in style_map:
						style_lines.append(f"{indent}  style {node_id} {style_map[node_class]}")
						used_style_keys.add(node_class)  # Track used style

			# Render nested subgraphs
			nested_subgraphs = [sid for sid, parent_id in subgraph_hierarchy.items() if parent_id == subgraph_id]
			for nested_id in sorted(nested_subgraphs):  # Sort for consistent output
				# Apply filtering if enabled - check if the subgraph itself or any node inside it is connected
				is_nested_connected = subgraph_id in connected_ids or any(
					nid in connected_ids for nid in subgraph_definitions[nested_id][2]
				)
				if self.config.mermaid_remove_unconnected and not is_nested_connected:
					continue
				render_subgraph(nested_id, indent + "  ")

			output_lines.append(f"{indent}end")
			# Apply style definition to subgraph *after* end
			if sg_type in style_map:
				style_lines.append(f"{indent}style {subgraph_id} {style_map[sg_type]}")
				used_style_keys.add(sg_type)  # Track used style

		# --- Define Global Nodes (Imports primarily) ---
		output_lines.append("\n  %% Global Nodes")
		for node_id in sorted(global_nodes):
			if node_id in node_definitions:
				# Apply filtering if enabled
				if self.config.mermaid_remove_unconnected and node_id not in connected_ids:
					continue
				if node_id in rendered_elements:
					continue
				rendered_elements.add(node_id)

				definition, node_class = node_definitions[node_id]
				output_lines.append(f"  {definition}")
				if node_class in style_map:
					style_lines.append(f"  style {node_id} {style_map[node_class]}")
					used_style_keys.add(node_class)  # Track used style

		# --- Render Top-Level Subgraphs ---
		output_lines.append("\n  %% Subgraphs")
		top_level_subgraphs = [sg_id for sg_id in subgraph_definitions if sg_id not in subgraph_hierarchy]
		for sg_id in sorted(top_level_subgraphs):
			# Apply filtering if enabled - check if the subgraph itself or any node inside it is connected
			is_sg_connected = sg_id in connected_ids or any(
				nid in connected_ids for nid in subgraph_definitions[sg_id][2]
			)
			if self.config.mermaid_remove_unconnected and not is_sg_connected:
				continue
			render_subgraph(sg_id)

		# --- Render Edges --- #
		output_lines.append("\n  %% Edges")
		link_styles = []
		call_edge_indices = []
		import_edge_indices = []
		declare_edge_indices = []

		filtered_edges = []
		for _i, (source_id, target_id, label, edge_type) in enumerate(edges):
			# Ensure both source and target were actually rendered (or are subgraphs that contain rendered nodes)
			source_exists = source_id in rendered_elements or source_id in subgraph_definitions
			target_exists = target_id in rendered_elements or target_id in subgraph_definitions

			if source_exists and target_exists:
				edge_str = ""
				if edge_type == "import":
					edge_str = f"  {source_id} -.->|{label}| {target_id}"
					import_edge_indices.append(len(filtered_edges))  # Index in the filtered list
				elif edge_type == "call":
					edge_str = f"  {source_id} -->|{label}| {target_id}"
					call_edge_indices.append(len(filtered_edges))
				elif edge_type == "declare":
					# Make declare edges less prominent
					edge_str = f"  {source_id} --- {target_id}"  # Simple line, no label needed visually
					declare_edge_indices.append(len(filtered_edges))
				else:  # Default or unknown edge type
					edge_str = f"  {source_id} --> {target_id}"

				if edge_str:
					filtered_edges.append(edge_str)

		output_lines.extend(sorted(filtered_edges))  # Sort for consistency

		# --- Apply Link Styles --- #
		if call_edge_indices or import_edge_indices or declare_edge_indices:
			output_lines.append("\n  %% Link Styles")
			link_styles.extend(
				[f"  linkStyle {idx} stroke:#28a745,stroke-width:2px;" for idx in call_edge_indices]
			)  # Green
			link_styles.extend(
				[
					f"  linkStyle {idx} stroke:#ffc107,stroke-width:1px,stroke-dasharray: 5 5;"
					for idx in import_edge_indices
				]
			)  # Yellow dashed
			link_styles.extend(
				[f"  linkStyle {idx} stroke:#adb5bd,stroke-width:1px;" for idx in declare_edge_indices]
			)  # Gray thin

			output_lines.extend(link_styles)

		# --- Generate Dynamic Legend (if enabled) ---
		legend_lines = []
		legend_style_lines = []
		if self.config.mermaid_show_legend and used_style_keys:
			legend_lines.append("\n  %% Legend")
			legend_lines.append("  subgraph Legend")
			legend_lines.append("    direction LR")

			# Define all possible legend items and their corresponding style keys
			legend_item_definitions = {
				"legend_module": ("moduleSubgraph", '["Module/File"]'),
				"legend_submodule": ("submoduleSubgraph", '["Sub-Module"]'),
				"legend_class": ("classSubgraph", '["Class"]'),
				"legend_func": ("funcNode", '("Function/Method")'),
				"legend_const": ("constNode", '["Constant"]'),
				"legend_var": ("varNode", '["Variable"]'),
				"legend_import_int": ("internalImportNode", '["Internal Import"]'),
				"legend_import_ext": ("externalImportNode", '(("External Import"))'),
			}

			for legend_id, (style_key, definition) in legend_item_definitions.items():
				# Only add legend item if its corresponding style was actually used in the graph
				if style_key in used_style_keys:
					legend_lines.append(f"    {legend_id}{definition}")
					# Also add its style command to the list of styles
					if style_key in style_map:
						legend_style_lines.append(f"  style {legend_id} {style_map[style_key]}")

			legend_lines.append("  end")
			legend_lines.append("")  # Add a blank line after legend

		# --- Assemble Final Output --- #
		mermaid_lines.extend(legend_lines)  # Add legend definitions (if any)
		mermaid_lines.extend(output_lines)  # Add main graph structure and edges

		# Append all collected style commands at the end
		all_style_lines = style_lines + legend_style_lines
		if all_style_lines:
			mermaid_lines.append("\n  %% Styles")
			mermaid_lines.extend(sorted(all_style_lines))

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
		target_path_str = metadata.get("target_path", "")
		original_path = metadata.get("original_path", "")
		command_arg = metadata.get("command_arg", "")

		# Debug logging to see what values we're receiving
		logger.debug(
			f"Metadata values for heading: "
			f"command_arg='{command_arg}', "
			f"original_path='{original_path}', "
			f"target_path='{target_path_str}'"
		)

		# Use the exact command argument if available
		if command_arg:
			repo_name = command_arg
		# Fall back to original path if available
		elif original_path:
			repo_name = original_path
		# Further fallback to just the directory name
		elif target_path_str:
			repo_name = Path(target_path_str).name
		else:
			repo_name = metadata.get("name", "Repository")

		content.append(f"# `{repo_name}` Documentation")
		content.append(f"\nGenerated on: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")

		if "description" in metadata:
			content.append("\n" + metadata.get("description", ""))

		# Add repository statistics
		if "stats" in metadata:
			stats = metadata["stats"]
			content.append("\n## Document Statistics")
			content.append(f"- Total files scanned: {stats.get('total_files_scanned', 0)}")
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

		# Add table of contents for the scanned files
		content.append("\n## Scanned Files")

		# Group entities by file
		files: dict[Path, list[LODEntity]] = {}

		# Get the target path from metadata
		target_path_str = metadata.get("target_path", "")
		target_path = Path(target_path_str) if target_path_str else None

		for entity in entities:
			file_path = Path(entity.metadata.get("file_path", ""))
			if not file_path.name:
				continue

			if file_path not in files:
				files[file_path] = []
			files[file_path].append(entity)

		# Create TOC entries with properly formatted relative paths
		for i, file_path in enumerate(sorted(files.keys()), 1):
			# Get path relative to the target directory
			try:
				if target_path and target_path.exists():
					# Get the relative path from the target directory
					rel_path = file_path.relative_to(target_path)

					# Format the path with a leading slash for files directly in the target directory
					rel_path_str = f"/{rel_path}"

					# Create a clean anchor ID by converting to lowercase and removing all special characters
					# including underscores, to create standard anchor IDs
					filename = file_path.name
					clean_filename = "".join(c.lower() if c.isalnum() else "" for c in filename)

					# Handle paths with subdirectories
					if len(rel_path.parts) > 1:
						# Get directory name and filename
						directory = rel_path.parts[-2]  # Last directory before the file
						anchor = f"{directory}{clean_filename}"  # e.g., "raginitpy"
					else:
						# Just use the clean filename for files at root
						anchor = clean_filename
				else:
					# Fall back to just the filename if target path is not available
					rel_path_str = f"/{file_path.name}"
					clean_filename = "".join(c.lower() if c.isalnum() else "" for c in file_path.name)
					anchor = clean_filename

				content.append(f"{i}. [{rel_path_str}](#{anchor})")
			except ValueError:
				# If relative_to fails, just use the filename
				rel_path_str = f"/{file_path.name}"
				clean_filename = "".join(c.lower() if c.isalnum() else "" for c in file_path.name)
				content.append(f"{i}. [{rel_path_str}](#{clean_filename})")

		# Add code documentation grouped by file
		content.append("\n## Code Documentation")

		# Helper function to format a single entity recursively
		def format_entity_recursive(entity: LODEntity, level: int) -> list[str]:
			"""Recursively formats an entity and its children into markdown documentation.

			Args:
				entity: The entity to format
				level: The current indentation level in the hierarchy

			Returns:
				A list of markdown-formatted strings representing the entity and its children
			"""
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
				# Format docstring lines with proper indentation
				entity_content.append(f"{indent}  >")
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
		for i, (file_path, file_entities) in enumerate(sorted(files.items()), 1):
			# Add a divider before each file section except the first one
			if not first_file:
				content.append("\n---")  # Horizontal rule
			first_file = False

			# Get path relative to the target directory
			try:
				if target_path and target_path.exists():
					# Get the relative path from the target directory
					rel_path = file_path.relative_to(target_path)

					# Format the path with a leading slash for files directly in the target directory
					rel_path_str = f"/{rel_path}"

					# Create a clean anchor ID by converting to lowercase and removing all special characters
					# including underscores, to create standard anchor IDs
					filename = file_path.name
					clean_filename = "".join(c.lower() if c.isalnum() else "" for c in filename)

					# Handle paths with subdirectories
					if len(rel_path.parts) > 1:
						# Get directory name and filename
						directory = rel_path.parts[-2]  # Last directory before the file
						anchor = f"{directory}{clean_filename}"  # e.g., "raginitpy"
					else:
						# Just use the clean filename for files at root
						anchor = clean_filename
				else:
					# Fall back to just the filename if target path is not available
					rel_path_str = f"/{file_path.name}"
					clean_filename = "".join(c.lower() if c.isalnum() else "" for c in file_path.name)
					anchor = clean_filename

				# Add a custom ID to the heading to match our anchor
				content.append(f"\n### {i}. {rel_path_str}")
			except ValueError:
				# If relative_to fails, just use the filename
				rel_path_str = f"/{file_path.name}"
				clean_filename = "".join(c.lower() if c.isalnum() else "" for c in file_path.name)
				content.append(f"\n### {i}. {rel_path_str}")

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
