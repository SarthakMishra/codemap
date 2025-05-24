"""Code documentation generator implementation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from codemap.config.config_schema import GenSchema
from codemap.processor.lod import LODEntity, LODLevel
from codemap.processor.tree_sitter.base import EntityType

logger = logging.getLogger(__name__)

# --- Constants --- #
SMALL_RANGE_THRESHOLD = 5  # Threshold for adding range comments in link styles
MAX_SKELETON_EARLY_CALLS = 5  # Maximum number of early function calls to show in skeleton


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


def _get_consecutive_ranges(indices: list[int]) -> list[tuple[int, int]]:
	"""Groups consecutive integers into ranges.

	Args:
	    indices: List of integers to group into consecutive ranges.

	Returns:
	    List of tuples where each tuple represents a range (start, end).
	    For single values, start == end.
	"""
	if not indices:
		return []

	ranges = []
	start = indices[0]
	end = indices[0]

	for i in range(1, len(indices)):
		if indices[i] == end + 1:
			# Consecutive number, extend current range
			end = indices[i]
		else:
			# Gap found, save current range and start new one
			ranges.append((start, end))
			start = indices[i]
			end = indices[i]

	# Add the final range
	ranges.append((start, end))
	return ranges


def _add_range_styles(link_styles: list[str], indices: list[int], style_def: str) -> None:
	"""Adds link styles efficiently by grouping consecutive ranges with comments.

	Args:
	    link_styles: List to append style declarations to.
	    indices: List of edge indices to style.
	    style_def: Style definition string.
	"""
	if not indices:
		return

	sorted_indices = sorted(indices)
	ranges = _get_consecutive_ranges(sorted_indices)

	for range_start, range_end in ranges:
		if range_start == range_end:
			# Single index
			link_styles.append(f"  linkStyle {range_start} {style_def};")
		elif range_end - range_start < SMALL_RANGE_THRESHOLD:
			# Small range, list individually with comment
			link_styles.append(f"  %% Styles for edges {range_start}-{range_end}")
			link_styles.extend(f"  linkStyle {idx} {style_def};" for idx in range(range_start, range_end + 1))
		else:
			# Large range, add comment to reduce visual clutter
			link_styles.append(f"  %% Styles for edges {range_start}-{range_end} ({range_end - range_start + 1} edges)")
			link_styles.extend(f"  linkStyle {idx} {style_def};" for idx in range(range_start, range_end + 1))


class CodeMapGenerator:
	"""Generates code documentation based on LOD (Level of Detail)."""

	def __init__(self, config: GenSchema) -> None:
		"""
		Initialize the code map generator.

		Args:
		    config: Generation configuration settings

		"""
		self.config = config

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
		# Node ID counter for generating short, sequential IDs
		node_id_counter = {"count": 0}
		# Subgraph ID counter for generating short subgraph IDs
		subgraph_id_counter = {"count": 0}
		# Mapping from short node ID to entity info for debugging/comments
		node_id_to_info: dict[str, tuple[str, str, str]] = {}
		# Map entity unique IDs to their assigned node IDs to prevent duplicates
		entity_to_node_id: dict[str, str] = {}  # entity_unique_id -> node_id

		internal_paths = {str(e.metadata.get("file_path")) for e in entities if e.metadata.get("file_path")}

		def get_entity_unique_id(entity: LODEntity) -> str:
			"""Generate a unique identifier for an entity based on its properties."""
			file_path = entity.metadata.get("file_path", "unknown")
			name = entity.name or ""
			entity_type = entity.entity_type.name
			start_line = entity.start_line
			end_line = entity.end_line
			return f"{file_path}:{start_line}-{end_line}:{entity_type}:{name}"

		def get_node_id(entity: LODEntity) -> str:
			"""Generates a short, unique node ID for an entity in Mermaid diagram format.

			Uses sequential numbering (n1, n2, n3, etc.) to keep node IDs clean and readable.
			The full path and entity information is preserved in comments and labels.
			Returns the existing ID if the entity has already been assigned one.

			Args:
				entity: The entity to generate an ID for.

			Returns:
				str: A short Mermaid-compatible node ID string like 'n1', 'n2', etc.
			"""
			# Check if this entity already has an assigned node ID
			entity_unique_id = get_entity_unique_id(entity)
			if entity_unique_id in entity_to_node_id:
				return entity_to_node_id[entity_unique_id]

			node_id_counter["count"] += 1
			short_id = f"n{node_id_counter['count']}"

			# Store the mapping to prevent duplicates
			entity_to_node_id[entity_unique_id] = short_id

			# Store mapping for debugging/comments
			file_path = entity.metadata.get("file_path", "unknown_file")
			name = entity.name or entity.entity_type.name
			node_id_to_info[short_id] = (file_path, str(entity.start_line), name)

			return short_id

		def get_subgraph_id(entity: LODEntity) -> str:
			"""Generates a short, unique subgraph ID for modules and classes.

			Returns the existing ID if the entity has already been assigned one.

			Args:
				entity: The entity to generate a subgraph ID for.

			Returns:
				str: A short subgraph ID like 'sg1', 'sg2', etc.
			"""
			# Check if this entity already has an assigned node ID
			entity_unique_id = get_entity_unique_id(entity)
			if entity_unique_id in entity_to_node_id:
				return entity_to_node_id[entity_unique_id]

			subgraph_id_counter["count"] += 1
			short_id = f"sg{subgraph_id_counter['count']}"

			# Store the mapping to prevent duplicates
			entity_to_node_id[entity_unique_id] = short_id

			return short_id

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

			# Use different ID generation for subgraphs vs nodes
			if entity.entity_type in (EntityType.MODULE, EntityType.CLASS):
				entity_node_id = get_subgraph_id(entity)
			else:
				entity_node_id = get_node_id(entity)

			# Skip if already processed, but allow UNKNOWN entities to process their children
			if entity_node_id in processed_ids:
				return

			# For UNKNOWN entities, skip creating nodes but still process children
			if entity.entity_type == EntityType.UNKNOWN:
				processed_ids.add(entity_node_id)
				# Process children recursively even for UNKNOWN entities
				for child in sorted(entity.children, key=lambda e: e.start_line):
					process_entity_recursive(child, current_subgraph_id)
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

			# --- Process Children Recursively and track their node IDs --- #
			child_node_ids = []  # Track child node IDs for declare edges
			for child in sorted(entity.children, key=lambda e: e.start_line):
				# Determine what the child's node ID would be before processing
				if child.entity_type in (EntityType.MODULE, EntityType.CLASS):
					child_node_id = get_subgraph_id(child)
				else:
					child_node_id = get_node_id(child)

				# Store the child node ID for declare edges (before processing to avoid duplicates)
				child_node_ids.append((child, child_node_id))

				# Now process the child recursively
				process_entity_recursive(child, next_subgraph_id)

			# --- Define Parent Edges (Declares) using tracked child node IDs --- #
			for child, child_node_id in child_node_ids:
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

		# --- Define Style Classes ---
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

		# Check if styling is enabled
		styling_enabled = self.config.mermaid_styled

		# Track nodes by their style class for bulk application
		nodes_by_class: dict[str, list[str]] = {}

		# --- Render Logic --- #
		rendered_elements = set()  # Track IDs of things actually rendered
		output_lines = []
		class_def_lines = []  # Collect classDef commands
		style_lines = []  # Collect individual style commands for subgraphs
		used_style_keys = set()  # Track which styles (funcNode, classSubgraph etc.) are used

		# Helper function to check if a subgraph has any content after filtering
		def subgraph_has_content(subgraph_id: str) -> bool:
			"""Check if a subgraph has any content (nodes or nested subgraphs) after filtering.

			Args:
				subgraph_id: The ID of the subgraph to check

			Returns:
				True if the subgraph has any content that would be rendered, False otherwise
			"""
			if subgraph_id not in subgraph_definitions:
				return False

			_, _, contained_node_ids = subgraph_definitions[subgraph_id]

			# Check if any nodes in this subgraph would be rendered
			for node_id in contained_node_ids:
				if node_id in node_definitions:
					# If filtering is disabled, all nodes are rendered
					if not self.config.mermaid_remove_unconnected:
						return True
					# If filtering is enabled, check if node is connected
					if node_id in connected_ids:
						return True

			# Check if any nested subgraphs have content
			nested_subgraphs = [sid for sid, parent_id in subgraph_hierarchy.items() if parent_id == subgraph_id]
			return any(subgraph_has_content(nested_id) for nested_id in nested_subgraphs)

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

			# Skip empty subgraphs when filtering is enabled
			if self.config.mermaid_remove_unconnected and not subgraph_has_content(subgraph_id):
				return

			rendered_elements.add(subgraph_id)

			label, sg_type, contained_node_ids = subgraph_definitions[subgraph_id]
			subgraph_comment = get_subgraph_comment(subgraph_id)
			if subgraph_comment:
				output_lines.append(f"{indent}{subgraph_comment}")
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
					inline_comment = get_inline_comment(node_id)
					if inline_comment:
						output_lines.append(f"{indent}  {inline_comment}")
					output_lines.append(f"{indent}  {definition}")
					if node_class in style_map and styling_enabled:
						# Track nodes by class for bulk application
						if node_class not in nodes_by_class:
							nodes_by_class[node_class] = []
						nodes_by_class[node_class].append(node_id)
						used_style_keys.add(node_class)  # Track used style

			# Render nested subgraphs
			nested_subgraphs = [sid for sid, parent_id in subgraph_hierarchy.items() if parent_id == subgraph_id]
			for nested_id in sorted(nested_subgraphs):  # Sort for consistent output
				# Apply filtering if enabled - use the comprehensive content check
				if self.config.mermaid_remove_unconnected and not subgraph_has_content(nested_id):
					continue
				render_subgraph(nested_id, indent + "  ")

			output_lines.append(f"{indent}end")
			# Apply style definition to subgraph *after* end (subgraphs still use individual styles)
			if sg_type in style_map:
				style_lines.append(
					f"  style {subgraph_id} {style_map[sg_type]}"
				)  # Always use 2-space indent for styles
				used_style_keys.add(sg_type)  # Track used style

		# Helper function to create standalone comments for nodes
		def get_inline_comment(node_id: str) -> str:
			"""Generate a standalone comment for a node ID."""
			if node_id in node_id_to_info:
				file_path, line, name = node_id_to_info[node_id]
				try:
					from pathlib import Path

					filename = Path(file_path).name
				except (ValueError, AttributeError, OSError):
					filename = "unknown"
				return f"%% {filename}:{line}"
			return ""

		# Helper function to create standalone comments for subgraphs
		def get_subgraph_comment(subgraph_id: str) -> str:
			"""Generate a standalone comment for a subgraph ID."""
			if subgraph_id in entity_map:
				entity = entity_map[subgraph_id]
				file_path = entity.metadata.get("file_path", "unknown")
				try:
					from pathlib import Path

					filename = Path(file_path).name
				except (ValueError, AttributeError, OSError):
					filename = "unknown"
				return f"%% {filename}:{entity.start_line}"
			return ""

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
				inline_comment = get_inline_comment(node_id)
				if inline_comment:
					output_lines.append(f"  {inline_comment}")
				output_lines.append(f"  {definition}")
				if node_class in style_map and getattr(self.config, "mermaid_styled", True):
					# Track nodes by class for bulk application
					if nodes_by_class is not None:
						if node_class not in nodes_by_class:
							nodes_by_class[node_class] = []
						nodes_by_class[node_class].append(node_id)
					used_style_keys.add(node_class)  # Track used style

		# --- Render Top-Level Subgraphs ---
		output_lines.append("\n  %% Subgraphs")
		top_level_subgraphs = [sg_id for sg_id in subgraph_definitions if sg_id not in subgraph_hierarchy]
		for sg_id in sorted(top_level_subgraphs):
			# Apply filtering if enabled - use the comprehensive content check
			if self.config.mermaid_remove_unconnected and not subgraph_has_content(sg_id):
				continue
			render_subgraph(sg_id, "  ")  # Use 2-space base indentation for top-level subgraphs

		# --- Render Edges --- #
		output_lines.append("\n  %% Edges")
		link_styles = []
		call_edge_indices = []
		import_edge_indices = []
		declare_edge_indices = []

		# Group edges by type for better organization
		call_edges = []
		import_edges = []
		declare_edges = []
		other_edges = []

		for _i, (source_id, target_id, label, edge_type) in enumerate(edges):
			# Ensure both source and target were actually rendered (or are subgraphs that contain rendered nodes)
			source_exists = source_id in rendered_elements or source_id in subgraph_definitions
			target_exists = target_id in rendered_elements or target_id in subgraph_definitions

			if source_exists and target_exists:
				if edge_type == "call":
					edge_str = f"  {source_id} -->|{label}| {target_id}"
					call_edges.append(edge_str)
				elif edge_type == "import":
					edge_str = f"  {source_id} -.->|{label}| {target_id}"
					import_edges.append(edge_str)
				elif edge_type == "declare":
					# Make declare edges less prominent
					edge_str = f"  {source_id} --- {target_id}"  # Simple line, no label needed visually
					declare_edges.append(edge_str)
				else:  # Default or unknown edge type
					edge_str = f"  {source_id} --> {target_id}"
					other_edges.append(edge_str)

		# Track starting indices for link styles
		edge_counter = 0

		# Render call edges
		if call_edges:
			output_lines.append("  %% Call edges")
			call_edge_indices = list(range(edge_counter, edge_counter + len(call_edges)))
			edge_counter += len(call_edges)
			output_lines.extend(sorted(call_edges))

		# Render import edges
		if import_edges:
			output_lines.append("  %% Import edges")
			import_edge_indices = list(range(edge_counter, edge_counter + len(import_edges)))
			edge_counter += len(import_edges)
			output_lines.extend(sorted(import_edges))

		# Render declare edges
		if declare_edges:
			output_lines.append("  %% Declaration edges")
			declare_edge_indices = list(range(edge_counter, edge_counter + len(declare_edges)))
			edge_counter += len(declare_edges)
			output_lines.extend(sorted(declare_edges))

		# Render other edges (if any)
		if other_edges:
			output_lines.append("  %% Other edges")
			output_lines.extend(sorted(other_edges))

		# --- Apply Link Styles (Optimized with Comments for Large Ranges) --- #
		if styling_enabled and (call_edge_indices or import_edge_indices or declare_edge_indices):
			output_lines.append("\n  %% Link Styles")

			# Use helper function to add styles with range comments for better readability
			if call_edge_indices:
				link_styles.append("  %% Call edges (green)")
				_add_range_styles(link_styles, call_edge_indices, "stroke:#28a745,stroke-width:2px")

			if import_edge_indices:
				link_styles.append("  %% Import edges (yellow dashed)")
				_add_range_styles(
					link_styles, import_edge_indices, "stroke:#ffc107,stroke-width:1px,stroke-dasharray: 5 5"
				)

			if declare_edge_indices:
				link_styles.append("  %% Declaration edges (gray)")
				_add_range_styles(link_styles, declare_edge_indices, "stroke:#adb5bd,stroke-width:1px")

			output_lines.extend(link_styles)

		# --- Generate Dynamic Legend (if enabled) ---
		legend_lines = []
		legend_style_lines = []
		if self.config.mermaid_show_legend and used_style_keys and styling_enabled:
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
					# Track legend nodes by class for bulk application (except subgraph styles)
					if style_key in ["funcNode", "constNode", "varNode", "internalImportNode", "externalImportNode"]:
						if style_key not in nodes_by_class:
							nodes_by_class[style_key] = []
						nodes_by_class[style_key].append(legend_id)
					# Subgraph-style legend items still use individual styles
					elif style_key in style_map:
						legend_style_lines.append(f"  style {legend_id} {style_map[style_key]}")

			legend_lines.append("  end")
			legend_lines.append("")  # Add a blank line after legend

		# --- Assemble Final Output --- #
		mermaid_lines.extend(legend_lines)  # Add legend definitions (if any)
		mermaid_lines.extend(output_lines)  # Add main graph structure and edges

		# --- Add Class Definitions (Consolidated Style Application) --- #
		if styling_enabled and nodes_by_class:
			mermaid_lines.append("\n  %% Class Definitions")
			for style_class, node_ids in sorted(nodes_by_class.items()):
				if node_ids and style_class in style_map:
					class_def_lines.append(f"  classDef {style_class} {style_map[style_class]}")
					# Apply class to all nodes of this type
					node_list = ",".join(sorted(node_ids))
					class_def_lines.append(f"  class {node_list} {style_class}")

			mermaid_lines.extend(class_def_lines)

		# Append individual style commands for subgraphs and legend subgraph-style items
		all_style_lines = style_lines + legend_style_lines
		if styling_enabled and all_style_lines:
			mermaid_lines.append("\n  %% Individual Styles")
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

		# Helper function to get comment syntax for a language
		def get_comment_syntax(language: str) -> str:
			"""Get the appropriate comment syntax for a programming language.

			Args:
				language: The programming language name

			Returns:
				The comment prefix for that language
			"""
			language_lower = language.lower() if language else ""

			# Languages using // for comments
			if language_lower in (
				"javascript",
				"java",
				"c",
				"cpp",
				"c++",
				"csharp",
				"c#",
				"go",
				"rust",
				"php",
				"kotlin",
				"swift",
				"typescript",
			):
				return "//"
			# Languages using -- for comments
			if language_lower in ("sql", "haskell", "lua"):
				return "--"
			# Languages using % for comments
			if language_lower in ("matlab", "octave"):
				return "%"
			# Languages using ; for comments
			if language_lower in ("assembly", "asm"):
				return ";"
			# Default to # for Python, Ruby, Shell, Perl, etc.
			return "#"

		# Helper function to reconstruct code with LOD filtering
		def reconstruct_code_with_lod(entity: LODEntity, current_indent: str = "") -> list[str]:
			"""Reconstructs code content based on LOD level.

			Args:
				entity: The entity to process
				current_indent: Current indentation level

			Returns:
				List of code lines with appropriate LOD filtering
			"""
			lines = []

			# Skip certain entity types based on LOD level
			if entity.entity_type == EntityType.IMPORT and self.config.lod_level.value < LODLevel.FULL.value:
				return []  # Skip imports except at FULL level

			if entity.entity_type == EntityType.COMMENT and self.config.lod_level.value < LODLevel.FULL.value:
				return []  # Skip comments except at FULL level

			# Handle different entity types
			if entity.entity_type == EntityType.MODULE:
				# For modules, process children without adding module declaration
				for child in sorted(entity.children, key=lambda e: e.start_line):
					if child.entity_type != EntityType.UNKNOWN:
						child_lines = reconstruct_code_with_lod(child, current_indent)
						lines.extend(child_lines)
						if child_lines:  # Add spacing between entities
							lines.append("")
					else:
						# For UNKNOWN entities, check if they contain constants/variables as direct children
						# This handles cases where expression_statement wrappers contain assignment nodes
						for grandchild in sorted(child.children, key=lambda e: e.start_line):
							if grandchild.entity_type in (EntityType.CONSTANT, EntityType.VARIABLE):
								grandchild_lines = reconstruct_code_with_lod(grandchild, current_indent)
								lines.extend(grandchild_lines)
								if grandchild_lines:  # Add spacing between entities
									lines.append("")

			elif entity.entity_type in (EntityType.CLASS, EntityType.FUNCTION, EntityType.METHOD):
				# Add signature/declaration
				if entity.signature:
					lines.append(f"{current_indent}{entity.signature}:")
				elif entity.content:
					# Extract first line as signature
					first_line = entity.content.split("\n")[0].strip()
					lines.append(f"{current_indent}{first_line}:")
				else:
					# Fallback
					entity_keyword = "class" if entity.entity_type == EntityType.CLASS else "def"
					lines.append(f"{current_indent}{entity_keyword} {entity.name}:")

				# Add docstring if available and level permits (but not at SKELETON/FULL level)
				if (
					entity.docstring
					and self.config.lod_level.value >= LODLevel.DOCS.value
					and self.config.lod_level.value < LODLevel.SKELETON.value
				):
					lines.append(f'{current_indent}    """')
					lines.extend(f"{current_indent}    {doc_line}" for doc_line in entity.docstring.strip().split("\n"))
					lines.append(f'{current_indent}    """')

				# Determine indentation for children
				child_indent = current_indent + "    "

				if self.config.lod_level.value >= LODLevel.FULL.value:
					# FULL level: include full implementation
					if entity.content and entity.entity_type != EntityType.MODULE:
						content_lines = entity.content.strip().split("\n")
						# Skip the first line (signature) as we already added it
						lines.extend(f"{current_indent}{content_line}" for content_line in content_lines[1:])
				elif self.config.lod_level.value >= LODLevel.SKELETON.value:
					# SKELETON level: show full content but filter out comments
					if entity.content and entity.entity_type != EntityType.MODULE:
						content_lines = entity.content.strip().split("\n")
						# Skip the first line (signature) as we already added it
						for content_line in content_lines[1:]:
							stripped = content_line.strip()

							# Skip lines that are entirely comments
							if stripped and any(stripped.startswith(prefix) for prefix in ("#", "//", "/*")):
								continue

							# Handle inline comments by removing them
							if stripped:
								cleaned_line = content_line
								# Remove inline comments (simple approach - look for comment markers)
								for comment_prefix in ["#", "//"]:
									if comment_prefix in content_line:
										# Find the comment position (avoiding strings)
										comment_pos = content_line.find(comment_prefix)
										if comment_pos != -1:
											# Simple check: if not in quotes, it's likely a comment
											before_comment = content_line[:comment_pos]
											if (
												before_comment.count('"') % 2 == 0
												and before_comment.count("'") % 2 == 0
											):
												cleaned_line = before_comment.rstrip()
												break

								if cleaned_line.strip():  # Only add if there's actual code content
									lines.append(f"{current_indent}{cleaned_line}")
							else:  # Keep empty lines for structure
								lines.append(f"{current_indent}{content_line}")
				else:
					# SIGNATURES/STRUCTURE/DOCS: process children as signatures only
					has_children = False
					for child in sorted(entity.children, key=lambda e: e.start_line):
						if child.entity_type != EntityType.UNKNOWN:
							child_lines = reconstruct_code_with_lod(child, child_indent)
							if child_lines:
								lines.extend(child_lines)
								has_children = True
						else:
							# For UNKNOWN entities, check if they contain constants/variables as direct children
							for grandchild in sorted(child.children, key=lambda e: e.start_line):
								if grandchild.entity_type in (EntityType.CONSTANT, EntityType.VARIABLE):
									grandchild_lines = reconstruct_code_with_lod(grandchild, child_indent)
									if grandchild_lines:
										lines.extend(grandchild_lines)
										has_children = True

					# If we have content but not showing full, add truncation comment
					comment_prefix = get_comment_syntax(entity.language or "")
					if entity.content and not has_children:
						lines.append(f"{child_indent}{comment_prefix} truncated for brevity")
					elif not has_children and entity.entity_type != EntityType.CLASS:
						# For functions/methods without children, add truncation comment
						lines.append(f"{child_indent}{comment_prefix} hidden for brevity")

			elif entity.entity_type in (EntityType.VARIABLE, EntityType.CONSTANT):
				# At FULL level, show everything; at SKELETON+ levels, show the declaration
				if entity.content:
					if self.config.lod_level.value >= LODLevel.FULL.value:
						# Show complete content at FULL level
						content_lines = entity.content.strip().split("\n")
						lines.extend(f"{current_indent}{content_line}" for content_line in content_lines)
					elif self.config.lod_level.value >= LODLevel.SKELETON.value:
						# Show just the first line (declaration) at SKELETON and higher levels
						first_line = entity.content.strip().split("\n")[0]
						lines.append(f"{current_indent}{first_line}")
				# If no content is available, skip the entity entirely

			elif entity.entity_type == EntityType.IMPORT:
				if entity.content:
					lines.append(f"{current_indent}{entity.content.strip()}")
				else:
					lines.append(f"{current_indent}import {entity.name}")

			elif entity.entity_type == EntityType.COMMENT and entity.content:
				lines.extend(f"{current_indent}{comment_line}" for comment_line in entity.content.strip().split("\n"))

			return lines

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

			# Generate code content using LOD-based reconstruction
			code_lines = []
			for entity in sorted_entities:
				# Process all entities and collect code lines
				entity_lines = reconstruct_code_with_lod(entity)
				code_lines.extend(entity_lines)

			# Remove trailing empty lines
			while code_lines and not code_lines[-1].strip():
				code_lines.pop()

			# Add the code block with proper language detection
			if code_lines:
				# Get language from the first entity with language info
				lang = ""
				for entity in sorted_entities:
					if entity.language:
						lang = entity.language
						break

				content.append(f"\n```{lang}")
				content.extend(code_lines)
				content.append("```")

		return "\n".join(content)
