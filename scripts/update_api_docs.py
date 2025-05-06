"""Script to generate the API documentation for the project."""

import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.logging import RichHandler

# --- Configuration ---
# Define a logger for this script
logger = logging.getLogger(__name__)
# Set logger level to DEBUG to see all messages
logger.setLevel(logging.DEBUG)

# Create and add RichHandler for colored console output
rich_handler = RichHandler(level=logging.DEBUG, show_time=False, show_path=False, markup=True)
logger.addHandler(rich_handler)
# Prevent logs from propagating to the root logger if basicConfig was called elsewhere
logger.propagate = False

SRC_ROOT = Path("src")
CODE_PACKAGE = "codemap"  # The main package name within src/
DOCS_API_ROOT = Path("docs/api")
MKDOCS_CONFIG_PATH = Path("mkdocs.yml")
API_NAV_TITLE = "API Reference"  # The title used in mkdocs.yml nav

# --- Helper Functions ---


def module_path_to_string(path_parts: tuple[str, ...]) -> str:
	"""Converts ('codemap', 'git', 'utils') to 'codemap.git.utils'."""
	return ".".join(path_parts)


def path_to_title(path_part: str) -> str:
	"""Converts 'commit_linter' to 'Commit Linter'."""
	return path_part.replace("_", " ").title()


def extract_module_description(init_file_path: Path) -> str | None:
	"""Extract the first line of the module docstring from an __init__.py file."""
	if not init_file_path.exists():
		return None

	try:
		content = init_file_path.read_text(encoding="utf-8")
		# Look for a docstring at the beginning of the file
		docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
		if docstring_match:
			# Extract the first line from the docstring
			docstring = docstring_match.group(1).strip()
			first_line = docstring.split("\n")[0].strip()
			return first_line if first_line else None
		return None
	except Exception as e:
		logger.warning(f"Error reading docstring from {init_file_path}: {e!s}")
		return None


def create_markdown_content(module_id: str, title: str, is_package: bool) -> str:
	"""Generates the content for a mkdocstrings markdown file."""
	options = [
		"members_order: source",
		"show_if_no_docstring: true",
		"show_signature_annotations: true",
		"separate_signature: true",
	]

	if is_package:
		# Add option to show submodules for package index files
		options.append("show_submodules: true")

	options_str = "\n".join(f"      {opt}" for opt in options)
	return f"""# {title}

::: {module_id}
    options:
{options_str}
"""


def create_package_index_content(module_id: str, title: str, description: str, children: dict) -> str:
	"""Generates a more concise index file for packages with links to sub-modules.

	Args:
		module_id: Full module ID (e.g., 'codemap.watcher')
		title: Package title (e.g., 'Watcher')
		description: Package description from docstring
		children: Dictionary of child modules with their descriptions

	Returns:
		Markdown content with module summary and links to sub-modules
	"""
	# Start with the title and description
	content = [f"# {title} Overview"]

	if description:
		content.append(f"\n{description}\n")
	else:
		content.append("\n")  # Add empty line if no description

	# Add a section about available modules if we have children
	if children:
		# Sort modules alphabetically
		sorted_modules = sorted(children.keys())

		for module_name in sorted_modules:
			# Skip internal items
			if module_name.startswith("_"):
				continue

			module_info = children.get(module_name, {})
			module_title = path_to_title(module_name)
			module_description = module_info.get("_description", "")

			# Create relative link to the module
			link_path = f"{module_name}/index.md" if module_info.get("_is_package", False) else f"{module_name}.md"

			# Add module entry with description if available
			if module_description:
				content.append(f"- [{module_title}]({link_path}) - {module_description}")
			else:
				content.append(f"- [{module_title}]({link_path})")

	return "\n".join(content)


def build_nested_nav(structure: dict[str, Any], current_rel_path: Path) -> list[Any]:
	"""Recursively builds the nested list structure for MkDocs nav."""
	nav_list = []

	logger.debug(f"Building nav for path: {current_rel_path}, keys: {list(structure.keys())}")

	# If this is a structure with children, process the children directly
	# This handles the top-level package structure which might only have _children
	children = structure.get("_children", {})
	if children and isinstance(children, dict):
		logger.debug(f"Processing children of {current_rel_path}: {list(children.keys())}")
		return build_nested_nav(children, current_rel_path)

	sorted_keys = sorted(structure.keys())

	for key in sorted_keys:
		# Skip internal marker keys explicitly
		if key in ("_is_package", "_is_file", "_children"):
			continue

		# Skip __main__ module as it's not needed in the API docs
		if key == "__main__":
			logger.debug("Skipping __main__ module in navigation")
			continue

		item = structure.get(key)
		if not isinstance(item, dict):
			continue  # Skip non-dictionary items

		title = path_to_title(key)
		item_rel_path = current_rel_path / key

		logger.debug(f"Processing key: {key}, path: {item_rel_path}")

		is_package = item.get("_is_package", False)
		is_file = item.get("_is_file", False)
		children_structure = item.get("_children", None)

		if is_package:
			index_md_path = (item_rel_path / "index.md").as_posix()
			package_nav_list = [{f"{title} Overview": index_md_path}]
			if children_structure:
				package_nav_list.extend(build_nested_nav(children_structure, item_rel_path))
			if len(package_nav_list) > 1:  # Only add package section if it has children besides Overview
				nav_list.append({title: package_nav_list})
				logger.debug(f"Added package: {title} with {len(package_nav_list)} items")
			else:  # Otherwise, just link to the overview
				nav_list.append({f"{title} Overview": index_md_path})
				logger.debug(f"Added package overview: {title}")
		elif is_file:
			md_path = item_rel_path.with_suffix(".md").as_posix()
			nav_list.append({title: md_path})
			logger.debug(f"Added file: {title} -> {md_path}")
		elif children_structure:  # Intermediate directory with children
			# Recursively build nav for children and extend the *current* list
			logger.debug(f"Processing intermediate directory: {key} with children: {list(children_structure.keys())}")
			children_nav = build_nested_nav(children_structure, item_rel_path)
			if children_nav:  # Only add if there are actually items
				nav_list.extend(children_nav)
				logger.debug(f"Added {len(children_nav)} items from {key} children")

	logger.debug(f"Returning nav_list for {current_rel_path} with {len(nav_list)} items")
	return nav_list


# --- Main Logic ---


def discover_modules(src_package_dir: Path) -> dict[str, Any]:
	"""Discovers Python modules and packages using a two-pass approach."""
	module_structure = {}
	all_py_files = list(src_package_dir.rglob("*.py"))  # Get all files first

	# Pass 1: Build the nested dictionary structure
	logger.info("Pass 1: Building directory structure...")
	for py_file in all_py_files:
		relative_path = py_file.relative_to(src_package_dir.parent)
		parts = relative_path.with_suffix("").parts

		current_level = module_structure
		for part in parts[:-1]:  # Iterate through directory parts
			node = current_level.setdefault(part, {})
			# Ensure _children exists for directory parts
			children = node.setdefault("_children", {})
			current_level = children

		# Ensure the final part exists as a placeholder dictionary for now
		filename_no_ext = parts[-1]
		if filename_no_ext != "__init__":
			current_level.setdefault(filename_no_ext, {})

			# Extract description from standalone Python file
			file_node = current_level[filename_no_ext]
			if isinstance(file_node, dict):
				description = extract_module_description(py_file)
				if description:
					file_node["_description"] = description
					logger.debug(f"Extracted description for standalone file '{filename_no_ext}': {description}")

	# Pass 2: Mark nodes as packages or files and collect descriptions
	logger.info("Pass 2: Marking packages and files...")
	for py_file in all_py_files:
		relative_path = py_file.relative_to(src_package_dir.parent)
		parts = relative_path.with_suffix("").parts
		filename_no_ext = parts[-1]

		# Find the parent node
		parent_level = module_structure
		# Navigate using .get for safety, stopping before the last part
		for part in parts[:-2]:
			parent_level = parent_level.get(part, {}).get("_children", {})
			if not isinstance(parent_level, dict):  # Safety check
				logger.error(f"Structure error navigating to parent for {relative_path}")
				parent_level = None
				break
		if parent_level is None:
			continue

		if filename_no_ext == "__init__":
			if parts[:-1]:  # Ensure it's not the top-level __init__
				parent_key = parts[-2]
				package_node = parent_level.get(parent_key)
				# ---- DEBUG LOGGING ----
				logger.debug(
					f"Marking package: parent_key='{parent_key}', node_type={type(package_node)}, node={package_node}"
				)
				# ---- END DEBUG ----
				if isinstance(package_node, dict):
					package_node["_is_package"] = True

					# Extract description from __init__.py file
					description = extract_module_description(py_file)
					if description:
						package_node["_description"] = description
						logger.debug(f"Extracted description for '{parent_key}': {description}")
				else:
					logger.warning(
						f"Could not find valid parent node '{parent_key}' to mark as package for {relative_path}"
					)

		else:
			# Find the direct parent node again for file assignment
			file_parent_level = module_structure
			for part in parts[:-1]:
				file_parent_level = file_parent_level.get(part, {}).get("_children", {})
				if not isinstance(file_parent_level, dict):  # Safety check
					logger.error(f"Structure error navigating to file parent for {relative_path}")
					file_parent_level = None
					break
			if file_parent_level is None:
				continue

			file_node = file_parent_level.get(filename_no_ext)
			if isinstance(file_node, dict):
				if "_is_package" in file_node:
					logger.warning(
						f"Naming conflict: Module file '{py_file}' has the same name as a package. Skipping file marking."
					)
				elif "_is_file" not in file_node:  # Avoid double-marking
					file_node["_is_file"] = True
					# Ensure _children is not present on a file node
					file_node.pop("_children", None)
			else:
				logger.warning(
					f"Could not find valid dictionary node '{filename_no_ext}' to mark as file for {relative_path}. Found: {type(file_node)}"
				)

	# Return the structure starting from the main package
	return module_structure.get(CODE_PACKAGE, {})


def generate_docs(structure: dict[str, Any], current_module_parts: tuple[str, ...], docs_dir: Path):
	"""Recursively generates markdown files for the discovered structure."""
	# If this structure has _children, process those directly first
	if "_children" in structure:
		# Process the children dictionary first
		children = structure.get("_children", {})
		if children and isinstance(children, dict):
			logger.debug(f"Processing children at module parts: {current_module_parts}")
			generate_docs(children, current_module_parts, docs_dir)
			return

	for key, item in structure.items():
		if key.startswith("_") or not isinstance(item, dict):
			continue

		# ---- DEBUG LOGGING (Keep for now if needed) ----
		logger.debug(f"GenerateDocs: Processing key='{key}', item_type={type(item)}, item_value={item}")
		# ---- END DEBUG ----

		title = path_to_title(key)
		new_module_parts = (*current_module_parts, key)
		module_id = module_path_to_string(new_module_parts)

		is_package = item.get("_is_package", False)
		is_file = item.get("_is_file", False)
		children_structure = item.get("_children", None)
		description = item.get("_description", "")

		if is_package:
			logger.debug(f"  -> Generating package index for: {module_id}")
			md_file_path = docs_dir / key / "index.md"
			md_file_path.parent.mkdir(parents=True, exist_ok=True)

			# Use the new concise index format for packages
			if children_structure:
				content = create_package_index_content(module_id, title, description, children_structure)
			else:
				# Fallback to standard content if no children
				content = create_markdown_content(module_id, f"{title} Overview", is_package=True)

			md_file_path.write_text(content + "\n", encoding="utf-8")
			logger.info(f"Generated: {md_file_path}")

			if children_structure:
				generate_docs(children_structure, new_module_parts, docs_dir / key)
		elif is_file:
			logger.debug(f"  -> Generating module file for: {module_id}")
			md_file_path = docs_dir / f"{key}.md"
			md_file_path.parent.mkdir(parents=True, exist_ok=True)
			content = create_markdown_content(module_id, title, is_package=False)
			md_file_path.write_text(content + "\n", encoding="utf-8")
			logger.info(f"Generated: {md_file_path}")
		elif children_structure:  # Handle intermediate directories
			logger.debug(f"  -> Recursing into intermediate directory: {key}")
			# Create directory if it doesn't exist
			child_dir = docs_dir / key
			child_dir.mkdir(parents=True, exist_ok=True)
			# Don't generate a file for the directory itself, just recurse
			generate_docs(children_structure, new_module_parts, docs_dir / key)


def create_api_docs(module_structure: dict):
	"""
	Creates API documentation for a single version.

	Args:
		module_structure: The module structure dictionary

	Returns:
		Nav structure for the API docs
	"""
	logger.info("Generating API documentation")

	# Create the directory
	docs_dir = DOCS_API_ROOT
	docs_dir.mkdir(parents=True, exist_ok=True)

	# Extract the package description
	package_description = module_structure.get("_description", "")

	# Create a concise index with links to main modules
	index_content = ["# API Reference"]

	if package_description:
		index_content.append(f"\n{package_description}\n")
	else:
		index_content.append("\n")

	# Create list of main modules with descriptions
	if "_children" in module_structure:
		children = module_structure.get("_children", {})
		# Sort keys for consistent order
		sorted_keys = sorted(children.keys())

		index_content.append("## Main Modules\n")

		for key in sorted_keys:
			# Skip internal keys and __main__
			if key.startswith("_") or key == "__main__":
				continue

			# Get the module node
			module_node = children.get(key, {})
			module_title = path_to_title(key)
			is_package = module_node.get("_is_package", False)
			link_path = f"{key}/index.md" if is_package else f"{key}.md"
			description = module_node.get("_description", "")

			# Add the module entry with description if available
			if description:
				index_content.append(f"- [{module_title}]({link_path}) - {description}")
			else:
				index_content.append(f"- [{module_title}]({link_path})")
	else:
		index_content.append("No modules found.")

	# Write the index file
	index_path = docs_dir / "index.md"
	index_path.write_text("\n".join(index_content) + "\n", encoding="utf-8")
	logger.info(f"Generated: {index_path}")

	# Generate the markdown files
	generate_docs(module_structure, (CODE_PACKAGE,), docs_dir)

	# Build the navigation structure
	api_nav = build_nested_nav(module_structure, Path("api"))

	return [
		{"Overview": "api/index.md"},
		*api_nav,
	]


def update_mkdocs_config(nav_structure: list[Any]):
	"""Updates the mkdocs.yml nav section by replacing only the API Reference section."""
	try:
		# Read the original file content
		with MKDOCS_CONFIG_PATH.open(encoding="utf-8") as f:
			content = f.read()

		# Create a backup just in case
		backup_path = MKDOCS_CONFIG_PATH.with_suffix(".bak")
		with backup_path.open("w", encoding="utf-8") as f:
			f.write(content)
		logger.info(f"Created backup at {backup_path}")

		# Split by lines for processing
		lines = content.splitlines()

		# Generate API Reference section
		api_ref_yaml = yaml.dump(nav_structure, default_flow_style=False, sort_keys=False, allow_unicode=True)
		# Create indented lines for the API Reference section
		api_ref_lines = ["- API Reference:"]
		api_ref_lines.extend(f"  {line}" for line in api_ref_yaml.splitlines() if line.strip())

		# Process file into sections
		new_lines = []
		in_nav = False
		in_api_ref = False
		api_ref_added = False
		api_ref_indent_level = 0
		i = 0

		while i < len(lines):
			line = lines[i]
			stripped = line.strip()

			# Track when we're in the nav section
			if stripped == "nav:":
				in_nav = True
				new_lines.append(line)
				i += 1
				continue

			# Detect start of API Reference section with proper indentation tracking
			if in_nav and not in_api_ref and stripped == "- API Reference:":
				in_api_ref = True
				api_ref_indent_level = len(line) - len(line.lstrip())
				logger.debug(f"Found API Reference at line {i} with indent level {api_ref_indent_level}")
				i += 1  # Skip this line
				continue

			# Skip all lines in the API Reference section until we reach the next top-level nav item
			if in_api_ref:
				current_indent = len(line) - len(line.lstrip())
				# Check if this line is still part of the API Reference section
				if stripped and current_indent > api_ref_indent_level:
					# Still in API Reference section, skip this line
					logger.debug(f"Skipping API Reference content line: {line}")
					i += 1
					continue
				# We've reached the end of the API Reference section
				# Add our new API Reference section
				if not api_ref_added:
					logger.debug("Adding new API Reference section")
					for ref_line in api_ref_lines:
						indent_spaces = " " * api_ref_indent_level
						new_lines.append(f"{indent_spaces}{ref_line}")
					api_ref_added = True

				in_api_ref = False
				# Don't increment i, continue naturally to add the next line
				if stripped:
					new_lines.append(line)
					i += 1
				continue

			# Add the line normally
			new_lines.append(line)
			i += 1

		# If we're still in API Reference section at the end, or never found it
		if in_api_ref or (in_nav and not api_ref_added):
			# Add the API Reference section at the end of nav
			logger.debug("Adding API Reference section at end of nav")
			for ref_line in api_ref_lines:
				# Use default indentation level if we never found API Reference section
				indent_level = max(0, api_ref_indent_level)
				indent_spaces = " " * indent_level
				new_lines.append(f"{indent_spaces}{ref_line}")

		# Write the updated content back to the file
		with MKDOCS_CONFIG_PATH.open("w", encoding="utf-8") as f:
			f.write("\n".join(new_lines))

		logger.info("Successfully updated mkdocs.yml API Reference section")
	except Exception:
		logger.exception("Error updating mkdocs.yml")


if __name__ == "__main__":
	src_package_dir = SRC_ROOT / CODE_PACKAGE
	if not src_package_dir.is_dir():
		logger.error(f"Source package directory not found: {src_package_dir}")
		sys.exit(1)

	# Discover all the modules
	logger.info(f"Discovering modules in: {src_package_dir}")
	module_structure = discover_modules(src_package_dir)

	if not module_structure:
		logger.warning("No modules discovered. Exiting.")
		sys.exit(0)

	# Clean existing API docs directory (optional)
	if DOCS_API_ROOT.exists():
		logger.warning(f"Removing existing API docs directory: {DOCS_API_ROOT}")
		shutil.rmtree(DOCS_API_ROOT)
	DOCS_API_ROOT.mkdir(parents=True)

	# Generate docs for a single version
	nav_structure = create_api_docs(module_structure)

	# Update the mkdocs.yml configuration
	logger.info("Updating mkdocs.yml with API structure...")
	update_mkdocs_config(nav_structure)

	logger.info("API documentation update process finished.")
