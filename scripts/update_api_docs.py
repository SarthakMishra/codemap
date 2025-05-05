"""Script to generate the API documentation for the project."""

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from packaging import version
from rich.logging import RichHandler

# --- Configuration ---
# Remove basicConfig, we'll configure the handler directly
# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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

# GitHub repository information
GITHUB_REPO_URL = "https://github.com/SarthakMishra/codemap"
MAIN_BRANCH = "main"
DEV_BRANCH = "dev"

# --- Helper Functions ---


def get_git_tags() -> tuple[str, str]:
	"""
	Get the latest stable and pre-release versions from Git tags.

	Returns:
		tuple: (stable_version, prerelease_version)
	"""
	try:
		# Get all tags sorted by version
		result = subprocess.run(["git", "tag"], capture_output=True, text=True, check=True)

		if result.returncode != 0:
			logger.warning("Failed to get Git tags")
			return "1.0.0", "0.1.0"

		tags = result.stdout.strip().split("\n")

		# Filter out empty tags
		tags = [tag.strip() for tag in tags if tag.strip()]

		if not tags:
			logger.warning("No Git tags found")
			return "1.0.0", "0.1.0"

		# Parse versions and separate into stable and pre-release
		stable_versions = []
		prerelease_versions = []

		for tag in tags:
			# Remove 'v' prefix if present
			clean_tag = tag.removeprefix("v")

			try:
				parsed_version = version.parse(clean_tag)

				if parsed_version.is_prerelease:
					prerelease_versions.append((parsed_version, tag))
				else:
					stable_versions.append((parsed_version, tag))
			except version.InvalidVersion:
				logger.warning(f"Invalid version tag: {tag}")

		# Get latest versions
		latest_stable = "1.0.0"
		latest_prerelease = "0.1.0"

		if stable_versions:
			stable_versions.sort(reverse=True)
			latest_stable = stable_versions[0][1]

		if prerelease_versions:
			prerelease_versions.sort(reverse=True)
			latest_prerelease = prerelease_versions[0][1]

		# Ensure version string format is consistent (with v prefix)
		latest_stable = latest_stable if latest_stable.startswith("v") else f"v{latest_stable}"
		latest_prerelease = latest_prerelease if latest_prerelease.startswith("v") else f"v{latest_prerelease}"

		logger.info(f"Latest stable version: {latest_stable}")
		logger.info(f"Latest pre-release version: {latest_prerelease}")

		return latest_stable, latest_prerelease

	except (subprocess.SubprocessError, subprocess.CalledProcessError) as e:
		logger.warning(f"Error getting Git tags: {e}")
		return "1.0.0", "0.1.0"


def get_version_info() -> tuple[str, bool]:
	"""
	Extract version information from __init__.py and determine if we're on dev branch.

	Returns:
		tuple: (version_string, is_dev_branch)
	"""
	init_file = SRC_ROOT / CODE_PACKAGE / "__init__.py"
	if not init_file.exists():
		logger.warning(f"Could not find {init_file} for version information")
		return "unknown", False

	try:
		# Read the __init__.py file
		content = init_file.read_text(encoding="utf-8")

		# Extract version
		version_match = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', content)
		version = version_match.group(1) if version_match else "unknown"

		# Determine current branch
		try:
			result = subprocess.run(
				["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
			)
			current_branch = result.stdout.strip()
			is_dev = current_branch == DEV_BRANCH or any(x in version.lower() for x in ["dev", "alpha", "beta", "rc"])

			logger.info(f"Detected version: {version} on branch: {current_branch}")
			return version, is_dev

		except (subprocess.SubprocessError, subprocess.CalledProcessError) as e:
			logger.warning(f"Error getting git branch: {e}")
			# Fall back to checking if version has dev, alpha, beta in it
			is_dev = any(x in version.lower() for x in ["dev", "alpha", "beta", "rc"])
			return version, is_dev

	except Exception as e:
		logger.warning(f"Error extracting version information: {e}")
		return "unknown", False


def get_remote_file_url(path: str, branch: str) -> str:
	"""Get the URL to the file in the GitHub repository."""
	return f"{GITHUB_REPO_URL}/blob/{branch}/{path}"


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


def create_markdown_content(module_id: str, title: str, is_package: bool, branch: str) -> str:
	"""Generates the content for a mkdocstrings markdown file."""
	# Figure out the path to the source file
	parts = module_id.split(".")
	rel_path = "/".join(parts[1:])  # Skip the top-level package name

	if is_package:
		source_path = (
			f"{SRC_ROOT}/{parts[0]}/{rel_path}/__init__.py" if rel_path else f"{SRC_ROOT}/{parts[0]}/__init__.py"
		)
	else:
		source_path = f"{SRC_ROOT}/{parts[0]}/{rel_path}.py" if rel_path else f"{SRC_ROOT}/{parts[0]}.py"

	# Generate source link
	source_url = get_remote_file_url(source_path, branch)

	# Basic options common to all
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

[View Source Code]({source_url})
"""


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


def generate_docs(structure: dict[str, Any], current_module_parts: tuple[str, ...], docs_dir: Path, branch: str):
	"""Recursively generates markdown files for the discovered structure."""
	# If this structure has _children, process those directly first
	if "_children" in structure:
		# Process the children dictionary first
		children = structure.get("_children", {})
		if children and isinstance(children, dict):
			logger.debug(f"Processing children at module parts: {current_module_parts}")
			generate_docs(children, current_module_parts, docs_dir, branch)
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

		if is_package:
			logger.debug(f"  -> Generating package index for: {module_id}")
			md_file_path = docs_dir / key / "index.md"
			md_file_path.parent.mkdir(parents=True, exist_ok=True)
			content = create_markdown_content(module_id, f"{title} Overview", is_package=True, branch=branch)
			md_file_path.write_text(content + "\n", encoding="utf-8")
			logger.info(f"Generated: {md_file_path}")
			if children_structure:
				generate_docs(children_structure, new_module_parts, docs_dir / key, branch)
		elif is_file:
			logger.debug(f"  -> Generating module file for: {module_id}")
			md_file_path = docs_dir / f"{key}.md"
			md_file_path.parent.mkdir(parents=True, exist_ok=True)
			content = create_markdown_content(module_id, title, is_package=False, branch=branch)
			md_file_path.write_text(content + "\n", encoding="utf-8")
			logger.info(f"Generated: {md_file_path}")
		elif children_structure:  # Handle intermediate directories
			logger.debug(f"  -> Recursing into intermediate directory: {key}")
			# Create directory if it doesn't exist
			child_dir = docs_dir / key
			child_dir.mkdir(parents=True, exist_ok=True)
			# Don't generate a file for the directory itself, just recurse
			generate_docs(children_structure, new_module_parts, docs_dir / key, branch)


def create_version_docs(stable_version: str, prerelease_version: str, module_structure: dict):
	"""
	Creates API documentation for both stable and pre-release versions.

	Args:
		stable_version: The stable version string
		prerelease_version: The pre-release version string
		module_structure: The module structure dictionary

	Returns:
		Dict with nav structures for both versions
	"""
	versions_nav = {}

	# Define version-specific directories and info
	versions_info = [
		{
			"title": "Stable",
			"display_title": f"Stable ({stable_version.lstrip('v')})",
			"version": stable_version,
			"branch": MAIN_BRANCH,
			"dir": DOCS_API_ROOT / "stable",
			"other_display_title": f"Pre-release ({prerelease_version.lstrip('v')})",
		},
		{
			"title": "Pre-release",
			"display_title": f"Pre-release ({prerelease_version.lstrip('v')})",
			"version": prerelease_version,
			"branch": DEV_BRANCH,
			"dir": DOCS_API_ROOT / "pre-release",
			"other_display_title": f"Stable ({stable_version.lstrip('v')})",
		},
	]

	# Process each version
	for version_info in versions_info:
		title = version_info["title"]
		display_title = version_info["display_title"]
		version = version_info["version"]
		branch = version_info["branch"]
		docs_dir = version_info["dir"]
		other_display_title = version_info["other_display_title"]

		logger.info(f"Generating documentation for {display_title}")

		# Create the directory for this version
		docs_dir.mkdir(parents=True, exist_ok=True)

		# Create list of main modules with descriptions
		main_modules_links = []
		if "_children" in module_structure:
			children = module_structure.get("_children", {})
			# Sort the keys to ensure consistent order
			for key in sorted(children.keys()):
				# Skip internal keys and __main__
				if key.startswith("_") or key == "__main__":
					continue

				# Get the module node
				module_node = children.get(key, {})

				# Create a link for each top-level module with description if available
				module_title = path_to_title(key)

				# Determine the correct link path based on whether it's a package
				is_package = module_node.get("_is_package", False)

				link_path = f"{key}/index.md" if is_package else f"{key}.md"

				# Check if this module has a description
				description = module_node.get("_description", "")
				if description:
					main_modules_links.append(f"- [{module_title}]({link_path}) - {description}")
				else:
					main_modules_links.append(f"- [{module_title}]({link_path})")

		module_links_text = "\n".join(main_modules_links) if main_modules_links else "No modules found."

		index_content = f"""# {display_title}

This section provides the auto-generated API documentation for the `{CODE_PACKAGE}` package.

## Version Information
- **Version:** {version.lstrip("v")}
- **Branch:** [{branch}]({get_remote_file_url(f"{SRC_ROOT}/{CODE_PACKAGE}", branch)})
- **Switch to:** [{other_display_title}](../{"stable" if "Pre-release" in title else "pre-release"}/index.md)

Navigate through the modules using the sidebar to explore the full API documentation.

## Main Modules

{module_links_text}
"""
		index_path = docs_dir / "index.md"
		index_path.write_text(index_content + "\n", encoding="utf-8")
		logger.info(f"Generated: {index_path}")

		# Generate the markdown files for this version
		generate_docs(module_structure, (CODE_PACKAGE,), docs_dir, branch)

		# Build the navigation structure for this version
		# For the path, use a relative path that works with the directory layout
		# Include 'api/' prefix in the path
		version_path = Path("api") / Path(docs_dir.name)
		api_nav = build_nested_nav(module_structure, version_path)

		# Prepare the nav structure including the overview
		version_nav_content = [
			{"Overview": (version_path / "index.md").as_posix()},
			*api_nav,
		]

		# Add to the versions_nav dictionary using the title (without version info) as key
		# This ensures we have "Stable" and "Pre-release" as section titles
		versions_nav[title] = version_nav_content

	return versions_nav


def create_api_index(stable_version: str, prerelease_version: str):
	"""
	Creates a top-level index.md file for the API Reference section.

	Args:
		stable_version: The stable version string
		prerelease_version: The pre-release version string
	"""
	logger.info("Generating top-level API index file")

	# Create the content for the index file
	content = f"""# API Reference

This section contains the auto-generated API documentation for the `{CODE_PACKAGE}` package.

## Available Versions

- [Stable ({stable_version.lstrip("v")})](stable/index.md) - Documentation for the latest stable release
- [Pre-release ({prerelease_version.lstrip("v")})](pre-release/index.md) - Documentation for the upcoming release

Choose a version from the navigation menu or the links above to explore the API documentation.
"""

	# Write the index file
	index_path = DOCS_API_ROOT / "index.md"
	index_path.write_text(content + "\n", encoding="utf-8")
	logger.info(f"Generated top-level index: {index_path}")


def update_mkdocs_config(versions_nav_structure: dict[str, list[Any]]):
	"""Updates the mkdocs.yml nav section, handling the mermaid tag safely."""
	mermaid_tag = "!!python/name:mermaid2.fence_mermaid_custom"
	placeholder = "__MERMAID_FORMAT_PLACEHOLDER__"

	raw_yaml_content = ""
	try:
		with MKDOCS_CONFIG_PATH.open(encoding="utf-8") as f:
			raw_yaml_content = f.read()
	except FileNotFoundError:
		logger.exception(f"mkdocs.yml not found at {MKDOCS_CONFIG_PATH}")
		return
	except OSError:
		logger.exception("Error reading mkdocs.yml file")
		return

	# Replace tag with placeholder before parsing (NO quotes)
	yaml_to_parse = raw_yaml_content.replace(mermaid_tag, placeholder)

	try:
		config_data = yaml.safe_load(yaml_to_parse)
	except yaml.YAMLError:
		logger.exception("Error parsing mkdocs.yml after placeholder replacement")
		return

	if not isinstance(config_data, dict) or "nav" not in config_data:
		logger.error("Invalid mkdocs.yml format: Missing 'nav' section.")
		return

	nav = config_data.get("nav", [])

	# Create the complete API reference section with all versions
	api_sections = [
		# Add the Overview as the first item
		{"Overview": "api/index.md"},
	]

	# Add version sections
	for version_title, version_content in versions_nav_structure.items():
		api_sections.append({version_title: version_content})

	# Find the 'Home' section first - that's where our API Reference needs to be
	for i, item in enumerate(nav):
		if isinstance(item, dict) and "Home" in item:
			home_section = item["Home"]

			# Look for API Reference within Home section
			api_ref_index = None
			for j, subsection in enumerate(home_section):
				if isinstance(subsection, dict) and "API Reference" in subsection:
					api_ref_index = j
					break

			if api_ref_index is not None:
				# Replace the entire API Reference section with our new structure
				logger.info(
					f"Replacing existing API Reference section with overview and {len(api_sections) - 1} versions"
				)
				home_section[api_ref_index] = {"API Reference": api_sections}
			else:
				# Add new API Reference section to Home
				logger.info(f"Adding new API Reference section with overview and {len(api_sections) - 1} versions")
				home_section.append({"API Reference": api_sections})

			# Update the nav with modified Home section
			nav[i] = {"Home": home_section}
			logger.info("Updated navigation structure successfully")
			break

	config_data["nav"] = nav

	# Dump the modified data back to a string
	try:
		dumped_yaml = yaml.dump(config_data, default_flow_style=False, sort_keys=False, allow_unicode=True)
		# Replace bare placeholder back with the original tag
		final_yaml_content = dumped_yaml.replace(placeholder, mermaid_tag)

		# Write the final string to the file
		with MKDOCS_CONFIG_PATH.open("w", encoding="utf-8") as f:
			f.write(final_yaml_content)
		logger.info("Successfully updated mkdocs.yml")
	except yaml.YAMLError:
		logger.exception("Error dumping mkdocs.yml data")
	except OSError:
		logger.exception("Error writing mkdocs.yml file")


if __name__ == "__main__":
	src_package_dir = SRC_ROOT / CODE_PACKAGE
	if not src_package_dir.is_dir():
		logger.error(f"Source package directory not found: {src_package_dir}")
		sys.exit(1)

	# Get Git tags for versioning
	stable_version, prerelease_version = get_git_tags()

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

	# Create the top-level index file first
	create_api_index(stable_version, prerelease_version)

	# Generate docs for both stable and pre-release versions
	versions_nav = create_version_docs(stable_version, prerelease_version, module_structure)

	# Update the mkdocs.yml configuration with both versions
	logger.info("Updating mkdocs.yml with both stable and pre-release versions...")
	update_mkdocs_config(versions_nav)

	logger.info("API documentation update process finished.")
