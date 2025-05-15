"""Direct parser for Python files to CodeStruct notation without Tree-sitter."""

import ast
import sys
from pathlib import Path
from typing import Any

from rich.console import Console


def get_docstring(node: ast.AST) -> str | None:
	"""Extract docstring from an AST node."""
	if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
		return None

	if not node.body:
		return None

	first = node.body[0]
	if isinstance(first, ast.Expr) and isinstance(first.value, ast.Str):
		# Get the first line of the docstring
		docstring = first.value.s.strip()
		lines = docstring.split("\n")
		if len(lines) > 1:
			return f"{lines[0]}..."
		return lines[0]
	return None


def get_imports(module_node: ast.Module) -> list[tuple[str, str]]:
	"""Extract imports from a module."""
	imports = []

	for node in module_node.body:
		if isinstance(node, ast.Import):
			imports.extend((name.name, "external") for name in node.names)
		elif isinstance(node, ast.ImportFrom) and node.module:
			imports.extend((f"{node.module}.{name.name}", "external") for name in node.names)

	# Classify imports
	classified_imports = []
	stdlib_modules = {
		"os",
		"sys",
		"re",
		"pathlib",
		"logging",
		"asyncio",
		"threading",
		"queue",
		"time",
		"collections",
		"typing",
	}

	for imp, _ in imports:
		base_module = imp.split(".")[0]
		if base_module in stdlib_modules:
			classified_imports.append((imp, "stdlib"))
		else:
			classified_imports.append((imp, "external"))

	return classified_imports


def get_class_info(cls_node: ast.ClassDef) -> dict[str, Any]:
	"""Extract information about a class."""
	methods = []
	attributes = []

	for node in cls_node.body:
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
			methods.append(get_function_info(node, is_method=True))
		elif isinstance(node, ast.Assign):
			attributes.extend(
				{
					"name": target.id,
					"type": None,  # Could try to infer type from the value
				}
				for target in node.targets
				if isinstance(target, ast.Name)
			)

	return {"name": cls_node.name, "docstring": get_docstring(cls_node), "methods": methods, "attributes": attributes}


def format_type_annotation(annotation_type: str) -> str:
	"""Format a type annotation, wrapping complex types in double quotes if needed.

	Args:
		annotation_type: The type annotation as a string

	Returns:
		Formatted type annotation
	"""
	# Check if the type annotation contains special characters that would conflict with CodeStruct syntax
	if any(char in annotation_type for char in "[]|,(){}"):
		return f'"{annotation_type}"'
	return annotation_type


def get_arg_info(arg: ast.arg) -> dict[str, Any]:
	"""Extract information about a function argument."""
	annotation_type = None
	if hasattr(arg, "annotation") and arg.annotation:
		if isinstance(arg.annotation, ast.Name) or (
			hasattr(arg.annotation, "__dict__") and "id" in dir(arg.annotation)
		):
			annotation_type = arg.annotation.id
		else:
			try:
				annotation_type = ast.unparse(arg.annotation)
			except (AttributeError, ValueError):
				annotation_type = str(arg.annotation)

	return {"name": arg.arg, "type": annotation_type}


def get_function_info(func_node: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool = False) -> dict[str, Any]:
	"""Extract information about a function or async function.

	Args:
		func_node: The function node from the AST
		is_method: Whether this is a method in a class

	Returns:
		Dict with function information
	"""
	args = []
	is_async = isinstance(func_node, ast.AsyncFunctionDef)

	# Process arguments
	for arg in func_node.args.args:
		if is_method and arg.arg in ("self", "cls"):
			continue
		arg_info = get_arg_info(arg)
		args.append(arg_info)

	# Get return type
	returns = None
	if func_node.returns:
		if isinstance(func_node.returns, ast.Name) or (
			hasattr(func_node.returns, "__dict__") and "id" in dir(func_node.returns)
		):
			returns = func_node.returns.id
		else:
			try:
				returns = ast.unparse(func_node.returns)
			except (AttributeError, ValueError):
				returns = str(func_node.returns)

	return {
		"name": func_node.name,
		"docstring": get_docstring(func_node),
		"args": args,
		"returns": returns,
		"is_async": is_async,
	}


def parse_file(file_path: Path) -> dict[str, Any]:
	"""Parse a Python file into a structured dictionary."""
	with Path(file_path).open("r", encoding="utf-8") as f:
		source = f.read()

	tree = ast.parse(source)

	# Process imports
	imports = get_imports(tree)

	# Process classes and functions
	classes = []
	functions = []

	for node in tree.body:
		if isinstance(node, ast.ClassDef):
			classes.append(get_class_info(node))
		elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
			functions.append(get_function_info(node))

	return {
		"file": str(file_path),
		"module": file_path.stem,
		"docstring": get_docstring(tree),
		"imports": imports,
		"classes": classes,
		"functions": functions,
	}


def to_codestruct(parsed_info: dict[str, Any]) -> str:
	"""Convert parsed information to CodeStruct notation."""
	lines = []

	# Module
	lines.append(f"module: {parsed_info['module']}")

	# Module docstring
	if parsed_info["docstring"]:
		lines.append(f"  doc: {parsed_info['docstring']}")

	# Group imports by source type
	imports_by_source = {}
	for imp, source_type in parsed_info["imports"]:
		if source_type not in imports_by_source:
			imports_by_source[source_type] = []
		imports_by_source[source_type].append(imp)

	# Add grouped imports using ampersand
	for source_type, imports in imports_by_source.items():
		# Sort imports alphabetically for consistency
		imports.sort()
		# For each source type, create an ampersand-separated list of imports
		grouped_imports = " & ".join(imports)
		lines.append(f"  import: {grouped_imports}")
		lines.append(f"    source: {source_type}")

	# Classes
	for cls in parsed_info["classes"]:
		lines.append(f"  class: {cls['name']}")

		if cls["docstring"]:
			lines.append(f"    doc: {cls['docstring']}")

		# Class attributes
		for attr in cls["attributes"]:
			attr_line = f"    attr: {attr['name']}"
			if attr["type"]:
				formatted_type = format_type_annotation(attr["type"])
				attr_line += f" [type: {formatted_type}]"
			lines.append(attr_line)

		# Class methods
		for method in cls["methods"]:
			method_line = f"    func: {method['name']}"
			# Add async type as an attribute instead of a prefix
			if method.get("is_async", False):
				method_line += " [type: Async]"
			lines.append(method_line)

			if method["docstring"]:
				lines.append(f"      doc: {method['docstring']}")

			# Method parameters
			for arg in method["args"]:
				arg_line = f"      param: {arg['name']}"
				if arg["type"]:
					formatted_type = format_type_annotation(arg["type"])
					arg_line += f" [type: {formatted_type}]"
				lines.append(arg_line)

			# Return type
			if method["returns"]:
				formatted_returns = format_type_annotation(method["returns"])
				lines.append(f"      returns: {formatted_returns}")
			else:
				lines.append("      returns: None")

	# Functions
	for func in parsed_info["functions"]:
		func_line = f"  func: {func['name']}"
		# Add async type as an attribute instead of a prefix
		if func.get("is_async", False):
			func_line += " [type: Async]"
		lines.append(func_line)

		if func["docstring"]:
			lines.append(f"    doc: {func['docstring']}")

		# Function parameters
		for arg in func["args"]:
			arg_line = f"    param: {arg['name']}"
			if arg["type"]:
				formatted_type = format_type_annotation(arg["type"])
				arg_line += f" [type: {formatted_type}]"
			lines.append(arg_line)

		# Return type
		if func["returns"]:
			formatted_returns = format_type_annotation(func["returns"])
			lines.append(f"    returns: {formatted_returns}")
		else:
			lines.append("    returns: None")

	return "\n".join(lines)


def main():
	"""Convert file_watcher.py to CodeStruct notation with direct parsing."""
	console = Console()

	# Path to the file we want to convert
	file_path = Path("src/codemap/watcher/file_watcher.py")

	if not file_path.exists():
		console.print(f"[red]File not found: {file_path}[/red]")
		return 1

	try:
		# Parse the file
		console.print("[blue]Parsing file...[/blue]")
		parsed_info = parse_file(file_path)

		# Count entities
		entity_count = (
			1  # Module
			+ len(parsed_info["imports"])
			+ len(parsed_info["classes"])
			+ sum(len(cls["methods"]) for cls in parsed_info["classes"])
			+ len(parsed_info["functions"])
		)
		console.print(f"[green]Extracted approximately {entity_count} entities[/green]")

		# Convert to CodeStruct
		console.print("[blue]Converting to CodeStruct...[/blue]")
		codestruct = to_codestruct(parsed_info)

		# Output the result
		output_path = Path("direct_parsed.cstxt")
		with output_path.open("w") as f:
			f.write(codestruct)

		console.print(f"[green]CodeStruct notation written to {output_path}[/green]")
		console.print("[blue]First 1000 characters of output:[/blue]")
		console.print(codestruct[:1000] + "..." if len(codestruct) > 1000 else codestruct)

		return 0

	except Exception as e:
		console.print(f"[red]Error: {e}[/red]")
		import traceback

		console.print(traceback.format_exc())
		return 1


if __name__ == "__main__":
	sys.exit(main())
