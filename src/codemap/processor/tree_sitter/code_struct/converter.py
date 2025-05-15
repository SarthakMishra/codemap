"""Convert EntitySchema to CodeStruct notation."""

import logging
import re
from pathlib import Path

from codemap.processor.tree_sitter.schema.entity import EntitySchema

logger = logging.getLogger(__name__)

# Type mappings from EntityType to CodeStruct notation
TYPE_TO_CODESTRUCT = {
	# Top-level entities
	"MODULE": "module:",
	"NAMESPACE": "namespace:",
	"PACKAGE": "module:",
	# Functions and methods
	"FUNCTION": "func:",
	"METHOD": "func:",
	# Classes and types
	"CLASS": "class:",
	"INTERFACE": "class:",
	"TYPE_ALIAS": "type_alias:",
	"UNION": "union:",
	# Variables and fields
	"VARIABLE": "var:",
	"CONSTANT": "const:",
	"CLASS_FIELD": "attr:",
	# Parameters
	"PARAMETER": "param:",
	"OPTIONAL": "optional:",
	# Others
	"IMPORT": "import:",
	"LAMBDA": "lambda:",
}

# Expanded type map to include all scalar data types
SCALAR_TYPE_MAP = {
	"INTEGER": "INTEGER",
	"FLOAT": "FLOAT",
	"STRING": "STRING",
	"CHAR": "CHAR",
	"BOOLEAN": "BOOLEAN",
	"NULL": "NULL",
	"SYMBOL": "SYMBOL",
	"COMPLEX": "COMPLEX",
	"DECIMAL": "DECIMAL",
}

# Collection types
COLLECTION_TYPE_MAP = {
	"ARRAY": "ARRAY",
	"LIST": "LIST",
	"TUPLE": "TUPLE",
	"SET": "SET",
	"RANGE": "RANGE",
}

# Associative types
ASSOCIATIVE_TYPE_MAP = {
	"MAP": "MAP",
	"DICT": "DICT",
	"OBJECT": "OBJECT",
	"TABLE": "TABLE",
}

# Combined type map for attributes
SIMPLE_TYPE_MAP = {
	**SCALAR_TYPE_MAP,
	**COLLECTION_TYPE_MAP,
	**ASSOCIATIVE_TYPE_MAP,
}

# Cache for file contents to avoid repeated file reads
_file_cache: dict[str, str] = {}

# Standard library modules for Python
PYTHON_STDLIB_MODULES = {
	"abc",
	"aifc",
	"argparse",
	"array",
	"ast",
	"asyncio",
	"atexit",
	"audioop",
	"base64",
	"bdb",
	"binascii",
	"bisect",
	"builtins",
	"bz2",
	"cProfile",
	"calendar",
	"cgi",
	"cgitb",
	"chunk",
	"cmath",
	"cmd",
	"code",
	"codecs",
	"codeop",
	"collections",
	"colorsys",
	"compileall",
	"concurrent",
	"configparser",
	"contextlib",
	"contextvars",
	"copy",
	"copyreg",
	"crypt",
	"csv",
	"ctypes",
	"curses",
	"dataclasses",
	"datetime",
	"dbm",
	"decimal",
	"difflib",
	"dis",
	"distutils",
	"doctest",
	"email",
	"encodings",
	"ensurepip",
	"enum",
	"errno",
	"faulthandler",
	"fcntl",
	"filecmp",
	"fileinput",
	"fnmatch",
	"formatter",
	"fractions",
	"ftplib",
	"functools",
	"gc",
	"getopt",
	"getpass",
	"gettext",
	"glob",
	"grp",
	"gzip",
	"hashlib",
	"heapq",
	"hmac",
	"html",
	"http",
	"idlelib",
	"imaplib",
	"imghdr",
	"importlib",
	"inspect",
	"io",
	"ipaddress",
	"itertools",
	"json",
	"keyword",
	"lib2to3",
	"linecache",
	"locale",
	"logging",
	"lzma",
	"macpath",
	"mailbox",
	"mailcap",
	"marshal",
	"math",
	"mimetypes",
	"mmap",
	"modulefinder",
	"msilib",
	"msvcrt",
	"multiprocessing",
	"netrc",
	"nis",
	"nntplib",
	"numbers",
	"operator",
	"optparse",
	"os",
	"ossaudiodev",
	"parser",
	"pathlib",
	"pdb",
	"pickle",
	"pickletools",
	"pipes",
	"pkgutil",
	"platform",
	"plistlib",
	"poplib",
	"posix",
	"pprint",
	"profile",
	"pstats",
	"pty",
	"pwd",
	"py_compile",
	"pyclbr",
	"pydoc",
	"queue",
	"quopri",
	"random",
	"re",
	"readline",
	"reprlib",
	"resource",
	"rlcompleter",
	"runpy",
	"sched",
	"secrets",
	"select",
	"selectors",
	"shelve",
	"shlex",
	"shutil",
	"signal",
	"site",
	"smtpd",
	"smtplib",
	"sndhdr",
	"socket",
	"socketserver",
	"spwd",
	"sqlite3",
	"ssl",
	"stat",
	"statistics",
	"string",
	"stringprep",
	"struct",
	"subprocess",
	"sunau",
	"symbol",
	"symtable",
	"sys",
	"sysconfig",
	"syslog",
	"tabnanny",
	"tarfile",
	"telnetlib",
	"tempfile",
	"termios",
	"test",
	"textwrap",
	"threading",
	"time",
	"timeit",
	"tkinter",
	"token",
	"tokenize",
	"trace",
	"traceback",
	"tracemalloc",
	"tty",
	"turtle",
	"turtledemo",
	"types",
	"typing",
	"unicodedata",
	"unittest",
	"urllib",
	"uu",
	"uuid",
	"venv",
	"warnings",
	"wave",
	"weakref",
	"webbrowser",
	"winreg",
	"winsound",
	"wsgiref",
	"xdrlib",
	"xml",
	"xmlrpc",
	"zipapp",
	"zipfile",
	"zipimport",
	"zlib",
}

# Common third-party libraries
COMMON_THIRD_PARTY = {
	"numpy",
	"pandas",
	"matplotlib",
	"tensorflow",
	"torch",
	"sklearn",
	"scikit-learn",
	"pytest",
	"django",
	"flask",
	"requests",
	"bs4",
	"beautifulsoup4",
	"sqlalchemy",
	"pillow",
	"opencv",
	"cv2",
	"aiohttp",
	"fastapi",
	"pydantic",
	"transformers",
	"huggingface_hub",
	"tiktoken",
	"openai",
	"langchain",
	"nltk",
	"spacy",
	"gensim",
	"anyio",
	"watchdog",
	"tree_sitter",
	"xxhash",
	"dotenv",
}


def get_codestruct_type(entity_type: str) -> str | None:
	"""Convert an entity type to a CodeStruct keyword.

	Args:
	    entity_type: The entity type from EntitySchema

	Returns:
	    The corresponding CodeStruct keyword or None if not a supported type
	"""
	# Check direct mappings first
	if entity_type in TYPE_TO_CODESTRUCT:
		return TYPE_TO_CODESTRUCT[entity_type]

	# Try generic mappings for core types
	prefix_mappings = {
		"FUNCTION": "func:",
		"METHOD": "func:",
		"CLASS": "class:",
		"VARIABLE": "var:",
		"CONSTANT": "const:",
		"PARAMETER": "param:",
		"IMPORT": "import:",
	}

	for prefix, mapped_type in prefix_mappings.items():
		if entity_type.startswith(prefix):
			return mapped_type

	# Skip unsupported types
	return None


def format_attributes(entity: EntitySchema) -> str:
	"""Format entity attributes for CodeStruct notation.

	Args:
	    entity: The entity schema with attributes

	Returns:
	    Formatted attribute string in [key: value, ...] format
	"""
	attributes = []

	# Add type attribute if we can infer it from the entity type
	if entity.type in SIMPLE_TYPE_MAP:
		attributes.append(f"type: {SIMPLE_TYPE_MAP[entity.type]}")

	# For functions/methods, add return type if available
	if entity.type in ["FUNCTION", "METHOD", "FUNC", "LAMBDA"] and hasattr(entity, "return_type"):
		# Use getattr with a default value to avoid the type error
		return_type = getattr(entity, "return_type", None)
		if return_type:
			attributes.append(f"returns: {return_type}")

	if not attributes:
		return ""

	return f" [{', '.join(attributes)}]"


def get_file_content(file_path: str) -> str:
	"""Get file content with caching.

	Args:
	    file_path: Path to the file

	Returns:
	    Content of the file as string
	"""
	if file_path in _file_cache:
		return _file_cache[file_path]

	try:
		with Path(file_path).open("r", encoding="utf-8") as f:
			content = f.read()
			_file_cache[file_path] = content
			return content
	except (FileNotFoundError, PermissionError) as e:
		logger.debug(f"Error reading file {file_path}: {e}")
		return ""


def extract_python_docstring(source_code: str, start_line: int, class_or_func_name: str) -> str | None:
	"""Extract Python docstring from source code.

	Args:
	    source_code: Source code content
	    start_line: Start line (0-indexed) of the entity
	    class_or_func_name: Name of the class or function

	Returns:
	    Extracted docstring or None
	"""
	lines = source_code.splitlines()

	# Find the line containing the class or function definition
	definition_line_idx = -1
	pattern = rf"(class|def)\s+{re.escape(class_or_func_name)}\s*[\(:]"

	for i in range(max(0, start_line - 5), min(len(lines), start_line + 5)):
		if re.search(pattern, lines[i]):
			definition_line_idx = i
			break

	if definition_line_idx == -1:
		return None

	# Look for triple-quoted string after the definition
	triple_quote_pattern = r'"""(.*?)"""'

	# Check for docstring on the next few lines
	combined_text = "\n".join(lines[definition_line_idx : definition_line_idx + 15])
	matches = re.search(triple_quote_pattern, combined_text, re.DOTALL)

	if matches:
		docstring = matches.group(1).strip()
		# Return the first line with ellipsis if there are more lines
		lines = docstring.split("\n")
		if len(lines) > 1:
			return f"{lines[0]}..."
		return lines[0]

	return None


def extract_doc_string(entity: EntitySchema) -> str | None:
	"""Extract documentation string for the entity if available.

	Args:
	    entity: The entity schema

	Returns:
	    Formatted documentation string or None
	"""
	# Get the node content from the entity metadata
	if not entity.metadata or not hasattr(entity.metadata, "file_path"):
		return None

	file_path = entity.metadata.file_path

	# Skip non-Python files for now
	if not file_path.endswith((".py", ".pyi")):
		return None

	# In a real implementation, we would directly access the entity's docstring
	# Since we don't have that, try to infer from the location and extract from source
	if entity.location and entity.name:
		try:
			# Get file content using cache
			source_code = get_file_content(file_path)
			if not source_code:
				return None

			# For Python, extract docstrings based on entity type
			if entity.type in ["CLASS", "FUNCTION", "METHOD", "MODULE"]:
				loc = entity.location
				start_line = loc.start_line - 1  # Convert to 0-indexed

				# For modules, extract the module-level docstring
				if entity.type == "MODULE":
					# Look for a module-level docstring at the beginning of the file
					matches = re.search(r'^"""(.*?)"""', source_code, re.DOTALL)
					if matches:
						docstring = matches.group(1).strip()
						if docstring:
							lines = docstring.split("\n")
							if len(lines) > 1:
								return f"{lines[0]}..."
							return lines[0]
				else:
					# For classes and functions, extract their specific docstrings
					docstring = extract_python_docstring(source_code, start_line, entity.name)
					if docstring:
						return docstring

		except (FileNotFoundError, PermissionError) as e:
			logger.debug(f"Error extracting docstring: {e}")

	# Fallback to a basic description if no docstring found
	if entity.name:
		if entity.type == "MODULE":
			module_path = Path(file_path).stem
			return f"{module_path} module"
		if entity.type == "CLASS":
			return f"{entity.name} class definition"
		if entity.type in ["FUNCTION", "METHOD"]:
			return f"{entity.name} function"

	return None


def extract_parameters_from_source(source_code: str, start_line: int, func_name: str) -> list[tuple[str, str | None]]:
	"""Extract function parameters from source code.

	Args:
	    source_code: Source code content
	    start_line: Start line (0-indexed) of the function
	    func_name: Name of the function

	Returns:
	    List of (param_name, param_type) tuples
	"""
	lines = source_code.splitlines()

	# Find the function definition line
	def_line_idx = -1
	pattern = rf"def\s+{re.escape(func_name)}\s*\("

	for i in range(max(0, start_line - 5), min(len(lines), start_line + 5)):
		if re.search(pattern, lines[i]):
			def_line_idx = i
			break

	if def_line_idx == -1:
		return []

	# Extract the parameter list
	# Find the opening parenthesis
	def_line = lines[def_line_idx]
	open_paren_idx = def_line.find("(")
	if open_paren_idx == -1:
		return []

	# Collect lines until we find the closing parenthesis
	param_text = def_line[open_paren_idx + 1 :]
	line_idx = def_line_idx

	# If the closing parenthesis is not on the same line, keep reading lines
	while ")" not in param_text and line_idx < len(lines) - 1:
		line_idx += 1
		next_line = lines[line_idx]
		param_text += next_line
		if ")" in next_line:
			param_text = param_text[: param_text.find(")")]
			break

	# Clean up the parameter text
	param_text = param_text.strip()
	if param_text.endswith("):"):
		param_text = param_text[:-2]
	elif param_text.endswith(")"):
		param_text = param_text[:-1]

	# No parameters
	if not param_text:
		return []

	# Split parameters and process each one
	params = []
	for raw_param in param_text.split(","):
		param_item = raw_param.strip()
		if not param_item or param_item in {"self", "cls"}:
			continue

		# Check for type annotation
		if ":" in param_item:
			name, type_hint = param_item.split(":", 1)
			name = name.strip()
			type_hint = type_hint.strip()

			# Handle default values
			if "=" in type_hint:
				type_hint = type_hint.split("=")[0].strip()

			params.append((name, type_hint))
		else:
			# No type annotation
			name = param_item.split("=")[0].strip() if "=" in param_item else param_item
			params.append((name, None))

	return params


def extract_return_type_from_source(source_code: str, start_line: int, func_name: str) -> str | None:
	"""Extract function return type from source code.

	Args:
	    source_code: Source code content
	    start_line: Start line (0-indexed) of the function
	    func_name: Name of the function

	Returns:
	    Return type as string or None
	"""
	lines = source_code.splitlines()

	# Find the function definition line
	def_line_idx = -1
	pattern = rf"def\s+{re.escape(func_name)}\s*\("

	for i in range(max(0, start_line - 5), min(len(lines), start_line + 5)):
		if re.search(pattern, lines[i]):
			def_line_idx = i
			break

	if def_line_idx == -1:
		return None

	# Look for -> return type in the function signature
	# This might span multiple lines, so we need to check a few lines
	signature_text = ""
	for i in range(def_line_idx, min(def_line_idx + 5, len(lines))):
		signature_text += lines[i]
		if "->" in signature_text and ":" in signature_text:
			# Extract return type
			return_section = signature_text.split("->")[1].split(":")[0].strip()
			if return_section:
				return return_section

		# If we found a colon without a return type, stop searching
		if ":" in lines[i] and "->" not in signature_text:
			break

	return None


def classify_import(import_name: str) -> str:
	"""Classify an import as stdlib, third-party, or project.

	Args:
	    import_name: Name of the imported module

	Returns:
	    Classification as string: 'stdlib', 'external', or 'internal'
	"""
	# Extract the base module name (e.g., 'pandas.core' -> 'pandas')
	base_module = import_name.split(".")[0].lower()

	if base_module in PYTHON_STDLIB_MODULES:
		return "stdlib"
	if base_module in COMMON_THIRD_PARTY:
		return "external"
	# Check if it might be an internal import from this project
	# This is a simplified approach - a real implementation would examine the project structure
	if base_module in {"codemap", "watcher", "processor"}:
		return "internal"

	return "external"  # Default to external if unknown


def find_class_methods_in_source(
	source_code: str, class_name: str, start_line: int, end_line: int
) -> list[tuple[str, int, int]]:
	"""Find methods defined in a class from source code.

	Args:
	    source_code: Source code content
	    class_name: Name of the class
	    start_line: Start line of class (0-indexed)
	    end_line: End line of class (0-indexed)

	Returns:
	    List of (method_name, start_line, end_line) tuples
	"""
	lines = source_code.splitlines()
	methods = []

	# Find the class definition line
	class_pattern = rf"class\s+{re.escape(class_name)}"
	class_def_line = -1

	for i in range(max(0, start_line - 3), min(len(lines), start_line + 3)):
		if re.search(class_pattern, lines[i]):
			class_def_line = i
			break

	if class_def_line == -1:
		return []

	# Find indentation level of class definition
	class_indent = len(lines[class_def_line]) - len(lines[class_def_line].lstrip())

	# Expected indentation for methods (class + 1 tab)
	method_indent = class_indent + 1

	# Search for method definitions within the class
	method_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
	current_method = None
	current_method_start = -1

	for i in range(class_def_line + 1, min(len(lines), end_line + 1)):
		line = lines[i]

		# Skip empty lines
		if not line.strip():
			continue

		# Calculate indent level
		indent = len(line) - len(line.lstrip())

		# If indent is less than class, we've exited the class
		if indent <= class_indent and line.strip():
			if current_method:
				methods.append((current_method, current_method_start, i - 1))
				current_method = None
			break

		# Check for method definition
		if indent == method_indent * 4:  # Assuming 4 spaces per indent level
			method_match = re.search(method_pattern, line)
			if method_match:
				if current_method:
					methods.append((current_method, current_method_start, i - 1))
				current_method = method_match.group(1)
				current_method_start = i
		# If we find a new method or class end, store the previous one
		elif indent == method_indent * 4 and current_method and line.strip():
			methods.append((current_method, current_method_start, i - 1))
			current_method = None

	# Add the last method if we're at the end of the file
	if current_method:
		methods.append((current_method, current_method_start, end_line))

	return methods


def entity_to_codestruct(
	entity: EntitySchema,
	indent_level: int = 0,
	indent_width: int = 2,
	debug_entities: set[str] | None = None,
	processed_imports: set[str] | None = None,
	params_map: dict[str, list[EntitySchema]] | None = None,
) -> str | None:
	"""Convert an EntitySchema to CodeStruct notation.

	Args:
	    entity: The entity schema to convert
	    indent_level: Current indentation level
	    indent_width: Number of spaces per indentation level
	    debug_entities: Set of entity IDs to include debug info for
	    processed_imports: Set of already processed imports to avoid duplication
	    params_map: Map of function IDs to their parameter entities

	Returns:
	    The entity in CodeStruct notation or None if entity type is not supported
	"""
	# Initialize collections if None
	if processed_imports is None:
		processed_imports = set()
	if params_map is None:
		params_map = {}

	# Get keyword based on entity type
	keyword = get_codestruct_type(entity.type)

	# Skip unsupported entity types
	if keyword is None:
		logger.debug(f"Skipping unsupported entity type: {entity.type}")
		return None

	# Skip anonymous entities (they're usually just syntax nodes)
	if entity.name is None or entity.name.lower() == "anonymous":
		logger.debug(f"Skipping anonymous entity: {entity.type}")
		return None

	# Skip duplicate imports
	if entity.type == "IMPORT":
		if entity.name in {"anonymous", "", None}:
			return None
		if entity.name in processed_imports:
			return None
		processed_imports.add(entity.name)

	result = []
	indent = " " * (indent_level * indent_width)

	# Format the entity declaration
	entity_name = entity.name or "anonymous"
	attributes = format_attributes(entity)
	declaration = f"{indent}{keyword} {entity_name}{attributes}"
	result.append(declaration)

	# Add documentation if available
	doc_string = extract_doc_string(entity)
	if doc_string:
		result.append(f"{indent}  doc: {doc_string}")

	# Only include location info for debugging if requested
	if debug_entities and entity.id in debug_entities and entity.location:
		loc = entity.location
		location_str = f"L{loc.start_line}-{loc.end_line}:C{loc.start_col}-{loc.end_col}"
		result.append(f"{indent}  # {location_str}")

	# Process imports with proper classification
	if entity.scope and entity.scope.imports:
		# Remove duplicate imports
		unique_imports = set()
		for imp in entity.scope.imports:
			if imp and imp != "anonymous" and imp not in processed_imports:
				unique_imports.add(imp)
				processed_imports.add(imp)

		for imp in unique_imports:
			# Improve import notation with proper classification
			result.append(f"{indent}  import: {imp}")

			# Use the classification function to determine import type
			import_type = classify_import(imp)
			result.append(f"{indent}    source: {import_type}")

	# For classes, try to extract methods directly from source if entity tree is incomplete
	if entity.type == "CLASS" and entity.metadata and entity.location:
		file_path = getattr(entity.metadata, "file_path", None)

		# Only proceed if we have the necessary information
		if file_path and entity.name:
			source_code = get_file_content(file_path)

			if source_code and (not entity.children or len(entity.children) < 2):  # noqa: PLR2004
				# Try to find class methods from source
				start_line = entity.location.start_line - 1  # Convert to 0-indexed
				end_line = entity.location.end_line - 1

				methods = find_class_methods_in_source(source_code, entity.name, start_line, end_line)

				# Add each method as a child element
				for method_name, method_start, _method_end in methods:
					if method_name in {"__init__", "__repr__", "__str__", "__eq__", "__lt__", "__gt__"}:
						# Skip built-in methods
						continue

					result.append(f"{indent}  func: {method_name}")

					# Try to extract parameters for this method
					params = extract_parameters_from_source(source_code, method_start, method_name)
					for param_name, param_type in params:
						if param_type:
							result.append(f"{indent}    param: {param_name} [type: {param_type}]")
						else:
							result.append(f"{indent}    param: {param_name}")

					# Try to extract return type
					return_type = extract_return_type_from_source(source_code, method_start, method_name)
					if return_type:
						result.append(f"{indent}    returns: {return_type}")
					else:
						result.append(f"{indent}    returns: None")

	# Process parameters for functions more thoroughly
	if entity.type in ["FUNCTION", "METHOD", "FUNC"]:
		file_path = getattr(entity.metadata, "file_path", None)
		source_code = get_file_content(file_path) if file_path else ""

		# Extract parameters from source if available
		params = []
		return_type = None

		if source_code and entity.name and entity.location:
			start_line = entity.location.start_line - 1  # Convert to 0-indexed
			params = extract_parameters_from_source(source_code, start_line, entity.name)
			return_type = extract_return_type_from_source(source_code, start_line, entity.name)

		# If parameters were extracted from source, use those
		if params:
			for param_name, param_type in params:
				if param_type:
					result.append(f"{indent}  param: {param_name} [type: {param_type}]")
				else:
					result.append(f"{indent}  param: {param_name}")
		# Otherwise fall back to params_map
		elif entity.id in params_map:
			for param in params_map[entity.id]:
				param_name = param.name or "unnamed"
				param_type = getattr(param, "type_hint", None)
				if param_type:
					result.append(f"{indent}  param: {param_name} [type: {param_type}]")
				else:
					result.append(f"{indent}  param: {param_name}")
		# If still no parameters, try to infer from children
		elif entity.children:
			param_children = [
				child for child in entity.children if child.type == "PARAMETER" or child.type.startswith("PARAMETER")
			]

			for param in param_children:
				param_name = param.name or "unnamed"
				param_type = getattr(param, "type_hint", None)
				if param_type:
					result.append(f"{indent}  param: {param_name} [type: {param_type}]")
				else:
					result.append(f"{indent}  param: {param_name}")

		# Handle return type
		if return_type:
			result.append(f"{indent}  returns: {return_type}")
		else:
			fallback_return = getattr(entity, "return_type", "Any")
			result.append(f"{indent}  returns: {fallback_return or 'None'}")

	# Improved processing of children
	valid_children = []
	if entity.children:
		for child in entity.children:
			# Skip parameters as they're handled separately
			if child.type == "PARAMETER" or child.type.startswith("PARAMETER"):
				continue

			child_struct = entity_to_codestruct(
				child, indent_level + 1, indent_width, debug_entities, processed_imports, params_map
			)
			if child_struct:  # Only add supported entity types
				valid_children.append(child_struct)

		# Add non-empty valid children
		result.extend(valid_children)

	return "\n".join(result)


def entities_to_codestruct(entities: list[EntitySchema], indent_width: int = 2, debug: bool = False) -> str:
	"""Convert a list of EntitySchema objects to CodeStruct notation.

	Args:
	    entities: List of entity schemas to convert
	    indent_width: Number of spaces per indentation level
	    debug: Whether to include debug info like line numbers

	Returns:
	    The entities in CodeStruct notation
	"""
	result = []

	# Create a set of entity IDs to debug if requested
	debug_entities = set()
	if debug:
		debug_entities = {entity.id for entity in entities}

	# Track processed imports to avoid duplication
	processed_imports = set()

	# Create a map of parent_id to child entities for proper nesting
	parent_to_children = {}
	root_entities = []

	# Map to identify parameters for functions
	params_map = {}

	# First organize entities by their parent relationship
	for entity in entities:
		# Skip anonymous entities at this stage
		if entity.name is None or entity.name.lower() == "anonymous":
			continue

		# Identify parameters
		if entity.type == "PARAMETER" or entity.type.startswith("PARAMETER"):
			if entity.parent_id:
				if entity.parent_id not in params_map:
					params_map[entity.parent_id] = []
				params_map[entity.parent_id].append(entity)
			continue

		if entity.parent_id:
			if entity.parent_id not in parent_to_children:
				parent_to_children[entity.parent_id] = []
			parent_to_children[entity.parent_id].append(entity)
		else:
			root_entities.append(entity)

	# Process only root entities (no parent)
	for entity in root_entities:
		# Skip duplicates (like repeated imports)
		if entity.type == "IMPORT" and entity.name in processed_imports:
			continue

		# Add child methods and attributes if this is a class
		if entity.id in parent_to_children:
			if not hasattr(entity, "children"):
				entity.children = []
			entity.children.extend(parent_to_children[entity.id])

		entity_struct = entity_to_codestruct(entity, 0, indent_width, debug_entities, processed_imports, params_map)
		if entity_struct:  # Only add supported entity types
			result.append(entity_struct)

	return "\n".join(result)


def minify_codestruct(codestruct: str) -> str:
	"""Minify CodeStruct notation for efficient context inclusion.

	Args:
	    codestruct: The original CodeStruct notation

	Returns:
	    Minified CodeStruct notation
	"""
	# Keyword shortening map
	keyword_map = {
		"dir:": "d:",
		"file:": "f:",
		"module:": "m:",
		"namespace:": "ns:",
		"class:": "cl:",
		"func:": "fn:",
		"lambda:": "lm:",
		"attr:": "at:",
		"param:": "p:",
		"returns:": "r:",
		"var:": "v:",
		"const:": "c:",
		"type_alias:": "ta:",
		"union:": "un:",
		"optional:": "opt:",
		"import:": "i:",
		"doc:": "dc:",
	}

	# Type abbreviations
	type_map = {
		"INTEGER": "INT",
		"STRING": "STR",
		"BOOLEAN": "BOOL",
		"FLOAT": "FLT",
		"external": "ext",
		"internal": "int",
		"stdlib": "std",
	}

	# Split into lines and filter empty lines
	lines = [line for line in codestruct.strip().split("\n") if line.strip()]
	if not lines:
		return ""

	# Process and build hierarchical structure
	# We'll use a dictionary to store {indent_level: [nodes at this level]}
	indent_to_nodes = {}
	parent_map = {}  # Maps a node index to its parent index

	# First pass: Group nodes by their indentation level
	for i, line in enumerate(lines):
		# Skip comment lines
		if line.strip().startswith("#"):
			continue

		indent = len(line) - len(line.lstrip())
		if indent not in indent_to_nodes:
			indent_to_nodes[indent] = []

		# Store the line index at this indentation level
		indent_to_nodes[indent].append(i)

		# Find parent for this node
		if indent > 0:
			# Find closest previous line with less indentation
			for j in range(i - 1, -1, -1):
				prev_indent = len(lines[j]) - len(lines[j].lstrip())
				if prev_indent < indent and not lines[j].strip().startswith("#"):
					parent_map[i] = j
					break

	# Apply shorthand transformations
	shortened_lines = []
	for line in lines:
		if line.strip().startswith("#"):
			continue

		# Clean the line
		cleaned = line.strip()

		# Apply keyword shortening
		for keyword, short in keyword_map.items():
			if cleaned.startswith(keyword):
				cleaned = cleaned.replace(keyword, short, 1)
				break

		# Apply type abbreviations
		for type_name, short in type_map.items():
			cleaned = cleaned.replace(f"type: {type_name}", f"t:{short}")
			cleaned = cleaned.replace(f"source: {type_name}", f"s:{short}")

		shortened_lines.append(cleaned)

	# Build the minified output with proper delimiters
	result = []
	processed = set()

	# Process top-level nodes first (minimum indentation)
	min_indent = min(indent_to_nodes.keys())
	top_nodes = indent_to_nodes[min_indent]

	for node_idx in top_nodes:
		if node_idx in processed:
			continue

		node_content = shortened_lines[node_idx]
		children = []

		# Find all direct children of this node
		for child_idx, parent_idx in parent_map.items():
			if parent_idx == node_idx:
				# Child gets added with a pipe delimiter
				child_content = shortened_lines[child_idx]
				child_grandchildren = []

				# Check for grandchildren
				for gc_idx, gc_parent_idx in parent_map.items():
					if gc_parent_idx == child_idx:
						child_grandchildren.append(shortened_lines[gc_idx])
						processed.add(gc_idx)

				if child_grandchildren:
					child_content += f"|{','.join(child_grandchildren)}"

				children.append(child_content)
				processed.add(child_idx)

		if children:
			node_content += f"|{','.join(children)}"

		result.append(node_content)
		processed.add(node_idx)

	return ";".join(result)


def generate_legend() -> str:
	"""Generate a legend to help interpret minified CodeStruct.

	Returns:
	    A string containing the legend explanation
	"""
	return """Format: Entity;Entity|Child,Child[Attribute,Attribute]
Keyword map: d=dir,f=file,m=module,cl=class,fn=func,at=attr,p=param,r=returns,v=var,c=const,
i=import,t=type,s=source,rf=ref
Type map: INT=INTEGER,STR=STRING,BOOL=BOOLEAN,FLT=FLOAT,ext=external,int=internal,std=stdlib
Delimiters: ;=entity separator, |=child separator, ,=attribute separator, []=attribute container"""
