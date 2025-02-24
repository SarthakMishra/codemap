"""Tree-sitter based code parsing and analysis."""

from __future__ import annotations

import fnmatch
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser

logger = logging.getLogger(__name__)

# Error messages
ERR_PARSER_INIT = "Failed to initialize Python parser"
ERR_FILE_PARSE = "Failed to parse file {file}"

# Constants
# Get the project root (2 levels up from this file: src/codemap/analyzer -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
VENDOR_PATH = PROJECT_ROOT / "vendor"
PYTHON_GRAMMAR_PATH = VENDOR_PATH / "tree-sitter-python"
LANGUAGES_SO = VENDOR_PATH / "my-languages.so"


class CodeParser:
    """Parses Python source code files using tree-sitter for syntax analysis."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the code parser.

        Args:
            config: Configuration dictionary with exclude_patterns and use_gitignore settings.

        Raises:
            RuntimeError: If Python parser could not be initialized.
        """
        self.config = config or {}
        self.gitignore_patterns: list[str] = []

        try:
            # Create vendor directory if it doesn't exist
            VENDOR_PATH.mkdir(parents=True, exist_ok=True)

            # Clone the Python grammar if it doesn't exist
            if not PYTHON_GRAMMAR_PATH.exists():
                logger.debug("Cloning Python grammar repository...")
                subprocess.run(
                    ["git", "clone", "https://github.com/tree-sitter/tree-sitter-python.git", str(PYTHON_GRAMMAR_PATH)],
                    check=True,
                )

            # Build the language library
            if not LANGUAGES_SO.exists():
                logger.debug("Building Python language library...")
                Language.build_library(str(LANGUAGES_SO), [str(PYTHON_GRAMMAR_PATH)])

            # Create and initialize the parser
            self.parser = Parser()
            PYTHON_LANGUAGE = Language(str(LANGUAGES_SO))  # Only pass the library path
            self.parser.set_language(PYTHON_LANGUAGE)
            logger.debug("Successfully initialized Python parser")

        except Exception as e:
            logger.error("Failed to initialize Python parser: %s", e)
            raise RuntimeError(ERR_PARSER_INIT) from e

        if self.config.get("use_gitignore", True):
            self._load_gitignore()

    def _load_gitignore(self) -> None:
        """Load patterns from .gitignore file if it exists."""
        gitignore_path = Path(".gitignore")
        self.gitignore_patterns = []  # Always start with empty list
        if gitignore_path.exists():
            with gitignore_path.open() as f:
                patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                self.gitignore_patterns.extend(patterns)

    def _matches_pattern(self, file_path: Path, pattern: str) -> bool:
        """Check if a file path matches a glob pattern.

        Args:
            file_path: Path to check
            pattern: Glob pattern to match against

        Returns:
            True if the path matches the pattern
        """
        # Convert pattern and path to forward slashes for consistency
        pattern = pattern.replace(os.sep, "/")
        path_str = str(file_path).replace(os.sep, "/")

        # Handle directory patterns (ending with /)
        if pattern.endswith("/"):
            pattern = pattern.rstrip("/")
            path_parts = Path(path_str).parts
            return any(part == pattern for part in path_parts)

        # For .dot patterns (like .venv), match against both name and any part of the path
        if pattern.startswith("."):
            path_parts = Path(path_str).parts
            if any(part == pattern for part in path_parts):
                return True
            if any(part == pattern[1:] for part in path_parts):
                return True
            return (
                fnmatch.fnmatch(file_path.name, pattern)
                or fnmatch.fnmatch(path_str, pattern)
                or fnmatch.fnmatch(path_str, f"*/{pattern}")
            )

        # For patterns with directory separators (like **/temp/*), match the full path
        if "/" in pattern:
            return fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path_str, f"**/{pattern}")

        # For simple patterns (like *.py), match against any part of the path
        path_parts = Path(path_str).parts
        return any(fnmatch.fnmatch(part, pattern) for part in path_parts)

    def should_parse(self, file_path: Path) -> bool:
        """Check if a file should be parsed based on configuration.

        Args:
            file_path: Path to check.

        Returns:
            True if the file should be parsed, False otherwise.
        """
        # Only parse Python files
        if file_path.suffix.lower() != ".py":
            return False

        # Check exclude patterns
        exclude_patterns = self.config.get("exclude_patterns", [])
        for pattern in exclude_patterns:
            if self._matches_pattern(file_path, pattern):
                return False

        # Check gitignore patterns if enabled
        if self.config.get("use_gitignore", True):
            for pattern in self.gitignore_patterns:
                if self._matches_pattern(file_path, pattern):
                    return False

        return True

    def parse_file(self, file_path: Path) -> dict[str, Any]:
        """Parse a file and extract its symbols.

        Args:
            file_path: Path to the file to parse.

        Returns:
            Dictionary containing the extracted symbols.

        Raises:
            OSError: If file cannot be read.
        """
        logger.debug("Parsing file: %s", file_path)

        # Initialize empty symbols dictionary
        symbols: dict[str, Any] = {
            "imports": [],
            "classes": [],
            "references": [],
            "bases": {},
            "attributes": {},
        }

        if not self.should_parse(file_path):
            return symbols

        try:
            content = file_path.read_bytes()
            tree = self.parser.parse(content)
            if not tree:
                logger.warning("Failed to parse file: %s", file_path)
                return symbols

            # Extract symbols
            self._visit_node(tree.root_node, symbols)
            logger.debug(
                "Found in %s: %d classes, %d imports, %d references",
                file_path,
                len(symbols["classes"]),
                len(symbols["imports"]),
                len(symbols["references"]),
            )
        except OSError:
            logger.exception("Failed to read file")
            raise
        except Exception as e:
            logger.exception("Failed to parse file: %s", e)
            return symbols

        return symbols

    def _process_class_node(self, node: Node, symbols: dict[str, Any]) -> None:
        """Process a class definition node.

        Args:
            node: The class definition node
            symbols: Dictionary to store extracted symbols
        """
        try:
            # Initialize class info
            class_name = None
            bases = []

            # Extract class name and base classes
            for child in node.children:
                if child.type == "identifier" and not class_name:
                    class_name = child.text.decode("utf-8")
                elif child.type == "argument_list":
                    for base_node in child.children:
                        if base_node.type == "identifier":
                            bases.append(base_node.text.decode("utf-8"))

            if not class_name:
                logger.warning("Could not extract class name from node")
                return

            # Skip private classes
            if class_name.startswith("_"):
                return

            # Initialize class data in symbols
            if "classes" not in symbols:
                symbols["classes"] = []
            if "bases" not in symbols:
                symbols["bases"] = {}
            if "attributes" not in symbols:
                symbols["attributes"] = {}

            # Add class info
            symbols["classes"].append(class_name)
            if bases:
                symbols["bases"][class_name] = bases

            # Process class body for attributes
            body_node = next((c for c in node.children if c.type == "block"), None)
            if body_node:
                symbols["attributes"][class_name] = self._extract_class_attributes(body_node)
                logger.debug("Extracted attributes for %s: %s", class_name, symbols["attributes"][class_name])

        except Exception as e:
            logger.debug("Error processing class node: %s", e)

    def _extract_class_attributes(self, body_node: Node) -> dict[str, str]:
        """Extract attributes from a class body node.

        Args:
            body_node: The class body node

        Returns:
            Dictionary mapping attribute names to their types
        """
        attributes = {}
        for stmt in body_node.children:
            try:
                # Handle different attribute definition patterns
                if stmt.type == "expression_statement":
                    expr = stmt.children[0] if stmt.children else None
                    if expr and expr.type == "assignment":
                        target = expr.children[0] if expr.children else None
                        if target and target.type == "identifier":
                            name = target.text.decode("utf-8")
                            if not name.startswith("_"):  # Skip private attributes
                                value = expr.children[-1] if len(expr.children) > 1 else None
                                if value:
                                    attr_type = self._infer_type(value)
                                    if attr_type:
                                        attributes[name] = attr_type
                elif stmt.type == "annotated_assignment":
                    target = stmt.children[0] if stmt.children else None
                    annotation = next((c for c in stmt.children if c.type == "type"), None)
                    if target and target.type == "identifier" and annotation:
                        name = target.text.decode("utf-8")
                        if not name.startswith("_"):  # Skip private attributes
                            attr_type = annotation.text.decode("utf-8")
                            attributes[name] = attr_type
            except Exception as e:
                logger.debug("Error processing attribute statement: %s", e)

        return attributes

    def _infer_type(self, node: Node) -> str | None:
        """Infer the type of a node.

        Args:
            node: The node to infer the type from

        Returns:
            The inferred type name or None if cannot be inferred
        """
        try:
            type_map = {
                "string": "str",
                "integer": "int",
                "float": "float",
                "true": "bool",
                "false": "bool",
                "none": "None",
                "list": "list",
                "dictionary": "dict",
                "set": "set",
                "tuple": "tuple",
            }

            if node.type in type_map:
                return type_map[node.type]
            if node.type == "call":
                func = node.children[0] if node.children else None
                if func and func.type == "identifier":
                    return func.text.decode("utf-8")
            elif node.type == "identifier":
                return node.text.decode("utf-8")

            return None
        except Exception as e:
            logger.debug("Error inferring type: %s", e)
            return None

    def _process_import_node(self, node: Node, symbols: dict[str, list[str]]) -> None:
        """Process an import statement node.

        Args:
            node: The import statement node
            symbols: Dictionary to store extracted symbols
        """
        try:
            for child in node.children:
                if child.type == "dotted_name":
                    symbols["imports"].append(child.text.decode("utf-8"))
        except (AttributeError, UnicodeDecodeError) as e:
            logger.debug("Error processing import node: %s", e)

    def _process_reference_node(self, node: Node, symbols: dict[str, list[str]]) -> None:
        """Process a reference node (call expression or identifier).

        Args:
            node: The reference node
            symbols: Dictionary to store extracted symbols
        """
        try:
            text = node.text.decode("utf-8")
            if text and not text.startswith("_"):
                symbols["references"].append(text)
        except (AttributeError, UnicodeDecodeError) as e:
            logger.debug("Error processing reference node: %s", e)

    def _visit_node(self, node: Node, symbols: dict[str, Any]) -> None:
        """Visit a node and process its contents.

        Args:
            node: The node to visit
            symbols: Dictionary to store extracted symbols
        """
        try:
            # Process the current node based on its type
            if node.type == "class_definition":
                self._process_class_node(node, symbols)
            elif node.type == "import_statement":
                self._process_import_node(node, symbols)
            elif node.type in ("call_expression", "identifier"):
                self._process_reference_node(node, symbols)

            # Recursively visit children
            for child in node.children:
                if child:  # Skip None children
                    self._visit_node(child, symbols)
        except Exception as e:
            logger.debug("Error visiting node %s: %s", node.type, e)
