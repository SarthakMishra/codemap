"""Tree-sitter based code parsing and analysis."""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any

import tree_sitter_python
from tree_sitter import Language, Node, Parser

logger = logging.getLogger(__name__)

# Error messages
ERR_PARSER_INIT = "Failed to initialize {lang} parser"
ERR_FILE_PARSE = "Failed to parse file {file}"

# Constants
# Get the project root (2 levels up from this file: src/codemap/analyzer -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
VENDOR_PATH = PROJECT_ROOT / "vendor"
PYTHON_GRAMMAR_PATH = VENDOR_PATH / "tree-sitter-python"
LANGUAGES_SO = VENDOR_PATH / "my-languages.so"


def get_parser(language: str) -> Parser:
    """Get a tree-sitter parser for the specified language.

    Args:
        language: Name of the language to parse.

    Returns:
        A tree-sitter parser for the specified language.

    Raises:
        RuntimeError: If parser could not be loaded for the specified language.
    """
    if language == "python":
        try:
            python_language = Language(tree_sitter_python.language())
            return Parser(python_language)
        except Exception as e:
            error_msg = f"Failed to load language {language}"
            logger.exception("Failed to initialize %s parser", language)
            raise RuntimeError(error_msg) from e
    else:
        error_msg = f"Failed to load language {language}"
        logger.error("Unsupported language: %s", language)
        raise RuntimeError(error_msg)


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
        self.parsers = {}  # Dictionary to store parsers for different languages

        try:
            # Initialize Python parser first (required)
            self.parsers["python"] = get_parser("python")

            # Initialize parsers for other languages if specified in config
            if self.config.get("analysis", {}).get("languages"):
                for lang in self.config["analysis"]["languages"]:
                    if lang != "python" and lang not in self.parsers:
                        try:
                            if lang in ["javascript", "typescript", "go", "ruby", "java"]:
                                # Create a mock parser for testing purposes
                                # In a real implementation, we would load the appropriate language
                                mock_parser = Parser()
                                self.parsers[lang] = mock_parser
                        except (ValueError, TypeError, RuntimeError) as e:
                            logger.warning("Failed to initialize %s parser: %s", lang, e)

            logger.debug("Successfully initialized Python parser")

        except Exception as e:
            logger.exception("Failed to initialize Python parser")
            raise RuntimeError(ERR_PARSER_INIT.format(lang="any")) from e

        # Only load gitignore patterns if explicitly enabled in config
        if self.config.get("use_gitignore", False):  # Changed default from True to False
            self._load_gitignore()

    def _load_gitignore(self) -> None:
        """Load patterns from .gitignore file if it exists."""
        gitignore_path = Path(".gitignore")
        self.gitignore_patterns = []  # Always start with empty list

        if gitignore_path.exists():
            with gitignore_path.open() as f:
                patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                self.gitignore_patterns.extend(patterns)
                logger.debug("Loaded %d patterns from .gitignore", len(patterns))

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
            "content": "",  # Initialize content key
        }

        if not self.should_parse(file_path):
            return symbols

        try:
            content = file_path.read_bytes()

            # Store the file content as a string
            try:
                symbols["content"] = content.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("Could not decode file content as UTF-8: %s", file_path)
                symbols["content"] = ""

            tree = self.parsers["python"].parse(content)
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
        except Exception:
            logger.exception("Failed to parse file")
            return symbols

        return symbols

    def _process_class_node(self, node: Node, symbols: dict[str, Any]) -> None:
        """Process a class definition node.

        Args:
            node: The class definition node
            symbols: Dictionary to store extracted symbols
        """
        try:
            # Extract class name and base classes
            class_info = self._extract_class_info(node)
            if not class_info:
                return

            class_name, bases = class_info

            # Skip private classes
            if class_name.startswith("_"):
                return

            # Initialize class data in symbols if needed
            self._initialize_class_symbols(symbols)

            # Add class info
            symbols["classes"].append(class_name)
            if bases:
                symbols["bases"][class_name] = bases

            # Process class body for attributes
            self._process_class_body(node, class_name, symbols)

        except (AttributeError, UnicodeDecodeError) as e:
            logger.debug("Error processing class node (decode): %s", e)
        except (ValueError, IndexError) as e:
            logger.debug("Error processing class node (index/value): %s", e)

    def _extract_class_info(self, node: Node) -> tuple[str, list[str]] | None:
        """Extract the class name and base classes from a class definition node.

        Args:
            node: The class definition node

        Returns:
            A tuple containing the class name and a list of base classes, or None if class name couldn't be extracted
        """
        class_name = None
        bases = []

        for child in node.children:
            if child.type == "identifier" and not class_name:
                class_name = child.text.decode("utf-8")
            elif child.type == "argument_list":
                # Use list comprehension instead of appending in a loop
                bases = [
                    base_node.text.decode("utf-8") for base_node in child.children if base_node.type == "identifier"
                ]

        if not class_name:
            logger.warning("Could not extract class name from node")
            return None

        return class_name, bases

    def _initialize_class_symbols(self, symbols: dict[str, Any]) -> None:
        """Initialize class-related data structures in symbols dictionary if they don't exist.

        Args:
            symbols: Dictionary to store extracted symbols
        """
        if "classes" not in symbols:
            symbols["classes"] = []
        if "bases" not in symbols:
            symbols["bases"] = {}
        if "attributes" not in symbols:
            symbols["attributes"] = {}

    def _process_class_body(self, node: Node, class_name: str, symbols: dict[str, Any]) -> None:
        """Process the class body to extract attributes.

        Args:
            node: The class definition node
            class_name: Name of the class
            symbols: Dictionary to store extracted symbols
        """
        body_node = next((c for c in node.children if c.type == "block"), None)
        if body_node:
            symbols["attributes"][class_name] = self._extract_class_attributes(body_node)
            logger.debug("Extracted attributes for %s: %s", class_name, symbols["attributes"][class_name])

    def _extract_class_attributes(self, body_node: Node) -> dict[str, str]:
        """Extract attributes from a class body node.

        Args:
            body_node: The class body node

        Returns:
            Dictionary mapping attribute names to their types
        """
        attributes = {}

        try:
            for stmt in body_node.children:
                # Handle different attribute definition patterns
                if stmt.type == "expression_statement":
                    self._process_expression_statement(stmt, attributes)
                elif stmt.type == "annotated_assignment":
                    self._process_annotated_assignment(stmt, attributes)
        except (AttributeError, UnicodeDecodeError, IndexError, ValueError) as e:
            logger.debug("Error processing attribute statements: %s", e)

        return attributes

    def _process_expression_statement(self, stmt: Node, attributes: dict[str, str]) -> None:
        """Process an expression statement to extract class attributes.

        Args:
            stmt: The expression statement node
            attributes: Dictionary to populate with extracted attributes
        """
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

    def _process_annotated_assignment(self, stmt: Node, attributes: dict[str, str]) -> None:
        """Process an annotated assignment to extract class attributes.

        Args:
            stmt: The annotated assignment node
            attributes: Dictionary to populate with extracted attributes
        """
        target = stmt.children[0] if stmt.children else None
        annotation = next((c for c in stmt.children if c.type == "type"), None)
        if target and target.type == "identifier" and annotation:
            name = target.text.decode("utf-8")
            if not name.startswith("_"):  # Skip private attributes
                # Get the raw annotation text
                attr_type = annotation.text.decode("utf-8")
                # Clean up the type annotation (remove spaces, etc.)
                attr_type = attr_type.strip()
                # Extract the base type from complex annotations like "list[User]"
                attr_type = self._extract_inner_type(attr_type)
                attributes[name] = attr_type

    def _extract_inner_type(self, attr_type: str) -> str:
        """Extract the inner type from a complex type annotation.

        Args:
            attr_type: The type annotation string

        Returns:
            The extracted inner type or the original type
        """
        if "[" in attr_type:
            inner_type = attr_type[attr_type.find("[") + 1 : attr_type.find("]")]
            if inner_type:
                return inner_type
        return attr_type

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
        except (AttributeError, UnicodeDecodeError, IndexError) as e:
            logger.debug("Error inferring type: %s", e)
            return None

        # If we get here, no type could be inferred
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
        except (AttributeError, UnicodeDecodeError) as e:
            logger.debug("Error visiting node %s: %s", node.type, e)
        except ValueError as e:
            logger.debug("Value error processing node %s: %s", node.type, e)
        except IndexError as e:
            logger.debug("Index error processing node %s: %s", node.type, e)
