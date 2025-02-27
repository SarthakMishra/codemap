"""Markdown documentation generation for the CodeMap tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codemap.analyzer.tree_parser import CodeParser


@dataclass
class TreeState:
    """State for tree generation."""

    included_files: set[Path]
    parser: CodeParser
    tree: list[str]
    max_depth: int


class MarkdownGenerator:
    """Generates markdown documentation from parsed code files."""

    # Maximum depth for directory traversal to prevent infinite recursion
    MAX_TREE_DEPTH = 20

    def __init__(self, repo_root: Path, config: dict[str, Any]) -> None:
        """Initialize the markdown generator.

        Args:
            repo_root: Root directory of the repository.
            config: Configuration dictionary for documentation generation.
        """
        self.repo_root = repo_root
        self.config = config

    def _find_repo_root(self, start_path: Path, max_levels: int = 5) -> Path:
        """Find the repository root by looking for .git directory.

        Args:
            start_path: Path to start searching from.
            max_levels: Maximum number of parent directories to check.

        Returns:
            Repository root path.
        """
        current_root = start_path
        for _ in range(max_levels):
            if (current_root / ".git").exists():
                return current_root
            if current_root.parent == current_root:  # reached filesystem root
                break
            current_root = current_root.parent
        return start_path

    def _should_process_path(self, path: Path, parser: CodeParser) -> bool:
        """Check if a path should be processed based on configuration.

        Args:
            path: Path to check.
            parser: CodeParser instance for checking parse rules.

        Returns:
            True if the path should be processed.
        """
        # Skip directories that match exclude patterns
        for pattern in self.config.get("exclude_patterns", []):
            if self._matches_pattern(path, pattern):
                return False

        # Skip files that match gitignore patterns if enabled
        if self.config.get("use_gitignore", True) and parser.gitignore_patterns:
            for pattern in parser.gitignore_patterns:
                if self._matches_pattern(path, pattern):
                    return False

        return True

    def _add_path_to_tree(
        self,
        path: Path,
        state: TreeState,
        prefix: str = "",
        depth: int = 0,
        *,
        is_last: bool = True,
    ) -> bool:
        """Add a path and its children to the tree representation.

        Args:
            path: Path to add.
            state: Tree generation state.
            prefix: Current tree prefix for formatting.
            depth: Current depth in tree.
            is_last: Whether this is the last item in its parent directory.

        Returns:
            True if all parseable files in this path (and subdirectories) are included.
        """
        if depth > state.max_depth:
            return False

        if not self._should_process_path(path, state.parser):
            return False

        # Special handling for root directory
        display_name = "root" if depth == 0 else path.name

        # Determine the prefix symbol based on whether this is the last item
        prefix_symbol = "└──" if is_last else "├──"

        if path.is_file():
            # Show all files with checkboxes
            is_included = path.resolve() in state.included_files
            can_parse = state.parser.should_parse(path)
            # Files that can be parsed should show inclusion status
            checkbox = "[x]" if (can_parse and is_included) else "[ ]"
            state.tree.append(f"{prefix}{prefix_symbol} {checkbox} {display_name}")
            return is_included if can_parse else True

        # Handle directory
        try:
            children = sorted(path.iterdir())
        except (PermissionError, OSError):
            # Skip directories we can't read or that have too many symlinks
            state.tree.append(f"{prefix}{prefix_symbol} [ ] {display_name}/")
            return False

        # Process all children first to determine if all are included
        all_parseable_included = True
        has_parseable = False

        # Process children but don't add to tree yet
        child_results = []
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            child_included = self._add_path_to_tree(
                child,
                TreeState(
                    included_files=state.included_files,
                    parser=state.parser,
                    tree=[],  # Temporary tree for child
                    max_depth=state.max_depth,
                ),
                prefix + ("    " if is_last else "│   "),
                depth + 1,
                is_last=is_last_child,
            )
            if state.parser.should_parse(child):
                has_parseable = True
                if not child_included:
                    all_parseable_included = False
            child_results.append((child, child_included, is_last_child))

        # Add directory with appropriate checkbox
        checkbox = "[x]" if (has_parseable and all_parseable_included) else "[ ]"
        state.tree.append(f"{prefix}{prefix_symbol} {checkbox} {display_name}/")

        # Now add all children to the real tree
        for child, _, is_last_child in child_results:
            self._add_path_to_tree(
                child,
                state,
                prefix + ("    " if is_last else "│   "),
                depth + 1,
                is_last=is_last_child,
            )

        return all_parseable_included

    def _generate_file_tree(self, parsed_files: dict[Path, dict[str, Any]], repo_root: Path) -> str:
        """Generate a tree representation of the repository structure with checkboxes.

        Args:
            parsed_files: Dictionary mapping file paths to their parsed contents.
            repo_root: Root directory of the repository.

        Returns:
            Generated tree representation with checkboxes.
        """
        tree = ["```markdown"]
        state = TreeState(
            included_files={file_path.resolve() for file_path in parsed_files},
            parser=CodeParser(self.config),
            tree=tree,
            max_depth=self.MAX_TREE_DEPTH,
        )

        root_path = self._find_repo_root(repo_root)
        self._add_path_to_tree(root_path, state)
        tree.append("```")
        return "\n".join(tree)

    def _matches_pattern(self, path: Path, pattern: str) -> bool:
        """Check if a path matches a glob pattern.

        Args:
            path: Path to check.
            pattern: Glob pattern to match against.

        Returns:
            True if the path matches the pattern.
        """
        import fnmatch
        import os

        # Convert pattern and path to forward slashes for consistency
        pattern = pattern.replace(os.sep, "/")
        path_str = str(path).replace(os.sep, "/")

        # Handle directory patterns (ending with /)
        if pattern.endswith("/"):
            return any(part == pattern.rstrip("/") for part in Path(path_str).parts)

        # Match against the full path and just the name
        return fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path.name, pattern)

    def _generate_header(self) -> str:
        """Generate the header section of the documentation.

        Returns:
            Header section as a string.
        """
        return "# Project Documentation\n\n"

    def _generate_overview(self, parsed_files: dict[Path, dict[str, Any]]) -> str:
        """Generate overview section.

        Args:
            parsed_files: Dictionary mapping file paths to their parsed contents.

        Returns:
            Generated overview section.
        """
        overview = ["## Overview\n"]
        if not parsed_files:
            overview.append("No files found for analysis.\n")
            return "\n".join(overview)

        overview.append("This documentation was generated by CodeMap.\n")
        overview.append(f"Total files analyzed: {len(parsed_files)}\n")

        overview.append("\n## File Structure\n")
        overview.append("Files marked with [x] are included in this documentation:\n")
        overview.append(
            self._generate_file_tree(parsed_files, next(iter(parsed_files)).parent if parsed_files else Path.cwd()),
        )

        return "\n".join(overview)

    def _generate_dependencies(self, parsed_files: dict[Path, dict[str, Any]]) -> str:
        """Generate dependencies section.

        Args:
            parsed_files: Dictionary mapping file paths to their parsed contents.

        Returns:
            Generated dependencies section.
        """
        deps = ["\n\n## Dependencies\n"]
        all_imports = set()
        for symbols in parsed_files.values():
            all_imports.update(symbols.get("imports", []))

        if all_imports:
            deps.append("### External Dependencies\n")
            deps.extend(f"- {imp}" for imp in sorted(all_imports))

        return "\n".join(deps)

    def _get_language_for_file(self, file_path: Path) -> str:
        """Get the markdown code block language identifier based on file extension.

        Args:
            file_path: Path to the file.

        Returns:
            Language identifier for markdown code block.
        """
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sql": "sql",
            ".xml": "xml",
            ".toml": "toml",
        }
        return extension_map.get(file_path.suffix.lower(), "")

    def _escape_markdown(self, text: str) -> str:
        """Escape markdown special characters that could affect formatting.

        Only escapes characters that could be interpreted as markdown syntax
        in regular text (not in code blocks or headings).

        Args:
            text: Text to escape.

        Returns:
            Escaped text.
        """
        # Only escape characters that could be interpreted as markdown syntax
        # in regular text (not in code blocks or headings)
        special_chars = ["*", "_", "`"]  # These are the main ones that affect inline formatting
        escaped_text = text
        for char in special_chars:
            escaped_text = escaped_text.replace(char, f"\\{char}")
        return escaped_text

    def _generate_file_documentation(self, file_path: Path, symbols: dict[str, Any]) -> str:
        """Generate documentation for a single file.

        Args:
            file_path: Path to the file being documented.
            symbols: Dictionary containing parsed symbols from the file.

        Returns:
            Generated markdown documentation for the file.
        """
        docs = []

        if "docstring" in symbols:
            # Escape docstrings since they can contain markdown formatting
            docs.append(self._escape_markdown(symbols["docstring"]))
            docs.append("")

        if "content" in symbols and symbols.get("content", "").strip():
            # Include the full content without truncation
            content = symbols["content"]
            language = self._get_language_for_file(file_path)
            docs.append(f"```{language}")
            docs.append(content)
            docs.append("```")
            docs.append("")

        if "classes" in symbols:
            for class_name in symbols["classes"]:
                # No need to escape in headings
                docs.append(f"#### {class_name}")
                if class_name in symbols.get("attributes", {}):
                    docs.append("\nAttributes:")
                    for attr, attr_type in symbols["attributes"][class_name].items():
                        docs.append(f"- {attr}: {attr_type}")
                docs.append("")

        if "functions" in symbols:
            for func_name in symbols["functions"]:
                # No need to escape in headings
                docs.append(f"#### {func_name}")
                docs.append("")

        return "\n".join(docs)

    def generate_documentation(self, parsed_files: dict[Path, dict[str, Any]]) -> str:
        """Generate markdown documentation from parsed files.

        Args:
            parsed_files: Dictionary mapping file paths to their parsed contents.

        Returns:
            Generated markdown documentation as a string.
        """
        markdown = [self._generate_header()]

        # Generate sections based on config
        sections = self.config.get("sections", ["overview", "dependencies", "details"])

        for section in sections:
            if section == "overview":
                markdown.append(self._generate_overview(parsed_files))
            elif section == "dependencies":
                markdown.append(self._generate_dependencies(parsed_files))
            elif section == "details":
                # Sort files by importance score if available, defaulting to 0
                sorted_files = sorted(
                    parsed_files.items(),
                    key=lambda x: (
                        # Primary sort by importance score (descending)
                        -(x[1].get("importance_score", 0) or 0),
                        # Secondary sort by file path (ascending) for stable ordering
                        str(x[0]),
                    ),
                )

                markdown.append("\n\n## Details\n")
                for file_path, symbols in sorted_files:
                    rel_path = file_path.relative_to(self.repo_root)
                    # No need to escape in headings
                    markdown.append(f"\n### {rel_path}\n")
                    file_docs = self._generate_file_documentation(file_path, symbols)
                    markdown.append(file_docs)
            else:
                # Custom section
                section_title = section.replace("_", " ").title()
                markdown.append(f"\n\n## {section_title}\n")

        return "\n".join(markdown)
