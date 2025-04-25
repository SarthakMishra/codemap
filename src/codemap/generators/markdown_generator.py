"""Markdown documentation generation for the CodeMap tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codemap.analyzer.tree_parser import CodeParser

# Constants
MAX_TREE_DEPTH = 5
MAX_CONTENT_LENGTH = 5000


@dataclass
class TreeState:
    """State for tree generation."""

    included_files: set[Path]
    parser: CodeParser
    tree: dict[str, Any]
    max_depth: int


class MarkdownGenerator:
    """Generates markdown documentation for a code repository."""

    def __init__(self, repo_path: Path, config: dict[str, Any] | None = None) -> None:
        """Initialize the markdown generator.

        Args:
            repo_path: Path to the repository or specific target to analyze
            config: Configuration dictionary

        """
        self.target_path = Path(repo_path).resolve()
        self.config = config or {}
        self.max_tree_depth = self.config.get("max_tree_depth", MAX_TREE_DEPTH)
        self.max_content_length = self.config.get("max_content_length", MAX_CONTENT_LENGTH)
        self.repo_root = self._find_repo_root(self.target_path)

        # Default excluded directories
        self.default_excluded = [
            "__pycache__",
            ".git",
            ".env",
            ".venv",
            "venv",
            "node_modules",
            "build",
            "dist",
        ]

    def _find_repo_root(self, start_path: Path) -> Path:
        """Find the root of the repository by looking for .git directory.

        Args:
            start_path: Path to start searching from

        Returns:
            Path to the repository root

        """
        # If the start_path is a file or a specific directory that exists,
        # use its parent or the path itself as the root
        if start_path.is_file():
            return start_path.parent
        if start_path.is_dir() and start_path.exists():
            return start_path

        # Otherwise, try to find the Git root
        current_path = start_path
        max_depth = 10  # Guard against infinite loop

        for _ in range(max_depth):
            if (current_path / ".git").exists():
                return current_path

            # Stop if we've reached the filesystem root
            parent_path = current_path.parent
            if parent_path == current_path:
                break

            current_path = parent_path

        # If we can't find a .git directory, just use the start_path
        return start_path

    def _should_process_path(self, path: Path, parser: CodeParser) -> bool:
        """Check if a path should be processed.

        Args:
            path: Path to check.
            parser: CodeParser instance for checking parse rules.

        Returns:
            True if the path should be processed.

        """
        # Skip files that match gitignore patterns if enabled
        if self.config.get("use_gitignore", True) and parser.file_filter.gitignore_patterns:
            for pattern in parser.file_filter.gitignore_patterns:
                if self._matches_gitignore_pattern(path, pattern, parser):
                    return False

        # Skip default excluded directories
        return all(excluded not in str(path) for excluded in self.default_excluded)

    def _matches_gitignore_pattern(self, path: Path, pattern: str, parser: CodeParser) -> bool:
        """Check if a path matches a gitignore pattern using the parser's file
        filter.

        Args:
            path: Path to check
            pattern: Gitignore pattern to match against
            parser: The CodeParser instance with the matching method

        Returns:
            True if the path matches the pattern

        """
        # Use the file_filter's matches_pattern method
        return parser.file_filter.matches_pattern(path, pattern)

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
            can_parse = state.parser.file_filter.should_parse(path)
            # Files that can be parsed should show inclusion status
            checkbox = "[x]" if (can_parse and is_included) else "[ ]"
            state.tree["content"][str(path)] = f"{prefix}{prefix_symbol} {checkbox} {display_name}"
            return is_included if can_parse else True

        # Handle directory
        try:
            children = sorted(path.iterdir())
        except (PermissionError, OSError):
            # Skip directories we can't read
            state.tree["content"][str(path)] = f"{prefix}{prefix_symbol} [ ] {display_name}/"
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
                    tree={},  # Temporary tree for child
                    max_depth=state.max_depth,
                ),
                prefix + ("    " if is_last else "│   "),
                depth + 1,
                is_last=is_last_child,
            )
            if state.parser.file_filter.should_parse(child):
                has_parseable = True
                if not child_included:
                    all_parseable_included = False
            child_results.append((child, child_included, is_last_child))

        # Add directory with appropriate checkbox
        checkbox = "[x]" if (has_parseable and all_parseable_included) else "[ ]"
        state.tree["content"][str(path)] = f"{prefix}{prefix_symbol} {checkbox} {display_name}/"

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

    def _generate_file_tree(self, tree: dict[str, Any]) -> str:
        """Generate a tree representation of the repository structure with
        checkboxes.

        Args:
            tree: Dictionary representing the tree structure or parsed files

        Returns:
            Generated tree representation with checkboxes.

        """
        # Check if we're getting a tree or parsed_files dictionary
        if "content" in tree:
            # It's a proper tree structure with content
            return "\n".join(list(tree["content"].values()))

        # It's a parsed_files dictionary, generate a simple list
        lines = []
        for file_path in sorted(tree.keys()):
            # Get relative path from target_path for better display
            # Determine if file is within target path first to avoid try-except in loop
            if str(file_path).startswith(str(self.target_path)):
                try:
                    # Ensure file_path is a Path object
                    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
                    rel_path = path_obj.relative_to(self.target_path)
                    lines.append(f"- {rel_path}")
                except ValueError:
                    # Fallback in case of unexpected error
                    lines.append(f"- {file_path}")
            else:
                # File is not under target_path, show full path
                lines.append(f"- {file_path}")
        return "\n".join(lines)

    def _generate_header(self) -> str:
        """Generate the document header."""
        return "# Code Map\n\n_Generated documentation of the codebase structure and files._\n\n"

    def _generate_overview(self, parsed_files: dict[Path, dict[str, Any]]) -> str:
        """Generate the overview section with file counts and repository
        structure.

        Args:
            parsed_files: Dictionary mapping file paths to their parsed contents.

        Returns:
            Generated overview section.

        """
        # Count files by language
        language_counts = {}
        for file_info in parsed_files.values():
            lang = file_info.get("language", "unknown")
            language_counts[lang] = language_counts.get(lang, 0) + 1

        overview = ["## Overview\n"]
        overview.append(f"**Total Files:** {len(parsed_files)}\n")

        if language_counts:
            overview.append("\n**Files by Language:**\n")
            for lang, count in sorted(language_counts.items()):
                overview.append(f"- {lang.capitalize()}: {count}\n")

        return "\n".join(overview)

    def _generate_file_documentation(self, file_path: Path, file_info: dict[str, Any]) -> str:
        """Generate documentation for a single file.

        Args:
            file_path: Path to the file.
            file_info: Dictionary with parsed file information.

        Returns:
            Generated file documentation.

        """
        rel_path = file_path.relative_to(self.repo_root)
        language = file_info.get("language", "unknown")

        docs = [f"### {rel_path}\n"]
        docs.append(f"**Language:** {language.capitalize()}\n")

        # Add classes if any (Python only)
        if language == "python" and file_info.get("classes"):
            docs.append("\n**Classes:**\n")
            class_items = [f"- `{cls}`\n" for cls in sorted(file_info.get("classes", []))]
            docs.extend(class_items)

        # Add imports if any (Python only)
        if language == "python" and file_info.get("imports"):
            docs.append("\n**Imports:**\n")
            import_items = [f"- `{imp}`\n" for imp in sorted(file_info.get("imports", []))]
            docs.extend(import_items)

        # Add file content with syntax highlighting
        content = file_info.get("content", "")
        if content:
            # Use the correct language for syntax highlighting
            highlight_lang = language if language != "unknown" else file_path.suffix.lstrip(".")
            # Truncate very large files if max_content_length is set and not zero (infinite)
            if self.max_content_length > 0 and len(content) > self.max_content_length:
                content = content[: self.max_content_length] + "\n...\n[Content truncated for brevity]"
            docs.append(f"\n```{highlight_lang}\n{content}\n```\n")

        return "\n".join(docs)

    def generate_documentation(self, parsed_files: dict[Path, dict[str, Any]]) -> str:
        """Generate markdown documentation for the codebase.

        Args:
            parsed_files: Dictionary of parsed files

        Returns:
            Markdown documentation as a string

        """
        parts: list[str] = []

        # Start directly with overview (removing header)
        parts.append(self._generate_overview(parsed_files))

        # Generate file tree
        parts.append("## 📁 Project Structure")
        parts.append(self.generate_tree(self.target_path, parsed_files=set(parsed_files.keys())))

        # Generate documentation for each file
        parts.append("## 📄 Files")
        for file_path, file_info in sorted(parsed_files.items()):
            parts.append(self._generate_file_documentation(file_path, file_info))

        return "\n\n".join(parts)

    def _add_clean_path_to_tree(
        self,
        path: Path,
        state: TreeState,
        prefix: str = "",
        depth: int = 0,
        *,
        is_last: bool = True,
    ) -> None:
        """Add a path to the clean tree representation (without checkboxes).

        Args:
            path: Path to add.
            state: Tree generation state.
            prefix: Current tree prefix for formatting.
            depth: Current depth in tree.
            is_last: Whether this is the last item in its parent directory.

        """
        if depth > state.max_depth:
            return

        if not self._should_process_path(path, state.parser):
            return

        # Special handling for root directory
        display_name = "root" if depth == 0 else path.name

        # Determine the prefix symbol based on whether this is the last item
        prefix_symbol = "└──" if is_last else "├──"

        if path.is_file():
            state.tree["content"][str(path)] = f"{prefix}{prefix_symbol} {display_name}"
            return

        # Handle directory
        try:
            children = sorted(path.iterdir())
        except (PermissionError, OSError):
            # Skip directories we can't read
            state.tree["content"][str(path)] = f"{prefix}{prefix_symbol} {display_name}/"
            return

        # Add directory to tree
        state.tree["content"][str(path)] = f"{prefix}{prefix_symbol} {display_name}/"

        # Process children
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            self._add_clean_path_to_tree(
                child,
                state,
                prefix + ("    " if is_last else "│   "),
                depth + 1,
                is_last=is_last_child,
            )

    def generate_tree(self, path: Path | None = None, parsed_files: set[Path] | None = None) -> str:
        """Generate a tree structure of the repository.

        Args:
            path: Root path to generate tree from, defaults to repo root
            parsed_files: Set of parsed file paths to highlight in the tree

        Returns:
            Markdown-formatted tree structure

        """
        # Use target path by default
        root_path = path or self.target_path

        # When parsed_files is specified, create a simplified tree showing only those files
        if parsed_files is not None:
            # Create a tree manually from parsed_files
            lines = []
            prefix = "└── "

            # Map of directory to files in that directory
            dir_files = {}

            # Group files by directory
            for file_path in sorted(parsed_files):
                # Get path relative to the root path
                # Skip files that are not under root_path
                if not self._is_path_under_root(file_path, root_path):
                    continue

                rel_path = file_path.relative_to(root_path)
                parent = str(rel_path.parent)
                # Add file to its parent directory
                if parent not in dir_files:
                    dir_files[parent] = []
                dir_files[parent].append(rel_path.name)

            # Build a simple tree of included files
            for directory, files in sorted(dir_files.items()):
                if directory != ".":
                    lines.append(f"{prefix}{directory}/")
                    for i, file in enumerate(sorted(files)):
                        file_prefix = "    └── " if i == len(files) - 1 else "    ├── "
                        lines.append(f"{prefix}{file_prefix}{file}")
                else:
                    # Root directory files
                    for i, file in enumerate(sorted(files)):
                        file_prefix = "└── " if i == len(files) - 1 else "├── "
                        lines.append(f"{file_prefix}{file}")

            return "\n".join(lines)

        # If no parsed_files specified, use the standard tree generation
        parser = CodeParser(self.config)

        state = TreeState(
            included_files=parsed_files or set(),
            parser=parser,
            tree={"name": root_path.name, "content": {}, "is_dir": True},
            max_depth=self.max_tree_depth,
        )

        # Use the clean path tree method to build the tree
        self._add_clean_path_to_tree(root_path, state)

        # Convert tree to markdown
        return self._generate_file_tree(state.tree)

    def _is_path_under_root(self, path: Path, root_path: Path) -> bool:
        """Check if a path is under the root path.

        Args:
            path: The path to check
            root_path: The root path

        Returns:
            True if the path is under the root path, False otherwise

        """
        try:
            path.relative_to(root_path)
        except ValueError:
            return False
        else:
            return True
