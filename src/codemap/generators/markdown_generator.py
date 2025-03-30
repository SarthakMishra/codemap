"""Markdown documentation generation for the CodeMap tool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codemap.analyzer.tree_parser import CodeParser

if TYPE_CHECKING:
    from pathlib import Path

# Constants
MAX_TREE_DEPTH = 20
MAX_CONTENT_LENGTH = 10000  # ~100 lines


@dataclass
class TreeState:
    """State for tree generation."""

    included_files: set[Path]
    parser: CodeParser
    tree: list[str]
    max_depth: int


class MarkdownGenerator:
    """Generates markdown documentation from parsed code files."""

    def __init__(self, repo_root: Path, config: dict[str, Any]) -> None:
        """Initialize the markdown generator.

        Args:
            repo_root: Root directory of the repository.
            config: Configuration dictionary for documentation generation.
        """
        self.repo_root = repo_root
        self.config = config
        self.max_tree_depth = MAX_TREE_DEPTH

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
        """Check if a path should be processed.

        Args:
            path: Path to check.
            parser: CodeParser instance for checking parse rules.

        Returns:
            True if the path should be processed.
        """
        # Skip files that match gitignore patterns if enabled
        if self.config.get("use_gitignore", True) and parser.gitignore_patterns:
            for pattern in parser.gitignore_patterns:
                if self._matches_gitignore_pattern(path, pattern, parser):
                    return False

        # Skip default excluded directories
        default_excluded = ["__pycache__", ".git", ".env", ".venv", "venv", "build", "dist"]
        return all(excluded not in str(path) for excluded in default_excluded)

    def _matches_gitignore_pattern(self, path: Path, pattern: str, parser: CodeParser) -> bool:
        """Check if a path matches a gitignore pattern using the parser's method.

        Args:
            path: Path to check
            pattern: Gitignore pattern to match against
            parser: The CodeParser instance with the matching method

        Returns:
            True if the path matches the pattern
        """
        # This is a wrapper around the private method to avoid the linter warning
        return parser.matches_pattern(path, pattern)

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
            # Skip directories we can't read
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
            max_depth=self.max_tree_depth,
        )

        root_path = self._find_repo_root(repo_root)
        self._add_path_to_tree(root_path, state)
        tree.append("```")
        return "\n".join(tree)

    def _generate_header(self) -> str:
        """Generate the document header."""
        return "# Code Map\n\n_Generated documentation of the codebase structure and files._\n\n"

    def _generate_overview(self, parsed_files: dict[Path, dict[str, Any]]) -> str:
        """Generate the overview section with file counts and repository structure.

        Args:
            parsed_files: Dictionary mapping file paths to their parsed contents.

        Returns:
            Generated overview section.
        """
        total_files = len(parsed_files)

        overview = ["## Overview\n"]
        overview.append(f"**Total Files:** {total_files}\n")
        overview.append("\n### Repository Structure\n")
        overview.append(self._generate_file_tree(parsed_files, self.repo_root))

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

        docs = [f"### {rel_path}\n"]

        # Add classes if any
        if file_info.get("classes"):
            docs.append("\n**Classes:**\n")
            class_items = [f"- `{cls}`\n" for cls in sorted(file_info.get("classes", []))]
            docs.extend(class_items)

        # Add imports if any
        if file_info.get("imports"):
            docs.append("\n**Imports:**\n")
            import_items = [f"- `{imp}`\n" for imp in sorted(file_info.get("imports", []))]
            docs.extend(import_items)

        # Add file content with syntax highlighting
        content = file_info.get("content", "")
        if content:
            file_ext = file_path.suffix.lstrip(".")
            # Truncate very large files
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "\n...\n[Content truncated for brevity]"
            docs.append(f"\n```{file_ext}\n{content}\n```\n")

        return "\n".join(docs)

    def generate_documentation(self, parsed_files: dict[Path, dict[str, Any]]) -> str:
        """Generate complete documentation for the codebase.

        Args:
            parsed_files: Dictionary mapping file paths to their parsed contents.

        Returns:
            Generated documentation.
        """
        sections = []

        # Add header
        sections.append(self._generate_header())

        # Add overview section
        sections.append(self._generate_overview(parsed_files))

        # Add file details
        sections.append("## File Details\n")
        for file_path, file_info in sorted(parsed_files.items()):
            sections.append(self._generate_file_documentation(file_path, file_info))

        return "\n\n".join(sections)

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
            state.tree.append(f"{prefix}{prefix_symbol} {display_name}")
            return

        # Handle directory
        try:
            children = sorted(path.iterdir())
        except (PermissionError, OSError):
            # Skip directories we can't read
            state.tree.append(f"{prefix}{prefix_symbol} {display_name}/")
            return

        # Add directory to tree
        state.tree.append(f"{prefix}{prefix_symbol} {display_name}/")

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

    def generate_tree(self, repo_root: Path) -> str:
        """Generate a clean tree representation without checkboxes.

        Args:
            repo_root: Root directory of the repository.

        Returns:
            Generated tree representation without checkboxes.
        """
        tree = []
        state = TreeState(
            included_files=set(),  # Not used for clean tree
            parser=CodeParser(self.config),
            tree=tree,
            max_depth=self.max_tree_depth,
        )

        root_path = self._find_repo_root(repo_root)
        self._add_clean_path_to_tree(root_path, state)
        return "\n".join(tree)
