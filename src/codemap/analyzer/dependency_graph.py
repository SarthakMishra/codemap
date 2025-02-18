"""Dependency analysis and graph generation for the codebase."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx.exception import NetworkXError

if TYPE_CHECKING:
    from pathlib import Path


class DependencyGraph:
    """Analyzes and builds dependency relationships between code files."""

    def __init__(self, repo_root: Path) -> None:
        """Initialize the dependency graph.

        Args:
            repo_root: Root directory of the repository.
        """
        self.repo_root = repo_root
        self.graph = nx.DiGraph()

    def build_graph(self, parsed_files: dict[Path, dict[str, Any]]) -> None:
        """Build a dependency graph from parsed files.

        Args:
            parsed_files: Dictionary mapping file paths to their parsed contents.
        """
        # Add all files as nodes first
        for file_path in parsed_files:
            self.graph.add_node(file_path)

        # Then add edges based on imports and references
        for file_path, symbols in parsed_files.items():
            if not symbols:  # Skip if parsing failed
                continue

            # Add edges based on imports and references
            for imp in symbols.get("imports", []):
                # Look for matching files
                imp_name = imp.split(".")[-1]  # Get the last part of the import
                for target_path in parsed_files:
                    if target_path.stem in {imp, imp_name}:
                        self.graph.add_edge(file_path, target_path)

            # Add edges from references if they exist
            for ref in symbols.get("references", []):
                target_file = ref.get("target_file")
                if target_file and target_file in parsed_files:
                    self.graph.add_edge(file_path, target_file)

    def _count_tokens(self, file_path: Path) -> int:
        """Rough estimation of tokens in a file.

        Args:
            file_path: Path to the file to count tokens in.

        Returns:
            Estimated number of tokens in the file.
        """
        try:
            with file_path.open(encoding="utf-8") as f:
                content = f.read()
                # Simple tokenization by whitespace
                return len(content.split())
        except OSError:
            return 0

    def get_important_files(self, token_limit: int) -> list[Path]:
        """Get the most important files based on dependencies.

        Args:
            token_limit: Maximum number of tokens to include.

        Returns:
            List of file paths sorted by importance.
        """
        if not self.graph.nodes:
            return []

        try:
            # Calculate PageRank scores
            scores = nx.pagerank(self.graph)

            # Sort files by score
            sorted_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Filter files based on token limit
            total_tokens = 0
            important_files = []
            for path, _ in sorted_files:
                tokens = self._count_tokens(path)
                if total_tokens + tokens <= token_limit:
                    important_files.append(path)
                    total_tokens += tokens
                else:
                    break
        except NetworkXError:
            # If PageRank fails (e.g., no edges), return all files
            return list(self.graph.nodes)
        else:
            return important_files
