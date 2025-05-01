"""
Tree-sitter based code analysis.

This module provides functionality for analyzing source code using tree-
sitter. It extracts structure and semantic information from code files
in various programming languages.

"""

from __future__ import annotations

from codemap.processor.tree_sitter.analyzer import TreeSitterAnalyzer
from codemap.processor.tree_sitter.base import EntityType

__all__ = ["EntityType", "TreeSitterAnalyzer"]
