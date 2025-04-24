"""Tree-sitter analysis module."""

from codemap.processor.analysis.tree_sitter.analyzer import TreeSitterAnalyzer, get_language_by_extension
from codemap.processor.analysis.tree_sitter.base import EntityType

__all__ = ["EntityType", "TreeSitterAnalyzer", "get_language_by_extension"]
