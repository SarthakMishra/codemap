"""
Language-specific configurations and handlers for tree-sitter analysis.

This module contains language-specific configurations and handlers that
define how to analyze different programming languages using tree-sitter.

"""

from __future__ import annotations

from codemap.processor.tree_sitter.languages.base import LanguageConfig, LanguageSyntaxHandler
from codemap.processor.tree_sitter.languages.javascript import JAVASCRIPT_CONFIG, JavaScriptSyntaxHandler
from codemap.processor.tree_sitter.languages.python import PYTHON_CONFIG, PythonSyntaxHandler
from codemap.processor.tree_sitter.languages.typescript import TYPESCRIPT_CONFIG, TypeScriptSyntaxHandler

__all__ = [
	"LANGUAGE_CONFIGS",
	"LANGUAGE_HANDLERS",
	"LanguageConfig",
	"LanguageSyntaxHandler",
]

# Dictionary of language configurations
LANGUAGE_CONFIGS = {
	"python": PYTHON_CONFIG,
	"javascript": JAVASCRIPT_CONFIG,
	"typescript": TYPESCRIPT_CONFIG,
}

# Dictionary of language handler classes
LANGUAGE_HANDLERS = {
	"python": PythonSyntaxHandler,
	"javascript": JavaScriptSyntaxHandler,
	"typescript": TypeScriptSyntaxHandler,
}
