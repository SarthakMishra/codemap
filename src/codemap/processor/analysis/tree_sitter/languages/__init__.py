"""Language-specific configurations for syntax chunking."""

from codemap.processor.analysis.tree_sitter.languages.base import LanguageConfig, LanguageSyntaxHandler
from codemap.processor.analysis.tree_sitter.languages.javascript import JAVASCRIPT_CONFIG, JavaScriptSyntaxHandler
from codemap.processor.analysis.tree_sitter.languages.python import PYTHON_CONFIG, PythonSyntaxHandler
from codemap.processor.analysis.tree_sitter.languages.typescript import TYPESCRIPT_CONFIG, TypeScriptSyntaxHandler

# Map language names to their configs
LANGUAGE_CONFIGS = {
	"python": PYTHON_CONFIG,
	"javascript": JAVASCRIPT_CONFIG,
	"typescript": TYPESCRIPT_CONFIG,
	# Add other languages when implemented
}

# Map language names to their handler classes
LANGUAGE_HANDLERS = {
	"python": PythonSyntaxHandler,
	"javascript": JavaScriptSyntaxHandler,
	"typescript": TypeScriptSyntaxHandler,
	# Add other language handlers when implemented
}

__all__ = ["LANGUAGE_CONFIGS", "LANGUAGE_HANDLERS", "LanguageConfig", "LanguageSyntaxHandler"]
