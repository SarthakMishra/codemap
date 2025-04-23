"""Language-specific configurations for syntax chunking."""

from codemap.processor.chunking.languages.base import LanguageConfig, LanguageSyntaxHandler
from codemap.processor.chunking.languages.python import PYTHON_CONFIG, PythonSyntaxHandler

# Map language names to their configs
LANGUAGE_CONFIGS = {
    "python": PYTHON_CONFIG,
    # Add other languages when implemented
}

# Map language names to their handler classes
LANGUAGE_HANDLERS = {
    "python": PythonSyntaxHandler,
    # Add other language handlers when implemented
}

__all__ = ["LANGUAGE_CONFIGS", "LANGUAGE_HANDLERS", "LanguageConfig", "LanguageSyntaxHandler"]
