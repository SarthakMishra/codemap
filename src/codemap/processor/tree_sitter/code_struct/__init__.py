"""CodeStruct notation converter module.

This module provides tools for converting code entities into CodeStruct notation,
a plain-text, human- and machine-readable format for describing code structure.
"""

from .converter import (
	entities_to_codestruct,
	entity_to_codestruct,
	generate_legend,
	minify_codestruct,
)

__all__ = [
	"entities_to_codestruct",
	"entity_to_codestruct",
	"generate_legend",
	"minify_codestruct",
]
