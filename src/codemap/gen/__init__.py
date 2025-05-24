"""
Code documentation generation package for CodeMap.

This package provides modules for generating LLM-optimized code context
and human-readable documentation.

"""

from .command import GenCommand, process_codebase
from .generator import CodeMapGenerator

__all__ = [
	"CodeMapGenerator",
	"GenCommand",
	# Functions
	"process_codebase",
]
