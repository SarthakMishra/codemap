"""
Code documentation generation package for CodeMap.

This package provides modules for generating LLM-optimized code context
and human-readable documentation.

"""

from .command import GenCommand, process_codebase
from .generator import CodeMapGenerator
from .models import GenConfig

__all__ = [
	"CodeMapGenerator",
	"GenCommand",
	# Classes
	"GenConfig",
	# Functions
	"process_codebase",
]
