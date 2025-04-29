"""
Code documentation generation package for CodeMap.

This package provides modules for generating LLM-optimized code context
and human-readable documentation.

"""

from .command import GenCommand, process_codebase
from .compressor import SemanticCompressor
from .generator import CodeMapGenerator
from .models import CompressionStrategy, DocFormat, GenConfig, GenerationMode

__all__ = [
	"CodeMapGenerator",
	"CompressionStrategy",
	"DocFormat",
	"GenCommand",
	# Classes
	"GenConfig",
	# Enums
	"GenerationMode",
	"SemanticCompressor",
	# Functions
	"process_codebase",
]
