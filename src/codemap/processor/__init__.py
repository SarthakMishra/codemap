"""
Code processing modules for CodeMap.

This package contains modules for processing and analyzing code:
- chunking: Strategies for breaking code into semantic chunks
- analysis: Tools for analyzing code structure and metadata
- embedding: Tools for generating vector embeddings of code

"""

from codemap.processor.pipeline import ProcessingPipeline

__all__ = ["ProcessingPipeline"]
