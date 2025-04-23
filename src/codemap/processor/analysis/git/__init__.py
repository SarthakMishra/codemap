"""Git metadata analysis module.

This module provides functionality for extracting Git-related metadata from code,
such as commit history, authorship, and last modification details.
"""

from codemap.processor.analysis.git.analyzer import GitMetadataAnalyzer
from codemap.processor.analysis.git.models import GitMetadata

__all__ = ["GitMetadata", "GitMetadataAnalyzer"]
