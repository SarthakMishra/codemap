"""Models for the code generation module."""

from __future__ import annotations

from dataclasses import dataclass

from codemap.processor.lod import LODLevel


@dataclass
class GenConfig:
	"""Configuration for the gen command."""

	lod_level: LODLevel
	max_content_length: int
	include_tree: bool
	semantic_analysis: bool
