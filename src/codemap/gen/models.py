"""Models for the code generation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codemap.processor.lod import LODLevel

if TYPE_CHECKING:
	from pathlib import Path
else:
	# Define as string literal for runtime if not type checking
	Path = "pathlib.Path"


@dataclass
class GenConfig:
	"""Configuration settings for the 'gen' command."""

	# Fields without default values
	max_content_length: int
	use_gitignore: bool
	output_dir: Path
	semantic_analysis: bool
	lod_level: LODLevel

	# Fields with default values
	include_tree: bool = True
	include_entity_graph: bool = True
	mermaid_entities: list[str] = field(default_factory=list)
	mermaid_relationships: list[str] = field(default_factory=list)
	mermaid_show_legend: bool = True
	mermaid_remove_unconnected: bool = False
