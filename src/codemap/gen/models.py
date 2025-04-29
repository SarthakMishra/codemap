"""Models for the code generation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GenerationMode(str, Enum):
	"""Generation mode for the gen command."""

	LLM = "llm"
	HUMAN = "human"


class CompressionStrategy(str, Enum):
	"""Compression strategy for the gen command."""

	SMART = "smart"
	AGGRESSIVE = "aggressive"
	MINIMAL = "minimal"
	NONE = "none"


class DocFormat(str, Enum):
	"""Documentation format for human-readable docs."""

	MARKDOWN = "markdown"
	MINTLIFY = "mintlify"
	MKDOCS = "mkdocs"
	MDX = "mdx"


@dataclass
class GenConfig:
	"""Configuration for the gen command."""

	mode: GenerationMode
	compression_strategy: CompressionStrategy
	doc_format: DocFormat
	token_limit: int
	max_content_length: int
	include_tree: bool
	semantic_analysis: bool
	auto_start_daemon: bool = field(default=False)
