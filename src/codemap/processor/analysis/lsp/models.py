"""Data models for LSP analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LSPReference:
	"""Represents a reference to another code entity found by the LSP."""

	target_name: str
	"""The name of the referenced entity."""

	target_uri: str
	"""The URI of the file containing the referenced entity."""

	target_range: dict[str, Any]
	"""The range of the referenced entity in the target file."""

	reference_type: str
	"""The type of reference (e.g., 'call', 'import', 'inheritance')."""


@dataclass(frozen=True)
class LSPTypeInfo:
	"""Type information for a code entity obtained from LSP."""

	type_name: str
	"""The name of the type."""

	is_built_in: bool = False
	"""Whether this is a built-in type."""

	type_hierarchy: list[str] = field(default_factory=list)
	"""Hierarchy of types (for inheritance)."""


@dataclass(frozen=True)
class LSPMetadata:
	"""LSP-related metadata for a code chunk."""

	symbol_references: list[LSPReference] = field(default_factory=list)
	"""References to other symbols found in this chunk (function calls, class usage, etc.)."""

	type_info: LSPTypeInfo | None = None
	"""Type information for this chunk, if applicable."""

	hover_text: str | None = None
	"""Hover information provided by LSP (can include type hints, docstrings, etc.)."""

	definition_uri: str | None = None
	"""URI to the definition of this symbol, if this chunk references another symbol."""

	is_definition: bool = True
	"""Whether this chunk is a definition (vs a reference)."""

	additional_attributes: dict[str, Any] = field(default_factory=dict)
	"""Additional LSP-provided attributes not covered by the above fields."""
