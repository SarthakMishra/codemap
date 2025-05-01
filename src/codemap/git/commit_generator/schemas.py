"""Schemas and data structures for commit message generation."""

from __future__ import annotations

from typing import TypedDict, cast

from codemap.git.diff_splitter import DiffChunk


class DiffChunkData(TypedDict, total=False):
	"""TypedDict representing the structure of a DiffChunk."""

	files: list[str]
	content: str
	description: str


# Define a schema for structured commit message output
COMMIT_MESSAGE_SCHEMA = {
	"type": "object",
	"properties": {
		"type": {
			"type": "string",
			"description": "The type of change (e.g., feat, fix, docs, style, refactor, perf, test, chore)",
		},
		"scope": {"type": ["string", "null"], "description": "The scope of the change (e.g., component affected)"},
		"description": {"type": "string", "description": "A short, imperative-tense description of the change"},
		"body": {
			"type": ["string", "null"],
			"description": "A longer description of the changes, explaining why and how",
		},
		"breaking": {"type": "boolean", "description": "Whether this is a breaking change", "default": False},
		"footers": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"token": {
						"type": "string",
						"description": "Footer token (e.g., 'BREAKING CHANGE', 'Fixes', 'Refs')",
					},
					"value": {"type": "string", "description": "Footer value"},
				},
				"required": ["token", "value"],
			},
			"default": [],
		},
	},
	"required": ["type", "description"],
}


class CommitMessageSchema(TypedDict):
	"""TypedDict representing the structured commit message output."""

	type: str
	scope: str | None
	description: str
	body: str | None
	breaking: bool
	footers: list[dict[str, str]]


def adapt_chunk_access(chunk: DiffChunk | DiffChunkData) -> DiffChunkData:
	"""
	Adapt chunk access to work with both DiffChunk objects and dictionaries.

	Args:
	    chunk: Chunk to adapt

	Returns:
	    Dictionary with chunk data

	"""
	if isinstance(chunk, DiffChunk):
		return DiffChunkData(
			files=chunk.files,
			content=chunk.content,
			description=chunk.description if chunk.description else "",
		)
	return cast("DiffChunkData", chunk)
