"""Git metadata models for version control information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from datetime import datetime


@dataclass(frozen=True)
class GitMetadata:
	"""
	Git-related metadata for a code chunk.

	Tracks version control information for the chunk, including both
	original authorship and last modification details.

	"""

	is_committed: bool
	"""Whether the chunk is committed to the repository."""

	commit_id: str
	"""Git commit hash where this chunk was last modified."""

	commit_message: str
	"""Message of the last commit."""

	timestamp: datetime
	"""Timestamp of the original commit."""

	branch: list[str]
	"""List of branch names where this chunk exists."""

	last_modified_at: datetime | None = None
	"""Timestamp of the last modification, if different from original commit."""
