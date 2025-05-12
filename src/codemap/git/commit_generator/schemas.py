"""Schemas and data structures for commit message generation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Footer(BaseModel):
	"""Footer token and value."""

	token: str = Field(description="Footer token (e.g., 'BREAKING CHANGE', 'Fixes', 'Refs')")
	value: str = Field(description="Footer value")


class CommitMessageSchema(BaseModel):
	"""Commit message schema for LLM output."""

	type: str = Field(description="The type of change (e.g., feat, fix, docs, style, refactor, perf, test, chore)")
	scope: str | None = Field(description="The scope of the change (e.g., component affected). This is optional.")
	description: str = Field(description="A short, imperative-tense description of the change")
	body: str | None = Field(description="A longer description of the changes. This is optional.")
	breaking: bool = Field(description="Whether this is a breaking change", default=False)
	footers: list[Footer] = Field(description="Footer tokens and values. This is optional.", default=[])
