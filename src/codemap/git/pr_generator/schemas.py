"""Schemas and data structures for PR generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

# Type definitions
WorkflowStrategySchema = Literal["github-flow", "gitflow", "trunk-based"]
BranchType = Literal["feature", "release", "hotfix", "bugfix", "docs"]


class PRContent(TypedDict):
	"""Pull request content type."""

	title: str
	description: str


@dataclass
class PullRequest:
	"""Represents a GitHub Pull Request."""

	branch: str
	title: str
	description: str
	url: str | None = None
	number: int | None = None
