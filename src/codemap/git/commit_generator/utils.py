"""Linting functionality for commit messages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codemap.git.commit_linter import create_linter
from codemap.utils.config_loader import ConfigLoader

if TYPE_CHECKING:
	from pathlib import Path


logger = logging.getLogger(__name__)


def lint_commit_message(
	message: str, repo_root: Path, config_loader: ConfigLoader | None = None
) -> tuple[bool, list[str]]:
	"""
	Lint a commit message using the CommitLinter.

	Args:
	    message: Commit message to lint
	    repo_root: Repository root path
	    config_loader: Optional ConfigLoader instance for dependency injection

	Returns:
	    Tuple of (is_valid, list_of_messages)

	"""
	try:
		# Create a linter using the commit convention config from config_loader
		# Use dependency injection if a config_loader is provided, otherwise create one based on repo_root
		linter = create_linter(
			config_path=str(repo_root / ".codemap.yml"), config_loader=config_loader, repo_root=repo_root
		)
		return linter.lint(message)
	except Exception as e:
		logger.exception("Error during commit message linting")
		# Return valid=False with error message to make the failure visible
		return False, [f"Linter failed to execute: {e!s}"]


def clean_message_for_linting(message: str) -> str:
	"""
	Clean a message before linting.

	Args:
	    message: Message to clean

	Returns:
	    Cleaned message

	"""
	# Basic cleaning
	message = message.strip()

	# Remove markdown code blocks and inline code that might come from LLM
	message = message.replace("```", "").replace("`", "")

	# Remove common prefixes the LLM might add
	prefixes_to_remove = ["commit message:", "message:", "response:"]
	for prefix in prefixes_to_remove:
		if message.lower().startswith(prefix):
			message = message[len(prefix) :].strip()

	# Process multi-line content carefully to preserve format
	lines = message.splitlines()
	if len(lines) > 0:
		# Keep the header line as is (first line)
		header = lines[0]
		# Join the rest into a body (if any)
		if len(lines) > 1:
			# Check if there's already a blank line between header and body
			body_start = 2 if lines[1].strip() == "" else 1

			if len(lines) > body_start:
				# Join body lines with proper line breaks
				body = "\n".join(lines[body_start:])
				# Return header + blank line + body
				return f"{header}\n\n{body}"

		# Just return the header if no body
		return header

	return message
