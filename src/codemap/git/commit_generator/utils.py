"""Utility functions for commit message generation."""

import logging
import re

from codemap.config import ConfigLoader
from codemap.git.commit_generator.schemas import CommitMessageSchema
from codemap.git.commit_linter.linter import CommitLinter

logger = logging.getLogger(__name__)


class CommitFormattingError(ValueError):
	"""Custom exception for errors during commit message formatting."""

	def __init__(self, message: str) -> None:
		"""Initialize the CommitFormattingError with a message."""
		super().__init__(message)


def clean_message_for_linting(message: str) -> str:
	"""
	Clean a commit message for linting.

	Removes extra newlines, trims whitespace, etc.

	Args:
	        message: The commit message to clean

	Returns:
	        The cleaned commit message

	"""
	# Replace multiple consecutive newlines with a single newline
	cleaned = re.sub(r"\n{3,}", "\n\n", message)
	# Trim leading and trailing whitespace
	return cleaned.strip()


def lint_commit_message(message: str, config_loader: ConfigLoader | None = None) -> tuple[bool, str | None]:
	"""
	Lint a commit message.

	Checks if it adheres to Conventional Commits format using internal CommitLinter.

	Args:
	        message: The commit message to lint
	        config_loader: Configuration loader instance

	Returns:
	        Tuple of (is_valid, error_message)

	"""
	# Get config loader if not provided
	if config_loader is None:
		config_loader = ConfigLoader.get_instance()

	try:
		# Create a CommitLinter instance with the config_loader
		linter = CommitLinter(config_loader=config_loader)

		# Lint the commit message
		is_valid, lint_messages = linter.lint(message)

		# Get error message if not valid
		error_message = None
		if not is_valid and lint_messages:
			error_message = "\n".join(lint_messages)

		return is_valid, error_message

	except Exception as e:
		# Handle any errors during linting
		logger.exception("Error linting commit message")
		return False, f"Linting failed: {e!s}"


def format_commit(commit: CommitMessageSchema, config_loader: ConfigLoader | None = None) -> str:
	"""
	Format a JSON string as a conventional commit message.

	Args:
	        commit: CommitMessageSchema object from LLM response
	        config_loader: Optional ConfigLoader for commit conventions

	Returns:
	        Formatted commit message string

	Raises:
	        JSONFormattingError: If JSON parsing or validation fails.

	"""
	try:
		# Extract components with validation/defaults
		commit_type = str(commit.type).lower().strip()

		# Check for valid commit type if config_loader is provided
		if config_loader:
			valid_types = config_loader.get.commit.convention.types
			if valid_types and commit_type not in valid_types:
				logger.warning("Invalid commit type: %s. Valid types: %s", commit_type, valid_types)
				# Try to find a valid type as fallback
				if "feat" in valid_types:
					commit_type = "feat"
				elif "fix" in valid_types:
					commit_type = "fix"
				elif len(valid_types) > 0:
					commit_type = valid_types[0]
				logger.debug("Using fallback commit type: %s", commit_type)

		scope = commit.scope
		if scope is not None:
			scope = str(scope).lower().strip()

		description = str(commit.description).strip()

		# Ensure description doesn't start with another type prefix
		if config_loader:
			valid_types = config_loader.get.commit.convention.types
			for valid_type in valid_types:
				if description.lower().startswith(f"{valid_type}:"):
					description = description.split(":", 1)[1].strip()
					break

		body = commit.body
		if body is not None:
			body = str(body).strip()
		is_breaking = bool(commit.breaking)

		# Format the header
		header = f"{commit_type}"
		if scope:
			header += f"({scope})"
		if is_breaking:
			header += "!"
		header += f": {description}"

		# Ensure compliance with commit format
		if ": " not in header:
			parts = header.split(":")
			if len(parts) == 2:  # type+scope and description # noqa: PLR2004
				header = f"{parts[0]}: {parts[1].strip()}"

		# Build the complete message
		message_parts = [header]

		# Add body if provided
		if body:
			message_parts.append("")  # Empty line between header and body
			message_parts.append(body)

		# Handle breaking change footers
		footers = commit.footers
		breaking_change_footers = []

		if isinstance(footers, list):
			breaking_change_footers = [
				footer
				for footer in footers
				if isinstance(footer, dict)
				and footer.get("token", "").upper() in ("BREAKING CHANGE", "BREAKING-CHANGE")
			]

		if breaking_change_footers:
			if not body:
				message_parts.append("")  # Empty line before footers if no body
			else:
				message_parts.append("")  # Empty line between body and footers

			for footer in breaking_change_footers:
				token = footer.get("token", "")
				value = footer.get("value", "")
				message_parts.append(f"{token}: {value}")

		message = "\n".join(message_parts)
		logger.debug("Formatted commit message: %s", message)
		return message

	except (TypeError, AttributeError) as e:
		# Catch parsing/attribute errors and raise the custom exception
		error_msg = f"Error processing commit message: {e}"
		logger.warning(error_msg)
		raise CommitFormattingError(error_msg) from e
	except CommitFormattingError:
		# Re-raise the validation errors triggered by _raise_validation_error
		raise
	except Exception as e:
		# Catch any other unexpected errors during formatting
		error_msg = f"Unexpected error formatting commit message: {e}"
		logger.exception(error_msg)  # Log unexpected errors with stack trace
		raise CommitFormattingError(error_msg) from e
