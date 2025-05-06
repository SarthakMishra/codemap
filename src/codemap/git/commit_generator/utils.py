"""Utility functions for commit message generation."""

import json
import logging
import re
from pathlib import Path

from codemap.git.commit_linter.linter import CommitLinter
from codemap.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class JSONFormattingError(ValueError):
	"""Custom exception for errors during JSON commit message formatting."""

	def __init__(self, message: str, original_content: str) -> None:
		"""Initialize the JSONFormattingError with a message and original content."""
		super().__init__(message)
		self.original_content = original_content


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


def lint_commit_message(
	message: str, repo_root: Path | None = None, config_loader: ConfigLoader | None = None
) -> tuple[bool, str | None]:
	"""
	Lint a commit message.

	Checks if it adheres to Conventional Commits format using internal CommitLinter.

	Args:
	        message: The commit message to lint
	        repo_root: Repository root path
	        config_loader: Configuration loader instance

	Returns:
	        Tuple of (is_valid, error_message)

	"""
	# Get config loader if not provided
	if config_loader is None:
		config_loader = ConfigLoader(repo_root=repo_root)

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


def format_commit_json(content: str, config_loader: ConfigLoader | None = None) -> str:
	"""
	Format a JSON string as a conventional commit message.

	Args:
	        content: JSON content string from LLM response
	        config_loader: Optional ConfigLoader for commit conventions

	Returns:
	        Formatted commit message string

	Raises:
	        JSONFormattingError: If JSON parsing or validation fails.

	"""
	original_content = content  # Store original content for error reporting

	def _raise_validation_error(message: str) -> None:
		"""Helper to raise JSONFormattingError with consistent message."""
		logger.warning("LLM response validation failed: %s", message)
		# Raise the custom exception with the original content
		raise JSONFormattingError(message, original_content)

	try:
		# Handle both direct JSON objects and strings containing JSON
		if not content.strip().startswith("{"):
			# Extract JSON if it's wrapped in other text
			json_match = re.search(r"({.*})", content, re.DOTALL)
			if json_match:
				content = json_match.group(1)
			else:
				# If no JSON object found within the string, raise error
				_raise_validation_error("Could not find JSON object in the response")

		message_data = json.loads(content)
		logger.debug("Parsed JSON: %s", message_data)

		# Check for simplified {"commit_message": "..."} format FIRST
		if isinstance(message_data, dict) and "commit_message" in message_data:
			commit_message_value = message_data["commit_message"]
			# Handle if commit_message itself is the nested structure
			if isinstance(commit_message_value, dict):
				logger.debug("Found nested structure within 'commit_message'. Processing nested dict.")
				message_data = commit_message_value  # Use the nested dict for further processing
			# Handle if commit_message is a string (original simplified format)
			elif isinstance(commit_message_value, str):
				logger.debug("Found simplified string format within 'commit_message'. Returning directly.")
				return commit_message_value.strip()
			# Handle if commit_message is neither dict nor string
			else:
				_raise_validation_error(f"Unexpected type for 'commit_message': {type(commit_message_value).__name__}")

		# Basic Schema Validation (now applied to potentially nested structure)
		if not isinstance(message_data, dict):
			_raise_validation_error("JSON response is not an object")

		if not message_data.get("type") or not message_data.get("description"):
			_raise_validation_error("Missing required fields ('type', 'description') in JSON response")

		# Extract components with validation/defaults
		commit_type = str(message_data["type"]).lower().strip()

		# Check for valid commit type if config_loader is provided
		if config_loader:
			valid_types = config_loader.get_commit_convention().get("types", [])
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

		scope = message_data.get("scope")
		if scope is not None:
			scope = str(scope).lower().strip()

		description = str(message_data["description"]).strip()

		# Ensure description doesn't start with another type prefix
		if config_loader:
			valid_types = config_loader.get_commit_convention().get("types", [])
			for valid_type in valid_types:
				if description.lower().startswith(f"{valid_type}:"):
					description = description.split(":", 1)[1].strip()
					break

		body = message_data.get("body")
		if body is not None:
			body = str(body).strip()
		is_breaking = bool(message_data.get("breaking", False))

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
		footers = message_data.get("footers", [])
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

	except (json.JSONDecodeError, TypeError, AttributeError) as e:
		# Catch parsing/attribute errors and raise the custom exception
		error_msg = f"Error processing commit message JSON: {e}"
		logger.warning(error_msg)
		raise JSONFormattingError(error_msg, original_content) from e
	except JSONFormattingError:
		# Re-raise the validation errors triggered by _raise_validation_error
		raise
	except Exception as e:
		# Catch any other unexpected errors during formatting
		error_msg = f"Unexpected error formatting commit message JSON: {e}"
		logger.exception(error_msg)  # Log unexpected errors with stack trace
		raise JSONFormattingError(error_msg, original_content) from e
