"""Utility functions for commit message generation."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from codemap.git.commit_generator.schemas import COMMIT_MESSAGE_SCHEMA
from codemap.git.commit_linter.linter import CommitLinter
from codemap.git.utils import GitError, run_git_command
from codemap.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


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


def save_working_directory_state(files: list[str], output_file: str) -> bool:
	"""
	Save the current state of specified files to a patch file.

	Args:
	        files: List of file paths
	        output_file: Path to output patch file

	Returns:
	        bool: Whether the operation was successful

	"""
	output_path = Path(output_file)

	try:
		if not files:
			# Nothing to save
			with output_path.open("w") as f:
				f.write("")
			return True

		# Generate diff for the specified files
		diff_cmd = ["git", "diff", "--", *files]
		diff_content = run_git_command(diff_cmd)

		# Write to output file
		with output_path.open("w") as f:
			f.write(diff_content)

		return True

	except (OSError, GitError):
		logger.exception("Error saving working directory state")
		return False


def restore_working_directory_state(patch_file: str) -> bool:
	"""
	Restore the working directory state from a patch file.

	Args:
	        patch_file: Path to patch file

	Returns:
	        bool: Whether the operation was successful

	"""
	patch_path = Path(patch_file)

	try:
		# Check if the patch file exists and is not empty
		if not patch_path.exists() or patch_path.stat().st_size == 0:
			return True  # Nothing to restore

		# Apply the patch
		run_git_command(["git", "apply", patch_file])
		return True

	except GitError:
		logger.exception("Error restoring working directory state")
		return False


def format_commit_json(content: str, config_loader: ConfigLoader | None = None) -> str:
	"""
	Format a JSON string as a conventional commit message.

	Args:
	        content: JSON content string from LLM response
	        config_loader: Optional ConfigLoader for commit conventions

	Returns:
	        Formatted commit message string

	"""

	def _raise_validation_error(message: str) -> None:
		"""Helper to raise ValueError with consistent message."""
		logger.warning("LLM response validation failed: %s", message)
		raise ValueError(message)

	try:
		# Handle both direct JSON objects and strings containing JSON
		if not content.strip().startswith("{"):
			# Extract JSON if it's wrapped in other text
			json_match = re.search(r"({.*})", content, re.DOTALL)
			if json_match:
				content = json_match.group(1)

		message_data = json.loads(content)
		logger.debug("Parsed JSON: %s", message_data)

		# Check for simplified {"commit_message": "..."} format
		if "commit_message" in message_data and isinstance(message_data["commit_message"], str):
			return message_data["commit_message"].strip()

		# Basic Schema Validation
		if not isinstance(message_data, dict):
			_raise_validation_error("JSON response is not an object")

		if not message_data.get("type") or not message_data.get("description"):
			_raise_validation_error("Missing required fields in JSON response")

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

	except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
		# If parsing or validation fails, return the content as-is, but cleaned
		logger.warning("Error formatting JSON to commit message: %s. Using raw content.", str(e))
		return content.strip()


def prepare_prompt(
	template: str,
	diff_content: str,
	file_info: dict[str, Any],
	convention: dict[str, Any],
	extra_context: dict[str, Any] | None = None,
) -> str:
	"""
	Prepare a prompt for LLM commit message generation.

	Args:
	        template: The prompt template string
	        diff_content: The diff content to include in the prompt
	        file_info: Dictionary of file information
	        convention: Commit convention configuration
	        extra_context: Additional context variables for the template

	Returns:
	        Formatted prompt string

	"""
	# Create a context dict with default values for template variables
	context = {
		"diff": diff_content,
		"files": file_info,
		"convention": convention,
		"schema": COMMIT_MESSAGE_SCHEMA,
		"original_message": "",  # Default value for original_message
		"lint_errors": "",  # Default value for lint_errors
	}

	# Update with any extra context provided
	if extra_context:
		context.update(extra_context)

	try:
		from string import Template

		template_obj = Template(template)
		return template_obj.safe_substitute(context)
	except (ValueError, KeyError) as e:
		logger.warning("Error formatting prompt template: %s", str(e))
		# Fallback to simple string formatting
		return template.format(**context)
