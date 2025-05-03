"""Generator module for commit messages."""

from __future__ import annotations

# Import collections.abc for type annotation
import json
import logging
import os
from typing import TYPE_CHECKING, Any

from codemap.git.diff_splitter import DiffChunk
from codemap.llm import LLMClient, LLMError
from codemap.utils.cli_utils import loading_spinner
from codemap.utils.config_loader import ConfigLoader

from .prompts import get_lint_prompt_template, prepare_lint_prompt, prepare_prompt
from .schemas import COMMIT_MESSAGE_SCHEMA

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


class CommitMessageGenerator:
	"""Generates commit messages using LLMs."""

	def __init__(
		self,
		repo_root: Path,
		llm_client: LLMClient,
		prompt_template: str,
		config_loader: ConfigLoader,
	) -> None:
		"""
		Initialize the commit message generator.

		Args:
		    repo_root: Root directory of the Git repository
		    llm_client: LLMClient instance to use
		    prompt_template: Custom prompt template to use
		    config_loader: ConfigLoader instance to use for configuration

		"""
		self.repo_root = repo_root
		self.prompt_template = prompt_template
		self._config_loader = config_loader
		self.client = llm_client

		# Add commit template to client
		self.client.set_template("commit", self.prompt_template)

	def extract_file_info(self, chunk: DiffChunk) -> dict[str, Any]:
		"""
		Extract file information from the diff chunk.

		Args:
		    chunk: Diff chunk object to extract information from

		Returns:
		    Dictionary with information about files

		"""
		file_info = {}
		files = chunk.files
		for file in files:
			if not isinstance(file, str):
				continue  # Skip non-string file entries
			file_path = self.repo_root / file
			if not file_path.exists():
				continue
			try:
				extension = file_path.suffix.lstrip(".")
				file_info[file] = {
					"extension": extension,
					"directory": str(file_path.parent.relative_to(self.repo_root)),
				}
				path_parts = file_path.parts
				if len(path_parts) > 1:
					if "src" in path_parts:
						idx = path_parts.index("src")
						if idx + 1 < len(path_parts):
							file_info[file]["module"] = path_parts[idx + 1]
					elif "tests" in path_parts:
						file_info[file]["module"] = "tests"
			except (ValueError, IndexError, TypeError):
				continue
		return file_info

	def get_commit_convention(self) -> dict[str, Any]:
		"""Get commit convention settings from config."""
		# Use the centralized ConfigLoader to get the convention
		return self._config_loader.get_commit_convention()

	def _prepare_prompt(self, chunk: DiffChunk) -> str:
		"""
		Prepare the prompt for the LLM.

		Args:
		    chunk: Diff chunk object to prepare prompt for

		Returns:
		    Prepared prompt with diff and file information

		"""
		file_info = self.extract_file_info(chunk)
		convention = self.get_commit_convention()

		# Get the diff content directly from the chunk object
		diff_content = chunk.content

		# Prepare and return the prompt
		return prepare_prompt(
			template=self.prompt_template,
			diff_content=diff_content,
			file_info=file_info,
			convention=convention,
		)

	def format_json_to_commit_message(self, content: str) -> str:
		"""
		Format a JSON string as a conventional commit message.

		Args:
		    content: JSON content string from LLM response

		Returns:
		    Formatted commit message string

		"""

		def _validate_type(value: Any, expected_type: type, field_name: str) -> None:  # noqa: ANN401
			"""Helper to validate type and raise TypeError."""
			if value is not None and not isinstance(value, expected_type):
				msg = f"Invalid type for {field_name}"
				raise TypeError(msg)

		def _raise_missing_fields() -> None:
			"""Helper to raise ValueError for missing required fields."""
			logger.warning("LLM response missing required fields (type/description).")
			msg = "Missing required fields in LLM JSON response"
			raise ValueError(msg)

		try:
			# Try to parse the content as JSON
			message_data = json.loads(content)

			# Basic Schema Validation
			if (
				not isinstance(message_data, dict)
				or not message_data.get("type")
				or not message_data.get("description")
			):
				_raise_missing_fields()

			# Extract components with validation/defaults
			commit_type = message_data["type"]
			scope = message_data.get("scope")
			description = message_data["description"]
			body = message_data.get("body")
			is_breaking = message_data.get("breaking", False)
			footers = message_data.get("footers", [])

			# Ensure basic types are correct using the helper
			_validate_type(commit_type, str, "type")
			_validate_type(description, str, "description")
			_validate_type(scope, str, "scope")
			_validate_type(body, str, "body")
			_validate_type(is_breaking, bool, "breaking flag")
			_validate_type(footers, list, "footers")

			# Format the header
			header = f"{commit_type}"
			if scope:
				header += f"({scope})"
			if is_breaking:
				header += "!"
			header += f": {description}"

			# Build the complete message
			message_parts = [header]

			# Add body if provided
			if body:
				message_parts.append("")  # Empty line between header and body
				message_parts.append(body)

			# Add breaking change footers
			breaking_change_footers = [
				footer
				for footer in footers
				if footer.get("token", "").upper() in ("BREAKING CHANGE", "BREAKING-CHANGE")
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

			return "\n".join(message_parts)

		except (json.JSONDecodeError, TypeError, AttributeError, ValueError) as e:
			# If parsing or validation fails, return the content as-is
			logger.warning(
				"Could not parse/validate JSON response: %s. Using raw content: %s", e, content[:100] + "..."
			)
			# Return the raw content, stripped, as fallback
			return content.strip()

	def fallback_generation(self, chunk: DiffChunk) -> str:
		"""
		Generate a fallback commit message without LLM.

		This is used when LLM-based generation fails or is disabled.

		Args:
		    chunk: Diff chunk object to generate message for

		Returns:
		    Generated commit message

		"""
		commit_type = "chore"

		# Get files directly from the chunk object
		files = chunk.files

		# Filter only strings (defensive, though DiffChunk.files should be list[str])
		string_files = [f for f in files if isinstance(f, str)]

		for file in string_files:
			if file.startswith("tests/"):
				commit_type = "test"
				break
			if file.startswith("docs/") or file.endswith(".md"):
				commit_type = "docs"
				break

		# Get content directly from the chunk object
		content = chunk.content

		if isinstance(content, str) and ("fix" in content.lower() or "bug" in content.lower()):
			commit_type = "fix"  # Be slightly smarter about 'fix' type

		# Use chunk description if available and seems specific (not just placeholder)
		chunk_desc = chunk.description
		placeholder_descs = ["update files", "changes in", "hunk in", "new file:"]
		# Ensure chunk_desc is not None before calling lower()
		use_chunk_desc = chunk_desc and not any(p in chunk_desc.lower() for p in placeholder_descs)

		if use_chunk_desc and chunk_desc:  # Add explicit check for chunk_desc
			description = chunk_desc
			# Attempt to extract a type from the chunk description if possible
			# Ensure chunk_desc is not None before calling lower() and split()
			if chunk_desc.lower().startswith(
				("feat", "fix", "refactor", "docs", "test", "chore", "style", "perf", "ci", "build")
			):
				parts = chunk_desc.split(":", 1)
				if len(parts) > 1:
					commit_type = parts[0].split("(")[0].strip().lower()  # Extract type before scope
					description = parts[1].strip()
		else:
			# Generate description based on file count/path if no specific chunk desc
			description = "update files"  # Default
			if string_files:
				if len(string_files) == 1:
					description = f"update {string_files[0]}"
				else:
					try:
						common_dir = os.path.commonpath(string_files)
						# Make common_dir relative to repo root if possible
						try:
							common_dir_rel = os.path.relpath(common_dir, self.repo_root)
							if common_dir_rel and common_dir_rel != ".":
								description = f"update files in {common_dir_rel}"
							else:
								description = f"update {len(string_files)} files"
						except ValueError:  # Happens if paths are on different drives (unlikely in repo)
							description = f"update {len(string_files)} files"

					except (ValueError, TypeError):  # commonpath fails on empty list or mixed types
						description = f"update {len(string_files)} files"

		message = f"{commit_type}: {description}"
		logger.debug("Generated fallback message: %s", message)
		return message

	def generate_message(self, chunk: DiffChunk) -> tuple[str, bool]:
		"""
		Generate a commit message for a single chunk.

		Args:
		    chunk: Diff chunk object to generate message for

		Returns:
		    Tuple of (commit_message, was_generated_by_llm)

		"""
		existing_desc = chunk.description

		# Check for existing description
		if existing_desc and isinstance(existing_desc, str):
			is_generic = existing_desc.startswith(("chore: update", "fix: update", "docs: update", "test: update"))
			is_llm_gen = getattr(chunk, "is_llm_generated", False) if isinstance(chunk, DiffChunk) else False

			if not is_generic and is_llm_gen:
				logger.debug("Chunk already has LLM-generated description: '%s'", existing_desc)
				return existing_desc, True  # Assume it was LLM generated previously

		# Try to generate a message using LLM
		try:
			# Prepare prompt for the model
			prompt = self._prepare_prompt(chunk)

			with loading_spinner("Generating commit message..."):
				result = self._call_llm_api(prompt=prompt)

			# Format the JSON into a conventional commit message
			message = self.format_json_to_commit_message(result)

			# Mark the chunk if possible
			if isinstance(chunk, DiffChunk):
				chunk.is_llm_generated = True  # Mark original object if it's the class type

			return message, True

		except (LLMError, ValueError, RuntimeError):
			# Handle errors gracefully
			logger.exception("Error during LLM generation")
			logger.info("Falling back to simple message generation.")
			message = self.fallback_generation(chunk)
			return message, False

	def _call_llm_api(self, prompt: str) -> str:
		"""
		Call the LLM API with the given prompt.

		Args:
		    prompt: Prompt to send to the LLM

		Returns:
		    Raw response content from the LLM

		Raises:
		    LLMError: If the API call fails

		"""
		# Directly use the generate_text method from the LLMClient
		return self.client.generate_text(prompt=prompt, json_schema=COMMIT_MESSAGE_SCHEMA)

	def generate_message_with_linting(
		self,
		chunk: DiffChunk,
		max_retries: int = 3,
	) -> tuple[str, bool, bool]:
		"""
		Generate a commit message with linting and regeneration attempts.

		Args:
		    chunk: Diff chunk object to generate message for
		    max_retries: Maximum number of regeneration retries for invalid messages

		Returns:
		    Tuple of (message, was_generated_by_llm, passed_linting)

		"""
		# First attempt to generate a message
		message, used_llm = self.generate_message(chunk)

		# If not generated by LLM, skip linting
		if not used_llm:
			logger.debug("Message was not generated by LLM, skipping linting.")
			return message, used_llm, True

		# Clean the message before linting
		# Assuming clean_message_for_linting and lint_commit_message are accessible
		# Either keep them in utils or move them here as private methods
		from .utils import clean_message_for_linting, lint_commit_message  # Keep imports local for now

		message = clean_message_for_linting(message)

		# Lint the message
		is_valid, lint_messages = lint_commit_message(message, self.repo_root)

		# If valid, return immediately
		if is_valid:
			logger.debug("Generated message passed linting checks.")
			return message, used_llm, True

		# Log the linting issues
		logger.warning("Commit message failed linting: %s", message)
		for lint_msg in lint_messages:
			logger.warning("Lint issue: %s", lint_msg)

		# Try to regenerate with more explicit instructions
		retries_left = max_retries
		regenerated_message = message

		while retries_left > 0 and not is_valid:
			retries_left -= 1

			try:
				# Get diff content directly from chunk object
				diff_content = chunk.content

				# Prepare the enhanced prompt for regeneration
				lint_template = get_lint_prompt_template()
				enhanced_prompt = prepare_lint_prompt(
					template=lint_template,
					diff_content=diff_content,
					file_info=self.extract_file_info(chunk),  # Use self
					convention=self.get_commit_convention(),  # Use self
					lint_messages=lint_messages,
				)

				# Use a loading spinner to show regeneration progress
				with loading_spinner(f"Commit message failed linting, regenerating (attempts left: {retries_left})..."):
					# Use the client to generate text with enhanced prompt
					# Call _call_llm_api directly (or client.generate_text if preferred)
					# Note: _call_llm_api uses COMMIT_MESSAGE_SCHEMA, lint prompt might not need it.
					# Using client.generate_text without schema for the lint prompt.
					result = self.client.generate_text(prompt=enhanced_prompt, json_schema=None)
					regenerated_message = self.format_json_to_commit_message(result)

				# Clean and Lint the regenerated message
				regenerated_message = clean_message_for_linting(regenerated_message)
				is_valid, lint_messages = lint_commit_message(regenerated_message, self.repo_root)

				if is_valid:
					logger.info("Successfully regenerated a valid commit message.")
					break

				logger.warning("Regenerated message still failed linting: %s", regenerated_message)
				for lint_msg in lint_messages:
					logger.warning("Lint issue: %s", lint_msg)

			except Exception:
				logger.exception("Error during message regeneration")
				# Break out of the loop on error
				break

		# If we exhausted retries or had an error, return the last message with linting status
		if not is_valid and retries_left == 0:
			logger.warning("Exhausted all regeneration attempts. Using the last generated message.")

		return regenerated_message, used_llm, is_valid
