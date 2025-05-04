"""Generator module for commit messages."""

from __future__ import annotations

# Import collections.abc for type annotation
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codemap.git.diff_splitter import DiffChunk
from codemap.llm import LLMClient
from codemap.utils.config_loader import ConfigLoader

from .prompts import get_lint_prompt_template, prepare_lint_prompt, prepare_prompt
from .schemas import COMMIT_MESSAGE_SCHEMA
from .utils import clean_message_for_linting, lint_commit_message

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)

MAX_DEBUG_CONTENT_LENGTH = 100
EXPECTED_PARTS_COUNT = 2  # Type+scope and description


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

		# Create a context dict with default values for template variables
		context = {
			"diff": diff_content,
			"files": file_info,
			"convention": convention,
			"schema": COMMIT_MESSAGE_SCHEMA,
			"original_message": "",  # Default value for original_message
			"lint_errors": "",  # Default value for lint_errors
		}

		# Prepare and return the prompt
		return prepare_prompt(
			template=self.prompt_template,
			diff_content=diff_content,
			file_info=file_info,
			convention=convention,
			extra_context=context,  # Pass the context with default values
		)

	def format_json_to_commit_message(self, content: str) -> str:
		"""
		Format a JSON string as a conventional commit message.

		Args:
		    content: JSON content string from LLM response

		Returns:
		    Formatted commit message string

		"""

		def _raise_validation_error(message: str) -> None:
			"""Helper to raise ValueError with consistent message."""
			logger.warning("LLM response validation failed: %s", message)
			msg = message
			raise ValueError(msg)

		try:
			# Try to parse the content as JSON
			debug_content = (
				content[:MAX_DEBUG_CONTENT_LENGTH] + "..." if len(content) > MAX_DEBUG_CONTENT_LENGTH else content
			)
			logger.debug("Parsing JSON content: %s", debug_content)

			# Handle both direct JSON objects and strings containing JSON
			if not content.strip().startswith("{"):
				# Extract JSON if it's wrapped in other text
				import re

				json_match = re.search(r"({.*})", content, re.DOTALL)
				if json_match:
					content = json_match.group(1)

			message_data = json.loads(content)
			logger.debug("Parsed JSON: %s", message_data)

			# Basic Schema Validation
			if not isinstance(message_data, dict):
				_raise_validation_error("JSON response is not an object")

			if not message_data.get("type") or not message_data.get("description"):
				_raise_validation_error("Missing required fields in JSON response")

			# Extract components with validation/defaults
			commit_type = str(message_data["type"]).lower().strip()

			# Check for valid commit type (from the config)
			valid_types = self._config_loader.get_commit_convention().get("types", [])
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

			description = str(message_data["description"]).lower().strip()

			# Ensure description doesn't start with another type prefix
			for valid_type in valid_types:
				if description.startswith(f"{valid_type}:"):
					# Remove the duplicate type prefix from description
					description = description.split(":", 1)[1].strip()
					logger.debug("Removed duplicate type prefix from description: %s", description)
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

			# Ensure compliance with commit format regex
			# The regex requires a space after the colon, and the format should be <type>(<scope>)!: <description>
			if ": " not in header:
				parts = header.split(":")
				if len(parts) == EXPECTED_PARTS_COUNT:
					header = f"{parts[0]}: {parts[1].strip()}"

			# Validation check against regex pattern
			import re

			from codemap.git.commit_linter.constants import COMMIT_REGEX

			# If header doesn't match the expected format, log and try to fix it
			if not COMMIT_REGEX.match(header):
				logger.warning("Generated header doesn't match commit format: %s", header)
				# As a fallback, recreate with a simpler format
				simple_header = f"{commit_type}"
				if scope:
					simple_header += f"({scope})"
				if is_breaking:
					simple_header += "!"
				simple_header += f": {description}"
				header = simple_header
				logger.debug("Fixed header to: %s", header)

			# Build the complete message
			message_parts = [header]

			# Add body if provided
			if body:
				message_parts.append("")  # Empty line between header and body
				message_parts.append(body)

			# Carefully filter only breaking change footers
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
		Generate a commit message for a diff chunk.

		Args:
		    chunk: Diff chunk to generate message for

		Returns:
		    Generated message and success flag

		"""
		# Prepare prompt with chunk data
		try:
			prompt = self._prepare_prompt(chunk)
			logger.debug("Prompt prepared successfully")

			# Generate message using configured LLM provider
			message = self._call_llm_api(prompt)
			logger.debug("LLM generated message: %s", message)

			# Return generated message with success flag
			return message, True
		except Exception:
			logger.exception("Error during LLM generation")
			# Fall back to heuristic generation
			return self.fallback_generation(chunk), False

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
		self, chunk: DiffChunk, retry_count: int = 1, max_retries: int = 3
	) -> tuple[str, bool, bool, list[str]]:
		"""
		Generate a commit message with linting verification.

		Args:
		        chunk: The DiffChunk to generate a message for
		        retry_count: Current retry count (default: 1)
		        max_retries: Maximum number of retries for linting (default: 3)

		Returns:
		        Tuple of (message, used_llm, passed_linting, lint_messages)

		"""
		# First, generate the initial message
		initial_lint_messages: list[str] = []  # Store initial messages
		try:
			message, used_llm = self.generate_message(chunk)
			logger.debug("Generated initial message: %s", message)

			# Clean the message before linting
			message = clean_message_for_linting(message)

			# Check if the message passes linting
			is_valid, initial_lint_messages = lint_commit_message(message, self.repo_root)
			logger.debug("Lint result: valid=%s, messages=%s", is_valid, initial_lint_messages)

			if is_valid or retry_count >= max_retries:
				# Return empty list if valid, or initial messages if max retries reached
				return message, used_llm, is_valid, [] if is_valid else initial_lint_messages

			# Prepare the diff content
			diff_content = chunk.content
			if not diff_content:
				diff_content = "Empty diff (likely modified binary files)"

			logger.info("Regenerating message with linting feedback (attempt %d/%d)", retry_count, max_retries)

			try:
				# Prepare the enhanced prompt for regeneration
				lint_template = get_lint_prompt_template()
				enhanced_prompt = prepare_lint_prompt(
					template=lint_template,
					diff_content=diff_content,
					file_info=self.extract_file_info(chunk),  # Use self
					convention=self.get_commit_convention(),  # Use self
					lint_messages=initial_lint_messages,  # Use initial messages for feedback
				)

				# Generate message with the enhanced prompt
				regenerated_message = self._call_llm_api(enhanced_prompt)
				logger.debug("Regenerated message (RAW LLM output): %s", regenerated_message)

				# Format from JSON to commit message format
				regenerated_message = self.format_json_to_commit_message(regenerated_message)
				logger.debug("Formatted message: %s", regenerated_message)

				# Clean and recheck linting
				cleaned_message = clean_message_for_linting(regenerated_message)
				logger.debug("Cleaned message for linting: %s", cleaned_message)

				# Check if the message passes linting
				final_is_valid, final_lint_messages = lint_commit_message(cleaned_message, self.repo_root)
				logger.debug("Regenerated lint result: valid=%s, messages=%s", final_is_valid, final_lint_messages)

				# Return final result and messages (empty if valid)
				return cleaned_message, True, final_is_valid, [] if final_is_valid else final_lint_messages
			except Exception:
				# If regeneration fails, log it and return the original message and its lint errors
				logger.exception("Error during message regeneration")
				return message, used_llm, False, initial_lint_messages  # Return original message and errors
		except Exception:
			# If generation fails completely, use a fallback (fallback doesn't lint, so return True, empty messages)
			logger.exception("Error during message generation")
			message = self.fallback_generation(chunk)
			return message, False, True, []  # Fallback assumes valid, no lint messages
