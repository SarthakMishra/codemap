"""Generator module for commit messages."""

from __future__ import annotations

# Import collections.abc for type annotation
import json
import logging
import os
from typing import TYPE_CHECKING, Any, cast

from codemap.git.diff_splitter import DiffChunk
from codemap.llm import LLMClient, LLMError
from codemap.utils.cli_utils import loading_spinner
from codemap.utils.config_loader import ConfigLoader

from .prompts import prepare_prompt
from .schemas import COMMIT_MESSAGE_SCHEMA, DiffChunkData, adapt_chunk_access
from .utils import generate_message_with_linting

if TYPE_CHECKING:
	import collections.abc
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

	def extract_file_info(self, chunk: DiffChunk | DiffChunkData) -> dict[str, Any]:
		"""
		Extract file information from the diff chunk.

		Args:
		    chunk: Diff chunk to extract information from

		Returns:
		    Dictionary with information about files

		"""
		file_info = {}
		files = chunk.files if isinstance(chunk, DiffChunk) else chunk.get("files", [])
		if not isinstance(files, list):
			try:
				# Convert to list only if it's actually iterable
				if hasattr(files, "__iter__") and not isinstance(files, str):
					files = list(cast("collections.abc.Iterable", files))
				else:
					files = []
			except (TypeError, ValueError):
				files = []

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

	def _prepare_prompt(self, chunk: DiffChunk | DiffChunkData) -> str:
		"""
		Prepare the prompt for the LLM.

		Args:
		    chunk: Diff chunk to prepare prompt for

		Returns:
		    Prepared prompt with diff and file information

		"""
		file_info = self.extract_file_info(chunk)
		convention = self.get_commit_convention()

		# Get the diff content from the chunk
		diff_content = chunk.content if isinstance(chunk, DiffChunk) else chunk.get("content", "")

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
		try:
			# Try to parse the content as JSON
			message_data = json.loads(content)

			# Extract components
			commit_type = message_data.get("type", "chore")
			scope = message_data.get("scope")
			description = message_data.get("description", "")
			body = message_data.get("body")
			is_breaking = message_data.get("breaking", False)
			footers = message_data.get("footers", [])

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

		except (json.JSONDecodeError, TypeError, AttributeError):
			# If parsing fails, return the content as-is
			logger.warning("Could not parse JSON response, using raw content")
			return content.strip()

	def fallback_generation(self, chunk: DiffChunk | DiffChunkData) -> str:
		"""
		Generate a fallback commit message without LLM.

		This is used when LLM-based generation fails or is disabled.

		Args:
		    chunk: Diff chunk to generate message for

		Returns:
		    Generated commit message

		"""
		commit_type = "chore"

		# Get files from the chunk
		files = chunk.files if isinstance(chunk, DiffChunk) else chunk.get("files", [])

		string_files = [f for f in files if isinstance(f, str)]  # Filter only strings for path operations

		for file in string_files:
			if file.startswith("tests/"):
				commit_type = "test"
				break
			if file.startswith("docs/") or file.endswith(".md"):
				commit_type = "docs"
				break

		# Get content from the chunk
		content = chunk.content if isinstance(chunk, DiffChunk) else chunk.get("content", "")

		if isinstance(content, str) and ("fix" in content.lower() or "bug" in content.lower()):
			commit_type = "fix"  # Be slightly smarter about 'fix' type

		description = "update files"  # Default description
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
		# Ensure fallback follows length constraints
		convention = self.get_commit_convention()
		max_length = convention.get("max_length", 72)
		# Ensure max_length is an integer to avoid TypeError in tests with mocks
		if not isinstance(max_length, int):
			max_length = 72  # Default max length if we can't get one from convention
		if len(message) > max_length:
			message = message[:max_length]

		return message

	def generate_message(self, chunk: DiffChunk | DiffChunkData) -> tuple[str, bool]:
		"""
		Generate a commit message for the given diff chunk.

		Args:
		    chunk: Diff chunk to generate message for

		Returns:
		    Tuple of (message, was_generated_by_llm)

		"""
		chunk_dict = adapt_chunk_access(chunk)
		existing_desc = chunk_dict.get("description")

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
			prompt = self._prepare_prompt(chunk_dict)

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
			message = self.fallback_generation(chunk_dict)
			return message, False

	def _call_llm_api(self, prompt: str) -> str:
		"""
		Call the LLM API with the given prompt.

		This is a helper method to centralize LLM API calls and make them easier to mock in tests.

		Args:
		    prompt: The prompt to send to the LLM

		Returns:
		    The LLM response text

		Raises:
		    LLMError: If there's an issue with the LLM API call

		"""
		return self.client.generate_text(
			prompt=prompt,
			json_schema=COMMIT_MESSAGE_SCHEMA,
		)

	def generate_message_with_linting(self, chunk: DiffChunk | DiffChunkData) -> tuple[str, bool, bool]:
		"""
		Generate a commit message with linting.

		Args:
		    chunk: Diff chunk to generate message for

		Returns:
		    Tuple of (message, was_generated_by_llm, passed_linting)

		"""
		return generate_message_with_linting(
			chunk=chunk,
			generator=self,
			repo_root=self.repo_root,
			max_retries=3,
		)
