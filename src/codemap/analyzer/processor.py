"""Documentation processor for CodeMap."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.progress import Progress

from codemap.utils.file_utils import count_tokens

if TYPE_CHECKING:
	from codemap.analyzer.tree_parser import CodeParser

logger = logging.getLogger(__name__)


class DocumentationProcessor:
	"""Processor for generating documentation data from code files."""

	def __init__(self, parser: CodeParser, token_limit: int = 10000) -> None:
		"""
		Initialize the documentation processor.

		Args:
		    parser: The code parser to use
		    token_limit: Maximum number of tokens allowed (0 means infinite)

		"""
		self.parser = parser
		self.token_limit = token_limit
		self.total_tokens = 0

	def process_file(self, file_path: Path, progress: Progress | None = None) -> tuple[dict[str, Any] | None, int]:
		"""
		Process a single file and return its info and updated token count.

		Args:
		    file_path: Path to the file to process
		    progress: Optional progress bar to update

		Returns:
		    Tuple of (file_info, new_total_tokens)

		"""
		try:
			if not self.parser.file_filter.should_parse(file_path):
				return None, self.total_tokens

			# Count tokens first before parsing
			tokens = count_tokens(file_path)

			# Only check token limit if it's greater than 0 (not infinite)
			if self.token_limit > 0 and self.total_tokens + tokens > self.token_limit:
				logger.warning("Token limit reached, skipping remaining files")
				return None, self.total_tokens

			file_info = self.parser.parse_file(file_path)

			if progress:
				progress.update(progress.task_ids[0], advance=1)

			self.total_tokens += tokens
			return file_info, self.total_tokens
		except (OSError, UnicodeDecodeError) as e:
			logger.warning("Failed to parse file %s: %s", file_path, e)
			return None, self.total_tokens

	def process_directory(self, target_path: Path) -> dict[Path, dict[str, Any]]:
		"""
		Process a directory and return parsed files.

		Args:
		    target_path: Path to process

		Returns:
		    Dictionary of parsed files

		"""
		parsed_files: dict[Path, dict[str, Any]] = {}

		with Progress() as progress:
			progress.add_task("Parsing files...", total=None)

			for root, _, files in os.walk(target_path):
				root_path = Path(root)
				for file in files:
					file_path = root_path / file
					file_info, _ = self.process_file(file_path, progress)
					if file_info is not None:
						parsed_files[file_path] = file_info
					# Only break if token_limit > 0 and we've exceeded it
					if self.token_limit > 0 and self.total_tokens >= self.token_limit:
						break
				# Only break if token_limit > 0 and we've exceeded it
				if self.token_limit > 0 and self.total_tokens >= self.token_limit:
					break

		return parsed_files

	def process(self, target_path: Path) -> dict[Path, dict[str, Any]]:
		"""
		Process files in the target path.

		Args:
		    target_path: Path to process (file or directory)

		Returns:
		    Dictionary of parsed files

		"""
		# Process files based on whether target is a file or directory
		if target_path.is_file():
			parsed_files = {}
			file_info, _ = self.process_file(target_path)
			if file_info is not None:
				parsed_files[target_path] = file_info
			return parsed_files
		return self.process_directory(target_path)
