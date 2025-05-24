"""Search result data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path

	from ast_grep_py import SgNode


@dataclass
class SearchResult:
	"""Represents a search result from ast-grep pattern matching."""

	file_path: Path
	pattern: str
	matched_text: str
	start_line: int
	end_line: int
	start_col: int
	end_col: int
	context: str
	node_kind: str

	@classmethod
	def from_ast_grep_match(cls, match: SgNode, file_path: Path, pattern: str) -> SearchResult:
		"""Create SearchResult from ast-grep match."""
		range_info = match.range()

		# Get surrounding context (3 lines before/after)
		try:
			lines = file_path.read_text(encoding="utf-8").splitlines()
			context_start = max(0, range_info.start.line - 3)
			context_end = min(len(lines), range_info.end.line + 4)
			context = "\n".join(lines[context_start:context_end])
		except (OSError, UnicodeDecodeError):
			context = match.text()

		return cls(
			file_path=file_path,
			pattern=pattern,
			matched_text=match.text(),
			start_line=range_info.start.line + 1,  # Convert to 1-based
			end_line=range_info.end.line + 1,
			start_col=range_info.start.column,
			end_col=range_info.end.column,
			context=context,
			node_kind=match.kind(),
		)

	def to_formatted_string(self) -> str:
		"""Format result for display."""
		result = f"## {self.file_path}:{self.start_line}-{self.end_line}\n\n"

		result += f"**Node Type:** `{self.node_kind}`\n\n"

		result += f"**Matched Code:**\n```{self._get_language()}\n{self.matched_text}\n```\n\n"

		if self.context != self.matched_text:
			result += f"**Context:**\n```{self._get_language()}\n{self.context}\n```\n\n"

		return result

	def _get_language(self) -> str:
		"""Get language for syntax highlighting."""
		ext = self.file_path.suffix
		lang_map = {
			".py": "python",
			".js": "javascript",
			".jsx": "javascript",
			".ts": "typescript",
			".tsx": "typescript",
			".java": "java",
			".cpp": "cpp",
			".c": "c",
			".rs": "rust",
			".go": "go",
			".rb": "ruby",
		}
		return lang_map.get(ext, "text")
