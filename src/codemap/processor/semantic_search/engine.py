"""Minimal ast-grep search engine for pattern matching."""

from __future__ import annotations

import logging
from pathlib import Path

from ast_grep_py import SgRoot

from .results import SearchResult

logger = logging.getLogger(__name__)


class AstGrepEngine:
	"""Minimal semantic code search engine using ast-grep pattern matching."""

	def __init__(self) -> None:
		"""Initialize the search engine."""
		self.supported_languages = {
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

	def search_pattern(
		self,
		pattern: str,
		file_paths: list[Path] | None = None,
		language: str | None = None,
		constraints: dict | None = None,
		limit: int = 50,
	) -> list[SearchResult]:
		"""Search for an ast-grep pattern across files.

		Args:
		    pattern: ast-grep pattern (e.g., "def $NAME($$$PARAMS): $$$BODY")
		    file_paths: Optional list of files to search, defaults to finding source files
		    language: Optional language override (python, javascript, etc.)
		    constraints: Optional constraints dict for pattern variables
		    limit: Maximum number of results to return

		Returns:
		    List of SearchResult objects

		Example patterns:
		    "def $NAME($$$PARAMS): $$$BODY"  # Find function definitions
		    "$FUNC($$$ARGS)"                # Find function calls
		    "class $NAME($$$BASES): $$$BODY" # Find class definitions
		    "import $MODULE"                # Find imports
		"""
		results = []

		# Get files to search
		if file_paths is None:
			file_paths = self._find_source_files()

		for file_path in file_paths:
			if len(results) >= limit:
				break

			file_language = language or self._detect_language(file_path)
			if not file_language:
				continue

			try:
				content = file_path.read_text(encoding="utf-8")
				root = SgRoot(content, file_language)

				# Search with constraints if provided
				if constraints:
					matches = root.root().find_all({"rule": {"pattern": pattern}, "constraints": constraints})
				else:
					matches = root.root().find_all(pattern=pattern)

				for match in matches:
					if len(results) >= limit:
						break

					results.append(SearchResult.from_ast_grep_match(match, file_path, pattern))

			except (OSError, UnicodeDecodeError, ValueError) as e:
				logger.warning(f"Failed to search {file_path}: {e}")

		return results

	def _find_source_files(self) -> list[Path]:
		"""Find source files in common directories."""
		files = []
		search_patterns = [
			"src/**/*.py",
			"**/*.py",
			"src/**/*.js",
			"**/*.js",
			"src/**/*.ts",
			"**/*.ts",
			"*.py",
			"*.js",
			"*.ts",
		]

		for pattern in search_patterns:
			files.extend(Path.cwd().glob(pattern))

		# Remove duplicates and limit to reasonable number
		unique_files = list({f.resolve() for f in files if f.is_file()})
		return unique_files[:200]  # Limit to prevent overwhelming searches

	def _detect_language(self, file_path: Path) -> str | None:
		"""Detect programming language from file extension."""
		return self.supported_languages.get(file_path.suffix)
