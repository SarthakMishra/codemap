"""Tests for SearchResult class."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

from codemap.processor.semantic_search.results import SearchResult


class TestSearchResult:
	"""Test cases for SearchResult class."""

	def test_from_ast_grep_match_basic(self):
		"""Test creating SearchResult from ast-grep match."""
		# Create a temporary file for testing
		with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
			f.write("def hello():\n    print('world')\n")
			temp_path = Path(f.name)

		try:
			# Mock ast-grep match object
			mock_match = Mock()
			mock_match.text.return_value = "def hello():\n    print('world')"
			mock_match.kind.return_value = "function_definition"

			# Mock range object
			mock_range = Mock()
			mock_start = Mock()
			mock_start.line = 0
			mock_start.column = 0
			mock_end = Mock()
			mock_end.line = 1
			mock_end.column = 20
			mock_range.start = mock_start
			mock_range.end = mock_end
			mock_match.range.return_value = mock_range

			pattern = "def $NAME(): $$$BODY"
			result = SearchResult.from_ast_grep_match(mock_match, temp_path, pattern)

			assert result.file_path == temp_path
			assert result.pattern == pattern
			assert result.matched_text == "def hello():\n    print('world')"
			assert result.start_line == 1  # 1-based
			assert result.end_line == 2  # 1-based
			assert result.start_col == 0
			assert result.end_col == 20
			assert result.node_kind == "function_definition"
			assert "def hello():" in result.context

		finally:
			temp_path.unlink()

	def test_from_ast_grep_match_with_context(self):
		"""Test SearchResult with surrounding context."""
		content = "# Header comment\ndef hello():\n    print('world')\n# Footer comment\n"

		with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
			f.write(content)
			temp_path = Path(f.name)

		try:
			mock_match = Mock()
			mock_match.text.return_value = "def hello():\n    print('world')"
			mock_match.kind.return_value = "function_definition"

			mock_range = Mock()
			mock_start = Mock()
			mock_start.line = 1  # Second line
			mock_start.column = 0
			mock_end = Mock()
			mock_end.line = 2
			mock_end.column = 20
			mock_range.start = mock_start
			mock_range.end = mock_end
			mock_match.range.return_value = mock_range

			result = SearchResult.from_ast_grep_match(mock_match, temp_path, "def $NAME(): $$$BODY")

			# Context should include surrounding lines
			assert "# Header comment" in result.context
			assert "# Footer comment" in result.context

		finally:
			temp_path.unlink()

	def test_from_ast_grep_match_file_error(self):
		"""Test SearchResult when file cannot be read."""
		# Non-existent file
		fake_path = Path("/non/existent/file.py")

		mock_match = Mock()
		mock_match.text.return_value = "def hello(): pass"
		mock_match.kind.return_value = "function_definition"

		mock_range = Mock()
		mock_start = Mock()
		mock_start.line = 0
		mock_start.column = 0
		mock_end = Mock()
		mock_end.line = 0
		mock_end.column = 15
		mock_range.start = mock_start
		mock_range.end = mock_end
		mock_match.range.return_value = mock_range

		result = SearchResult.from_ast_grep_match(mock_match, fake_path, "def $NAME(): $$$BODY")

		# Should fallback to match text when file can't be read
		assert result.context == "def hello(): pass"

	def test_to_formatted_string_python(self):
		"""Test formatted string output for Python file."""
		result = SearchResult(
			file_path=Path("test.py"),
			pattern="def $NAME(): $$$BODY",
			matched_text="def hello():\n    print('world')",
			start_line=1,
			end_line=2,
			start_col=0,
			end_col=20,
			context="def hello():\n    print('world')",
			node_kind="function_definition",
		)

		formatted = result.to_formatted_string()

		assert "test.py:1-2" in formatted
		assert "function_definition" in formatted
		assert "```python" in formatted
		assert "def hello():" in formatted

	def test_to_formatted_string_javascript(self):
		"""Test formatted string output for JavaScript file."""
		result = SearchResult(
			file_path=Path("test.js"),
			pattern="function $NAME() { $$$BODY }",
			matched_text="function hello() {\n    console.log('world');\n}",
			start_line=5,
			end_line=7,
			start_col=0,
			end_col=25,
			context="function hello() {\n    console.log('world');\n}",
			node_kind="function_declaration",
		)

		formatted = result.to_formatted_string()

		assert "test.js:5-7" in formatted
		assert "function_declaration" in formatted
		assert "```javascript" in formatted
		assert "function hello()" in formatted

	def test_to_formatted_string_with_different_context(self):
		"""Test formatted string when context differs from matched text."""
		result = SearchResult(
			file_path=Path("test.py"),
			pattern="def $NAME(): $$$BODY",
			matched_text="def hello():\n    print('world')",
			start_line=2,
			end_line=3,
			start_col=0,
			end_col=20,
			context="# Comment\ndef hello():\n    print('world')\n# End",
			node_kind="function_definition",
		)

		formatted = result.to_formatted_string()

		# Should show both matched code and context
		assert "**Matched Code:**" in formatted
		assert "**Context:**" in formatted
		assert "# Comment" in formatted
		assert "# End" in formatted

	def test_get_language_mapping(self):
		"""Test language detection from file extensions."""
		test_cases = [
			("test.py", "python"),
			("test.js", "javascript"),
			("test.jsx", "javascript"),
			("test.ts", "typescript"),
			("test.tsx", "typescript"),
			("test.java", "java"),
			("test.cpp", "cpp"),
			("test.c", "c"),
			("test.rs", "rust"),
			("test.go", "go"),
			("test.rb", "ruby"),
			("test.unknown", "text"),
		]

		for filename, expected_lang in test_cases:
			result = SearchResult(
				file_path=Path(filename),
				pattern="",
				matched_text="",
				start_line=1,
				end_line=1,
				start_col=0,
				end_col=0,
				context="",
				node_kind="",
			)

			assert result._get_language() == expected_lang

	def test_dataclass_equality(self):
		"""Test that SearchResult instances can be compared for equality."""
		result1 = SearchResult(
			file_path=Path("test.py"),
			pattern="def $NAME(): $$$BODY",
			matched_text="def hello(): pass",
			start_line=1,
			end_line=1,
			start_col=0,
			end_col=15,
			context="def hello(): pass",
			node_kind="function_definition",
		)

		result2 = SearchResult(
			file_path=Path("test.py"),
			pattern="def $NAME(): $$$BODY",
			matched_text="def hello(): pass",
			start_line=1,
			end_line=1,
			start_col=0,
			end_col=15,
			context="def hello(): pass",
			node_kind="function_definition",
		)

		result3 = SearchResult(
			file_path=Path("test.py"),
			pattern="def $NAME(): $$$BODY",
			matched_text="def world(): pass",  # Different text
			start_line=1,
			end_line=1,
			start_col=0,
			end_col=15,
			context="def world(): pass",
			node_kind="function_definition",
		)

		assert result1 == result2
		assert result1 != result3
