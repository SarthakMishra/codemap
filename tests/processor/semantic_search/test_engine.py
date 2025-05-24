"""Tests for AstGrepEngine class."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from codemap.processor.semantic_search.engine import AstGrepEngine
from codemap.processor.semantic_search.results import SearchResult


class TestAstGrepEngine:
	"""Test cases for AstGrepEngine class."""

	def setup_method(self):
		"""Set up test fixtures."""
		self.engine = AstGrepEngine()

	def test_init(self):
		"""Test engine initialization."""
		assert self.engine.supported_languages == {
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

	def test_detect_language(self):
		"""Test language detection from file extensions."""
		test_cases = [
			(Path("test.py"), "python"),
			(Path("test.js"), "javascript"),
			(Path("test.jsx"), "javascript"),
			(Path("test.ts"), "typescript"),
			(Path("test.tsx"), "typescript"),
			(Path("test.java"), "java"),
			(Path("test.cpp"), "cpp"),
			(Path("test.c"), "c"),
			(Path("test.rs"), "rust"),
			(Path("test.go"), "go"),
			(Path("test.rb"), "ruby"),
			(Path("test.unknown"), None),
		]

		for file_path, expected_lang in test_cases:
			assert self.engine._detect_language(file_path) == expected_lang

	def test_find_source_files(self):
		"""Test finding source files in the project."""
		# Create temporary directory structure
		with tempfile.TemporaryDirectory() as temp_dir:
			temp_path = Path(temp_dir)

			# Create some test files
			(temp_path / "test.py").write_text("def hello(): pass")
			(temp_path / "script.js").write_text("function hello() {}")
			(temp_path / "readme.txt").write_text("Not a source file")
			src_dir = temp_path / "src"
			src_dir.mkdir()
			(src_dir / "main.py").write_text("def main(): pass")

			# Change to temp directory
			original_cwd = Path.cwd()
			try:
				import os

				os.chdir(temp_path)

				files = self.engine._find_source_files()

				# Should find Python and JS files
				file_names = [f.name for f in files]
				assert "test.py" in file_names
				assert "script.js" in file_names
				assert "main.py" in file_names
				assert "readme.txt" not in file_names

			finally:
				os.chdir(original_cwd)

	@patch("codemap.processor.semantic_search.engine.SgRoot")
	def test_search_pattern_basic(self, mock_sg_root):
		"""Test basic pattern search functionality."""
		# Create test file
		with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
			f.write("def hello():\n    print('world')\n")
			temp_path = Path(f.name)

		try:
			# Mock ast-grep components
			mock_match = Mock()
			mock_match.text.return_value = "def hello():\n    print('world')"
			mock_match.kind.return_value = "function_definition"

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

			mock_root_node = Mock()
			mock_root_node.find_all.return_value = [mock_match]

			mock_sg_instance = Mock()
			mock_sg_instance.root.return_value = mock_root_node
			mock_sg_root.return_value = mock_sg_instance

			# Execute search
			results = self.engine.search_pattern(
				pattern="def $NAME(): $$$BODY", file_paths=[temp_path], language="python"
			)

			# Verify results
			assert len(results) == 1
			assert isinstance(results[0], SearchResult)
			assert results[0].pattern == "def $NAME(): $$$BODY"
			assert results[0].file_path == temp_path
			assert results[0].matched_text == "def hello():\n    print('world')"

			# Verify mock calls
			mock_sg_root.assert_called_once()
			mock_root_node.find_all.assert_called_once_with(pattern="def $NAME(): $$$BODY")

		finally:
			temp_path.unlink()

	@patch("codemap.processor.semantic_search.engine.SgRoot")
	def test_search_pattern_with_constraints(self, mock_sg_root):
		"""Test pattern search with constraints."""
		with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
			f.write("def test_hello():\n    pass\n")
			temp_path = Path(f.name)

		try:
			mock_match = Mock()
			mock_match.text.return_value = "def test_hello():\n    pass"
			mock_match.kind.return_value = "function_definition"

			mock_range = Mock()
			mock_start = Mock()
			mock_start.line = 0
			mock_start.column = 0
			mock_end = Mock()
			mock_end.line = 1
			mock_end.column = 15
			mock_range.start = mock_start
			mock_range.end = mock_end
			mock_match.range.return_value = mock_range

			mock_root_node = Mock()
			mock_root_node.find_all.return_value = [mock_match]

			mock_sg_instance = Mock()
			mock_sg_instance.root.return_value = mock_root_node
			mock_sg_root.return_value = mock_sg_instance

			constraints = {"NAME": {"regex": "^test_"}}
			results = self.engine.search_pattern(
				pattern="def $NAME(): $$$BODY", file_paths=[temp_path], constraints=constraints
			)

			assert len(results) == 1

			# Verify constraints were passed correctly
			expected_call = {"rule": {"pattern": "def $NAME(): $$$BODY"}, "constraints": constraints}
			mock_root_node.find_all.assert_called_once_with(expected_call)

		finally:
			temp_path.unlink()

	@patch("codemap.processor.semantic_search.engine.SgRoot")
	def test_search_pattern_auto_detect_language(self, mock_sg_root):
		"""Test pattern search with automatic language detection."""
		with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
			f.write("function hello() {\n    console.log('world');\n}")
			temp_path = Path(f.name)

		try:
			mock_match = Mock()
			mock_match.text.return_value = "function hello() {\n    console.log('world');\n}"
			mock_match.kind.return_value = "function_declaration"

			mock_range = Mock()
			mock_start = Mock()
			mock_start.line = 0
			mock_start.column = 0
			mock_end = Mock()
			mock_end.line = 2
			mock_end.column = 1
			mock_range.start = mock_start
			mock_range.end = mock_end
			mock_match.range.return_value = mock_range

			mock_root_node = Mock()
			mock_root_node.find_all.return_value = [mock_match]

			mock_sg_instance = Mock()
			mock_sg_instance.root.return_value = mock_root_node
			mock_sg_root.return_value = mock_sg_instance

			# Execute search without specifying language
			results = self.engine.search_pattern(pattern="function $NAME() { $$$BODY }", file_paths=[temp_path])

			assert len(results) == 1
			assert results[0].file_path == temp_path

			# Verify SgRoot was called with JavaScript
			mock_sg_root.assert_called_once()
			args, kwargs = mock_sg_root.call_args
			assert len(args) == 2
			assert args[1] == "javascript"  # Language should be auto-detected

		finally:
			temp_path.unlink()

	def test_search_pattern_limit(self):
		"""Test search result limiting."""
		# Create multiple test files
		temp_files = []
		try:
			for i in range(5):
				with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
					f.write(f"def func_{i}():\n    pass\n")
					temp_files.append(Path(f.name))

			with patch("codemap.processor.semantic_search.engine.SgRoot") as mock_sg_root:
				# Mock multiple matches
				matches = []
				for i in range(5):
					mock_match = Mock()
					mock_match.text.return_value = f"def func_{i}():\n    pass"
					mock_match.kind.return_value = "function_definition"

					mock_range = Mock()
					mock_start = Mock()
					mock_start.line = 0
					mock_start.column = 0
					mock_end = Mock()
					mock_end.line = 1
					mock_end.column = 10
					mock_range.start = mock_start
					mock_range.end = mock_end
					mock_match.range.return_value = mock_range
					matches.append(mock_match)

				mock_root_node = Mock()
				mock_root_node.find_all.return_value = matches

				mock_sg_instance = Mock()
				mock_sg_instance.root.return_value = mock_root_node
				mock_sg_root.return_value = mock_sg_instance

				# Search with limit
				results = self.engine.search_pattern(pattern="def $NAME(): $$$BODY", file_paths=temp_files, limit=3)

				# Should respect the limit
				assert len(results) == 3

		finally:
			for temp_file in temp_files:
				temp_file.unlink()

	def test_search_pattern_file_read_error(self):
		"""Test search behavior when file cannot be read."""
		# Create a file and then make it unreadable
		with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
			f.write("def hello(): pass")
			temp_path = Path(f.name)

		try:
			# Make file unreadable by deleting it
			temp_path.unlink()

			results = self.engine.search_pattern(pattern="def $NAME(): $$$BODY", file_paths=[temp_path])

			# Should handle the error gracefully and return empty results
			assert len(results) == 0

		except FileNotFoundError:
			# This is expected behavior
			pass

	def test_search_pattern_unsupported_language(self):
		"""Test search with unsupported file type."""
		with tempfile.NamedTemporaryFile(mode="w", suffix=".unknown", delete=False) as f:
			f.write("some content")
			temp_path = Path(f.name)

		try:
			results = self.engine.search_pattern(pattern="def $NAME(): $$$BODY", file_paths=[temp_path])

			# Should skip unsupported files
			assert len(results) == 0

		finally:
			temp_path.unlink()

	def test_search_pattern_no_files(self):
		"""Test search with empty file list."""
		results = self.engine.search_pattern(pattern="def $NAME(): $$$BODY", file_paths=[])

		assert len(results) == 0

	@patch.object(AstGrepEngine, "_find_source_files")
	def test_search_pattern_auto_find_files(self, mock_find_files):
		"""Test search with automatic file discovery."""
		# Mock file discovery
		with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
			f.write("def hello(): pass")
			temp_path = Path(f.name)

		mock_find_files.return_value = [temp_path]

		try:
			with patch("codemap.processor.semantic_search.engine.SgRoot") as mock_sg_root:
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

				mock_root_node = Mock()
				mock_root_node.find_all.return_value = [mock_match]

				mock_sg_instance = Mock()
				mock_sg_instance.root.return_value = mock_root_node
				mock_sg_root.return_value = mock_sg_instance

				# Search without providing file_paths
				results = self.engine.search_pattern(pattern="def $NAME(): $$$BODY")

				assert len(results) == 1
				mock_find_files.assert_called_once()

		finally:
			temp_path.unlink()
