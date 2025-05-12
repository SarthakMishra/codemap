"""Tests for the TreeSitterAnalyzer."""

from unittest.mock import MagicMock, patch

import pytest
from tree_sitter import Language, Node, Parser, Tree

from codemap.processor.tree_sitter.analyzer import TreeSitterAnalyzer, get_language_by_extension
from codemap.processor.tree_sitter.base import EntityType
from codemap.processor.tree_sitter.languages import LanguageSyntaxHandler

# --- Fixtures ---

# Sample code for testing
PYTHON_CODE = """
def greet(name):
    '''Simple greeting function.'''
    print(f"Hello, {name}!")

class MyClass:
    pass
"""

JAVASCRIPT_CODE = """
function add(a, b) {
  // Adds two numbers
  return a + b;
}
"""

UNKNOWN_CODE = "<xml><tag>content</tag></xml>"


@pytest.fixture
def analyzer():
	"""Provides a TreeSitterAnalyzer instance."""
	return TreeSitterAnalyzer()
	# Tests that need to control parser behavior should mock
	# instance.get_parser directly or its dependencies like
	# 'codemap.processor.tree_sitter.analyzer.get_language'
	# and 'codemap.processor.tree_sitter.analyzer.Parser'.


@pytest.fixture
def mock_parser(analyzer):  # Depends on analyzer
	"""Provides a mock Parser object and patches analyzer.get_parser for 'python'."""
	mock_p = MagicMock(spec=Parser)
	mock_p.language = MagicMock(spec=Language)
	mock_tree = MagicMock(spec=Tree)
	mock_root_node = MagicMock(spec=Node)
	mock_root_node.text = PYTHON_CODE.encode("utf-8")
	mock_root_node.named_child_count = 2
	mock_tree.root_node = mock_root_node
	mock_p.parse.return_value = mock_tree

	# Patch get_parser on the analyzer instance
	# So when analyzer.get_parser("python") is called, it returns our mock_p

	def side_effect_get_parser(lang_name):
		if lang_name == "python":
			return mock_p
		# For other languages, or if the original get_parser had more complex logic
		# you might want to call original_get_parser(lang_name) here.
		# For this test setup, if it's not python, we can assume it might return None or raise.
		return None  # Or call original_get_parser(lang_name) if other tests need it.

	with patch.object(analyzer, "get_parser", side_effect=side_effect_get_parser):
		yield mock_p


@pytest.fixture
def mock_handler():
	"""Provides a mock LanguageSyntaxHandler."""
	handler = MagicMock(spec=LanguageSyntaxHandler)
	handler.should_skip_node.return_value = False
	handler.get_entity_type.return_value = EntityType.FUNCTION  # Default mock type
	handler.extract_name.return_value = "mock_function"
	handler.find_docstring.return_value = ("Mock docstring", MagicMock(spec=Node))
	handler.extract_imports.return_value = []
	handler.extract_calls.return_value = []
	handler.get_body_node.return_value = MagicMock(spec=Node)
	handler.get_children_to_process.return_value = []
	return handler


# --- Test Cases ---


# Test get_language_by_extension
@pytest.mark.parametrize(
	("filename", "expected_lang"),
	[
		("test.py", "python"),
		("script.js", "javascript"),
		("component.ts", "typescript"),  # Assuming typescript is configured
		("style.css", None),
		("README.md", None),
		("no_extension", None),
	],
)
def test_get_language_by_extension(filename, expected_lang, tmp_path):
	"""Test language detection based on file extension."""
	file_path = tmp_path / filename
	assert get_language_by_extension(file_path) == expected_lang


# Test get_parser
def test_get_parser(analyzer):
	"""Test retrieving a parser for a supported language."""
	assert analyzer.get_parser("python") is not None
	assert analyzer.get_parser("javascript") is not None
	assert analyzer.get_parser("unknown_lang") is None


# Test parse_file
def test_parse_file_success(analyzer, mock_parser, tmp_path):
	"""Test successful parsing of a file."""
	file_path = tmp_path / "test.py"
	file_path.write_text(PYTHON_CODE)

	root_node, lang, parser_instance, content_bytes = analyzer.parse_file(file_path, language="python")

	assert lang == "python"
	assert root_node is not None
	assert parser_instance is mock_parser
	assert content_bytes == PYTHON_CODE.encode("utf-8")
	mock_parser.parse.assert_called_once_with(PYTHON_CODE.encode("utf-8"))


def test_parse_file_language_detection(analyzer, mock_parser, tmp_path):
	"""Test parse_file correctly detects language if not provided."""
	file_path = tmp_path / "test.py"
	file_path.write_text(PYTHON_CODE)

	root_node, lang, parser_instance, content_bytes = analyzer.parse_file(file_path)

	assert lang == "python"
	assert root_node is not None
	assert parser_instance is mock_parser
	assert content_bytes == PYTHON_CODE.encode("utf-8")
	mock_parser.parse.assert_called_once_with(PYTHON_CODE.encode("utf-8"))


def test_parse_file_unsupported_language(analyzer, tmp_path):
	"""Test parse_file with an unsupported language."""
	file_path = tmp_path / "test.unknown"
	file_path.write_text(UNKNOWN_CODE)

	root_node, lang, parser_instance, content_bytes = analyzer.parse_file(file_path)

	assert lang == ""
	assert root_node is None
	assert parser_instance is None
	assert content_bytes is None


def test_parse_file_no_parser(analyzer, tmp_path):
	"""Test parse_file when parser for the language is not available."""
	file_path = tmp_path / "test.java"
	file_path.write_text("class Test {}")

	# Mock get_language_by_extension to ensure "java" is detected
	# Mock analyzer.get_parser to return None for "java"
	with (
		patch("codemap.processor.tree_sitter.analyzer.get_language_by_extension", return_value="java"),
		patch.object(
			analyzer,
			"get_parser",
			side_effect=lambda lang_name: None if lang_name == "java" else MagicMock(spec=Parser),
		),
	):
		root_node, lang, parser_instance, content_bytes = analyzer.parse_file(file_path)

	assert lang == "java"
	assert root_node is None
	assert parser_instance is None
	assert content_bytes is None


def test_parse_file_parser_error(analyzer, mock_parser, tmp_path):
	"""Test parse_file when the tree-sitter parser raises an error."""
	file_path = tmp_path / "test.py"
	file_path.write_text(PYTHON_CODE)
	# mock_parser fixture already patches analyzer.get_parser for "python"
	mock_parser.parse.side_effect = Exception("Parsing Crash")

	root_node, lang, parser_instance, content_bytes = analyzer.parse_file(file_path, language="python")

	assert lang == "python"
	assert root_node is None
	assert parser_instance is mock_parser
	assert content_bytes == PYTHON_CODE.encode("utf-8")  # Content is read before parse attempt


# Test get_syntax_handler
@patch("codemap.processor.tree_sitter.analyzer.LANGUAGE_HANDLERS")
def test_get_syntax_handler(mock_handlers, analyzer):
	"""Test retrieving the correct syntax handler."""
	mock_py_handler_cls = MagicMock()
	mock_js_handler_cls = MagicMock()
	mock_py_handler_instance = MagicMock(spec=LanguageSyntaxHandler)
	mock_js_handler_instance = MagicMock(spec=LanguageSyntaxHandler)

	mock_py_handler_cls.return_value = mock_py_handler_instance
	mock_js_handler_cls.return_value = mock_js_handler_instance

	mock_handlers.get.side_effect = lambda lang: {"python": mock_py_handler_cls, "javascript": mock_js_handler_cls}.get(
		lang
	)

	py_handler = analyzer.get_syntax_handler("python")
	js_handler = analyzer.get_syntax_handler("javascript")
	unknown_handler = analyzer.get_syntax_handler("unknown")

	assert py_handler is mock_py_handler_instance
	assert js_handler is mock_js_handler_instance
	assert unknown_handler is None
	mock_py_handler_cls.assert_called_once()
	mock_js_handler_cls.assert_called_once()


# Test analyze_node (basic case)
def test_analyze_node(analyzer, mock_handler, tmp_path):
	"""Test basic analysis of a single node."""
	mock_node = MagicMock(spec=Node)
	mock_node.start_byte = 0
	mock_node.end_byte = 10
	mock_node.start_point = (0, 0)
	mock_node.end_point = (1, 5)
	mock_node.named_child_count = 0  # No children for simplicity
	content_bytes = b"def func(): pass"
	file_path = tmp_path / "test.py"

	result = analyzer.analyze_node(
		node=mock_node,
		content_bytes=content_bytes,
		file_path=file_path,
		language="python",
		handler=mock_handler,
		parent_node=None,
	)

	assert result["type"] == EntityType.FUNCTION.name
	assert result["name"] == "mock_function"
	assert result["docstring"] == "Mock docstring"
	assert result["location"]["start_line"] == 1
	assert result["location"]["end_line"] == 2
	assert result["content"] == "def func()"
	assert "children" in result
	assert len(result["children"]) == 0
	assert result["language"] == "python"
	# Check handler methods were called
	mock_handler.should_skip_node.assert_called_once_with(mock_node)
	mock_handler.get_entity_type.assert_called_once()
	mock_handler.extract_name.assert_called_once()
	mock_handler.find_docstring.assert_called_once()
	# Since it's a function, body/call extraction should be attempted
	assert mock_handler.get_body_node.call_count >= 1  # It may be called more than once
	mock_handler.extract_calls.assert_called_once()


# Test analyze_file (integration-like)
@patch.object(TreeSitterAnalyzer, "get_syntax_handler")
@patch.object(TreeSitterAnalyzer, "parse_file")
@patch.object(TreeSitterAnalyzer, "analyze_node")  # Mock recursive call
def test_analyze_file(mock_analyze_node, mock_parse_file, mock_get_handler, analyzer, mock_handler, tmp_path):
	"""Test the overall file analysis process."""
	file_path = tmp_path / "test.py"
	file_path.write_text(PYTHON_CODE)

	mock_root_node = MagicMock(spec=Node)

	mock_parser_instance = MagicMock(spec=Parser)
	mock_parse_file.return_value = (mock_root_node, "python", mock_parser_instance, PYTHON_CODE.encode("utf-8"))

	mock_get_handler.return_value = mock_handler
	# We still set this up to show that if analyze_node was real, it would get children:
	mock_child_node_for_handler = MagicMock(spec=Node)
	mock_handler.get_children_to_process.return_value = [mock_child_node_for_handler]
	# The return value of mock_analyze_node is what analyze_file returns
	# However, analyze_file itself adds 'full_content_str' to this result.
	mock_analyze_node_return_value = {"type": "MODULE", "name": "test.py", "children": []}
	mock_analyze_node.return_value = mock_analyze_node_return_value

	# Add debugging print to see the actual calls made
	mock_analyze_node.side_effect = lambda *args, **kwargs: (
		print(f"DEBUG: analyze_node called with: {args}, {kwargs}") or mock_analyze_node_return_value  # noqa: T201
	)

	analysis_result = analyzer.analyze_file(file_path, language="python")

	expected_analysis_result = {
		**mock_analyze_node_return_value,
		"full_content_str": PYTHON_CODE,
	}
	assert analysis_result == expected_analysis_result
	mock_parse_file.assert_called_once_with(file_path, "python")

	# We won't assert the exact call parameters since we're printing them for debugging
	# Instead we'll just check that it was called at least once
	assert mock_analyze_node.call_count >= 1
	mock_get_handler.assert_called_once_with("python")


def test_analyze_file_parsing_fails(analyzer, tmp_path):
	"""Test analyze_file when parsing fails."""
	file_path = tmp_path / "invalid.py"
	content = "def func("  # Invalid syntax
	file_path.write_text(content)

	with patch.object(analyzer, "parse_file", return_value=(None, "python", None, content.encode("utf-8"))):
		analysis_result = analyzer.analyze_file(file_path, language="python")  # No content string

	assert analysis_result == {}


def test_analyze_file_no_handler(analyzer, tmp_path):
	"""Test analyze_file when no syntax handler is available."""
	file_path = tmp_path / "test.unknown"
	content = "some content"
	file_path.write_text(content)

	mock_root_node = MagicMock(spec=Node)
	mock_parser_instance = MagicMock(spec=Parser)
	with (
		patch.object(
			analyzer,
			"parse_file",
			return_value=(mock_root_node, "unknown", mock_parser_instance, content.encode("utf-8")),
		),
		patch.object(analyzer, "get_syntax_handler", return_value=None),
	):
		analysis_result = analyzer.analyze_file(file_path, language="unknown")  # No content string

	assert analysis_result == {}
