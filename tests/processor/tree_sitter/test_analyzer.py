"""Tests for the TreeSitterAnalyzer."""

from unittest.mock import MagicMock, call, patch

import pytest
from tree_sitter import Language, Node, Parser, Tree  # Import Tree

from codemap.processor.tree_sitter.analyzer import LANGUAGE_NAMES, TreeSitterAnalyzer, get_language_by_extension
from codemap.processor.tree_sitter.base import EntityType
from codemap.processor.tree_sitter.languages import LANGUAGE_CONFIGS, LanguageSyntaxHandler

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
	# Prevent actual parser loading during tests by patching _load_parsers
	with patch.object(TreeSitterAnalyzer, "_load_parsers", return_value=None):
		instance = TreeSitterAnalyzer()
		# Manually add mock parsers if needed for specific tests
		instance.parsers = {
			"python": MagicMock(spec=Parser),
			"javascript": MagicMock(spec=Parser),
		}
		yield instance  # Use yield to ensure cleanup if needed


@pytest.fixture
def mock_parser(analyzer):  # Depends on analyzer to get a target parser
	"""Provides a mock Parser object."""
	parser = analyzer.parsers["python"]  # Get one of the mock parsers
	parser.language = MagicMock(spec=Language)
	parser.parse.return_value = MagicMock(spec=Tree)
	parser.parse.return_value.root_node = MagicMock(spec=Node)
	parser.parse.return_value.root_node.text = PYTHON_CODE.encode("utf-8")  # Add text attribute
	parser.parse.return_value.root_node.named_child_count = 2  # Simulate children
	# Add more mock attributes/methods to root_node as needed
	return parser


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


# Test Analyzer Initialization
@patch("codemap.processor.tree_sitter.analyzer.get_language")
@patch("codemap.processor.tree_sitter.analyzer.Parser")
def test_analyzer_init_loads_parsers(mock_parser_cls, mock_get_lang):
	"""Test that __init__ calls _load_parsers and attempts to load languages."""
	# Mock get_language to return a mock Language object
	mock_lang_obj = MagicMock(spec=Language)
	mock_get_lang.return_value = mock_lang_obj
	# Mock Parser constructor
	mock_parser_instance = MagicMock(spec=Parser)
	mock_parser_cls.return_value = mock_parser_instance

	analyzer = TreeSitterAnalyzer()  # Call init directly

	# Check that get_language and Parser were called for configured languages
	expected_calls = []
	for lang, ts_lang_name in LANGUAGE_NAMES.items():
		if lang in LANGUAGE_CONFIGS:  # Only check configured languages
			expected_calls.append(call(ts_lang_name))

	mock_get_lang.assert_has_calls(expected_calls, any_order=True)
	assert mock_parser_cls.call_count == len(expected_calls)
	# Check that the parser's language was set
	assert mock_parser_instance.language == mock_lang_obj
	# Check that parsers are stored
	for lang in LANGUAGE_NAMES:
		if lang in LANGUAGE_CONFIGS:
			assert lang in analyzer.parsers
			assert analyzer.parsers[lang] is mock_parser_instance


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
	analyzer.parsers["python"] = mock_parser  # Use the configured mock parser

	root_node, lang = analyzer.parse_file(file_path, PYTHON_CODE, "python")

	assert lang == "python"
	assert root_node is not None
	mock_parser.parse.assert_called_once_with(PYTHON_CODE.encode("utf-8"))


def test_parse_file_language_detection(analyzer, mock_parser, tmp_path):
	"""Test parse_file correctly detects language if not provided."""
	file_path = tmp_path / "test.py"
	file_path.write_text(PYTHON_CODE)
	analyzer.parsers["python"] = mock_parser

	# Call without specifying language
	root_node, lang = analyzer.parse_file(file_path, PYTHON_CODE)

	assert lang == "python"
	assert root_node is not None
	mock_parser.parse.assert_called_once_with(PYTHON_CODE.encode("utf-8"))


def test_parse_file_unsupported_language(analyzer, tmp_path):
	"""Test parse_file with an unsupported language."""
	file_path = tmp_path / "test.unknown"
	file_path.write_text(UNKNOWN_CODE)

	root_node, lang = analyzer.parse_file(file_path, UNKNOWN_CODE)

	assert lang == ""
	assert root_node is None


def test_parse_file_no_parser(analyzer, tmp_path):
	"""Test parse_file when parser for the language is not available."""
	file_path = tmp_path / "test.java"  # Assume java parser wasn't loaded
	file_path.write_text("class Test {}")
	# Ensure no parser exists for java
	if "java" in analyzer.parsers:
		del analyzer.parsers["java"]

	with patch("codemap.processor.tree_sitter.analyzer.get_language_by_extension", return_value="java"):
		root_node, lang = analyzer.parse_file(file_path, "class Test {}")

	assert lang == "java"
	assert root_node is None


def test_parse_file_parser_error(analyzer, mock_parser, tmp_path):
	"""Test parse_file when the tree-sitter parser raises an error."""
	file_path = tmp_path / "test.py"
	file_path.write_text(PYTHON_CODE)
	analyzer.parsers["python"] = mock_parser
	mock_parser.parse.side_effect = Exception("Parsing Crash")

	root_node, lang = analyzer.parse_file(file_path, PYTHON_CODE, "python")

	assert lang == "python"
	assert root_node is None


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

	# Mock parse_file result
	mock_root_node = MagicMock(spec=Node)
	mock_root_node.named_child_count = 1
	mock_child_node = MagicMock(spec=Node)  # The single child
	mock_root_node.children = [mock_child_node]  # Simulate children
	mock_parse_file.return_value = (mock_root_node, "python")

	# Mock get_syntax_handler
	mock_get_handler.return_value = mock_handler

	# Configure handler mock to return the child node
	mock_handler.get_children_to_process.return_value = [mock_child_node]

	# Mock analyze_node to return a simplified result
	mock_analyze_node.return_value = {"type": "FUNCTION", "name": "child_func", "children": []}

	analysis_result = analyzer.analyze_file(file_path, PYTHON_CODE, "python")

	assert analysis_result["success"] is True
	assert analysis_result["language"] == "python"
	assert analysis_result["file"] == str(file_path)
	assert "children" in analysis_result
	assert len(analysis_result["children"]) > 0  # Root node has children

	# Verify mocks were called
	mock_parse_file.assert_called_once_with(file_path, PYTHON_CODE, "python")
	mock_get_handler.assert_called_once_with("python")
	# analyze_node should be called for the root's children (mocked here)
	# Check it was called at least once, exact number depends on handler/mock structure
	mock_analyze_node.assert_called()


def test_analyze_file_parsing_fails(analyzer, tmp_path):
	"""Test analyze_file when parsing fails."""
	file_path = tmp_path / "invalid.py"
	content = "def func("  # Invalid syntax
	file_path.write_text(content)

	# Simulate parse_file returning None
	with patch.object(analyzer, "parse_file", return_value=(None, "python")):
		analysis_result = analyzer.analyze_file(file_path, content, "python")

	assert analysis_result["success"] is False
	assert analysis_result["file"] == str(file_path)
	assert "error" in analysis_result


def test_analyze_file_no_handler(analyzer, tmp_path):
	"""Test analyze_file when no syntax handler is available."""
	file_path = tmp_path / "test.unknown"
	content = "some content"
	file_path.write_text(content)

	# Simulate parse_file succeeding but get_handler returning None
	mock_root_node = MagicMock(spec=Node)
	with (
		patch.object(analyzer, "parse_file", return_value=(mock_root_node, "unknown")),
		patch.object(analyzer, "get_syntax_handler", return_value=None),
	):
		analysis_result = analyzer.analyze_file(file_path, content, "unknown")

	assert analysis_result["success"] is False
	assert analysis_result["file"] == str(file_path)
	assert "error" in analysis_result
	assert analysis_result["error"] == "No handler for language unknown"
