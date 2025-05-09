"""Tests for the LLM RAG formatter module."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.markdown import Markdown

from codemap.llm.rag.ask.formatter import format_ask_response, format_content_for_context, print_ask_result


@pytest.fixture
def mock_console():
	"""Create a mock Rich console for testing output."""
	string_io = StringIO()
	return Console(file=string_io, highlight=False)


@pytest.mark.unit
def test_format_ask_response_text():
	"""Test formatting simple text result."""
	result = "This is a simple text answer"

	formatted = format_ask_response(result)

	assert isinstance(formatted, Markdown)
	assert result in formatted.markup


@pytest.mark.unit
def test_format_ask_response_none():
	"""Test formatting None result."""
	formatted = format_ask_response(None)

	assert isinstance(formatted, Markdown)
	assert "No response generated" in formatted.markup


@pytest.mark.unit
def test_print_ask_result_with_context(mock_console):
	"""Test printing structured result with context to console."""
	result = {
		"answer": "The answer is 42",
		"context": [
			{"file_path": "source1.py", "start_line": 10, "end_line": 15, "distance": 0.1},
			{"file_path": "source2.py", "start_line": 20, "end_line": 25, "distance": 0.2},
		],
	}

	with patch("codemap.llm.rag.ask.formatter.rich_print") as mock_print:
		print_ask_result(result)

		# Verify print was called at least twice (answer and context)
		assert mock_print.call_count >= 2


@pytest.mark.unit
def test_print_ask_result_without_context(mock_console):
	"""Test printing structured result without context."""
	result = {"answer": "The answer is 42"}

	with patch("codemap.llm.rag.ask.formatter.rich_print") as mock_print:
		print_ask_result(result)

		# Should only print the answer, not context
		assert mock_print.call_count == 1


@pytest.mark.unit
def test_format_content_for_context_empty():
	"""Test formatting empty context items."""
	result = format_content_for_context([])

	assert "No relevant code found" in result


@pytest.mark.unit
def test_format_content_for_context_with_items():
	"""Test formatting context items."""
	context_items = [
		{"file_path": "file1.py", "start_line": 10, "end_line": 20, "content": "def function1():\n    return 42"},
		{"file_path": "file2.py", "start_line": 5, "end_line": 15, "content": "def function2():\n    return 'hello'"},
	]

	result = format_content_for_context(context_items)

	# Should include file paths and line numbers
	assert "file1.py" in result
	assert "lines 10-20" in result
	assert "file2.py" in result
	assert "lines 5-15" in result

	# Should include content
	assert "def function1():" in result
	assert "return 42" in result
	assert "def function2():" in result
	assert "return 'hello'" in result
