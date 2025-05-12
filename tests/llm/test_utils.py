"""Tests for the LLM utils module."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from codemap.config import ConfigLoader
from codemap.llm.utils import load_prompt_template


@pytest.fixture
def mock_config_loader():
	"""Fixture to create a mock ConfigLoader."""
	mock_loader = Mock(spec=ConfigLoader)
	# Create a mock for the 'get' property that returns a Pydantic model with llm attribute
	mock_get = MagicMock()
	mock_llm = MagicMock()

	# Set up the llm config properties
	mock_llm.model = "openai:gpt-4o-mini"
	mock_llm.temperature = 0.7
	mock_llm.max_output_tokens = 1000

	# Attach the llm config to the get property
	type(mock_get).llm = type(mock_get).embedding = type(mock_get).rag = MagicMock()
	mock_get.llm = mock_llm

	# Make 'get' a property
	type(mock_loader).get = property(lambda _: mock_get)

	return mock_loader


@pytest.mark.unit
def test_load_prompt_template_none():
	"""Test load_prompt_template with None path."""
	assert load_prompt_template(None) is None


@pytest.mark.unit
def test_load_prompt_template_exists():
	"""Test load_prompt_template with existing file."""
	template_content = "This is a test template for {param}"
	mock_file = MagicMock()
	mock_file.__enter__.return_value.read.return_value = template_content

	with patch("pathlib.Path.open", return_value=mock_file), patch("pathlib.Path.exists", return_value=True):
		result = load_prompt_template("test_template.txt")
		assert result == template_content


@pytest.mark.unit
def test_load_prompt_template_not_exists():
	"""Test load_prompt_template with non-existent file."""
	with patch("pathlib.Path.exists", return_value=False):
		result = load_prompt_template("nonexistent.txt")
		assert result is None
