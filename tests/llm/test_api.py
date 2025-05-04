"""Tests for the LLM API module."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from codemap.llm.api import call_llm_api, extract_content_from_response
from codemap.llm.errors import LLMError
from codemap.utils.config_loader import ConfigLoader


@pytest.fixture
def mock_config_loader():
	"""Fixture to create a mock ConfigLoader."""
	mock_loader = Mock(spec=ConfigLoader)
	mock_loader.get_llm_config.return_value = {
		"temperature": 0.7,
		"max_tokens": 1000,
	}
	return mock_loader


@pytest.mark.unit
def test_call_llm_api_litellm_not_installed():
	"""Test handling of missing litellm dependency."""
	with patch.dict(sys.modules, {"litellm": None}), pytest.raises(LLMError, match="LiteLLM library not installed"):
		call_llm_api(prompt="Test prompt", model="openai/gpt-4", api_key="test-key")


@pytest.mark.unit
def test_call_llm_api_success():
	"""Test successful API call."""
	mock_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Generated content"))])

	with patch("litellm.completion") as mock_completion:
		mock_completion.return_value = mock_response

		result = call_llm_api(prompt="Test prompt", model="openai/gpt-4", api_key="test-key")

		# Verify API was called correctly
		mock_completion.assert_called_once()
		call_args = mock_completion.call_args[1]
		assert call_args["model"] == "openai/gpt-4"
		assert call_args["api_key"] == "test-key"
		assert call_args["messages"][0]["content"] == "Test prompt"

		# Verify result
		assert result == "Generated content"


@pytest.mark.unit
def test_call_llm_api_with_config():
	"""Test API call with configuration from ConfigLoader."""
	mock_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Generated with config"))])

	mock_loader = Mock(spec=ConfigLoader)
	mock_loader.get_llm_config.return_value = {
		"temperature": 0.5,
		"max_tokens": 2000,
	}

	with patch("litellm.completion") as mock_completion:
		mock_completion.return_value = mock_response

		result = call_llm_api(prompt="Test prompt", model="openai/gpt-4", api_key="test-key", config_loader=mock_loader)

		# Verify config was used
		call_args = mock_completion.call_args[1]
		assert call_args["temperature"] == 0.5
		assert call_args["max_tokens"] == 2000

		# Verify result
		assert result == "Generated with config"


@pytest.mark.unit
def test_call_llm_api_with_api_base():
	"""Test API call with custom API base URL."""
	mock_response = SimpleNamespace(
		choices=[SimpleNamespace(message=SimpleNamespace(content="Generated with custom API base"))]
	)

	with patch("litellm.completion") as mock_completion:
		mock_completion.return_value = mock_response

		result = call_llm_api(
			prompt="Test prompt", model="openai/gpt-4", api_key="test-key", api_base="https://custom-api.example.com"
		)

		# Verify custom API base was used
		call_args = mock_completion.call_args[1]
		assert call_args["api_base"] == "https://custom-api.example.com"

		# Verify result
		assert result == "Generated with custom API base"


@pytest.mark.unit
def test_call_llm_api_with_json_schema():
	"""Test API call with JSON schema for response validation."""
	mock_response = SimpleNamespace(
		choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer": "Yes", "confidence": 0.9}'))]
	)

	schema = {"type": "object", "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}}}

	with (
		patch("litellm.completion") as mock_completion,
		patch("litellm.enable_json_schema_validation", value=True, create=True),
	):
		mock_completion.return_value = mock_response

		result = call_llm_api(prompt="Test prompt", model="openai/gpt-4", api_key="test-key", json_schema=schema)

		# Verify JSON schema was used
		call_args = mock_completion.call_args[1]
		assert call_args["response_format"]["type"] == "json_object"
		assert call_args["response_format"]["schema"] == schema

		# Verify result
		assert result == '{"answer": "Yes", "confidence": 0.9}'


@pytest.mark.unit
def test_call_llm_api_error():
	"""Test handling of API call errors."""
	with (
		patch("litellm.completion", side_effect=Exception("API error")),
		pytest.raises(LLMError, match="LLM API call failed: API error"),
	):
		call_llm_api(prompt="Test prompt", model="openai/gpt-4", api_key="test-key")


@pytest.mark.unit
def test_call_llm_api_empty_response():
	"""Test handling of empty response content."""
	# Create a response with empty content
	mock_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])

	with patch("litellm.completion") as mock_completion:
		mock_completion.return_value = mock_response

		with pytest.raises(LLMError, match="Failed to extract content from LLM response"):
			call_llm_api(prompt="Test prompt", model="openai/gpt-4", api_key="test-key")


@pytest.mark.unit
def test_extract_content_from_response_object():
	"""Test extracting content from response object."""
	# Test with a SimpleNamespace object (like from OpenAI)
	response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Response content"))])

	content = extract_content_from_response(response)
	assert content == "Response content"


@pytest.mark.unit
def test_extract_content_from_response_dict():
	"""Test extracting content from response dictionary."""
	# Test with a dictionary (common format)
	response = {"choices": [{"message": {"content": "Dictionary content"}}]}

	content = extract_content_from_response(response)
	assert content == "Dictionary content"


@pytest.mark.unit
def test_extract_content_from_response_text():
	"""Test extracting content from response with text attribute."""
	# Test with an object that has a text attribute
	response = SimpleNamespace(text="Direct text content")

	content = extract_content_from_response(response)
	assert content == "Direct text content"


@pytest.mark.unit
def test_extract_content_from_response_empty():
	"""Test extracting content from empty response."""
	# Test with None
	assert extract_content_from_response(None) == ""

	# Test with empty dict
	assert extract_content_from_response({}) == ""

	# Test with empty object
	assert extract_content_from_response(SimpleNamespace()) == ""
