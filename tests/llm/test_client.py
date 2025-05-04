"""Tests for the LLM client module."""

from __future__ import annotations

import json
from typing import ClassVar
from unittest.mock import Mock, patch

import pytest

from codemap.llm.client import LLMClient
from codemap.llm.config import LLMConfig
from codemap.llm.errors import LLMError
from codemap.utils.config_loader import ConfigLoader


@pytest.fixture
def mock_config_loader():
	"""Fixture to create a mock ConfigLoader."""
	mock_loader = Mock(spec=ConfigLoader)
	mock_loader.get_llm_config.return_value = {
		"model": "openai/gpt-4",
		"provider": "openai",
		"temperature": 0.7,
		"max_tokens": 1000,
	}
	return mock_loader


@pytest.fixture
def llm_client(mock_config_loader):
	"""Fixture to create an LLMClient with mocked dependencies."""
	config = LLMConfig(
		model="openai/gpt-4",
		provider="openai",
		api_key="test-key",
	)
	return LLMClient(config=config, config_loader=mock_config_loader)


@pytest.mark.unit
def test_llm_client_initialization():
	"""Test that the LLMClient initializes correctly with various parameters."""
	# Test with explicitly provided config
	config = LLMConfig(model="openai/gpt-4", provider="openai", api_key="test-key")
	client = LLMClient(config=config)
	assert client.config.model == "openai/gpt-4"
	assert client.config.provider == "openai"
	assert client.config.get_api_key() == "test-key"

	# Test with default config
	client = LLMClient()
	assert client.config is not None

	# Test with config loader
	mock_loader = Mock(spec=ConfigLoader)
	client = LLMClient(config_loader=mock_loader)
	assert client.config_loader == mock_loader


@pytest.mark.unit
def test_template_management():
	"""Test template setting and retrieval."""
	client = LLMClient()

	# Test setting and getting a template
	client.set_template("test", "This is a {test} template")
	assert client.get_template("test") == "This is a {test} template"

	# Test getting a non-existent template
	with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
		client.get_template("nonexistent")

	# Test default templates
	class CustomClient(LLMClient):
		DEFAULT_TEMPLATES: ClassVar[dict[str, str]] = {"default": "Default template {var}"}

	custom_client = CustomClient()
	assert custom_client.get_template("default") == "Default template {var}"


@pytest.mark.unit
def test_get_default_model(llm_client, mock_config_loader):
	"""Test retrieving the default model."""
	# Mock config_loader returning a model
	assert llm_client._get_default_model() == "openai/gpt-4"

	# Test without config_loader
	client = LLMClient()
	assert client._get_default_model() == "openai/gpt-4o-mini"  # Hardcoded fallback


@pytest.mark.unit
def test_generate_text(llm_client):
	"""Test text generation with LLM API."""
	with patch("codemap.llm.api.litellm", create=True) as mock_litellm:
		# Mock the completion method directly
		mock_response = Mock()
		mock_response.choices = [Mock(message=Mock(content="Generated text response"))]
		mock_litellm.completion.return_value = mock_response

		# Mock the import check
		mock_litellm.__bool__.return_value = True

		# Use patch.object to directly modify the call_llm_api function's behavior
		with patch("codemap.llm.client.call_llm_api") as mock_call_api:
			mock_call_api.return_value = "Generated text response"

			result = llm_client.generate_text("Test prompt")

			# Verify API was called with correct parameters
			mock_call_api.assert_called_once()
			call_args = mock_call_api.call_args[1]
			assert call_args["prompt"] == "Test prompt"
			assert call_args["model"] == "openai/gpt-4"
			assert call_args["api_key"] == "test-key"

			# Verify result
			assert result == "Generated text response"


@pytest.mark.unit
def test_generate_text_no_api_key():
	"""Test text generation fails properly without API key."""
	config = LLMConfig(model="openai/gpt-4", provider="openai", api_key=None)
	client = LLMClient(config=config)

	with pytest.raises(LLMError, match="No API key available for openai provider"):
		client.generate_text("Test prompt")


@pytest.mark.unit
def test_generate_from_template(llm_client):
	"""Test generating text from a template."""
	# Set up a template
	llm_client.set_template("test_template", "This is a {type} template with {value}")

	with patch.object(llm_client, "generate_text") as mock_generate:
		mock_generate.return_value = "Generated from template"

		# Call with template variables
		result = llm_client.generate_from_template("test_template", {"type": "unit test", "value": "variables"})

		# Verify generate_text was called with formatted prompt
		mock_generate.assert_called_once()
		called_prompt = mock_generate.call_args[1]["prompt"]
		assert called_prompt == "This is a unit test template with variables"

		# Verify result
		assert result == "Generated from template"


@pytest.mark.unit
def test_generate_from_template_missing():
	"""Test generating text from a non-existent template raises error."""
	client = LLMClient()

	with pytest.raises(ValueError, match="Template 'missing' not found"):
		client.generate_from_template("missing", {"some": "vars"})


@pytest.mark.unit
def test_generate_text_with_json_schema(llm_client):
	"""Test generation with JSON schema validation."""
	schema = {
		"type": "object",
		"properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}},
		"required": ["answer"],
	}

	# Use patch.object to directly modify the call_llm_api function's behavior
	with patch("codemap.llm.client.call_llm_api") as mock_call_api:
		mock_call_api.return_value = json.dumps({"answer": "Yes", "confidence": 0.9})

		result = llm_client.generate_text("Test prompt requiring JSON response", json_schema=schema)

		# Verify API was called with correct schema
		mock_call_api.assert_called_once()
		call_args = mock_call_api.call_args[1]
		assert call_args["json_schema"] == schema

		# Verify result
		assert json.loads(result)["answer"] == "Yes"
		assert json.loads(result)["confidence"] == 0.9
