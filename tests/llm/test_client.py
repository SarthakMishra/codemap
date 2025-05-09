"""Tests for the LLM client module."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from codemap.config import ConfigLoader
from codemap.llm.api import MessageDict
from codemap.llm.client import LLMClient
from codemap.llm.errors import LLMError


@pytest.fixture
def mock_config_loader():
	"""Fixture to create a mock ConfigLoader."""
	mock_config = Mock()
	# Mock the pydantic schema structure
	mock_config.get = Mock()
	mock_config.get.llm = Mock()
	mock_config.get.llm.model = "openai/gpt-4"
	mock_config.get.llm.temperature = 0.7
	mock_config.get.llm.max_output_tokens = 1000
	mock_config.get.llm.model_dump.return_value = {
		"temperature": 0.7,
		"max_output_tokens": 1000,
	}

	return mock_config


@pytest.fixture
def llm_client(mock_config_loader):
	"""Fixture to create an LLMClient with mocked dependencies."""
	return LLMClient(config_loader=mock_config_loader)


@pytest.mark.unit
def test_llm_client_initialization(mock_config_loader):
	"""Test that the LLMClient initializes correctly with various parameters."""
	# Test with config_loader
	client = LLMClient(config_loader=mock_config_loader)
	assert client.config_loader == mock_config_loader
	assert client._templates == {}


@pytest.mark.unit
def test_template_management():
	"""Test template setting and retrieval."""
	# Create a config loader mock
	mock_config = Mock(spec=ConfigLoader)
	client = LLMClient(config_loader=mock_config)

	# Test setting and getting a template
	client.set_template("test", "This is a {test} template")
	assert client._templates["test"] == "This is a {test} template"

	# Test template not found
	assert "nonexistent" not in client._templates

	# Test default templates
	class CustomClient(LLMClient):
		DEFAULT_TEMPLATES: ClassVar[dict[str, str]] = {"default": "Default template {var}"}

	custom_client = CustomClient(config_loader=mock_config)
	assert custom_client._templates["default"] == "Default template {var}"


@pytest.mark.unit
def test_completion(llm_client):
	"""Test completion with LLM API."""
	messages: list[MessageDict] = [{"role": "user", "content": "Test prompt"}]

	with patch("codemap.llm.client.call_llm_api") as mock_call_api:
		mock_call_api.return_value = "Generated text response"

		result = llm_client.completion(messages=messages)

		# Verify API was called with correct parameters
		mock_call_api.assert_called_once()
		call_args = mock_call_api.call_args[1]
		assert call_args["messages"] == messages
		assert call_args["config_loader"] == llm_client.config_loader

		# Verify result
		assert result == "Generated text response"


@pytest.mark.unit
def test_completion_with_pydantic_model(llm_client):
	"""Test completion with Pydantic model validation."""

	# Define a simple Pydantic model for testing
	class TestResponse(BaseModel):
		answer: str
		confidence: float

	messages: list[MessageDict] = [{"role": "user", "content": "Test prompt"}]

	with patch("codemap.llm.client.call_llm_api") as mock_call_api:
		mock_output = TestResponse(answer="Generated response", confidence=0.9)
		mock_call_api.return_value = mock_output

		result = llm_client.completion(messages=messages, pydantic_model=TestResponse)

		# Verify API was called with correct parameters
		mock_call_api.assert_called_once()
		call_args = mock_call_api.call_args[1]
		assert call_args["messages"] == messages
		assert call_args["pydantic_model"] == TestResponse

		# Verify result is the Pydantic model
		assert isinstance(result, TestResponse)
		assert result.answer == "Generated response"
		assert result.confidence == 0.9


@pytest.mark.unit
def test_completion_error(mock_config_loader):
	"""Test error handling during completion."""
	client = LLMClient(config_loader=mock_config_loader)
	messages: list[MessageDict] = [{"role": "user", "content": "Test prompt"}]

	with (
		patch("codemap.llm.client.call_llm_api", side_effect=LLMError("API Error")),
		pytest.raises(LLMError, match="API Error"),
	):
		client.completion(messages=messages)
