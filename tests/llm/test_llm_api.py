"""Tests for the LLM API module functionality."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from codemap.llm.api import call_llm_api, extract_content_from_response
from codemap.llm.errors import LLMError
from tests.base import LLMTestBase


@pytest.mark.unit
@pytest.mark.llm
class TestLLMAPI(LLMTestBase):
	"""Test cases for LLM API functionality."""

	def test_extract_content_from_response_openai_format(self) -> None:
		"""Test content extraction from OpenAI-like response format."""
		# Create a mock response in OpenAI format
		mock_response = MagicMock()
		mock_response.choices = [MagicMock(message=MagicMock(content="This is the extracted content"))]

		# Act: Extract content
		content = extract_content_from_response(mock_response)

		# Assert: Verify content was extracted correctly
		assert content == "This is the extracted content"

	def test_extract_content_from_response_dict_format(self) -> None:
		"""Test content extraction from dictionary response format."""
		# Create a mock response as dictionary
		mock_response = {"choices": [{"message": {"content": "This is from dictionary format"}}]}

		# Act: Extract content
		content = extract_content_from_response(mock_response)

		# Assert: Verify content was extracted correctly
		assert content == "This is from dictionary format"

	def test_extract_content_from_response_text_format(self) -> None:
		"""Test content extraction from text attribute format."""
		# Create a mock response with text attribute
		mock_response = MagicMock()
		mock_response.text = "This is from text attribute"

		# Act: Extract content
		content = extract_content_from_response(mock_response)

		# Assert: Verify content was extracted correctly
		assert content == "This is from text attribute"

	def test_extract_content_from_response_empty(self) -> None:
		"""Test content extraction from empty or invalid response."""
		# Test with None
		assert extract_content_from_response(None) == ""

		# Test with empty dict
		assert extract_content_from_response({}) == ""

		# Test with invalid format
		mock_response = MagicMock()
		mock_response.choices = []
		# Make sure mock doesn't have a text attribute
		del mock_response.text
		assert extract_content_from_response(mock_response) == ""

	def test_call_llm_api_success(self) -> None:
		"""Test successful LLM API call."""
		# Arrange: Set up mock response
		mock_response = MagicMock()
		mock_response.choices = [MagicMock(message=MagicMock(content="LLM generated content"))]

		# Create a mock litellm module
		mock_litellm = MagicMock()
		mock_litellm.completion.return_value = mock_response

		# Act: Call the API with additional parameters
		with patch.dict(sys.modules, {"litellm": mock_litellm}):
			# Note: We can't actually modify the type annotation in the original function,
			# but for testing purposes we'll send the correct params and ignore the linter
			result = call_llm_api(
				prompt="Test prompt",
				model="gpt-3.5-turbo",
				api_key="test-api-key",
				api_base="https://api.example.com",
				temperature=0.7,  # type: ignore[arg-type]
				max_tokens=500,  # type: ignore[arg-type]
				stream=False,  # type: ignore[arg-type]
			)

		# Assert: Verify results
		assert result == "LLM generated content"
		mock_litellm.completion.assert_called_once()
		# Check that parameters were passed correctly
		call_args = mock_litellm.completion.call_args[1]
		assert call_args["model"] == "gpt-3.5-turbo"
		assert call_args["api_key"] == "test-api-key"
		assert call_args["api_base"] == "https://api.example.com"
		assert call_args["temperature"] == 0.7
		assert call_args["max_tokens"] == 500
		assert call_args["stream"] is False
		assert call_args["messages"][0]["content"] == "Test prompt"

	def test_call_llm_api_with_json_schema(self) -> None:
		"""Test LLM API call with JSON schema."""
		# Arrange: Set up mock response
		mock_response = MagicMock()
		mock_response.choices = [MagicMock(message=MagicMock(content='{"result": "success"}'))]

		# Create a mock litellm module
		mock_litellm = MagicMock()
		mock_litellm.completion.return_value = mock_response

		# Create a test JSON schema
		json_schema = {"type": "object", "properties": {"result": {"type": "string"}}}

		# Act: Call the API with JSON schema
		with patch.dict(sys.modules, {"litellm": mock_litellm}):
			result = call_llm_api(
				prompt="Generate JSON",
				model="gpt-3.5-turbo",
				api_key="test-api-key",
				json_schema=json_schema,
				temperature=0.2,  # type: ignore[arg-type]
			)

		# Assert: Verify results
		assert result == '{"result": "success"}'
		# Verify that JSON schema was included in request
		call_args = mock_litellm.completion.call_args[1]
		assert call_args["response_format"]["type"] == "json_object"
		assert call_args["response_format"]["schema"] == json_schema
		# Verify the temperature was passed correctly
		assert call_args["temperature"] == 0.2
		# Check that enable_json_schema_validation was set
		assert mock_litellm.enable_json_schema_validation is True

	def test_call_llm_api_empty_response(self) -> None:
		"""Test handling of empty responses from LLM API."""
		# Arrange: Set up empty mock response
		mock_response = MagicMock()
		mock_response.choices = []
		# Make sure the text attribute is not present, otherwise it won't be empty
		type(mock_response).text = None

		# Create a mock litellm module
		mock_litellm = MagicMock()
		mock_litellm.completion.return_value = mock_response

		# Act & Assert: Call should raise LLMError
		with (
			patch.dict(sys.modules, {"litellm": mock_litellm}),
			pytest.raises(LLMError, match="Failed to extract content from LLM response"),
		):
			call_llm_api(prompt="Test prompt", model="gpt-3.5-turbo", api_key="test-api-key")

	def test_call_llm_api_exception(self) -> None:
		"""Test handling of exceptions during LLM API call."""
		# Create a mock litellm module
		mock_litellm = MagicMock()
		# Configure mock to raise exception
		mock_litellm.completion.side_effect = Exception("API call failed")

		# Act & Assert: Verify exception is caught and wrapped
		with (
			patch.dict(sys.modules, {"litellm": mock_litellm}),
			pytest.raises(LLMError, match="LLM API call failed: API call failed"),
		):
			call_llm_api(prompt="Test prompt", model="gpt-3.5-turbo", api_key="test-api-key")

	def test_call_llm_api_import_error(self) -> None:
		"""Test handling of missing litellm library."""
		# Arrange: Simulate import error
		with patch.dict(sys.modules, {"litellm": None}), pytest.raises(LLMError, match="LiteLLM library not installed"):
			# Act & Assert: Verify proper error is raised
			call_llm_api(prompt="Test prompt", model="gpt-3.5-turbo", api_key="test-api-key")
