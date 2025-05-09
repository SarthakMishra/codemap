"""Tests for the LLM API module."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from codemap.config import ConfigLoader
from codemap.llm.api import MessageDict, call_llm_api
from codemap.llm.errors import LLMError


# Renamed to avoid pytest collecting it as a test class
class ModelForValidation(BaseModel):
	"""Model for Pydantic validation in tests."""

	answer: str
	confidence: float


@pytest.fixture
def mock_config_loader():
	"""Fixture to create a mock ConfigLoader."""
	mock_loader = Mock(spec=ConfigLoader)

	# Mock the get property to return a mock object with llm attribute
	mock_get = Mock()
	mock_llm = Mock()

	# Set up the llm config properties
	mock_llm.model = "openai:gpt-4o-mini"
	mock_llm.temperature = 0.7
	mock_llm.max_output_tokens = 1000

	# Attach the llm config to the get property
	mock_get.llm = mock_llm
	mock_loader.get = mock_get

	return mock_loader


@pytest.mark.unit
def test_call_llm_api_pydantic_ai_not_installed():
	"""Test handling of missing pydantic-ai dependency."""
	with (
		patch("codemap.llm.api.Agent", None),
		pytest.raises(LLMError, match="Pydantic-AI library or its required types .* not installed"),
	):
		call_llm_api(
			messages=[{"role": "user", "content": "Test prompt"}],
			config_loader=Mock(spec=ConfigLoader),
		)


@pytest.mark.unit
def test_call_llm_api_success(mock_config_loader):
	"""Test successful API call."""
	# Create a mock run result
	mock_run = MagicMock()
	mock_run.output = "Generated content"

	# Create a mock agent
	mock_agent = MagicMock()
	mock_agent.run_sync.return_value = mock_run

	with (
		patch("codemap.llm.api.Agent", return_value=mock_agent),
		patch("codemap.llm.api.End"),
		patch("codemap.llm.api.FinalResult"),
		patch("codemap.llm.api.ModelSettings"),
	):
		result = call_llm_api(
			messages=[{"role": "user", "content": "Test prompt"}],
			config_loader=mock_config_loader,
		)

		# Verify agent was created correctly
		mock_agent.run_sync.assert_called_once()

		# Verify result
		assert result == "Generated content"


@pytest.mark.unit
def test_call_llm_api_with_system_prompt(mock_config_loader):
	"""Test API call with a system prompt."""
	# Create a mock run result
	mock_run = MagicMock()
	mock_run.output = "Generated with system prompt"

	# Create a mock agent
	mock_agent = MagicMock()
	mock_agent.run_sync.return_value = mock_run

	with (
		patch("codemap.llm.api.Agent", return_value=mock_agent),
		patch("codemap.llm.api.End"),
		patch("codemap.llm.api.FinalResult"),
		patch("codemap.llm.api.ModelSettings"),
	):
		messages: list[MessageDict] = [
			{"role": "system", "content": "Custom system prompt"},
			{"role": "user", "content": "Test prompt"},
		]

		result = call_llm_api(
			messages=messages,
			config_loader=mock_config_loader,
		)

		# Verify the run_sync method was called
		mock_agent.run_sync.assert_called_once()

		# Verify result matches expected output
		assert result == "Generated with system prompt"


@pytest.mark.unit
def test_call_llm_api_with_pydantic_model(mock_config_loader):
	"""Test API call with Pydantic model for structured output."""
	# Create a mock structured output
	test_data = {"answer": "Yes", "confidence": 0.9}

	# Create a mock run result
	mock_run = MagicMock()
	mock_run.output = test_data

	# Create a mock agent
	mock_agent = MagicMock()
	mock_agent.run_sync.return_value = mock_run

	with (
		patch("codemap.llm.api.Agent", return_value=mock_agent),
		patch("codemap.llm.api.End"),
		patch("codemap.llm.api.FinalResult"),
		patch("codemap.llm.api.ModelSettings"),
		patch("codemap.llm.api.validate_schema", return_value=ModelForValidation(**test_data)),
	):
		result = call_llm_api(
			messages=[{"role": "user", "content": "Test prompt"}],
			config_loader=mock_config_loader,
			pydantic_model=ModelForValidation,
		)

		# Verify the run_sync method was called
		mock_agent.run_sync.assert_called_once()

		# Verify result
		assert isinstance(result, ModelForValidation)
		assert result.answer == "Yes"
		assert result.confidence == 0.9


@pytest.mark.unit
def test_call_llm_api_no_user_content(mock_config_loader):
	"""Test handling when no user content is provided."""
	with (
		patch("codemap.llm.api.Agent"),
		patch("codemap.llm.api.End"),
		patch("codemap.llm.api.FinalResult"),
		patch("codemap.llm.api.ModelSettings"),
		pytest.raises(LLMError, match="No user content found in messages"),
	):
		call_llm_api(
			messages=[{"role": "system", "content": "System message only"}],
			config_loader=mock_config_loader,
		)


@pytest.mark.unit
def test_call_llm_api_last_message_not_user(mock_config_loader):
	"""Test handling when the last message is not from the user."""
	with (
		patch("codemap.llm.api.Agent"),
		patch("codemap.llm.api.End"),
		patch("codemap.llm.api.FinalResult"),
		patch("codemap.llm.api.ModelSettings"),
		pytest.raises(LLMError, match="Last message is not an user prompt"),
	):
		call_llm_api(
			messages=[
				{"role": "user", "content": "User message"},
				{"role": "system", "content": "System message at the end"},
			],
			config_loader=mock_config_loader,
		)


@pytest.mark.unit
def test_call_llm_api_empty_response(mock_config_loader):
	"""Test handling of empty response."""
	# Create a mock run result with None output
	mock_run = MagicMock()
	mock_run.output = None

	# Create a mock agent
	mock_agent = MagicMock()
	mock_agent.run_sync.return_value = mock_run

	with (
		patch("codemap.llm.api.Agent", return_value=mock_agent),
		patch("codemap.llm.api.End"),
		patch("codemap.llm.api.FinalResult"),
		patch("codemap.llm.api.ModelSettings"),
		pytest.raises(LLMError, match="Pydantic-AI call succeeded but returned no structured data or text"),
	):
		call_llm_api(
			messages=[{"role": "user", "content": "Test prompt"}],
			config_loader=mock_config_loader,
		)


@pytest.mark.unit
def test_call_llm_api_error(mock_config_loader):
	"""Test handling of API call errors."""
	# Create a mock agent that raises an exception
	mock_agent = MagicMock()
	mock_agent.run_sync.side_effect = Exception("API error")

	with (
		patch("codemap.llm.api.Agent", return_value=mock_agent),
		patch("codemap.llm.api.End"),
		patch("codemap.llm.api.FinalResult"),
		patch("codemap.llm.api.ModelSettings"),
		pytest.raises(LLMError, match="Pydantic-AI LLM API call failed: API error"),
	):
		call_llm_api(
			messages=[{"role": "user", "content": "Test prompt"}],
			config_loader=mock_config_loader,
		)
