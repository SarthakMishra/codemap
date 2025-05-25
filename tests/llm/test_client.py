"""Tests for the LLM client module."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from codemap.config import ConfigLoader
from codemap.llm.api import MessageDict
from codemap.llm.client import CompletionStatus, LLMClient
from codemap.llm.errors import LLMError


@pytest.fixture
def mock_config_loader() -> Mock:
	"""Fixture to create a mock ConfigLoader."""
	mock_config = Mock()
	# Mock the pydantic schema structure
	mock_config.get = Mock()
	mock_config.get.llm = Mock()
	mock_config.get.llm.model = "openai:gpt-4o-mini"
	mock_config.get.llm.temperature = 0.7
	mock_config.get.llm.max_output_tokens = 1000
	mock_config.get.llm.max_input_tokens = 10000
	mock_config.get.llm.max_requests = 5
	mock_config.get.llm.model_dump.return_value = {
		"temperature": 0.7,
		"max_output_tokens": 1000,
		"max_input_tokens": 10000,
		"max_requests": 5,
	}

	return mock_config


@pytest.fixture
def llm_client(mock_config_loader: Mock) -> LLMClient:
	"""Fixture to create an LLMClient with mocked dependencies."""
	return LLMClient(config_loader=mock_config_loader)


@pytest.mark.unit
def test_llm_client_initialization(mock_config_loader: Mock) -> None:
	"""Test that the LLMClient initializes correctly with various parameters."""
	# Test with config_loader
	client = LLMClient(config_loader=mock_config_loader)
	assert client.config_loader == mock_config_loader
	assert client._templates == {}


@pytest.mark.unit
def test_template_management() -> None:
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
def test_completion(llm_client: LLMClient) -> None:
	"""Test completion with LLM API."""
	messages: list[MessageDict] = [{"role": "user", "content": "Test prompt"}]

	with (
		patch("codemap.llm.client.call_llm_api") as mock_call_api,
		patch("codemap.llm.client.get_agent") as mock_get_agent,
	):
		mock_call_api.return_value = "Generated text response"
		mock_get_agent.return_value = Mock()  # Mock agent

		result = llm_client.completion(messages=messages)

		# Verify API was called with correct parameters
		mock_call_api.assert_called_once()
		call_args = mock_call_api.call_args[1]
		assert call_args["messages"] == messages
		assert call_args["config_loader"] == llm_client.config_loader

		# Verify result
		assert result == "Generated text response"


@pytest.mark.unit
def test_completion_with_pydantic_model(llm_client: LLMClient) -> None:
	"""Test completion with Pydantic model validation."""

	# Define a simple Pydantic model for testing
	class TestResponse(BaseModel):
		answer: str
		confidence: float

	messages: list[MessageDict] = [{"role": "user", "content": "Test prompt"}]

	with (
		patch("codemap.llm.client.call_llm_api") as mock_call_api,
		patch("codemap.llm.client.get_agent") as mock_get_agent,
	):
		mock_output = TestResponse(answer="Generated response", confidence=0.9)
		mock_call_api.return_value = mock_output
		mock_get_agent.return_value = Mock()  # Mock agent

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
def test_completion_error(mock_config_loader: Mock) -> None:
	"""Test error handling during completion."""
	client = LLMClient(config_loader=mock_config_loader)
	messages: list[MessageDict] = [{"role": "user", "content": "Test prompt"}]

	with (
		patch("codemap.llm.client.call_llm_api", side_effect=LLMError("API Error")),
		patch("codemap.llm.client.get_agent") as mock_get_agent,
	):
		mock_get_agent.return_value = Mock()  # Mock agent
		with pytest.raises(LLMError, match="API Error"):
			client.completion(messages=messages)


@pytest.mark.unit
def test_iterative_completion_first_iteration_complete(llm_client: LLMClient) -> None:
	"""Test iterative completion that completes on the first iteration."""
	question = "What is this code doing?"
	system_prompt = "You are a code analysis assistant."

	# Mock the completion method to return a response
	with (
		patch("codemap.llm.client.LLMClient.completion") as mock_completion,
		patch("codemap.llm.client.LLMClient.check_completion") as mock_check_completion,
	):
		mock_completion.return_value = "This code implements a sorting algorithm."
		mock_check_completion.return_value = CompletionStatus(
			is_complete=True, final_response="This code implements a bubble sort algorithm with O(n²) complexity."
		)

		result = llm_client.iterative_completion(question=question, system_prompt=system_prompt)

		# Should call completion once
		assert mock_completion.call_count == 1

		# Should call check_completion once
		assert mock_check_completion.call_count == 1

		# Should return the final response from completion status
		assert result == "This code implements a bubble sort algorithm with O(n²) complexity."

		# Verify the completion call had the expected content (the exact message count may vary due to implementation details)
		first_completion_call = mock_completion.call_args_list[0][1]
		messages = first_completion_call["messages"]
		# The implementation adds the assistant response to conversation_messages before calling check_completion
		# So we see system + user + assistant messages in the mock
		assert any(msg["role"] == "system" and system_prompt in msg["content"] for msg in messages)
		assert any(msg["role"] == "user" and question in msg["content"] for msg in messages)


@pytest.mark.unit
def test_iterative_completion_multiple_iterations(llm_client: LLMClient) -> None:
	"""Test iterative completion that requires multiple iterations."""
	question = "Analyze this complex codebase"
	system_prompt = "You are a comprehensive code analyzer."

	with (
		patch("codemap.llm.client.LLMClient.completion") as mock_completion,
		patch("codemap.llm.client.LLMClient.check_completion") as mock_check_completion,
	):
		# First iteration - incomplete
		# Second iteration - complete
		mock_completion.side_effect = [
			"Initial analysis of the codebase structure...",
			"Detailed analysis with security recommendations...",
		]

		mock_check_completion.side_effect = [
			CompletionStatus(is_complete=False, suggestion="Please provide more details about security aspects"),
			CompletionStatus(
				is_complete=True,
				final_response="Complete analysis: The codebase has good structure but needs security improvements.",
			),
		]

		result = llm_client.iterative_completion(question=question, system_prompt=system_prompt, max_iterations=3)

		# Should call completion twice
		assert mock_completion.call_count == 2

		# Should call check_completion twice
		assert mock_check_completion.call_count == 2

		# Should return the final response
		assert result == "Complete analysis: The codebase has good structure but needs security improvements."


@pytest.mark.unit
def test_iterative_completion_max_iterations_reached(llm_client: LLMClient) -> None:
	"""Test iterative completion that reaches max iterations without completion."""
	question = "Complex analysis task"
	system_prompt = "Analyze everything in detail."

	with (
		patch("codemap.llm.client.LLMClient.completion") as mock_completion,
		patch("codemap.llm.client.LLMClient.check_completion") as mock_check_completion,
	):
		# All iterations return incomplete
		mock_completion.side_effect = ["Partial analysis 1...", "Partial analysis 2...", "Final partial analysis..."]

		# All completion checks return incomplete, including the final forced one
		mock_check_completion.side_effect = [
			CompletionStatus(is_complete=False, suggestion="Need more data"),
			CompletionStatus(is_complete=False, suggestion="Still need more"),
			CompletionStatus(is_complete=False, suggestion="Still incomplete"),
		]

		result = llm_client.iterative_completion(question=question, system_prompt=system_prompt, max_iterations=3)

		# Should call completion 3 times (max iterations)
		assert mock_completion.call_count == 3

		# Should call check_completion 3 times
		assert mock_check_completion.call_count == 3

		# Should return the last completion response when max iterations reached
		assert result == "Final partial analysis..."

		# Verify the last check_completion call has is_last_iteration=True
		last_check_call = mock_check_completion.call_args_list[-1]
		assert last_check_call[1]["is_last_iteration"] is True


@pytest.mark.unit
def test_iterative_completion_with_tools(llm_client: LLMClient) -> None:
	"""Test iterative completion with tools provided."""
	question = "Search and analyze code patterns"
	system_prompt = "Use tools to analyze the codebase."
	mock_tools: list[Any] = [Mock(), Mock()]

	with (
		patch("codemap.llm.client.LLMClient.completion") as mock_completion,
		patch("codemap.llm.client.LLMClient.check_completion") as mock_check_completion,
	):
		mock_completion.return_value = "Analysis complete using provided tools."
		mock_check_completion.return_value = CompletionStatus(
			is_complete=True, final_response="Tool-based analysis shows good code patterns."
		)

		result = llm_client.iterative_completion(question=question, system_prompt=system_prompt, tools=mock_tools)  # type: ignore[arg-type]

		# Verify tools were passed to completion
		call_args = mock_completion.call_args[1]
		assert call_args["tools"] == mock_tools

		assert result == "Tool-based analysis shows good code patterns."


@pytest.mark.unit
def test_iterative_completion_no_final_response_in_status(llm_client: LLMClient) -> None:
	"""Test iterative completion when completion status has no final response."""
	question = "Simple question"
	system_prompt = "Simple analysis."

	with (
		patch("codemap.llm.client.LLMClient.completion") as mock_completion,
		patch("codemap.llm.client.LLMClient.check_completion") as mock_check_completion,
	):
		mock_completion.return_value = "Simple analysis result."
		mock_check_completion.return_value = CompletionStatus(
			is_complete=True,
			# No final_response provided
		)

		result = llm_client.iterative_completion(question=question, system_prompt=system_prompt)

		# Should fallback to the current response when no final_response in status
		assert result == "Simple analysis result."


@pytest.mark.unit
def test_iterative_completion_conversation_history(llm_client: LLMClient) -> None:
	"""Test that conversation history is maintained correctly across iterations."""
	question = "Multi-step analysis"
	system_prompt = "Analyze step by step."

	with (
		patch("codemap.llm.client.LLMClient.completion") as mock_completion,
		patch("codemap.llm.client.LLMClient.check_completion") as mock_check_completion,
	):
		mock_completion.side_effect = ["Step 1 analysis...", "Step 2 analysis..."]

		mock_check_completion.side_effect = [
			CompletionStatus(is_complete=False, suggestion="Continue with step 2"),
			CompletionStatus(is_complete=True, final_response="Complete multi-step analysis"),
		]

		llm_client.iterative_completion(question=question, system_prompt=system_prompt)

		# Check the second completion call to verify conversation history
		second_call_args = mock_completion.call_args_list[1][1]
		messages = second_call_args["messages"]

		# Should have: system, user, assistant (from first), user (continuation)
		assert len(messages) == 4
		assert messages[0]["role"] == "system"
		assert messages[1]["role"] == "user"  # Original question
		assert messages[2]["role"] == "assistant"  # First response
		assert messages[2]["content"] == "Step 1 analysis..."
		assert messages[3]["role"] == "user"  # Continuation prompt
		assert "continue" in messages[3]["content"].lower()


@pytest.mark.unit
def test_iterative_completion_empty_messages_fallback(llm_client: LLMClient) -> None:
	"""Test fallback behavior when no messages are available."""
	question = "Test question"
	system_prompt = "Test prompt"

	with (
		patch("codemap.llm.client.LLMClient.completion") as mock_completion,
		patch("codemap.llm.client.LLMClient.check_completion") as mock_check_completion,
	):
		# Mock completion to return None/empty somehow (edge case)
		mock_completion.return_value = ""
		mock_check_completion.return_value = CompletionStatus(is_complete=False)

		result = llm_client.iterative_completion(question=question, system_prompt=system_prompt, max_iterations=1)

		# Should return the actual fallback message used in the implementation
		assert result == "Unable to generate a complete response within iteration limits."
