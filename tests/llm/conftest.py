"""Shared fixtures for LLM tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from codemap.llm.client import LLMClient
from codemap.llm.config import LLMConfig
from codemap.utils.config_loader import ConfigLoader


@pytest.fixture
def mock_config_loader():
	"""Create a mock ConfigLoader for tests."""
	mock_loader = Mock(spec=ConfigLoader)
	mock_loader.get_llm_config.return_value = {
		"model": "openai/gpt-4",
		"provider": "openai",
		"temperature": 0.7,
		"max_tokens": 1000,
	}
	return mock_loader


@pytest.fixture
def mock_llm_client():
	"""Create a mock LLMClient for tests."""
	client = Mock(spec=LLMClient)
	client.generate_text.return_value = "Generated text response"
	return client


@pytest.fixture
def llm_config():
	"""Create a test LLMConfig."""
	return LLMConfig(
		model="openai/gpt-4",
		provider="openai",
		api_key="test-key",
	)


@pytest.fixture
def llm_client(llm_config, mock_config_loader):
	"""Create a real LLMClient with mocked dependencies."""
	return LLMClient(
		config=llm_config,
		config_loader=mock_config_loader,
	)
