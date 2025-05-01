"""Tests for LLM utility functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from codemap.llm.client import LLMClient
from codemap.llm.config import LLMConfig
from codemap.llm.errors import LLMError
from codemap.llm.utils import (
	create_client,
	extract_content_from_response,
	generate_text,
	get_llm_client,
	load_prompt_template,
)
from tests.base import FileSystemTestBase, LLMTestBase


@pytest.mark.unit
@pytest.mark.fs
class TestLoadPromptTemplate(FileSystemTestBase):
	"""Test cases for load_prompt_template."""

	def test_load_prompt_template_success(self) -> None:
		"""Test loading a prompt template successfully."""
		template_content = "This is a test template."
		template_path = self.create_test_file("my_template.txt", template_content)

		result = load_prompt_template(str(template_path))
		assert result == template_content

	def test_load_prompt_template_none_path(self) -> None:
		"""Test loading with None path returns None."""
		result = load_prompt_template(None)
		assert result is None

	def test_load_prompt_template_not_found(self) -> None:
		"""Test loading a non-existent template returns None and logs warning."""
		non_existent_path = self.temp_dir / "not_found.txt"
		with patch("codemap.llm.utils.logger.warning") as mock_warning:
			result = load_prompt_template(str(non_existent_path))
			assert result is None
			mock_warning.assert_called_once_with("Could not load prompt template: %s", str(non_existent_path))

	@patch("pathlib.Path.open", side_effect=OSError("Permission denied"))
	def test_load_prompt_template_os_error(self, _mock_open: MagicMock) -> None:
		"""Test loading with OSError returns None and logs warning."""
		# We don't need to create the file as open is mocked
		template_path_str = "some/template.txt"
		with patch("codemap.llm.utils.logger.warning") as mock_warning:
			result = load_prompt_template(template_path_str)
			assert result is None
			mock_warning.assert_called_once_with("Could not load prompt template: %s", template_path_str)


@pytest.mark.unit
class TestLLMClientCreation(LLMTestBase):
	"""Test cases for LLM client creation functions."""

	@patch("codemap.llm.utils.LLMClient")
	@patch("codemap.llm.utils.get_llm_config")
	def test_get_llm_client_success(self, mock_get_config: MagicMock, mock_client_class: MagicMock) -> None:
		"""Test get_llm_client success case."""
		mock_config = LLMConfig(model="test-model")
		mock_get_config.return_value = mock_config
		mock_client_instance = MagicMock(spec=LLMClient)
		mock_client_class.return_value = mock_client_instance

		client = get_llm_client()

		mock_get_config.assert_called_once_with()
		mock_client_class.assert_called_once_with(config=mock_config)
		assert client == mock_client_instance

	@patch("codemap.llm.utils.get_llm_config", side_effect=LLMError("Config error"))
	def test_get_llm_client_llm_error(self, _mock_get_config: MagicMock) -> None:
		"""Test get_llm_client raises RuntimeError on LLMError."""
		with pytest.raises(RuntimeError, match="Failed to create LLM client: Config error"):
			get_llm_client()

	@patch("codemap.llm.utils.LLMClient")
	@patch("codemap.llm.utils.get_llm_config")
	def test_create_client_success_no_overrides(self, mock_get_config: MagicMock, mock_client_class: MagicMock) -> None:
		"""Test create_client success with no overrides."""
		mock_config = LLMConfig(model="default-model")
		mock_get_config.return_value = mock_config
		mock_client_instance = MagicMock(spec=LLMClient)
		mock_client_class.return_value = mock_client_instance
		repo_path = Path("/fake/repo")

		client = create_client(repo_path=repo_path)

		mock_get_config.assert_called_once_with(overrides={"model": None, "api_key": None, "api_base": None})
		mock_client_class.assert_called_once_with(config=mock_config, repo_path=repo_path)
		assert client == mock_client_instance

	@patch("codemap.llm.utils.LLMClient")
	@patch("codemap.llm.utils.get_llm_config")
	def test_create_client_success_with_overrides(
		self, mock_get_config: MagicMock, mock_client_class: MagicMock
	) -> None:
		"""Test create_client success with overrides."""
		mock_config = LLMConfig(model="override-model", api_key="override-key")
		mock_get_config.return_value = mock_config
		mock_client_instance = MagicMock(spec=LLMClient)
		mock_client_class.return_value = mock_client_instance
		repo_path = Path("/fake/repo")

		client = create_client(
			repo_path=repo_path,
			model="override-model",
			api_key="override-key",
			api_base="override-base",
		)

		mock_get_config.assert_called_once_with(
			overrides={
				"model": "override-model",
				"api_key": "override-key",
				"api_base": "override-base",
			}
		)
		mock_client_class.assert_called_once_with(config=mock_config, repo_path=repo_path)
		assert client == mock_client_instance

	@patch("codemap.llm.utils.get_llm_config", side_effect=Exception("Unexpected error"))
	def test_create_client_exception(self, _mock_get_config: MagicMock) -> None:
		"""Test create_client raises RuntimeError on generic exception."""
		with pytest.raises(RuntimeError, match="Failed to create LLM client: Unexpected error"):
			create_client(model="test")

	@patch("codemap.llm.utils.create_client")
	def test_generate_text_success(self, mock_create_client: MagicMock) -> None:
		"""Test generate_text successfully generates text."""
		mock_client = MagicMock(spec=LLMClient)
		mock_create_client.return_value = mock_client
		mock_client.generate_text.return_value = "Generated response"

		prompt = "Write a story."
		model = "gpt-test"
		api_key = "sk-key"
		api_base = "https://api.test.com"
		kwargs = {"temperature": 0.5}

		result = generate_text(prompt=prompt, model=model, api_key=api_key, api_base=api_base, **kwargs)

		mock_create_client.assert_called_once_with(model=model, api_key=api_key, api_base=api_base)
		mock_client.generate_text.assert_called_once_with(prompt=prompt, temperature=0.5)
		assert result == "Generated response"

	@patch("codemap.llm.utils.create_client")
	def test_generate_text_llm_error(self, mock_create_client: MagicMock) -> None:
		"""Test generate_text raises RuntimeError on LLMError from client."""
		mock_client = MagicMock(spec=LLMClient)
		mock_create_client.return_value = mock_client
		mock_client.generate_text.side_effect = LLMError("Generation failed")

		with pytest.raises(RuntimeError, match="Failed to generate text with LLM: Generation failed"):
			generate_text(prompt="Test prompt")

	# This test relies on the actual implementation in api.py
	# Could also mock _extract_content if needed
	def test_extract_content_from_response(self) -> None:
		"""Test extract_content_from_response wrapper."""
		# Example response structures (adapt based on actual expected responses)
		response_dict = {"choices": [{"message": {"content": "Hello from dict"}}]}
		response_obj = Mock()
		response_obj.choices = [Mock()]
		response_obj.choices[0].message.content = "Hello from obj"

		assert extract_content_from_response(response_dict) == "Hello from dict"
		assert extract_content_from_response(response_obj) == "Hello from obj"
