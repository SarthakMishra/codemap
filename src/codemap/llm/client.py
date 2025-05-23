"""LLM client for unified access to language models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codemap.config import ConfigLoader

from .api import MessageDict, PydanticModelT, call_llm_api

if TYPE_CHECKING:
	from pathlib import Path

	from pydantic_ai.tools import Tool

logger = logging.getLogger(__name__)


class LLMClient:
	"""Client for interacting with LLM services in a unified way."""

	# Default templates - empty in base class
	DEFAULT_TEMPLATES: ClassVar[dict[str, str]] = {}

	def __init__(
		self,
		config_loader: ConfigLoader,
		repo_path: Path | None = None,
	) -> None:
		"""
		Initialize the LLM client.

		Args:
		    config_loader: ConfigLoader instance to use
		    repo_path: Path to the repository (for loading configuration)
		"""
		self.repo_path = repo_path
		self.config_loader = config_loader
		self._templates = self.DEFAULT_TEMPLATES.copy()

	def set_template(self, name: str, template: str) -> None:
		"""
		Set a prompt template.

		Args:
		    name: Template name
		    template: Template content

		"""
		self._templates[name] = template

	def completion(
		self,
		messages: list[MessageDict],
		tools: list[Tool] | None = None,
		pydantic_model: type[PydanticModelT] | None = None,
	) -> str | PydanticModelT:
		"""
		Generate text using the configured LLM.

		Args:
		    messages: List of messages to send to the LLM
		    tools: Optional list of tools to use.
		    pydantic_model: Optional Pydantic model for response validation

		Returns:
		    Generated text or Pydantic model instance

		Raises:
		    LLMError: If the API call fails

		"""
		# Call the API
		return call_llm_api(
			messages=messages,
			tools=tools,
			pydantic_model=pydantic_model,
			config_loader=self.config_loader,
		)
