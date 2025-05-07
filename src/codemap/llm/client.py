"""LLM client for unified access to language models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codemap.config import ConfigLoader

from .api import M, MessageDict, call_llm_api

if TYPE_CHECKING:
	from pathlib import Path

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

	def get_template(self, name: str) -> str:
		"""
		Get a prompt template.

		Args:
		    name: Template name

		Returns:
		    Template content

		Raises:
		    ValueError: If template doesn't exist

		"""
		if name not in self._templates:
			msg = f"Template '{name}' not found"
			raise ValueError(msg)
		return self._templates[name]

	async def completion(
		self,
		messages: list[MessageDict],
		model: str | None = None,
		pydantic_model: type[M] | None = None,
		**kwargs: dict[str, str | int | float | bool | None],
	) -> str | M:
		"""
		Generate text using the configured LLM.

		Args:
		    messages: List of messages to send to the LLM
		    model: Optional model override
		    pydantic_model: Optional Pydantic model for response validation
		    **kwargs: Additional parameters to pass to the LLM API

		Returns:
		    Generated text

		Raises:
		    LLMError: If the API call fails

		"""
		# Get API configuration
		model_to_use = model or self.config_loader.get.llm.model

		# Call the API
		return await call_llm_api(
			messages=messages,
			model=model_to_use,
			output_schema=pydantic_model,
			config_loader=self.config_loader,
			**kwargs,
		)
