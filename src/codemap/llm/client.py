"""LLM client for unified access to language models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from .api import call_llm_api
from .config import DEFAULT_MODEL, LLMConfig
from .errors import LLMError

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


class LLMClient:
	"""Client for interacting with LLM services in a unified way."""

	# Default templates - empty in base class
	DEFAULT_TEMPLATES: ClassVar[dict[str, str]] = {}

	def __init__(
		self,
		config: LLMConfig | None = None,
		repo_path: Path | None = None,
	) -> None:
		"""
		Initialize the LLM client.

		Args:
		    config: LLM configuration
		    repo_path: Repository path (for loading configuration)

		"""
		self.config = config or LLMConfig()
		self.repo_path = repo_path
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

	def generate_text(
		self,
		prompt: str,
		model: str | None = None,
		json_schema: dict | None = None,
		**kwargs: dict[str, str | int | float | bool | None],
	) -> str:
		"""
		Generate text using the configured LLM.

		Args:
		    prompt: Prompt to send to the LLM
		    model: Optional model override
		    json_schema: Optional JSON schema for response validation
		    **kwargs: Additional parameters to pass to the LLM API

		Returns:
		    Generated text

		Raises:
		    LLMError: If the API call fails

		"""
		# Get API configuration
		model_to_use = model or self.config.model or DEFAULT_MODEL
		api_key = self.config.get_api_key()

		if not api_key:
			msg = f"No API key available for {self.config.provider or 'default'} provider"
			raise LLMError(msg)

		# Call the API
		return call_llm_api(
			prompt=prompt,
			model=model_to_use,
			api_key=api_key,
			api_base=self.config.api_base,
			json_schema=json_schema,
			**kwargs,
		)

	def generate_from_template(
		self,
		template_name: str,
		template_vars: dict[str, Any],
		model: str | None = None,
		json_schema: dict | None = None,
		**kwargs: dict[str, str | int | float | bool | None],
	) -> str:
		"""
		Generate text using a named template.

		Args:
		    template_name: Name of the template to use
		    template_vars: Variables to format the template with
		    model: Optional model override
		    json_schema: Optional JSON schema for response validation
		    **kwargs: Additional parameters to pass to the LLM API

		Returns:
		    Generated text

		Raises:
		    LLMError: If the API call fails
		    ValueError: If the template doesn't exist

		"""
		# Get and format the template
		template = self.get_template(template_name)
		prompt = template.format(**template_vars)

		# Generate text
		return self.generate_text(
			prompt=prompt,
			model=model,
			json_schema=json_schema,
			**kwargs,
		)
