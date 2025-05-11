"""Utility functions for working with LLMs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_prompt_template(template_path: str | None) -> str | None:
	"""
	Load custom prompt template from file.

	Args:
	    template_path: Path to prompt template file

	Returns:
	    Loaded template or None if loading failed

	"""
	if not template_path:
		return None

	try:
		template_file = Path(template_path)
		with template_file.open("r") as f:
			return f.read()
	except OSError:
		logger.warning("Could not load prompt template: %s", template_path)
		return None


# Define a type for the response that covers all expected formats
LLMResponseType = dict[str, Any] | Mapping[str, Any] | object


def is_ollama_model(model_name: str) -> bool:
	"""Check if the model name is an Ollama model."""
	return model_name.startswith("ollama:")
