"""API interaction for LLM services."""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict, TypeVar

from pydantic import BaseModel

from codemap.config import ConfigLoader

# Import Pydantic-AI
try:
	from pydantic_ai import Agent
	from pydantic_ai.result import FinalResult
	from pydantic_ai.settings import ModelSettings
	from pydantic_graph import End
except ImportError:
	Agent = None
	FinalResult = None
	End = None
	ModelSettings = None

from .errors import LLMError

logger = logging.getLogger(__name__)

# ResponseType alias is no longer strictly needed for pydantic-ai's RunResult but kept if other parts use it.
ResponseType = dict[str, Any] | Any


class MessageDict(TypedDict):
	"""Typed dictionary for LLM message structure."""

	role: Literal["user", "system"]
	content: str


# TypeVar for dynamic return type based on output_model
M = TypeVar("M", bound=BaseModel)


async def call_llm_api(
	model: str,
	messages: list[MessageDict],
	config_loader: ConfigLoader,
	output_schema: type[M] | None = None,
	**kwargs: dict[str, str | int | float | bool | None],
) -> str | M:
	"""
	Call an LLM API using pydantic-ai.

	Args:
	    model: The model identifier (e.g., "openai:gpt-4o")
	    messages: The list of messages to send to the LLM
	    config_loader: ConfigLoader instance for additional configuration
	    output_schema: Optional Pydantic model to structure the output.
	                  If provided, the function will return an instance of this model.
	                  Otherwise, it returns a string.
	    **kwargs: Additional parameters (e.g., temperature, max_tokens) to pass to the LLM.

	Returns:
	    The generated response, either as a string or an instance of the output_model.

	Raises:
	    LLMError: If pydantic-ai is not installed or the API call fails.
	"""
	if Agent is None or End is None or FinalResult is None:  # Check all imports
		msg = "Pydantic-AI library or its required types (AgentNode, End, FinalResult) not installed/found."
		logger.exception(msg)
		raise LLMError(msg) from None

	# Determine system prompt
	system_prompt_str = (
		"You are an AI programming assistant. Follow the user's requirements carefully and to the letter."
	)

	for msg in messages:
		if msg["role"] == "system":
			system_prompt_str = msg["content"]
			break

	# If an output_model is specified, pydantic-ai handles instructing the LLM for structured output.
	# So, no need to manually add schema instructions to the system_prompt_str here.

	# Determine the result type for the Pydantic-AI Agent
	current_output_type: type = str
	if output_schema:
		current_output_type = output_schema

	try:
		# Initialize Pydantic-AI Agent
		agent = Agent(
			model=model,
			system_prompt=system_prompt_str,
			output_type=current_output_type,
		)

		run_settings = {
			"temperature": config_loader.get.llm.temperature,
			"max_tokens": config_loader.get.llm.max_output_tokens,
		}
		run_settings.update(kwargs)

		logger.debug(
			"Calling Pydantic-AI Agent with model: %s, system_prompt: '%s...', output_type: %s, params: %s",
			model,
			system_prompt_str[:100],
			current_output_type.__name__,
			run_settings,
		)

		if not any(msg.get("role") == "user" for msg in messages):
			msg = "No user content found in messages for Pydantic-AI agent."
			logger.exception(msg)
			raise LLMError(msg)

		if not messages or messages[-1].get("role") != "user":
			msg = "Last message is not an user prompt"
			logger.exception(msg)
			raise LLMError(msg)

		user_prompt = messages[-1]["content"]

		final_data: str | M | None = None

		if ModelSettings is None:
			msg = "ModelSettings not found in pydantic-ai. Install the correct version."
			logger.exception(msg)
			raise LLMError(msg)

		async with agent.iter(user_prompt=user_prompt, model_settings=ModelSettings(**run_settings)) as run:
			async for chunk in run:
				if isinstance(chunk, End) and isinstance(chunk.data, FinalResult):
					final_data = chunk.data.output  # Assuming FinalResult has an 'output' attribute
					break

		if final_data is not None:
			if output_schema and not isinstance(final_data, output_schema):
				# This case should ideally be handled by pydantic-ai if output_type is set
				msg = (
					f"Pydantic-AI returned unexpected type. Expected {output_schema.__name__}, "
					f"got {type(final_data).__name__}."
				)
				raise LLMError(msg)
			return final_data

		msg = "Pydantic-AI call succeeded but returned no structured data or text chunks."
		logger.error(msg)
		raise LLMError(msg)

	except ImportError:
		msg = "Pydantic-AI library not installed. Install it with 'uv add pydantic-ai'."
		logger.exception(msg)
		raise LLMError(msg) from None
	except Exception as e:
		logger.exception("Pydantic-AI LLM API call failed")
		msg = f"Pydantic-AI LLM API call failed: {e}"
		raise LLMError(msg) from e
