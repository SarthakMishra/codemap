"""API interaction for LLM services."""

from __future__ import annotations

import logging
from typing import Literal, TypedDict, TypeVar

from pydantic import BaseModel, ValidationError

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

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)


class MessageDict(TypedDict):
	"""Typed dictionary for LLM message structure."""

	role: Literal["user", "system"]
	content: str


def validate_schema(model: type[PydanticModelT], input_data: str | object) -> PydanticModelT:
	"""Validate the schema of the input data."""
	if isinstance(input_data, str):
		return model.model_validate_json(input_data)
	return model.model_validate(input_data)


def call_llm_api(
	messages: list[MessageDict],
	config_loader: ConfigLoader,
	pydantic_model: type[PydanticModelT] | None = None,
) -> str | PydanticModelT:
	"""
	Call an LLM API using pydantic-ai.

	Args:
	    messages: The list of messages to send to the LLM
	    config_loader: ConfigLoader instance for additional configuration
	    pydantic_model: Optional Pydantic model class to structure the output.
	                  If provided, the function will return an instance of this model.
	                  Otherwise, it returns a string.

	Returns:
	    The generated response, either as a string or an instance of the pydantic_model.

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

	# Determine the output_type for the Pydantic-AI Agent
	agent_output_type: type = pydantic_model if pydantic_model else str

	try:
		# Initialize Pydantic-AI Agent
		agent = Agent(
			model=config_loader.get.llm.model,
			system_prompt=system_prompt_str,
			output_type=agent_output_type,
		)

		run_settings = {
			"temperature": config_loader.get.llm.temperature,
			"max_tokens": config_loader.get.llm.max_output_tokens,
		}

		logger.debug(
			"Calling Pydantic-AI Agent with model: %s, system_prompt: '%s...', params: %s",
			config_loader.get.llm.model,
			system_prompt_str[:100],
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

		if ModelSettings is None:
			msg = "ModelSettings not found in pydantic-ai. Install the correct version."
			logger.exception(msg)
			raise LLMError(msg)

		# Run the agent and validate the output
		run = agent.run_sync(user_prompt=user_prompt, model_settings=ModelSettings(**run_settings))

		if run.output is not None:
			if pydantic_model:
				try:
					return validate_schema(pydantic_model, run.output)
				except ValidationError as e:
					raise LLMError from e
			elif isinstance(run.output, (str, BaseModel)):
				return run.output  # type: ignore[return-value]

		msg = "Pydantic-AI call succeeded but returned no structured data or text."
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
