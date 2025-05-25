"""API interaction for LLM services."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, TypedDict, TypeVar, cast

from pydantic import BaseModel, ValidationError

# Import Pydantic-AI
from pydantic_ai import Agent
from pydantic_ai.result import FinalResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
from pydantic_graph import End

from codemap.config import ConfigLoader
from codemap.llm.utils import is_ollama_model

from .errors import LLMError

if TYPE_CHECKING:
	from pydantic_ai import Agent as AgentType
	from pydantic_ai.tools import Tool

logger = logging.getLogger(__name__)

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)


class MessageDict(TypedDict):
	"""Typed dictionary for LLM message structure."""

	role: Literal["user", "system", "assistant"]
	content: str


def validate_schema(model: type[PydanticModelT], input_data: str | object) -> PydanticModelT:
	"""Validate the schema of the input data."""
	if isinstance(input_data, str):
		return cast("PydanticModelT", model.model_validate_json(input_data))
	return cast("PydanticModelT", model.model_validate(input_data))


def get_agent(
	config_loader: ConfigLoader,
	tools: list[Tool] | None = None,
	system_prompt_str: str | None = None,
	output_type: type = str,
) -> AgentType:
	"""Initialize and return a Pydantic-AI Agent for LLM interactions.

	Args:
		config_loader: Configuration loader instance containing LLM settings.
		tools: Optional list of tools to enable for the agent.
		system_prompt_str: Optional system prompt to guide the agent's behavior.
			Defaults to a standard programming assistant prompt if not provided.
		output_type: Type for structuring the agent's output. Defaults to str.

	Returns:
		An initialized Pydantic-AI Agent instance configured with the specified settings.

	Raises:
		LLMError: If the Pydantic-AI library or required types are not installed.
	"""
	if Agent is None:
		msg = "Pydantic-AI library or its required types (Agent) not installed/found."
		logger.exception(msg)
		raise LLMError(msg) from None

	# Determine system prompt
	if system_prompt_str is None:
		system_prompt_str = (
			"You are an AI programming assistant. Follow the user's requirements carefully and to the letter."
		)

	# Convert None to empty list if tools is None
	agent_tools: list[Tool] = tools or []

	# Initialize Pydantic-AI Agent
	model_name = config_loader.get.llm.model

	if is_ollama_model(model_name):
		from pydantic_ai.models.openai import OpenAIModel
		from pydantic_ai.providers.openai import OpenAIProvider

		model_name = model_name.split(":", 1)[1]

		base_url = config_loader.get.llm.base_url
		if base_url is None:
			base_url = "http://localhost:11434/v1"

		ollama_model = OpenAIModel(model_name=model_name, provider=OpenAIProvider(base_url=base_url))

		agent = Agent(
			ollama_model,
			tools=agent_tools,
			system_prompt=system_prompt_str,
			output_type=output_type,
		)
	else:
		agent = Agent(
			model=config_loader.get.llm.model,
			tools=agent_tools,
			system_prompt=system_prompt_str,
			output_type=output_type,
		)

	return agent


def call_llm_api(
	messages: list[MessageDict],
	config_loader: ConfigLoader,
	agent: AgentType | None = None,
	tools: list[Tool] | None = None,
	pydantic_model: type[PydanticModelT] | None = None,
) -> str | PydanticModelT:
	"""
	Call an LLM API using pydantic-ai.

	Args:
	    messages: The list of messages to send to the LLM
	    config_loader: ConfigLoader instance for additional configuration
	    agent: Optional Pydantic-AI Agent instance to use for the API call.
	           If None, a new agent will be created using get_agent().
	    tools: Optional list of tools to use.
	    pydantic_model: Optional Pydantic model class to structure the output.
	                  If provided, the function will return an instance of this model.
	                  Otherwise, it returns a string.

	Returns:
	    The generated response, either as a string or an instance of the pydantic_model.

	Raises:
	    LLMError: If pydantic-ai is not installed or the API call fails.
	"""
	if Agent is None or End is None or FinalResult is None or UsageLimits is None:  # Check all imports
		msg = "Pydantic-AI library or its required types (AgentNode, End, FinalResult) not installed/found."
		logger.exception(msg)
		raise LLMError(msg) from None

	# Determine system prompt from messages
	system_prompt_str = None
	for message in messages:
		if message["role"] == "system":
			system_prompt_str = message["content"]
			break

	# Determine the output_type for the Pydantic-AI Agent
	agent_output_type: type = pydantic_model if pydantic_model else str

	try:
		# Create agent if not provided using the get_agent function
		if agent is None:
			agent = get_agent(
				config_loader=config_loader,
				tools=tools,
				system_prompt_str=system_prompt_str,
				output_type=agent_output_type,
			)

		if not any(message.get("role") == "user" for message in messages):
			error_msg = "No user content found in messages for Pydantic-AI agent."
			logger.exception(error_msg)
			raise LLMError(error_msg)

		if not messages or messages[-1].get("role") != "user":
			error_msg = "Last message is not an user prompt"
			logger.exception(error_msg)
			raise LLMError(error_msg)

		user_prompt = messages[-1]["content"]

		if ModelSettings is None:
			error_msg = "ModelSettings not found in pydantic-ai. Install the correct version."
			logger.exception(error_msg)
			raise LLMError(error_msg)

		# Run the agent and validate the output
		model_settings = ModelSettings(
			temperature=float(config_loader.get.llm.temperature),
			max_tokens=int(config_loader.get.llm.max_output_tokens),
			parallel_tool_calls=True,
		)

		usage_limits = UsageLimits(
			request_tokens_limit=int(config_loader.get.llm.max_input_tokens),
			response_tokens_limit=int(config_loader.get.llm.max_output_tokens),
			request_limit=int(config_loader.get.llm.max_requests),
		)

		run = agent.run_sync(
			user_prompt=user_prompt,
			model_settings=model_settings,
			usage_limits=usage_limits,
		)

		if run.output is not None:
			if pydantic_model:
				try:
					return validate_schema(pydantic_model, run.output)
				except ValidationError as e:
					raise LLMError from e
			elif isinstance(run.output, str):
				return run.output
			elif isinstance(run.output, BaseModel):
				# This shouldn't happen when pydantic_model is None, but handle it
				return str(run.output)

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
