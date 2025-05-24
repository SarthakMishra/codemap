"""LLM client for unified access to language models."""

from __future__ import annotations

import hashlib
import logging
import textwrap
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from codemap.config import ConfigLoader
from codemap.utils.cli_utils import progress_indicator

from .api import MessageDict, PydanticModelT, call_llm_api, get_agent

if TYPE_CHECKING:
	from pathlib import Path

	from pydantic_ai import Agent as AgentType
	from pydantic_ai.tools import Tool


class CompletionStatus(BaseModel):
	"""Structured response for completion status check."""

	is_complete: bool = Field(description="Whether the task/response is complete")
	suggestion: str | None = Field(default=None, description="Optional suggestion if incomplete")
	final_response: str | None = Field(default=None, description="Final comprehensive response if complete")


logger = logging.getLogger(__name__)


class LLMClient:
	"""Client for interacting with LLM services in a unified way."""

	# Class-level agent cache to share agents across instances with same config
	_agent_cache: ClassVar[dict[str, AgentType]] = {}
	# Default templates - empty in base class
	DEFAULT_TEMPLATES: ClassVar[dict[str, str]] = {}

	# Use slots for better memory efficiency
	__slots__ = ("_current_agent_key", "_templates", "config_loader", "repo_path")

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
		# Only copy templates if they exist and are non-empty
		self._templates = self.DEFAULT_TEMPLATES.copy() if self.DEFAULT_TEMPLATES else {}
		self._current_agent_key: str | None = None

	def set_template(self, name: str, template: str) -> None:
		"""
		Set a prompt template.

		Args:
		    name: Template name
		    template: Template content
		"""
		self._templates[name] = template

	def _create_agent_key(
		self,
		tools: list[Tool] | None = None,
		system_prompt_str: str | None = None,
		output_type: type = str,
	) -> str:
		"""Create a unique key for agent caching based on configuration.

		Args:
		    tools: Optional list of tools
		    system_prompt_str: Optional system prompt
		    output_type: Output type for the agent

		Returns:
		    A unique string key for this agent configuration
		"""
		# Create a deterministic key from agent configuration
		key_parts = [
			str(output_type),
			system_prompt_str or "",
			str(len(tools) if tools else 0),
		]

		# Add tool signatures for more specific caching
		if tools:
			tool_signatures = [getattr(tool, "__name__", str(tool)) for tool in tools]
			key_parts.extend(sorted(tool_signatures))

		key_string = "|".join(key_parts)
		return hashlib.sha256(key_string.encode()).hexdigest()

	def _extract_system_prompt(self, messages: list[MessageDict]) -> str | None:
		"""Extract system prompt from messages efficiently.

		Args:
		    messages: List of message dictionaries

		Returns:
		    System prompt string if found, None otherwise
		"""
		# Early return if no messages
		if not messages:
			return None

		# Check first message first (most common case)
		if messages[0]["role"] == "system":
			return messages[0]["content"]

		# Only scan remaining messages if first isn't system
		for msg in messages[1:]:
			if msg["role"] == "system":
				return msg["content"]

		return None

	def get_agent(
		self,
		tools: list[Tool] | None = None,
		system_prompt_str: str | None = None,
		output_type: type = str,
	) -> AgentType:
		"""Get or retrieve cached LLM agent.

		Args:
			tools: Optional list of tools to enable for the agent
			system_prompt_str: Optional system prompt to guide the agent's behavior
			output_type: Type for structuring the agent's output. Defaults to str.

		Returns:
			An initialized Pydantic-AI Agent instance configured with the specified settings.
		"""
		agent_key = self._create_agent_key(tools, system_prompt_str, output_type)

		# Return cached agent if available
		if agent_key in self._agent_cache:
			self._current_agent_key = agent_key
			return self._agent_cache[agent_key]

		# Create new agent and cache it
		agent = get_agent(
			self.config_loader,
			tools,
			system_prompt_str,
			output_type,
		)

		self._agent_cache[agent_key] = agent
		self._current_agent_key = agent_key
		return agent

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
		# Extract system prompt efficiently
		system_prompt_str = self._extract_system_prompt(messages)

		# Determine the output_type for the Pydantic-AI Agent
		agent_output_type: type = pydantic_model if pydantic_model else str

		# Get agent (cached or new)
		agent = self.get_agent(
			tools=tools,
			system_prompt_str=system_prompt_str,
			output_type=agent_output_type,
		)

		# Call the API
		return call_llm_api(
			messages=messages,
			tools=tools,
			agent=agent,
			pydantic_model=pydantic_model,
			config_loader=self.config_loader,
		)

	@classmethod
	def clear_agent_cache(cls) -> None:
		"""Clear the agent cache. Useful for testing or memory management."""
		cls._agent_cache.clear()

	@classmethod
	def get_cache_stats(cls) -> dict[str, int]:
		"""Get cache statistics for monitoring performance.

		Returns:
		    Dictionary with cache size and other stats
		"""
		return {
			"cache_size": len(cls._agent_cache),
			"cached_agents": len(cls._agent_cache),
		}

	def check_completion(
		self,
		response: str,
		original_question: str | None = None,
		system_prompt: str | None = None,
		is_last_iteration: bool = False,
	) -> CompletionStatus:
		"""Check if a response appears complete and generate final response if needed.

		This method uses structured LLM output to analyze completion status and
		generates a comprehensive final response in one step when complete.

		Args:
		    response: The response text to analyze
		    original_question: Optional original question for context
		    system_prompt: Optional original system prompt for task context
		    is_last_iteration: Whether this is the final iteration (forces completion)

		Returns:
		    CompletionStatus with completion assessment, optional suggestions, and final response
		"""
		# Build additional instruction for last iteration
		last_iteration_instruction = ""
		if is_last_iteration:
			last_iteration_instruction = textwrap.dedent("""
				🚨 CRITICAL: THIS IS THE LAST AND FINAL ITERATION 🚨

				You MUST mark this as complete and generate a final comprehensive response.
				This is your final chance - you cannot request more iterations.
				If you mark this as incomplete, you will have FAILED at your task.

				Generate the best possible final response based on all available information.
				It's better to provide a complete response with the available information
				than to leave the user with no answer at all.
			""")

		# Build the system prompt for completion analysis and final response generation
		check_completion_prompt = textwrap.dedent("""
			You are a task completion analyzer and response finalizer. Your job is to:

			1. Analyze the provided response to determine if it is complete and satisfactory
			2. If COMPLETE: Generate a final, comprehensive, self-contained response
			3. If INCOMPLETE: Provide suggestions for what's missing

			You will be provided with:
			- The original system prompt that defined the task requirements
			- The user's original question
			- The response that needs to be analyzed

			Completion Analysis Factors:
			- Has the original question been fully answered according to the system prompt?
			- Is the response complete and actionable?
			- Are there obvious gaps or missing information?
			- Does the response contain necessary details, code examples, and explanations?
			- Is the response self-contained (doesn't reference unseen tool outputs)?
			- Are there concrete recommendations or next steps provided?
			- Does the response fulfill the role and requirements specified in the original system prompt?

			Be conservative - if there's any doubt about completeness, mark as incomplete.

			If COMPLETE: Generate a final comprehensive response that:
			- Includes all key findings and analysis
			- Contains specific code examples and file paths
			- Provides actionable recommendations
			- Is completely self-contained
			- Doesn't reference previous messages or tool calls
			- Fulfills the requirements set by the original system prompt

			If INCOMPLETE: Provide specific suggestions for what needs to be added.
			{last_iteration_instruction}

			Following JSON Schema must be followed for Output:
			{model_schema}

			Return your answer as valid JSON that matches the schema.
		""").format(
			last_iteration_instruction=last_iteration_instruction, model_schema=CompletionStatus.model_json_schema()
		)

		# Prepare messages for the completion check and final response generation
		messages: list[MessageDict] = [
			{"role": "system", "content": check_completion_prompt},
		]

		# Build the user content with all available context
		user_content_parts = []

		if system_prompt:
			user_content_parts.append(f"Original System Prompt: {system_prompt}")

		if original_question:
			user_content_parts.append(f"Original Question: {original_question}")

		user_content_parts.append(f"Response to Analyze: {response}")

		messages.append(
			{
				"role": "user",
				"content": "\n\n".join(user_content_parts),
			}
		)

		try:
			# Get completion status and potential final response from the LLM
			result = self.completion(
				messages=messages,
				pydantic_model=CompletionStatus,
			)

			if isinstance(result, CompletionStatus):
				suggestion_text = f" - {result.suggestion}" if result.suggestion else ""
				status_text = "complete" if result.is_complete else "incomplete"
				final_response_info = " (with final response)" if result.final_response else ""
				logger.debug(f"LLM completion check result: {status_text}{suggestion_text}{final_response_info}")
				return result

			# Fallback if we didn't get the expected type
			logger.warning(f"Unexpected completion check result type: {type(result)}")
			return CompletionStatus(
				is_complete=False,
				suggestion="Unable to properly assess completion status, continue with analysis",
			)

		except Exception as e:
			msg = f"Failed to get completion status from LLM: {e}"
			logger.exception(msg)
			# Return a fallback completion status
			return CompletionStatus(
				is_complete=False,
				suggestion="Error during completion check, continue with current approach",
			)

	def iterative_completion(
		self,
		question: str,
		system_prompt: str,
		tools: list[Tool] | None = None,
		max_iterations: int = 6,
	) -> str:
		"""Perform iterative completion with automatic completion checking.

		This method handles the common pattern of:
		1. Initializing conversation with system prompt and question
		2. Iterating with LLM calls using provided tools
		3. Checking completion status after each iteration
		4. Generating final response when complete or max iterations reached

		Args:
		    question: The user's question or task
		    system_prompt: System prompt defining the task requirements
		    tools: Optional list of tools to use during completion
		    max_iterations: Maximum number of iterations before forcing completion
		    progress_callback: Optional callback for progress updates

		Returns:
		    Final comprehensive response from the iterative process
		"""
		# Initialize conversation with system prompt and user question
		conversation_messages: list[MessageDict] = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": f"Here's my question about the codebase: {question}"},
		]

		final_answer = None
		iteration_count = 0

		with progress_indicator(message="Processing iterative completion...", style="spinner", transient=True):
			while iteration_count < max_iterations:
				iteration_count += 1
				logger.debug(f"Iteration {iteration_count}/{max_iterations}")

				# For the first iteration, use the original completion method
				if iteration_count == 1:
					current_response = self.completion(
						messages=conversation_messages,
						tools=tools,
					)
				else:
					# For subsequent iterations, continue with the conversation
					continuation_messages: list[MessageDict] = [
						*conversation_messages,
						{"role": "user", "content": "Please continue your analysis or provide more details."},
					]
					current_response = self.completion(
						messages=continuation_messages,
						tools=tools,
					)

				logger.debug(f"Iteration {iteration_count} response: {current_response[:200]}...")

				# Add the response to conversation history
				conversation_messages.append({"role": "assistant", "content": current_response})

				# Check if the response is complete
				completion_status = self.check_completion(
					response=current_response,
					original_question=question,
					system_prompt=system_prompt,
					is_last_iteration=(iteration_count >= max_iterations),
				)

				logger.debug(f"Completion status: {completion_status.is_complete}")
				if completion_status.suggestion:
					logger.debug(f"Completion suggestion: {completion_status.suggestion}")

				if completion_status.is_complete:
					logger.info(f"Response marked as complete after {iteration_count} iterations")

					# Use the final response from completion check if available
					if completion_status.final_response:
						final_answer = completion_status.final_response
						logger.debug("Using final response from completion check")
					else:
						# Fallback to current response if no final response was generated
						final_answer = current_response
						logger.debug("Using current response as final answer (no final response from completion check)")
					break

				# Continue iteration - the response is not complete
				suggestion = completion_status.suggestion or "Continue with your analysis"
				logger.debug(f"Response not complete, continuing iteration {iteration_count + 1}: {suggestion}")

		# If we've reached max iterations without completion
		if final_answer is None and conversation_messages:
			logger.warning(f"Reached max iterations ({max_iterations}) without completion")
			# Use the last assistant response
			for msg in reversed(conversation_messages):
				if msg["role"] == "assistant":
					final_answer = msg["content"]
					break

			if not final_answer:
				final_answer = "Unable to generate a complete response within iteration limits."

		logger.debug(f"Final iterative response: {final_answer}")
		return final_answer or "No response generated."
