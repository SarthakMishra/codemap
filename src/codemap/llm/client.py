"""LLM client for unified access to language models."""

from __future__ import annotations

import hashlib
import json
import logging
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from codemap.config import ConfigLoader
from codemap.utils.cli_utils import progress_indicator

from .api import MessageDict, PydanticModelT, call_llm_api, get_agent

if TYPE_CHECKING:
	from pydantic_ai import Agent as AgentType
	from pydantic_ai.agent import AgentRunResult
	from pydantic_ai.tools import Tool


class CompletionStatus(BaseModel):
	"""Structured response for completion status check."""

	is_complete: bool = Field(description="Whether the task/response is complete")
	suggestion: str | None = Field(default=None, description="Optional suggestion if incomplete")
	final_response: str | None = Field(default=None, description="Final comprehensive response if complete")


class ToolCallSummary(BaseModel):
	"""Summary of tool calls for a specific tool."""

	tool_name: str = Field(description="Name of the tool")
	call_count: int = Field(default=0, description="Number of times the tool was called")
	total_duration: float = Field(default=0.0, description="Total duration in seconds")
	success_count: int = Field(default=0, description="Number of successful calls")
	error_count: int = Field(default=0, description="Number of failed calls")


class UsageSummary(BaseModel):
	"""Comprehensive usage summary for LLM interactions."""

	# Token usage
	request_tokens: int = Field(default=0, description="Total tokens in requests")
	response_tokens: int = Field(default=0, description="Total tokens in responses")
	total_tokens: int = Field(default=0, description="Total tokens used")

	# Request counts
	requests: int = Field(default=0, description="Total number of requests made")

	# Tool usage
	tool_calls: list[ToolCallSummary] = Field(default_factory=list, description="Summary of tool calls")
	total_tool_calls: int = Field(default=0, description="Total number of tool calls")

	# Cost estimation (if available)
	estimated_cost: float | None = Field(default=None, description="Estimated cost in USD")

	# Timing information
	response_time: float | None = Field(default=None, description="Total response time in seconds")

	# Iteration tracking for iterative completion
	iterations: int = Field(default=1, description="Number of iterations in iterative completion")

	def add_usage(self, usage_data: dict) -> None:
		"""Add usage data from a Pydantic-AI run result.

		Args:
			usage_data: Usage data from agent.run().usage()
		"""
		self.requests += usage_data.get("requests", 0)
		self.request_tokens += usage_data.get("request_tokens", 0) or 0
		self.response_tokens += usage_data.get("response_tokens", 0) or 0
		self.total_tokens += usage_data.get("total_tokens", 0) or 0

	def add_tool_call(self, tool_name: str, duration: float = 0.0, success: bool = True) -> None:
		"""Add a tool call to the summary.

		Args:
			tool_name: Name of the tool that was called
			duration: Duration of the tool call in seconds
			success: Whether the tool call was successful
		"""
		# Find existing tool summary or create new one
		tool_summary = None
		for tool in self.tool_calls:
			if tool.tool_name == tool_name:
				tool_summary = tool
				break

		if tool_summary is None:
			tool_summary = ToolCallSummary(tool_name=tool_name)
			self.tool_calls.append(tool_summary)

		# Update statistics
		tool_summary.call_count += 1
		tool_summary.total_duration += duration
		if success:
			tool_summary.success_count += 1
		else:
			tool_summary.error_count += 1

		self.total_tool_calls += 1

	def estimate_cost(self, model_name: str) -> None:
		"""Estimate the cost of the usage based on the model.

		Args:
			model_name: Name of the model used (e.g., "openai:gpt-4o", "google-gla:gemini-2.0-flash")
		"""
		# First try our internal cost calculation using the model prices JSON
		cost = self._calculate_cost_from_internal_data(model_name)
		if cost is not None:
			self.estimated_cost = cost
			return

		# Fallback to tokencost library
		try:
			from tokencost import calculate_cost_by_tokens

			# Map provider:model to tokencost-compatible name
			tokencost_model_name = self._map_to_tokencost_name(model_name)

			if self.request_tokens > 0 and self.response_tokens > 0 and tokencost_model_name:
				# Use calculate_cost_by_tokens since we have token counts, not actual text
				prompt_cost = calculate_cost_by_tokens(
					num_tokens=self.request_tokens,
					model=tokencost_model_name,
					token_type="input",  # noqa: S106
				)
				completion_cost = calculate_cost_by_tokens(
					num_tokens=self.response_tokens,
					model=tokencost_model_name,
					token_type="output",  # noqa: S106
				)
				# Convert Decimal to float
				self.estimated_cost = float(prompt_cost + completion_cost)
			else:
				logger.warning("Cannot estimate cost: missing token information or unsupported model")
		except ImportError:
			logger.debug("tokencost not available for cost estimation")
		except (ValueError, KeyError, TypeError) as e:
			logger.debug(f"Error estimating cost with tokencost: {e}")

	def _calculate_cost_from_internal_data(self, model_name: str) -> float | None:
		"""Calculate cost using our internal model prices JSON data.

		Args:
			model_name: Full model name with provider (e.g., "google-gla:gemini-2.0-flash")

		Returns:
			Calculated cost in USD, or None if model not found
		"""
		try:
			# Load the model prices JSON
			model_prices_path = Path(__file__).parent / "model_prices_and_context_window.json"
			if not model_prices_path.exists():
				return None

			with model_prices_path.open("r", encoding="utf-8") as f:
				model_prices = json.load(f)

			# Extract model name from provider:model format
			if ":" in model_name:
				_, model_key = model_name.split(":", 1)
			else:
				model_key = model_name

			# Try different variations of the model name
			possible_keys = [
				model_key,  # e.g., "gemini-2.0-flash"
				f"gemini/{model_key}",  # e.g., "gemini/gemini-2.0-flash"
			]

			model_data = None
			for key in possible_keys:
				if key in model_prices:
					model_data = model_prices[key]
					break

			if not model_data:
				logger.debug(f"Model {model_name} not found in pricing data")
				return None

			# Extract cost per token
			input_cost_per_token = model_data.get("input_cost_per_token")
			output_cost_per_token = model_data.get("output_cost_per_token")

			if input_cost_per_token is None or output_cost_per_token is None:
				logger.debug(f"Cost data incomplete for model {model_name}")
				return None

			# Calculate total cost
			if self.request_tokens > 0 and self.response_tokens > 0:
				input_cost = self.request_tokens * input_cost_per_token
				output_cost = self.response_tokens * output_cost_per_token
				total_cost = input_cost + output_cost
				logger.debug(
					f"Cost calculated for {model_name}: ${total_cost:.6f} "
					f"(input: ${input_cost:.6f}, output: ${output_cost:.6f})"
				)
				return total_cost

			return None

		except (json.JSONDecodeError, OSError, KeyError) as e:
			logger.debug(f"Error calculating cost from internal data: {e}")
			return None

	def _map_to_tokencost_name(self, model_name: str) -> str | None:
		"""Map provider:model names to tokencost-compatible names.

		Args:
			model_name: Full model name with provider (e.g., "openai:gpt-4o")

		Returns:
			Tokencost-compatible model name, or None if not mappable
		"""
		# If no provider prefix, return as-is
		if ":" not in model_name:
			return model_name

		provider, model_key = model_name.split(":", 1)

		# Common mappings from provider:model to tokencost names
		mappings = {
			# OpenAI models
			"openai": {
				"gpt-4o": "gpt-4o",
				"gpt-4o-mini": "gpt-4o-mini",
				"gpt-4": "gpt-4",
				"gpt-3.5-turbo": "gpt-3.5-turbo",
			},
			# Anthropic models
			"anthropic": {
				"claude-3-5-sonnet": "claude-3-5-sonnet-latest",
				"claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
				"claude-3-7-sonnet": "claude-3-7-sonnet-latest",
				"claude-3-opus": "claude-3-opus-20240229",
			},
			# Google models (vertex AI, gemini)
			"google-gla": {
				"gemini-2.0-flash": "gemini-2.0-flash",
				"gemini-1.5-pro": "gemini-1.5-pro",
				"gemini-1.5-flash": "gemini-1.5-flash",
			},
			"vertex_ai": {
				"gemini-2.0-flash": "gemini-2.0-flash",
				"gemini-1.5-pro": "gemini-1.5-pro",
			},
		}

		return mappings.get(provider, {}).get(model_key)


logger = logging.getLogger(__name__)


class LLMClient:
	"""Client for interacting with LLM services in a unified way."""

	# Class-level agent cache to share agents across instances with same config
	_agent_cache: ClassVar[dict[str, AgentType]] = {}
	# Default templates - empty in base class
	DEFAULT_TEMPLATES: ClassVar[dict[str, str]] = {}

	# Use slots for better memory efficiency
	__slots__ = ("_current_agent_key", "_current_usage", "_templates", "config_loader", "repo_path")

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
		self._current_usage = UsageSummary()

	def get_usage_summary(self) -> UsageSummary:
		"""Get the current usage summary.

		Returns:
			Current usage summary with all tracked metrics
		"""
		return self._current_usage

	def reset_usage_tracking(self) -> None:
		"""Reset the usage tracking for a new session."""
		self._current_usage = UsageSummary()

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
		track_usage: bool = True,
	) -> str | PydanticModelT:
		"""
		Generate text using the configured LLM.

		Args:
		    messages: List of messages to send to the LLM
		    tools: Optional list of tools to use.
		    pydantic_model: Optional Pydantic model for response validation
		    track_usage: Whether to track usage statistics

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

		# For tracking, we need to use the agent.run method directly
		if track_usage:
			logger.debug(f"Tracking usage enabled, tools provided: {len(tools) if tools else 0}")
			# Find the user message
			user_prompt = None
			for msg in reversed(messages):
				if msg["role"] == "user":
					user_prompt = msg["content"]
					break

			if user_prompt:
				logger.debug(f"Found user prompt for tracking: {user_prompt[:100]}...")
				try:
					# Use agent.run_sync to get usage information
					result = agent.run_sync(user_prompt)
					logger.debug(f"Successfully got result from agent.run_sync, result type: {type(result)}")

					# Extract usage information
					usage = result.usage()
					usage_dict = {
						"requests": usage.requests,
						"request_tokens": usage.request_tokens,
						"response_tokens": usage.response_tokens,
						"total_tokens": usage.total_tokens,
					}

					# Track the usage
					self._current_usage.add_usage(usage_dict)

					# Track tool calls if any tools were used
					if tools:
						logger.debug(f"Attempting to track tool calls with {len(tools)} tools")
						try:
							# Parse the result messages to count actual tool calls
							self._track_tool_calls_from_result(result, tools)
						except (AttributeError, TypeError) as tool_error:
							logger.debug(f"Error tracking tool calls: {tool_error}")
					else:
						logger.debug("No tools provided for tracking")

					return result.output
				except (AttributeError, TypeError, ValueError) as e:
					logger.warning(f"Failed to track usage, falling back to legacy API: {e}")
			else:
				logger.debug("No user prompt found for usage tracking")

		# Fallback to original API call
		logger.debug("Using fallback API call (no usage tracking)")
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
				track_usage=False,  # Don't track usage for internal completion checks
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

		Returns:
		    Final comprehensive response from the iterative process
		"""
		import time

		# Track start time for response time calculation
		start_time = time.time()

		# Reset usage tracking for this iterative completion
		self.reset_usage_tracking()

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

		# Calculate total response time
		end_time = time.time()
		response_time = end_time - start_time

		# Update usage summary with timing and iteration info
		self._current_usage.iterations = iteration_count
		self._current_usage.response_time = response_time

		# Estimate cost if we have model information
		model_name = self.config_loader.get.llm.model
		if model_name:
			self._current_usage.estimate_cost(model_name)

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

		logger.debug(f"Final iterative response: {final_answer} (took {response_time:.2f}s)")
		return final_answer or "No response generated."

	def _track_tool_calls_from_result(self, result: AgentRunResult, tools: list[Tool]) -> None:
		"""Track tool calls from Pydantic-AI result using proper message structure.

		Args:
			result: The Pydantic-AI run result
			tools: List of available tools
		"""
		try:
			# Get all messages from the result
			all_messages = result.all_messages()
			logger.debug(f"Analyzing {len(all_messages)} messages for tool calls")

			# Create a mapping of tool names for validation - use the correct attribute
			tool_names = {tool.name for tool in tools}
			logger.debug(f"Available tool names: {tool_names}")

			# Track tool calls from messages
			for message in all_messages:
				# Check if this is a ModelResponse (kind='response')
				if (
					hasattr(message, "kind")
					and message.kind == "response"
					and hasattr(message, "parts")
					and message.parts
				):
					for part in message.parts:
						# Look for ToolCallPart objects - check the class name or specific attributes
						part_type = type(part).__name__
						if part_type == "ToolCallPart" and hasattr(part, "tool_name") and hasattr(part, "args"):
							# This is a ToolCallPart
							tool_name = getattr(part, "tool_name", None)
							if tool_name and isinstance(tool_name, str) and tool_name in tool_names:
								self._current_usage.add_tool_call(tool_name, success=True)
								logger.debug(f"Tracked tool call: {tool_name}")

			# Fallback heuristic if no tool calls were found in messages
			if self._current_usage.total_tool_calls == 0 and tools:
				usage = result.usage()
				logger.debug(f"No explicit tool calls found, using heuristics. Requests: {usage.requests}")

				# If we have multiple requests, tools were likely used
				if usage.requests > 1:
					# Estimate tool calls based on extra requests
					estimated_tool_calls = usage.requests - 1
					logger.debug(f"Estimating {estimated_tool_calls} tool calls from request count")

					# Track the most likely tools to have been used - use correct tool name
					for _i, tool in enumerate(tools[:estimated_tool_calls]):
						tool_name = tool.name  # Use the correct attribute
						self._current_usage.add_tool_call(tool_name, success=True)
						logger.debug(f"Estimated tool call: {tool_name}")

				# Alternative: check output content for tool usage indicators
				elif hasattr(result, "output") and result.output:
					output_text = str(result.output).lower()
					tool_indicators = [
						"based on",
						"codebase",
						"components",
						"structure",
						"analysis",
						"directory",
						"file",
						"found",
						"located",
						"processing",
					]

					if any(indicator in output_text for indicator in tool_indicators):
						# Assume the first tool was used - use correct tool name
						first_tool = tools[0]
						tool_name = first_tool.name  # Use the correct attribute
						self._current_usage.add_tool_call(tool_name, success=True)
						logger.debug(f"Inferred tool usage from content: {tool_name}")

		except (AttributeError, TypeError, ValueError) as e:
			logger.debug(f"Error in tool call tracking: {e}")
			# Final fallback: if tools were provided, assume at least one was used
			if tools:
				first_tool = tools[0]
				tool_name = first_tool.name  # Use the correct attribute
				self._current_usage.add_tool_call(tool_name, success=True)
				logger.debug(f"Fallback tool tracking: {tool_name}")
