"""Command for asking questions about the codebase using RAG."""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, TypedDict, cast

from litellm import completion

from codemap.db.client import DatabaseClient
from codemap.db.engine import get_session
from codemap.db.models import ChatHistory
from codemap.llm import create_client
from codemap.processor.pipeline import ProcessingPipeline
from codemap.utils.cli_utils import loading_spinner  # Import spinner
from codemap.utils.config_loader import ConfigLoader

from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Define a constant for truncation length
MAX_TOOL_RESPONSE_LENGTH = 5000


class AskResult(TypedDict):
	"""Structured result for the ask command."""

	answer: str | None
	context: list[dict[str, Any]]


class AskCommand:
	"""
	Handles the logic for the `codemap ask` command.

	Interacts with the ProcessingPipeline, DatabaseClient, and an LLM to
	answer questions about the codebase using RAG. Maintains conversation
	history for interactive sessions.

	"""

	def __init__(
		self,
		# question: str, # Removed from init, handled by run()
		repo_path: Path | None = None,
		model: str | None = None,
		api_key: str | None = None,
		api_base: str | None = None,
	) -> None:
		"""Initializes the AskCommand, setting up clients and pipeline."""
		# self.question = question # Removed
		self.repo_path = repo_path or Path.cwd()
		self.session_id = str(uuid.uuid4())  # Unique session ID for DB logging
		self.messages: list[dict[str, Any]] = [
			{"role": "system", "content": SYSTEM_PROMPT}
		]  # Initialize message history

		self.config_loader = ConfigLoader.get_instance(repo_root=self.repo_path)
		self.db_client = DatabaseClient()  # Uses config path by default
		self.llm_client = create_client(
			repo_path=self.repo_path,
			model=model,  # create_client handles defaults/config
			api_key=api_key,
			api_base=api_base,
		)
		# Initialize ProcessingPipeline correctly
		try:
			with loading_spinner("Initializing processing pipeline..."):
				self.pipeline = ProcessingPipeline(
					repo_path=self.repo_path,
					config_loader=self.config_loader,
					sync_on_init=False,  # Avoid potentially long sync on every ask
				)
			logger.info("ProcessingPipeline initialized correctly.")
		except Exception:
			logger.exception("Failed to initialize ProcessingPipeline")
			self.pipeline = None

		self._setup_tools()
		logger.info(f"AskCommand initialized for session {self.session_id}")

	def _setup_tools(self) -> None:
		"""Defines the tools available to the LLM."""
		if not self.pipeline:
			logger.warning("ProcessingPipeline not available, no tools will be configured.")
			self.tools = []
			self.available_functions = {}
			return

		# Tool definitions compatible with LiteLLM
		# Split long descriptions
		semantic_desc = (
			"Search the codebase for code snippets semantically similar to the user's query. "
			"Use this for finding code related to concepts, functionality, or examples, "
			"when the exact structure or name is unknown."
		)
		graph_desc = (
			"Execute a *valid Cypher query* against the code graph database. "
			"Use this ONLY when the user asks a question that translates *directly* to a Cypher query "
			"(e.g., 'Find functions calling Foo.bar', 'List methods in class Bar', 'Show files importing module X'). "
			"The 'query' parameter MUST be a syntactically correct Cypher string, "
			"like 'MATCH (f:Function)-[:CALLS]->(m:Method {{name: \"Foo.bar\"}}) RETURN f.name'. "
			"Note: The node label for code elements is 'CodeEntity', "
			"filter by 'entity_type' property (e.g., 'CLASS', 'FUNCTION', 'MODULE', 'INTERFACE', 'VARIABLE'). "
			"Example class query: MATCH (c:CodeEntity {{name: 'MyClass', entity_type: 'CLASS'}}) RETURN c. "
			"DO NOT pass natural language questions to this tool."
		)
		graph_enhanced_desc = (
			"Search the codebase using a combination of semantic search and graph traversal. "
			"This is the preferred tool for most complex questions requiring understanding of code relationships "
			"and context, like 'How is data processed from function A to function B?', "
			"'Where is class C used?', or 'Find code related to feature Y and its dependencies'. "
			"Takes a natural language query as input."
		)
		self.tools = [
			{
				"type": "function",
				"function": {
					"name": "semantic_search",
					"description": semantic_desc,
					"parameters": {
						"type": "object",
						"properties": {
							"query": {
								"type": "string",
								"description": "The semantic search query based on the user's question.",
							},
							"limit": {
								"type": "integer",
								"description": "Maximum number of results to return.",
								"default": 5,
							},
						},
						"required": ["query"],
					},
				},
			},
			{
				"type": "function",
				"function": {
					"name": "graph_query",
					"description": graph_desc,
					"parameters": {
						"type": "object",
						"properties": {"query": {"type": "string", "description": "The Cypher query to execute."}},
						"required": ["query"],
					},
				},
			},
			{
				"type": "function",
				"function": {
					"name": "graph_enhanced_search",
					"description": graph_enhanced_desc,
					"parameters": {
						"type": "object",
						"properties": {
							"query": {
								"type": "string",
								"description": "The main query describing what information is needed.",
							},
							"k_vector": {
								"type": "integer",
								"description": "Number of initial similar chunks to retrieve via vector search.",
								"default": 5,
							},
							"graph_depth": {
								"type": "integer",
								"description": "Maximum depth for graph traversal from initial hits.",
								"default": 1,
							},
							"include_source_code": {
								"type": "boolean",
								"description": "Whether to include source code snippets for related graph entities.",
								"default": False,
							},
						},
						"required": ["query"],
					},
				},
			},
		]
		self.available_functions = {
			"semantic_search": self._tool_semantic_search,
			"graph_query": self._tool_graph_query,
			"graph_enhanced_search": self._tool_graph_enhanced_search,  # Register new tool
		}
		logger.debug(f"Configured tools: {[tool['function']['name'] for tool in self.tools]}")

	# --- Tool Implementation Wrappers ---
	def _tool_semantic_search(self, query: str, limit: int = 5) -> str:
		"""Wrapper for ProcessingPipeline.semantic_search for LLM tool use."""
		if not self.pipeline:
			return json.dumps({"error": "Processing pipeline is not available."})
		try:
			# Spinner might be too quick here, adding log message instead
			logger.info(f"Executing tool 'semantic_search' with query: '{query}', limit: {limit}")
			results = self.pipeline.semantic_search(query, k=limit)
			# Format results as a JSON string for the LLM
			# Provide richer, structured context from results
			formatted_results = []
			if results:  # Check if results is not None and has items
				for r in results:
					# Extract relevant fields safely
					metadata = r.get("metadata", {})
					formatted_results.append(
						{
							"file_path": metadata.get("file_path", "N/A"),
							"start_line": metadata.get("start_line", -1),
							"end_line": metadata.get("end_line", -1),
							"content_preview": metadata.get("content", "")[:200] + "..."
							if metadata.get("content")
							else "",  # Truncate long content
							"distance": r.get("distance", -1.0),
						}
					)

			logger.debug(f"Semantic search returned {len(formatted_results)} results.")
			json_output = json.dumps(formatted_results)
			if len(json_output) > MAX_TOOL_RESPONSE_LENGTH:
				logger.warning(f"Semantic search response truncated from {len(json_output)} chars.")
				return json_output[:MAX_TOOL_RESPONSE_LENGTH] + "... [truncated]"
			return json_output
		except Exception:
			logger.exception("Error during semantic_search tool")
			return json.dumps({"error": "An error occurred during semantic search."})

	def _tool_graph_query(self, query: str) -> str:
		"""Wrapper for ProcessingPipeline.graph_query for LLM tool use."""
		# --- Input Validation --- #
		required_keywords = ["MATCH", "RETURN", "WHERE", "CREATE", "MERGE", "CALL", "UNWIND"]
		query_upper = query.upper()
		if not any(keyword in query_upper for keyword in required_keywords):
			error_msg = (
				"Invalid input: graph_query requires a valid Cypher query string "
				"containing keywords like MATCH, RETURN, etc., not natural language."
			)
			logger.warning(f"Graph query validation failed: '{query}'. {error_msg}")
			return json.dumps({"error": error_msg})
		# --- End Validation --- #

		if not self.pipeline:
			return json.dumps({"error": "Processing pipeline is not available."})
		try:
			logger.info(f"Executing tool 'graph_query' with query: '{query}'")
			results = self.pipeline.graph_query(query)
			# Format results as a JSON string for the LLM
			num_results = len(results) if results is not None else 0  # Check for None before len()
			logger.debug(f"Graph query returned {num_results} results.")
			json_output = json.dumps(results)
			if len(json_output) > MAX_TOOL_RESPONSE_LENGTH:
				logger.warning(f"Graph query response truncated from {len(json_output)} chars.")
				return json_output[:MAX_TOOL_RESPONSE_LENGTH] + "... [truncated]"
			return json_output
		except Exception:
			logger.exception("Error during graph_query tool")
			return json.dumps({"error": "An error occurred during graph query."})

	def _tool_graph_enhanced_search(
		self,
		query: str,
		k_vector: int = 5,
		graph_depth: int = 1,
		include_source_code: bool = False,
	) -> str:
		"""Wrapper for ProcessingPipeline.graph_enhanced_search for LLM tool use."""
		if not self.pipeline:
			return json.dumps({"error": "Processing pipeline is not available."})
		try:
			logger.info(
				f"Executing tool 'graph_enhanced_search' with query: '{query}', "
				f"k_vector={k_vector}, graph_depth={graph_depth}, include_source_code={include_source_code}"
			)
			results = self.pipeline.graph_enhanced_search(
				query,
				k_vector=k_vector,
				graph_depth=graph_depth,
				include_source_code=include_source_code,
			)
			num_results = len(results) if results is not None else 0
			logger.debug(f"Graph-enhanced search returned {num_results} results.")
			# Consider summarizing or truncating the results if they are too large
			json_output = json.dumps(results)
			if len(json_output) > MAX_TOOL_RESPONSE_LENGTH:
				logger.warning(f"Graph-enhanced search response truncated from {len(json_output)} chars.")
				# Simple truncation for now, might break JSON structure
				return json_output[:MAX_TOOL_RESPONSE_LENGTH] + "... [truncated]"
			return json_output
		except Exception:
			logger.exception("Error during graph_enhanced_search tool")
			return json.dumps({"error": "An error occurred during graph-enhanced search."})

	# --- Main Execution Logic ---
	def run(self, question: str) -> AskResult:
		"""Executes one turn of the ask command, returning the answer and context."""
		logger.info(f"Processing question for session {self.session_id}: '{question}'")

		if not self.pipeline:
			return AskResult(answer="Processing pipeline not available.", context=[])

		# Add the current user question to the message history
		self.messages.append({"role": "user", "content": question})

		llm_config = self.config_loader.get_llm_config()
		llm_params = {
			"temperature": llm_config.get("temperature", 0.7),
			"max_tokens": llm_config.get("max_tokens", 1024),
		}
		model_name = llm_config.get("model", "openai/gpt-4o-mini")
		api_base = llm_config.get("api_base")
		api_key_env_var = llm_config.get("api_key_env")
		api_key = None
		if isinstance(api_key_env_var, str):
			api_key = os.getenv(api_key_env_var)
		api_key = api_key or llm_config.get("api_key")  # Fallback to direct config key

		# Store user query in DB (associated with this turn)
		db_entry_id = None
		try:
			# Use a unique turn ID or just the session ID? Using session ID for now.
			db_entry = self.db_client.add_chat_message(session_id=self.session_id, user_query=question)
			db_entry_id = db_entry.id if db_entry else None
			if db_entry_id:
				logger.debug(f"Stored current query turn with DB ID: {db_entry_id}")
			else:
				logger.warning("Failed to get DB entry ID for current query turn.")
		except Exception:
			logger.exception("Failed to store current query turn in DB")

		final_answer = None
		executed_tool_info_this_turn = []  # Track tools for *this* turn

		def raise_validation_error(message: str) -> None:
			logger.error(message)
			raise ValueError(message)

		def safe_get_message(response_obj: object) -> object | None:
			"""Safely extract the message from a litellm response object."""
			if not response_obj:
				return None
			try:
				if hasattr(response_obj, "choices") and response_obj.choices:  # type: ignore[attr-defined]
					choice = response_obj.choices[0]  # type: ignore[attr-defined]
					if hasattr(choice, "message"):
						return choice.message
					if isinstance(choice, dict) and "message" in choice:
						return choice["message"]
				if isinstance(response_obj, dict) and "choices" in response_obj and response_obj["choices"]:
					choice = response_obj["choices"][0]
					if isinstance(choice, dict) and "message" in choice:
						return choice["message"]
				logger.warning(f"Unrecognized response format: {type(response_obj)}")
				return None
			except Exception:
				logger.exception("Error extracting message from response")
				return None

		try:
			# Use the current message history
			with loading_spinner("Waiting for LLM response..."):
				response = completion(
					model=model_name,
					messages=self.messages,  # Use internal history
					tools=self.tools,
					api_base=api_base,
					api_key=api_key,
					stream=False,
					**llm_params,
				)

			response_message = safe_get_message(response)
			if not response_message:
				raise_validation_error("Could not extract a valid message from LLM response")

			# Add response message to history safely
			response_message_dict = {}
			model_dump_func: Any = getattr(response_message, "model_dump", None)
			if model_dump_func is not None:
				if callable(model_dump_func):
					response_message_dict = model_dump_func()  # type: ignore[misc]
				else:
					logger.warning("model_dump attribute is not callable.")
					response_message_dict = {"role": "assistant", "content": str(response_message)}
			elif isinstance(response_message, dict):
				response_message_dict = response_message
			else:
				# Fallback: Treat as string if not dict or dumpable
				response_message_dict = {"role": "assistant", "content": str(response_message)}
			# Ensure the appended item conforms to the list type
			self.messages.append(cast("dict[str, Any]", response_message_dict))

			# --- Tool Calling Loop ---
			tool_calls = getattr(response_message, "tool_calls", None)
			while tool_calls:
				logger.info(f"LLM requested {len(tool_calls)} tool calls.")

				tool_responses = []
				# Ensure tool_calls is iterable before looping
				if not isinstance(tool_calls, list):
					logger.warning(f"Expected tool_calls to be a list, got {type(tool_calls)}. Stopping tool loop.")
					tool_calls = []

				for tool_call in tool_calls:
					function_obj = getattr(tool_call, "function", None)
					if not function_obj or not hasattr(function_obj, "name"):
						logger.error(f"Invalid tool call structure received: {tool_call}")
						continue

					function_name = function_obj.name
					function_to_call = self.available_functions.get(function_name)

					tool_call_id = getattr(tool_call, "id", None)
					args_string = getattr(function_obj, "arguments", "{}")

					logger.debug(
						f"Attempting call to tool '{function_name}' (ID: {tool_call_id}) with args: {args_string}"
					)

					tool_info = {
						"tool_call_id": tool_call_id,
						"function_name": function_name,
						"arguments": args_string,
						"response": None,
					}

					if not function_to_call:
						logger.error(f"LLM requested unknown function: {function_name}")
						function_response = json.dumps({"error": f"Function '{function_name}' not found."})
						tool_info["response"] = function_response
					else:
						try:
							# Add spinner for individual tool call if desired
							# with loading_spinner(f"Running tool: {function_name}..."):
							function_args = json.loads(args_string)
							function_response = function_to_call(**function_args)
							tool_info["response"] = function_response
							logger.debug(
								f"Tool '{function_name}' (ID: {tool_call_id}) executed successfully. "
								f"Response length: {len(function_response)}"
							)
						except json.JSONDecodeError:
							logger.exception(f"Invalid JSON arguments for tool {function_name}: {args_string}.")
							function_response = json.dumps({"error": "Invalid JSON arguments"})
							tool_info["response"] = function_response
						except Exception:
							logger.exception(f"Error calling function {function_name}")
							function_response = json.dumps({"error": "Error executing tool"})
							tool_info["response"] = function_response

					tool_responses.append(
						{
							"tool_call_id": tool_call_id,
							"role": "tool",
							"name": function_name,
							"content": function_response,
						}
					)
					executed_tool_info_this_turn.append(tool_info)

				# Add tool responses to message history
				self.messages.extend(tool_responses)

				logger.info("Sending tool results back to LLM.")
				with loading_spinner("Waiting for LLM response after tool use..."):
					response = completion(
						model=model_name,
						messages=self.messages,
						tools=self.tools,
						api_base=api_base,
						api_key=api_key,
						stream=False,
						**llm_params,
					)

				response_message = safe_get_message(response)
				if not response_message:
					raise_validation_error("Could not extract a valid message from LLM response after tool use")

				# Add final response message to history safely
				response_message_dict = {}
				model_dump_func: Any = getattr(response_message, "model_dump", None)
				if model_dump_func is not None:
					if callable(model_dump_func):
						response_message_dict = model_dump_func()  # type: ignore[misc]
					else:
						logger.warning("model_dump attribute is not callable in final response.")
						response_message_dict = {"role": "assistant", "content": str(response_message)}
				elif isinstance(response_message, dict):
					response_message_dict = response_message
				else:
					# Fallback: Treat as string if not dict or dumpable
					response_message_dict = {"role": "assistant", "content": str(response_message)}
				# Ensure the appended item conforms to the list type
				self.messages.append(cast("dict[str, Any]", response_message_dict))

				# Check for more tool calls
				tool_calls = getattr(response_message, "tool_calls", None)

			# --- End Tool Calling Loop ---

			# Get the final content safely from the last assistant message in history
			final_answer = (
				self.messages[-1].get("content")
				if self.messages and self.messages[-1].get("role") == "assistant"
				else "No response content available"
			)

			logger.info(f"LLM final answer received for session {self.session_id}.")

		except Exception as e:
			logger.exception("Error during LLM interaction")
			final_answer = f"An error occurred during processing: {e}"

		# --- Update Database for this turn ---
		if db_entry_id is not None:
			try:
				with get_session(self.db_client.engine) as session:
					db_entry = session.get(ChatHistory, db_entry_id)
					if db_entry:
						db_entry.ai_response = final_answer
						# Store tool calls/responses specifically for this turn
						db_entry.tool_calls = (
							json.dumps(executed_tool_info_this_turn) if executed_tool_info_this_turn else None
						)
						# Context might be the full message history up to this point, or just this turn's tools?
						# Storing this turn's tool info for simplicity
						db_entry.context = (
							json.dumps(executed_tool_info_this_turn) if executed_tool_info_this_turn else None
						)
						session.add(db_entry)
						session.commit()
						logger.debug(f"Updated DB entry {db_entry_id} for current turn.")
					else:
						logger.warning(f"Could not find DB entry {db_entry_id} to update for current turn.")
			except Exception:
				logger.exception(f"Failed to update DB entry {db_entry_id} for current turn")
		else:
			logger.warning("No DB entry ID for current turn, cannot update history.")

		# Return structured result for this turn
		return AskResult(answer=final_answer, context=executed_tool_info_this_turn)
