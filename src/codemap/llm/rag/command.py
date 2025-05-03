"""Command for asking questions about the codebase using RAG."""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, TypedDict

from litellm import completion
from sqlalchemy.exc import SQLAlchemyError

from codemap.db.client import DatabaseClient
from codemap.db.engine import get_session
from codemap.db.models import ChatHistory
from codemap.llm import create_client
from codemap.processor.pipeline import ProcessingPipeline
from codemap.utils.cli_utils import loading_spinner, progress_indicator
from codemap.utils.config_loader import ConfigLoader

from .formatter import format_content_for_context
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Define a constant for truncation length
MAX_CONTEXT_LENGTH = 8000


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
		repo_path: Path | None = None,
		model: str | None = None,
		api_key: str | None = None,
		api_base: str | None = None,
	) -> None:
		"""Initializes the AskCommand, setting up clients and pipeline."""
		self.repo_path = repo_path or Path.cwd()
		self.session_id = str(uuid.uuid4())  # Unique session ID for DB logging

		self.db_client = DatabaseClient()  # Uses config path by default
		self.llm_client = create_client(
			repo_path=self.repo_path,
			model=model,  # create_client handles defaults/config
			api_key=api_key,
			api_base=api_base,
		)
		# Initialize ProcessingPipeline correctly
		try:
			# Show a spinner while initializing the pipeline
			with progress_indicator(message="Initializing processing pipeline...", style="spinner", transient=True):
				# Get ConfigLoader instance first, ensuring it knows the repo root
				self.config_loader = ConfigLoader.get_instance(repo_root=self.repo_path)
				self.pipeline = ProcessingPipeline(
					repo_path=self.repo_path,  # Pipeline still needs repo_path directly
					config_loader=self.config_loader,  # Pass the correctly initialized loader
					sync_on_init=False,  # Keep sync off during 'ask' init
					progress=None,  # No progress context with spinner style
					task_id=None,  # No task_id with spinner style
				)
			# Progress context manager handles completion message
			logger.info("ProcessingPipeline initialization attempt complete.")
		except Exception:
			logger.exception("Failed to initialize ProcessingPipeline")
			self.pipeline = None

		logger.info(f"AskCommand initialized for session {self.session_id}")

	def _retrieve_context(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
		"""Retrieve relevant code chunks based on the query."""
		if not self.pipeline:
			logger.warning("ProcessingPipeline not available, no context will be retrieved.")
			return []

		try:
			logger.info(f"Retrieving context for query: '{query}', limit: {limit}")
			results = self.pipeline.semantic_search(query, k=limit)

			# Format results for the LLM
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
							"content": metadata.get("content", ""),
							"distance": r.get("distance", -1.0),
						}
					)

			logger.debug(f"Semantic search returned {len(formatted_results)} results.")
			return formatted_results
		except Exception:
			logger.exception("Error retrieving context")
			return []

	def _extract_answer_from_response(self, response: Any) -> str | None:  # noqa: ANN401
		"""Safely extract the answer text from a litellm response object."""
		if response is None:
			return None

		try:
			# Handle different response object structures
			if hasattr(response, "choices") and response.choices:
				# Try to access as object with attributes
				choice = response.choices[0]
				if hasattr(choice, "message"):
					message = choice.message
					if hasattr(message, "content"):
						return message.content

			# Try to access as dictionary
			if isinstance(response, dict) and "choices" in response and response["choices"]:
				choice = response["choices"][0]
				if isinstance(choice, dict) and "message" in choice:
					message = choice["message"]
					if isinstance(message, dict) and "content" in message:
						return message["content"]

			# If we got here, we couldn't extract the answer
			logger.warning(f"Couldn't extract answer from response of type {type(response)}")
			return None

		# Fix for BLE001 (blind exception) - catch specific exceptions
		except (AttributeError, IndexError, KeyError, TypeError) as e:
			logger.warning(f"Error extracting answer from response: {e}")
			return None

	def run(self, question: str) -> AskResult:
		"""Executes one turn of the ask command, returning the answer and context."""
		logger.info(f"Processing question for session {self.session_id}: '{question}'")

		if not self.pipeline:
			return AskResult(answer="Processing pipeline not available.", context=[])

		# Retrieve relevant context first
		context = self._retrieve_context(question, limit=5)

		# Format context for inclusion in prompt
		context_text = format_content_for_context(context)
		if len(context_text) > MAX_CONTEXT_LENGTH:
			logger.warning(f"Context too long ({len(context_text)} chars), truncating.")
			context_text = context_text[:MAX_CONTEXT_LENGTH] + "... [truncated]"

		# Construct messages with context included
		messages = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{
				"role": "user",
				"content": (
					f"Here's my question about the codebase: {question}\n\n"
					f"Relevant context from the codebase:\n{context_text}"
				),
			},
		]

		# Store user query in DB
		db_entry_id = None
		try:
			db_entry = self.db_client.add_chat_message(session_id=self.session_id, user_query=question)
			db_entry_id = db_entry.id if db_entry else None
			if db_entry_id:
				logger.debug(f"Stored current query turn with DB ID: {db_entry_id}")
			else:
				logger.warning("Failed to get DB entry ID for current query turn.")
		except Exception:
			logger.exception("Failed to store current query turn in DB")

		# Get LLM config
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

		# Call LLM with context
		try:
			with loading_spinner("Waiting for LLM response..."):
				response = completion(
					model=model_name,
					messages=messages,
					api_base=api_base,
					api_key=api_key,
					stream=False,
					**llm_params,
				)

			# Extract answer from response
			answer = self._extract_answer_from_response(response)

			# Update DB with answer
			if db_entry_id and answer:
				try:
					# Access the engine directly from the database client
					with get_session(engine_instance=self.db_client.engine) as session:
						db_entry = session.get(ChatHistory, db_entry_id)
						if db_entry:
							db_entry.ai_response = answer
							session.commit()
							logger.debug(f"Updated DB entry {db_entry_id} with AI response")
				except (SQLAlchemyError, TypeError, AttributeError) as e:
					logger.warning(f"Failed to update DB entry {db_entry_id} with AI response: {e}")

			return AskResult(answer=answer, context=context)
		except Exception as e:
			logger.exception("Error during LLM completion")
			return AskResult(answer=f"Error: {e!s}", context=context)
