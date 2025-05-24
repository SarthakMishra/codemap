"""Command for asking questions about the codebase using RAG."""

import logging
import uuid
from typing import Any, TypedDict

from codemap.config import ConfigLoader
from codemap.db.client import DatabaseClient
from codemap.llm.api import MessageDict
from codemap.llm.client import LLMClient
from codemap.llm.rag.interactive import RagUI
from codemap.llm.rag.tools import read_file_tool, semantic_retrieval_tool, web_search_tool
from codemap.processor.pipeline import ProcessingPipeline
from codemap.utils.cli_utils import progress_indicator

from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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

	def __init__(self) -> None:
		"""Initialize the AskCommand with lazy-loaded dependencies."""
		self.config_loader = ConfigLoader.get_instance()
		self.ui = RagUI()
		self.session_id = str(uuid.uuid4())
		self._db_client: DatabaseClient | None = None
		self._llm_client: LLMClient | None = None
		self._pipeline: ProcessingPipeline | None = None
		self._max_context_length: int = 8000  # Default value
		self._max_context_results: int = 10  # Default value

	@property
	def db_client(self) -> DatabaseClient:
		"""Lazily initialize and return a DatabaseClient instance."""
		if self._db_client is None:
			self._db_client = DatabaseClient()
		return self._db_client

	@property
	def llm_client(self) -> LLMClient:
		"""Lazily initialize and return an LLMClient instance."""
		if self._llm_client is None:
			self._llm_client = LLMClient(config_loader=self.config_loader)
		return self._llm_client

	@property
	def pipeline(self) -> ProcessingPipeline | None:
		"""Lazily initialize and return a ProcessingPipeline instance, or None if initialization fails."""
		if self._pipeline is None:
			try:
				with progress_indicator(message="Initializing processing pipeline...", style="spinner", transient=True):
					self._pipeline = ProcessingPipeline(config_loader=self.config_loader)
				logger.info("ProcessingPipeline initialization complete.")
			except Exception:
				logger.exception("Failed to initialize ProcessingPipeline")

		return self._pipeline

	@property
	def max_context_length(self) -> int:
		"""Return the maximum context length for RAG, using config or default."""
		cached = getattr(self, "_max_context_length", None)
		if cached is not None:
			return cached
		try:
			rag_config = self.config_loader.get.rag
			value = getattr(rag_config, "max_context_length", None)
			if value is not None:
				self._max_context_length = value
				return value
		except (AttributeError, TypeError) as e:
			logger.debug("Error reading max_context_length from config: %s", e)
		return self._max_context_length

	@property
	def max_context_results(self) -> int:
		"""Return the maximum number of context results for RAG, using config or default."""
		cached = getattr(self, "_max_context_results", None)
		if cached is not None:
			return cached
		try:
			rag_config = self.config_loader.get.rag
			value = getattr(rag_config, "max_context_results", None)
			if value is not None:
				self._max_context_results = value
				return value
		except (AttributeError, TypeError) as e:
			logger.debug("Error reading max_context_results from config: %s", e)
		return self._max_context_results

	async def initialize(self) -> None:
		"""Perform asynchronous initialization for the command, especially the pipeline."""
		if self.pipeline and not self.pipeline.is_async_initialized:
			try:
				# Show a spinner while initializing the pipeline asynchronously
				with progress_indicator(
					message="Initializing async components (pipeline)...", style="spinner", transient=True
				):
					await self.pipeline.async_init(sync_on_init=True)
				logger.info("ProcessingPipeline async initialization complete.")
			except Exception:
				logger.exception("Failed during async initialization of ProcessingPipeline")
				# Optionally set pipeline to None or handle the error appropriately
				self._pipeline = None
		elif not self.pipeline:
			logger.error("Cannot perform async initialization: ProcessingPipeline failed to initialize earlier.")
		else:
			logger.info("AskCommand async components already initialized.")

	async def run(self, question: str) -> AskResult:
		"""Executes one turn of the ask command, returning the answer and context."""
		logger.info(f"Processing question for session {self.session_id}: '{question}'")

		# Ensure async initialization happened (idempotent check inside)
		await self.initialize()

		if not self.pipeline:
			return AskResult(answer="Processing pipeline not available.", context=[])

		# Construct prompt text from the context and question
		messages: list[MessageDict] = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": f"Here's my question about the codebase: {question}"},
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

		# Call LLM with context
		try:
			with progress_indicator("Waiting for LLM response..."):
				answer = self.llm_client.completion(
					messages=messages,
					tools=[read_file_tool, semantic_retrieval_tool, web_search_tool()],
				)
			logger.debug(f"LLM response: {answer}")

			# Update DB with answer using the dedicated client method
			if db_entry_id and answer:
				# The update_chat_response method handles its own exceptions and returns success/failure
				success = self.db_client.update_chat_response(message_id=db_entry_id, ai_response=answer)
				if not success:
					logger.warning(f"Failed to update DB entry {db_entry_id} via client method.")

			return AskResult(answer=answer, context=[])
		except Exception as e:  # Keep the outer exception for LLM call errors
			logger.exception("Error during LLM completion")
			return AskResult(answer=f"Error: {e!s}", context=[])
