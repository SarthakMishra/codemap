"""Command for asking questions about the codebase using RAG."""

import logging
import uuid
from typing import Any, TypedDict

from codemap.config import ConfigLoader
from codemap.db.client import DatabaseClient
from codemap.git.utils import get_repo_root
from codemap.llm.client import LLMClient
from codemap.processor.pipeline import ProcessingPipeline
from codemap.utils.cli_utils import progress_indicator

from .formatter import format_content_for_context
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Default values for configuration
DEFAULT_MAX_CONTEXT_LENGTH = 8000
DEFAULT_MAX_CONTEXT_RESULTS = 10


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
	) -> None:
		"""Initializes the AskCommand, setting up clients and pipeline."""
		self.session_id = str(uuid.uuid4())  # Unique session ID for DB logging

		self.config_loader = ConfigLoader.get_instance()

		if self.config_loader.get.repo_root is None:
			self.repo_root = get_repo_root()
		else:
			self.repo_root = self.config_loader.get.repo_root

		# Get RAG configuration
		rag_config = self.config_loader.get.rag
		self.max_context_length = rag_config.max_context_length
		self.max_context_results = rag_config.max_context_results
		logger.debug(
			f"Using max_context_length: {self.max_context_length}, max_context_results: {self.max_context_results}"
		)

		self.db_client = DatabaseClient()  # Uses config path by default
		self.llm_client = LLMClient(config_loader=self.config_loader)
		# Initialize ProcessingPipeline correctly
		try:
			# Show a spinner while initializing the pipeline
			with progress_indicator(message="Initializing processing pipeline...", style="spinner", transient=True):
				# Use the provided ConfigLoader instance
				self.pipeline = ProcessingPipeline(
					config_loader=self.config_loader,
				)

			# Progress context manager handles completion message
			logger.info("ProcessingPipeline initialization complete.")
		except Exception:
			logger.exception("Failed to initialize ProcessingPipeline")
			self.pipeline = None

		logger.info(f"AskCommand initialized for session {self.session_id}")

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
				self.pipeline = None
		elif not self.pipeline:
			logger.error("Cannot perform async initialization: ProcessingPipeline failed to initialize earlier.")
		else:
			logger.info("AskCommand async components already initialized.")

	async def _retrieve_context(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
		"""Retrieve relevant code chunks based on the query."""
		if not self.pipeline:
			logger.warning("ProcessingPipeline not available, no context will be retrieved.")
			return []

		# Use configured limit or default
		actual_limit = limit or self.max_context_results

		try:
			logger.info(f"Retrieving context for query: '{query}', limit: {actual_limit}")
			# Use synchronous method to get results (pipeline.semantic_search is async)
			# Now call await directly as this method is async
			# import asyncio
			# results = asyncio.run(self.pipeline.semantic_search(query, k=actual_limit))
			results = await self.pipeline.semantic_search(query, k=actual_limit)

			# Format results for the LLM
			formatted_results = []
			if results:  # Check if results is not None and has items
				for r in results:
					# Extract relevant fields from payload
					payload = r.get("payload", {})

					# Get file content from repo using file_path, start_line, and end_line
					file_path = payload.get("file_path", "N/A")
					start_line = payload.get("start_line", -1)
					end_line = payload.get("end_line", -1)

					# Get content from repository if needed and build a content representation
					# For now, we'll use a simple representation that includes metadata
					entity_type = payload.get("entity_type", "")
					entity_name = payload.get("entity_name", "")
					language = payload.get("language", "")

					# Build a content representation from the metadata
					content_parts = []
					content_parts.append(f"Type: {entity_type}")
					if entity_name:
						content_parts.append(f"Name: {entity_name}")

					# Get the file content from the repo
					try:
						if self.repo_root and file_path and start_line > 0 and end_line > 0:
							repo_file_path = self.repo_root / file_path
							if repo_file_path.exists():
								with repo_file_path.open(encoding="utf-8") as f:
									file_content = f.read()
								lines = file_content.splitlines()
								if start_line <= len(lines) and end_line <= len(lines):
									code_content = "\n".join(lines[start_line - 1 : end_line])
									if language:
										content_parts.append(f"```{language}\n{code_content}\n```")
									else:
										content_parts.append(f"```\n{code_content}\n```")
					except Exception:
						logger.exception(f"Error reading file content for {file_path}")

					content = "\n\n".join(content_parts)

					formatted_results.append(
						{
							"file_path": file_path,
							"start_line": start_line,
							"end_line": end_line,
							"content": content,
							"score": r.get("score", -1.0),
						}
					)

			logger.debug(f"Semantic search returned {len(formatted_results)} results.")
			return formatted_results
		except Exception:
			logger.exception("Error retrieving context")
			return []

	async def run(self, question: str) -> AskResult:
		"""Executes one turn of the ask command, returning the answer and context."""
		logger.info(f"Processing question for session {self.session_id}: '{question}'")

		# Ensure async initialization happened (idempotent check inside)
		await self.initialize()

		if not self.pipeline:
			return AskResult(answer="Processing pipeline not available.", context=[])

		# Retrieve relevant context first
		context = await self._retrieve_context(question)

		# Format context for inclusion in prompt
		context_text = format_content_for_context(context)
		if len(context_text) > self.max_context_length:
			logger.warning(f"Context too long ({len(context_text)} chars), truncating.")
			context_text = context_text[: self.max_context_length] + "... [truncated]"

		# Construct prompt text from the context and question
		prompt = (
			f"System: {SYSTEM_PROMPT}\n\n"
			f"User: Here's my question about the codebase: {question}\n\n"
			f"Relevant context from the codebase:\n{context_text}"
		)

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

		# Get LLM config from the injected ConfigLoader
		# At this point self.config_loader is guaranteed to be a valid instance
		llm_config = self.config_loader.get.llm.model_dump()

		# Call LLM with context
		try:
			with progress_indicator("Waiting for LLM response..."):
				answer = self.llm_client.completion(
					messages=[{"role": "user", "content": prompt}],
					**llm_config,
				)
			logger.debug(f"LLM response: {answer}")

			# Update DB with answer using the dedicated client method
			if db_entry_id and answer:
				# The update_chat_response method handles its own exceptions and returns success/failure
				success = self.db_client.update_chat_response(message_id=db_entry_id, ai_response=answer)
				if not success:
					logger.warning(f"Failed to update DB entry {db_entry_id} via client method.")

			return AskResult(answer=answer, context=context)
		except Exception as e:  # Keep the outer exception for LLM call errors
			logger.exception("Error during LLM completion")
			return AskResult(answer=f"Error: {e!s}", context=context)
