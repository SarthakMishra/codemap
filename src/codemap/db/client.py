"""Client interface for interacting with the database in CodeMap."""

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from sqlalchemy import asc
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import select

from .engine import create_db_and_tables, get_engine, get_session
from .models import ChatHistory

if TYPE_CHECKING:
	from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class DatabaseClient:
	"""Provides high-level methods to interact with the CodeMap database."""

	def __init__(self) -> None:
		"""
		Initializes the database client.

		It sets up the client in an uninitialized state. The actual
		initialization needs to be performed by calling the async initialize()
		method or waiting for _initialize_engine_if_needed to run when
		required.

		"""
		self.engine: Engine | None = None  # Initialize engine as None
		self.initialized = False  # Flag to track initialization status
		self._init_task: asyncio.Task[None] | None = None  # Store reference to initialization task

		# Initialize engine in event loop if possible
		try:
			if asyncio.get_event_loop().is_running():
				self._init_task = asyncio.create_task(self.initialize())
			else:
				# In sync context, create a new event loop to initialize
				loop = asyncio.new_event_loop()
				loop.run_until_complete(self.initialize())
				loop.close()
		except RuntimeError:
			# No event loop available, will initialize on first use
			logger.debug("No event loop available during DatabaseClient init, will initialize on demand")

	async def initialize(self) -> None:
		"""
		Asynchronously initialize the database client.

		This should be called after creating the client and before using it.

		"""
		await self._initialize_engine()
		self.initialized = True
		logger.info("Database client successfully initialized")

	async def _initialize_engine(self) -> None:
		"""Asynchronously gets the engine and creates tables."""
		if self.engine is None:
			try:
				self.engine = await get_engine()  # Await the async function
				# create_db_and_tables is synchronous, run it after engine is ready
				create_db_and_tables(self.engine)
				logger.info("Database client initialized with PostgreSQL engine.")
			except RuntimeError:
				logger.exception("Failed to initialize database engine")
				# Decide how to handle this error - maybe raise, maybe set engine to None?
				# For now, re-raising to make the failure explicit.
				raise
			except Exception:
				logger.exception("An unexpected error occurred during database initialization")
				raise

	async def _initialize_engine_if_needed(self) -> None:
		"""Ensure engine is initialized before use."""
		if not self.initialized or self.engine is None:
			await self.initialize()

	async def cleanup(self) -> None:
		"""
		Asynchronously cleanup the database client resources.

		This should be called before discarding the client.

		"""
		if self.engine:
			# No need to close Engine in SQLAlchemy 2.0, but dispose will close connections
			if hasattr(self.engine, "dispose"):
				self.engine.dispose()
			self.engine = None
		self.initialized = False
		logger.info("Database client cleaned up")

	# Ensure engine is initialized before DB operations
	async def _ensure_engine_initialized(self) -> None:
		"""Ensures the database engine is properly initialized.

		This method checks if the engine is initialized and attempts to initialize it if not.
		If initialization fails, it logs an error and raises a RuntimeError.

		Raises:
		    RuntimeError: If database client initialization fails after attempting to initialize.

		Returns:
		    None: This method doesn't return anything but ensures engine is ready for use.
		"""
		if not self.initialized or self.engine is None:
			await self._initialize_engine_if_needed()
			if not self.initialized or self.engine is None:
				msg = "Database client initialization failed."
				logger.error(msg)
				raise RuntimeError(msg)

	def add_chat_message(
		self,
		session_id: str,
		user_query: str,
		ai_response: str | None = None,
		context: str | None = None,
		tool_calls: str | None = None,
	) -> ChatHistory:
		"""
		Adds a chat message to the history.

		Args:
		    session_id (str): The session identifier.
		    user_query (str): The user's query.
		    ai_response (Optional[str]): The AI's response.
		    context (Optional[str]): JSON string of context used.
		    tool_calls (Optional[str]): JSON string of tool calls made.

		Returns:
		    ChatHistory: The newly created chat history record.

		"""
		# Ensure engine is initialized - run in a new event loop if needed
		if not self.initialized or self.engine is None:
			loop = asyncio.new_event_loop()
			try:
				loop.run_until_complete(self._ensure_engine_initialized())
			finally:
				loop.close()

		if self.engine is None:
			# This should ideally not happen if _ensure_engine_initialized worked
			msg = "Database engine is not initialized after check."
			logger.error(msg)
			raise RuntimeError(msg)

		chat_entry = ChatHistory(
			session_id=session_id,
			user_query=user_query,
			ai_response=ai_response,
			context=context,
			tool_calls=tool_calls,
		)
		try:
			with get_session(self.engine) as session:
				session.add(chat_entry)
				session.commit()
				session.refresh(chat_entry)
				logger.debug(f"Added chat message for session {session_id} to DB (ID: {chat_entry.id}).")
				return chat_entry
		except Exception:
			logger.exception("Error adding chat message")
			raise  # Re-raise after logging

	def get_chat_history(self, session_id: str, limit: int = 50) -> Sequence[ChatHistory]:
		"""
		Retrieves chat history for a session, ordered chronologically.

		Args:
		    session_id (str): The session identifier.
		    limit (int): The maximum number of messages to return.

		Returns:
		    Sequence[ChatHistory]: A sequence of chat history records.

		"""
		# Ensure engine is initialized - run in a new event loop if needed
		if not self.initialized or self.engine is None:
			loop = asyncio.new_event_loop()
			try:
				loop.run_until_complete(self._ensure_engine_initialized())
			finally:
				loop.close()

		if self.engine is None:
			# This should ideally not happen if _ensure_engine_initialized worked
			msg = "Database engine is not initialized after check."
			logger.error(msg)
			raise RuntimeError(msg)

		statement = (
			select(ChatHistory)
			.where(ChatHistory.session_id == session_id)
			# Using type ignore as the linter incorrectly flags the type
			.order_by(asc(ChatHistory.timestamp))  # type: ignore[arg-type]
			.limit(limit)
		)
		try:
			with get_session(self.engine) as session:
				results: Sequence[ChatHistory] = session.exec(statement).all()  # type: ignore[assignment]
				logger.debug(f"Retrieved {len(results)} messages for session {session_id}.")
				return results
		except Exception:
			logger.exception("Error retrieving chat history")
			raise

	def update_chat_response(self, message_id: int, ai_response: str) -> bool:
		"""
		Updates the AI response for a specific chat message.

		Args:
		        message_id (int): The primary key ID of the chat message to update.
		        ai_response (str): The new AI response text.

		Returns:
		        bool: True if the update was successful, False otherwise.

		"""
		# Ensure engine is initialized - run in a new event loop if needed
		if not self.initialized or self.engine is None:
			loop = asyncio.new_event_loop()
			try:
				loop.run_until_complete(self._ensure_engine_initialized())
			finally:
				loop.close()

		if self.engine is None:
			logger.error("Cannot update chat response: Database engine not initialized.")
			return False

		try:
			with get_session(self.engine) as session:
				db_entry = session.get(ChatHistory, message_id)  # type: ignore[arg-type]
				if db_entry:
					db_entry.ai_response = ai_response
					session.commit()
					logger.debug(f"Updated DB entry {message_id} with AI response.")
					return True
				logger.warning(f"Chat message with ID {message_id} not found for update.")
				return False
		except SQLAlchemyError:
			logger.exception(f"Database error updating chat response for message ID {message_id}")
			return False
		except Exception:
			logger.exception(f"Unexpected error updating chat response for message ID {message_id}")
			return False
