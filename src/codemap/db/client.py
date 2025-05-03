"""Client interface for interacting with the database in CodeMap."""

import logging
from pathlib import Path

from sqlalchemy import asc
from sqlmodel import select

from codemap.utils.config_loader import ConfigLoader

from .engine import create_db_and_tables, get_engine, get_session
from .models import ChatHistory

logger = logging.getLogger(__name__)


class DatabaseClient:
	"""Provides high-level methods to interact with the CodeMap database."""

	def __init__(self, db_path: str | None = None) -> None:
		"""
		Initializes the database client.

		If db_path is not provided, it reads the cache_dir from the configuration
		and constructs the path as cache_dir/codemap.db.
		It also ensures the database engine is initialized and tables are created.

		Args:
		    db_path (Optional[str]): Explicit path to the database file. Overrides config if provided.

		"""
		if db_path is None:
			# Use ConfigLoader to get the cache_dir
			config = ConfigLoader.get_instance()
			cache_dir = config.get("cache_dir")
			if not cache_dir:
				msg = "'cache_dir' not found in configuration. Cannot determine default database path."
				raise ValueError(msg)
			# Construct path relative to cache_dir
			db_path_obj = Path(cache_dir) / "codemap.db"
		else:
			# Use the explicitly provided path
			db_path_obj = Path(db_path)

		self.engine = get_engine(db_path_obj)
		create_db_and_tables(self.engine)  # Ensure tables exist on initialization
		logger.info(f"Database client initialized using path: {db_path_obj}")

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

	def get_chat_history(self, session_id: str, limit: int = 50) -> list[ChatHistory]:
		"""
		Retrieves chat history for a session, ordered chronologically.

		Args:
		    session_id (str): The session identifier.
		    limit (int): The maximum number of messages to return.

		Returns:
		    List[ChatHistory]: A list of chat history records.

		"""
		statement = (
			select(ChatHistory)
			.where(ChatHistory.session_id == session_id)
			# Without this ignore, pyright incorrectly complains that ChatHistory.timestamp is
			# a datetime object rather than a SQLAlchemy Column
			.order_by(asc(ChatHistory.timestamp))  # type: ignore # noqa: PGH003
			.limit(limit)
		)
		try:
			with get_session(self.engine) as session:
				results = session.exec(statement).all()
				logger.debug(f"Retrieved {len(results)} messages for session {session_id}.")
				return list(results)
		except Exception:
			logger.exception("Error retrieving chat history")
			raise
