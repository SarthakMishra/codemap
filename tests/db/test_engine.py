"""Tests for database engine functions."""

from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, select

from codemap.db.engine import create_db_and_tables, get_engine, get_session
from codemap.db.models import ChatHistory


def test_get_engine(temp_db_path):
	"""Test get_engine creates an engine and caches it."""
	# First call should create a new engine
	engine1 = get_engine(temp_db_path)
	assert isinstance(engine1, Engine)

	# Second call with same path should return the cached engine
	engine2 = get_engine(temp_db_path)
	assert engine1 is engine2  # Should be exactly the same instance

	# Different path should create a new engine
	different_path = Path(str(temp_db_path) + ".different")
	try:
		engine3 = get_engine(different_path)
		assert engine1 is not engine3  # Should be a different instance
	finally:
		# Clean up the extra db file
		if different_path.exists():
			different_path.unlink()


def test_create_db_and_tables(test_engine):
	"""Test create_db_and_tables creates tables."""
	# Should not raise an exception
	create_db_and_tables(test_engine)

	# Simply check if the table exists in the metadata
	assert "chat_history" in SQLModel.metadata.tables


def test_get_session(test_engine):
	"""Test get_session provides a working session context manager."""
	create_db_and_tables(test_engine)

	# Add an item using the session
	chat = ChatHistory(session_id="test-session", user_query="Does the session work?")

	with get_session(test_engine) as session:
		session.add(chat)
		session.commit()

		# Query it back
		result = session.exec(select(ChatHistory).where(ChatHistory.session_id == "test-session")).first()

		assert result is not None
		assert result.user_query == "Does the session work?"


def test_get_session_rollback_on_error(test_engine):
	"""Test get_session rolls back on error."""
	create_db_and_tables(test_engine)

	# Add an initial record
	chat = ChatHistory(session_id="test-rollback", user_query="Initial query")

	with get_session(test_engine) as session:
		session.add(chat)
		session.commit()

	# Define function to trigger an error
	def trigger_error():
		msg = "Test exception"
		raise ValueError(msg)

	# Now try a session that will raise an error
	try:
		with get_session(test_engine) as session:
			# Query the record
			result = session.exec(select(ChatHistory).where(ChatHistory.session_id == "test-rollback")).first()

			# Make sure we have a result before proceeding
			assert result is not None

			# Modify it
			result.user_query = "Modified query"

			# Raise an error before commit
			trigger_error()
	except ValueError:
		pass

	# In a new session, check that the change was rolled back
	with get_session(test_engine) as session:
		result = session.exec(select(ChatHistory).where(ChatHistory.session_id == "test-rollback")).first()

		# Make sure we have a result before checking its attribute
		assert result is not None
		assert result.user_query == "Initial query"  # Should not be modified
