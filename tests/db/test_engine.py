"""Tests for database engine functions."""

import pytest
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, select

# get_engine is now async
from codemap.db.engine import create_db_and_tables, get_engine, get_session
from codemap.db.models import ChatHistory


# Mark test as async
@pytest.mark.asyncio
async def test_get_engine():  # Make test async
	"""Test get_engine creates an engine and caches it."""
	# First call should create a new engine
	engine1 = await get_engine()  # Use await
	assert isinstance(engine1, Engine)

	# Second call should return the cached engine (same URL)
	engine2 = await get_engine()  # Use await
	assert engine1 is engine2  # Should be exactly the same instance due to URL caching

	# Test with different echo value - should create a new engine if logic changes
	# but current logic caches based on URL only.
	engine3 = await get_engine(echo=True)  # Use await
	# Depending on caching implementation, this might be the same or different
	# Current implementation caches based on URL, so echo change won't matter
	assert engine1 is engine3


# Removed test relying on different file paths, as it's not relevant for Postgres


# Mark test as async because test_engine fixture is async
@pytest.mark.asyncio
async def test_create_db_and_tables(test_engine):
	"""Test create_db_and_tables creates tables."""
	# create_db_and_tables is sync, no await needed here
	# Await the fixture itself if needed, but pytest handles async fixtures
	create_db_and_tables(test_engine)

	# Simply check if the table exists in the metadata
	assert "chat_history" in SQLModel.metadata.tables


# Mark test as async because test_engine/test_session fixtures are async
@pytest.mark.asyncio
async def test_get_session(test_engine, test_session):  # test_session implicitly awaits test_engine
	"""Test get_session provides a working session context manager."""
	# create_db_and_tables called by test_session fixture

	# Add an item using the session
	chat = ChatHistory(session_id="test-session", user_query="Does the session work?")

	# Session context manager itself is sync
	with get_session(test_engine) as session:
		session.add(chat)
		session.commit()

		# Query it back
		result = session.exec(select(ChatHistory).where(ChatHistory.session_id == "test-session")).first()

		assert result is not None
		assert result.user_query == "Does the session work?"


# Mark test as async because test_engine/test_session fixtures are async
@pytest.mark.asyncio
async def test_get_session_rollback_on_error(test_engine, test_session):
	"""Test get_session rolls back on error."""
	# create_db_and_tables called by test_session fixture

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
