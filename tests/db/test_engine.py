"""Tests for database engine functions."""

import pytest
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, select

# get_engine is now async
from codemap.db.engine import create_db_and_tables, get_engine, get_session
from codemap.db.models import ChatHistory
from tests.conftest import skip_db_tests


# Mark test as async
@pytest.mark.asyncio
@skip_db_tests
async def test_get_engine():
	"""Test get_engine creates an engine and caches it."""
	# First call should create a new engine
	engine1 = await get_engine()
	assert isinstance(engine1, Engine)

	# Second call should return the cached engine (same URL)
	engine2 = await get_engine()
	assert engine1 is engine2  # Should be exactly the same instance due to URL caching

	# Test with different echo value - should create a new engine if logic changes
	# but current logic caches based on URL only.
	engine3 = await get_engine(echo=True)
	# Depending on caching implementation, this might be the same or different
	# Current implementation caches based on URL, so echo change won't matter
	assert engine1 is engine3


# Separate test for create_db_and_tables that doesn't rely on fixtures
@pytest.mark.asyncio
@skip_db_tests
async def test_create_db_and_tables():
	"""Test create_db_and_tables creates tables."""
	# Get a fresh engine for this test
	engine = await get_engine()

	# Call create_db_and_tables with the actual engine
	create_db_and_tables(engine)

	# Check if the table exists in the metadata
	assert "chat_history" in SQLModel.metadata.tables


# Test get_session independently without fixtures
@pytest.mark.asyncio
@skip_db_tests
async def test_get_session():
	"""Test get_session provides a working session context manager."""
	# Get a fresh engine for this test
	engine = await get_engine()

	# Add an item using get_session
	chat = ChatHistory(session_id="test-session-get", user_query="Does the session work?")

	with get_session(engine) as session:
		session.add(chat)
		session.commit()

		# Query it back using the session
		result = session.exec(select(ChatHistory).where(ChatHistory.session_id == "test-session-get")).first()

		assert result is not None
		assert result.user_query == "Does the session work?"


# Test rollback functionality independently without fixtures
@pytest.mark.asyncio
@skip_db_tests
async def test_get_session_rollback_on_error():
	"""Test get_session rolls back on error."""
	# Get a fresh engine for this test
	engine = await get_engine()

	# Ensure tables are created
	create_db_and_tables(engine)

	# Add an initial record
	chat = ChatHistory(session_id="test-rollback-error", user_query="Initial query")

	with get_session(engine) as session:
		session.add(chat)
		session.commit()

	# Define function to trigger an error
	def trigger_error():
		msg = "Test exception"
		raise ValueError(msg)

	# Now try a session that will raise an error
	try:
		with get_session(engine) as session:
			# Query the record
			result = session.exec(select(ChatHistory).where(ChatHistory.session_id == "test-rollback-error")).first()

			# Make sure we have a result before proceeding
			assert result is not None

			# Modify it
			result.user_query = "Modified query"

			# Raise an error before commit
			trigger_error()
	except ValueError:
		pass

	# In a new session, check that the change was rolled back
	with get_session(engine) as session:
		result = session.exec(select(ChatHistory).where(ChatHistory.session_id == "test-rollback-error")).first()

		# Make sure we have a result before checking its attribute
		assert result is not None
		assert result.user_query == "Initial query"  # Should not be modified
