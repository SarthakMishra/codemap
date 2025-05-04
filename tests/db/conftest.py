"""Fixtures for database tests."""

import asyncio

import pytest
from sqlmodel import Session

from codemap.db.engine import create_db_and_tables, get_engine
from codemap.db.models import ChatHistory


@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for session scope fixtures."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture(scope="session")
async def test_engine():
	"""
	Creates a test database engine for PostgreSQL.

	Relies on get_engine to ensure the container is running. Scope is
	session to avoid starting/stopping container repeatedly.

	"""
	# Set echo=True to see SQL statements during tests (useful for debugging)
	return await get_engine(echo=True)
	# Optionally add logic here to ensure container is ready if get_engine doesn't suffice
	# For now, assume get_engine handles it.
	# No explicit cleanup needed here if engine is managed externally/by docker_utils


@pytest.fixture
async def test_session(test_engine):
	"""Creates a test database session with tables."""
	# Ensure tables exist for this engine instance (idempotent)
	# create_db_and_tables is sync, call it directly
	create_db_and_tables(test_engine)

	# Use a synchronous session for now, matching the sync create_engine in get_engine
	# If get_engine is changed to create_async_engine, this needs to change too.
	with Session(test_engine) as session:
		yield session
		# Rollback any changes made during the test
		session.rollback()


@pytest.fixture
def sample_chat_history():
	"""Returns a sample ChatHistory object for testing."""
	return ChatHistory(
		session_id="test-session-123",
		user_query="What is the meaning of life?",
		ai_response="42",
		context='{"file": "universe.py"}',
		tool_calls='[{"name": "lookup", "args": {"topic": "meaning of life"}}]',
	)
