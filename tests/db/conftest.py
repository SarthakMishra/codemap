"""Fixtures for database tests."""

import pytest
import pytest_asyncio
from sqlmodel import Session

from codemap.db.engine import create_db_and_tables, get_engine
from codemap.db.models import ChatHistory

# Configure pytest-asyncio to use session-scoped event loops by default
# This replaces the custom event_loop fixture
pytestmark = pytest.mark.asyncio(scope="session")


@pytest_asyncio.fixture(scope="session")
async def test_engine():
	"""
	Creates a test database engine for PostgreSQL.

	Relies on get_engine to ensure the container is running. Scope is
	session to avoid starting/stopping container repeatedly.

	"""
	# Set echo=True to see SQL statements during tests (useful for debugging)
	engine = await get_engine(echo=True)
	# Create tables once the engine is ready
	create_db_and_tables(engine)
	yield engine
	# Clean up any resources if needed
	if hasattr(engine, "dispose"):
		engine.dispose()


@pytest.fixture
def test_session(test_engine):
	"""Creates a test database session with tables."""
	# We need to await the test_engine fixture in a synchronous context
	# This will get the actual engine that was returned by the async fixture
	# Create a session using the engine
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
