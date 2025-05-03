"""Fixtures for database tests."""

import os
import tempfile
from pathlib import Path

import pytest
from sqlmodel import Session

from codemap.db.engine import create_db_and_tables, get_engine
from codemap.db.models import ChatHistory


@pytest.fixture
def temp_db_path():
	"""Creates a temporary database path that will be cleaned up after tests."""
	fd, path = tempfile.mkstemp(suffix=".db")
	os.close(fd)  # Close file descriptor to avoid resource leaks
	db_path = Path(path)
	yield db_path
	# Clean up after tests
	if db_path.exists():
		db_path.unlink()


@pytest.fixture
def test_engine(temp_db_path):
	"""Creates a test database engine using a temporary file."""
	# Set echo=True to see SQL statements during tests (useful for debugging)
	return get_engine(temp_db_path, echo=True)


@pytest.fixture
def test_session(test_engine):
	"""Creates a test database session with tables."""
	create_db_and_tables(test_engine)
	with Session(test_engine) as session:
		yield session


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
