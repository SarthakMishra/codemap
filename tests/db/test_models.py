"""Tests for database models."""

from datetime import UTC, datetime

import pytest
from sqlmodel import select

from codemap.db.engine import get_engine, get_session
from codemap.db.models import ChatHistory
from tests.conftest import skip_db_tests


def test_chat_history_model():
	"""Test that the ChatHistory model has the expected attributes."""
	# Create a model instance
	chat = ChatHistory(session_id="test-session", user_query="Test query", ai_response="Test response")

	# Check attributes
	assert chat.session_id == "test-session"
	assert chat.user_query == "Test query"
	assert chat.ai_response == "Test response"
	assert chat.context is None
	assert chat.tool_calls is None
	assert chat.id is None  # ID will be set by database


def test_chat_history_timestamp_default():
	"""Test that timestamp gets a default value."""
	before = datetime.now(UTC)
	chat = ChatHistory(session_id="test-session", user_query="Test query")
	after = datetime.now(UTC)

	# Check that timestamp is between before and after
	assert chat.timestamp is not None
	# Convert to UTC for comparison if needed
	chat_time = chat.timestamp.replace(tzinfo=UTC) if chat.timestamp.tzinfo is None else chat.timestamp  # pylint: disable=no-member

	assert before <= chat_time <= after


@pytest.mark.asyncio
@skip_db_tests
async def test_chat_history_db_interaction():
	"""Test ChatHistory database interactions."""
	# Create a fresh sample chat history
	sample_chat = ChatHistory(
		session_id="test-db-session",
		user_query="What is the meaning of life?",
		ai_response="42",
		context='{"file": "universe.py"}',
		tool_calls='[{"name": "lookup", "args": {"topic": "meaning of life"}}]',
	)

	# Get a fresh engine
	engine = await get_engine()

	# Create tables before testing
	from sqlmodel import SQLModel

	SQLModel.metadata.create_all(engine)

	# Use the engine directly
	with get_session(engine) as session:
		# Add to session
		session.add(sample_chat)
		session.commit()

		# ID should be set after commit
		assert sample_chat.id is not None

		# Query back
		result = session.exec(select(ChatHistory).where(ChatHistory.session_id == "test-db-session")).first()

		assert result is not None
		assert result.user_query == "What is the meaning of life?"
		assert result.ai_response == "42"

		# Test update
		result.ai_response = "The answer is 42"
		session.commit()

		# Query again
		updated = session.exec(select(ChatHistory).where(ChatHistory.id == result.id)).first()

		# Make sure updated is not None before checking its attributes
		assert updated is not None
		assert updated.ai_response == "The answer is 42"

		# Test delete
		session.delete(result)
		session.commit()

		# Query again - should be gone
		deleted = session.exec(select(ChatHistory).where(ChatHistory.id == result.id)).first()

		assert deleted is None
