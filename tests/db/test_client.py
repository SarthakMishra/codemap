"""Tests for DatabaseClient."""

import json
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import select

from codemap.db.client import DatabaseClient
from codemap.db.models import ChatHistory


@pytest.fixture
def mock_config_loader():
	"""Mock the ConfigLoader for testing."""
	with patch("codemap.db.client.ConfigLoader") as mock_config_class:
		mock_config = MagicMock()
		mock_config.get.return_value = ".test_cache"
		mock_config_class.get_instance.return_value = mock_config
		yield mock_config


def test_client_init_with_explicit_path(temp_db_path):
	"""Test DatabaseClient initialization with explicit path."""
	client = DatabaseClient(db_path=str(temp_db_path))
	assert client.engine is not None

	# The engine should be configured for the right path
	assert str(temp_db_path) in str(client.engine.url)


def test_client_init_with_config_path(temp_db_path, mock_config_loader):
	"""Test DatabaseClient initialization using path from config."""
	with patch("codemap.db.client.get_engine") as mock_get_engine:
		mock_get_engine.return_value = MagicMock()
		with patch("codemap.db.client.create_db_and_tables"):
			DatabaseClient()

			# Should have called config loader
			mock_config_loader.get.assert_called_once_with("cache_dir")

			# Should have constructed path with cache_dir
			expected_path = ".test_cache/codemap.db"
			args, _ = mock_get_engine.call_args
			assert expected_path in str(args[0])


def test_client_add_chat_message(temp_db_path):
	"""Test adding a chat message."""
	client = DatabaseClient(db_path=str(temp_db_path))

	# Add a message
	result = client.add_chat_message(
		session_id="test-client-session",
		user_query="Test query from client",
		ai_response="Test response from client",
		context=json.dumps({"test": "context"}),
		tool_calls=json.dumps([{"name": "test_tool"}]),
	)

	# Should return a ChatHistory object
	assert isinstance(result, ChatHistory)
	assert result.id is not None
	assert result.session_id == "test-client-session"
	assert result.user_query == "Test query from client"
	assert result.ai_response == "Test response from client"
	assert result.context == json.dumps({"test": "context"})
	assert result.tool_calls == json.dumps([{"name": "test_tool"}])

	# Verify it's in the database
	with patch("codemap.db.client.get_session"):
		# Create a real session to verify
		from codemap.db.engine import get_session

		with get_session(client.engine) as session:
			db_result = session.exec(select(ChatHistory).where(ChatHistory.id == result.id)).first()

			assert db_result is not None
			assert db_result.user_query == "Test query from client"


def test_client_get_chat_history(temp_db_path):
	"""Test retrieving chat history."""
	client = DatabaseClient(db_path=str(temp_db_path))

	# Add a few messages
	session_id = "test-history-session"
	client.add_chat_message(session_id=session_id, user_query="Query 1")
	client.add_chat_message(session_id=session_id, user_query="Query 2")
	client.add_chat_message(session_id=session_id, user_query="Query 3")

	# Different session
	client.add_chat_message(session_id="other-session", user_query="Other query")

	# Get history for our session
	history = client.get_chat_history(session_id)

	# Should have 3 items for this session
	assert len(history) == 3

	# Should be in order (oldest first)
	assert history[0].user_query == "Query 1"
	assert history[1].user_query == "Query 2"
	assert history[2].user_query == "Query 3"

	# Test limit
	limited_history = client.get_chat_history(session_id, limit=2)
	assert len(limited_history) == 2
	assert limited_history[0].user_query == "Query 1"
	assert limited_history[1].user_query == "Query 2"


def test_client_error_handling(temp_db_path):
	"""Test error handling in client methods."""
	client = DatabaseClient(db_path=str(temp_db_path))

	# Test add_chat_message error
	with patch("codemap.db.client.get_session") as mock_get_session:
		mock_session = MagicMock()
		mock_session.__enter__.return_value = mock_session
		mock_session.add.side_effect = Exception("Test exception")
		mock_get_session.return_value = mock_session

		with pytest.raises(Exception, match="Test exception") as exc_info:
			client.add_chat_message(session_id="error-session", user_query="Error query")
		assert "Test exception" in str(exc_info.value)

	# Test get_chat_history error
	with patch("codemap.db.client.get_session") as mock_get_session:
		mock_session = MagicMock()
		mock_session.__enter__.return_value = mock_session
		mock_session.exec.side_effect = Exception("Query exception")
		mock_get_session.return_value = mock_session

		with pytest.raises(Exception, match="Query exception") as exc_info:
			client.get_chat_history("error-session")
		assert "Query exception" in str(exc_info.value)
