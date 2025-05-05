"""Tests for the batch processor module."""

from unittest.mock import MagicMock, patch

from codemap.git.diff_splitter import DiffChunk
from codemap.git.semantic_grouping import SemanticGroup
from codemap.git.semantic_grouping.batch_processor import batch_generate_messages
from codemap.utils.config_loader import ConfigLoader


class TestBatchProcessor:
	"""Test cases for the batch processor module."""

	def test_batch_generate_messages_empty_list(self):
		"""Test batch_generate_messages with an empty list of groups."""
		config_loader = ConfigLoader()
		groups = []
		result = batch_generate_messages(groups, "test template", config_loader)
		assert result == []

	@patch("litellm.batch_completion")
	async def test_batch_generate_messages(self, mock_batch_completion):
		"""Test batch_generate_messages with mock batch_completion."""
		# Create mock response
		mock_response = [
			MagicMock(choices=[MagicMock(message=MagicMock(content="test message 1"))]),
			MagicMock(choices=[MagicMock(message=MagicMock(content="test message 2"))]),
		]
		mock_batch_completion.return_value = mock_response

		# Create test groups
		groups = [
			SemanticGroup(
				chunks=[DiffChunk(files=["file1.py"], content="content1")], files=["file1.py"], content="content1"
			),
			SemanticGroup(
				chunks=[DiffChunk(files=["file2.py"], content="content2")], files=["file2.py"], content="content2"
			),
		]

		# Create mock config_loader
		config_loader = MagicMock()
		config_loader.get.return_value = {"use_lod_context": True, "model": "test-model"}
		config_loader.get_api_key_for_model.return_value = "test-api-key"

		# Call the function
		result = batch_generate_messages(groups, "test template", config_loader)

		# Verify results
		assert len(result) == 2
		assert result[0].message == "test message 1"
		assert result[1].message == "test message 2"

		# Verify mock was called
		mock_batch_completion.assert_called_once()

	@patch("litellm.batch_completion")
	async def test_batch_generate_messages_exception(self, mock_batch_completion):
		"""Test batch_generate_messages with an exception from batch_completion."""
		# Make the mock raise an exception
		mock_batch_completion.side_effect = Exception("Test exception")

		# Create test groups
		groups = [
			SemanticGroup(
				chunks=[DiffChunk(files=["file1.py"], content="content1")], files=["file1.py"], content="content1"
			),
			SemanticGroup(
				chunks=[DiffChunk(files=["file2.py"], content="content2")], files=["file2.py"], content="content2"
			),
		]

		# Create mock config_loader
		config_loader = MagicMock()
		config_loader.get.return_value = {"use_lod_context": True, "model": "test-model"}
		config_loader.get_api_key_for_model.return_value = "test-api-key"

		# Call the function - should not raise exception
		result = batch_generate_messages(groups, "test template", config_loader)

		# Verify results - should have fallback messages
		assert len(result) == 2
		assert result[0].message == "update: changes to 1 files"
		assert result[1].message == "update: changes to 1 files"
