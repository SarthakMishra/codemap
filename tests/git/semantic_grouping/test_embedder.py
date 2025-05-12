"""Tests for the semantic_grouping.embedder module."""

from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest

from codemap.config.config_loader import ConfigLoader
from codemap.git.diff_splitter import DiffChunk
from codemap.git.semantic_grouping.embedder import DiffEmbedder


@pytest.fixture
def mock_config_loader():
	"""Mock for ConfigLoader to avoid actual dependency."""
	mock_cl = Mock(spec=ConfigLoader)  # Using spec is good practice

	# Mock the .get property to return an object that has an .embedding attribute
	app_config_mock = Mock()
	type(mock_cl).get = PropertyMock(return_value=app_config_mock)  # Mock .get as a property

	# Mock the .embedding attribute on the object returned by .get
	embedding_settings_mock = Mock()
	app_config_mock.embedding = embedding_settings_mock

	# Set model_name on the .embedding object to be a string
	embedding_settings_mock.model_name = "mock_model_for_testing"

	# If other attributes of embedding settings are read by the real generate_embedding
	# before it hits the StaticModel.from_pretrained, they might need mocking too.
	# For example:
	# embedding_settings_mock.batch_size = 32
	# embedding_settings_mock.dimension = 1024

	return mock_cl


@pytest.fixture
def mock_generate_embeddings_batch():
	"""Mock for generate_embedding function used by DiffEmbedder."""
	with patch("codemap.git.semantic_grouping.embedder.generate_embedding") as mock_gen:
		# Configure the mock to return a fixed embedding
		mock_gen.return_value = [[0.1, 0.2, 0.3]]
		yield mock_gen


@pytest.mark.asyncio
async def test_preprocess_diff(mock_config_loader):
	"""Test diff preprocessing to clean up diff formatting."""
	# Create embedder with mock config loader
	embedder = DiffEmbedder(config_loader=mock_config_loader)

	diff_text = (
		"diff --git a/file.py b/file.py\n"
		"index abc123..def456 100644\n"
		"--- a/file.py\n"
		"+++ b/file.py\n"
		"@@ -10,5 +10,6 @@ class Example:\n"
		" def existing():\n"
		"     return True\n"
		"-def removed():\n"
		"-    pass\n"
		"+def added():\n"
		"+    return False\n"
		" # A comment\n"
	)

	result = embedder.preprocess_diff(diff_text)

	# Check that diff headers and metadata are removed
	assert "diff --git" not in result
	assert "index" not in result
	assert "--- a/" not in result
	assert "+++ b/" not in result
	assert "@@ -10,5 +10,6 @@" not in result

	# Check that +/- are removed from content lines but content remains
	assert "def existing():" in result
	assert "def removed():" in result  # Content without the -
	assert "def added():" in result  # Content without the +
	assert "# A comment" in result

	# The processed diff should have these lines, with proper indentation
	# Get actual lines for better diagnosis
	actual_lines = result.splitlines()

	# Test each line individually, preserving whitespace
	assert "def existing():" in actual_lines
	assert "    return True" in actual_lines
	assert "def removed():" in actual_lines
	assert "    pass" in actual_lines
	assert "def added():" in actual_lines
	assert "    return False" in actual_lines
	assert "# A comment" in actual_lines


@pytest.mark.asyncio
async def test_embed_chunk(mock_config_loader, mock_generate_embeddings_batch):
	"""Test embedding a diff chunk."""
	embedder = DiffEmbedder(config_loader=mock_config_loader)

	# Prepare expected processed content

	# Create a test chunk
	chunk = DiffChunk(
		files=["file.py"], content=("diff --git a/file.py b/file.py\n+def new_function():\n+    return 42\n")
	)

	# Embed the chunk
	embedding = await embedder.embed_chunk(chunk)

	# Check that the embedding has the expected shape
	assert isinstance(embedding, np.ndarray)
	assert embedding.shape == (3,)  # Shape from our mock

	# Verify the mock was called (mock_generate_embeddings_batch is the patched generate_embedding)
	mock_generate_embeddings_batch.assert_called_once()
	assert mock_generate_embeddings_batch.called

	call_args_list = mock_generate_embeddings_batch.call_args_list[0]
	# generate_embedding takes (texts: list[str], config_loader: ConfigLoader | None = None)
	# First positional arg is the list of texts
	assert isinstance(call_args_list[0][0], list)
	# Second positional arg (or kwarg) is config_loader
	if len(call_args_list[0]) > 1:
		assert call_args_list[0][1] == mock_config_loader
	elif "config_loader" in call_args_list[1]:
		assert call_args_list[1]["config_loader"] == mock_config_loader


@pytest.mark.asyncio
async def test_embed_chunk_empty_content(mock_config_loader, mock_generate_embeddings_batch):
	"""Test embedding a chunk with empty content."""
	embedder = DiffEmbedder(config_loader=mock_config_loader)

	# Create a test chunk with empty content
	chunk = DiffChunk(files=["file1.py", "file2.py"], content="")

	# Embed the chunk
	embedding = await embedder.embed_chunk(chunk)

	# Should still return an embedding
	assert isinstance(embedding, np.ndarray)

	# Just verify the mock was called - we're expecting it to use filenames
	mock_generate_embeddings_batch.assert_called_once()

	# Check that config_loader was passed
	call_args_list = mock_generate_embeddings_batch.call_args_list[0]
	if len(call_args_list[0]) > 1:
		assert call_args_list[0][1] == mock_config_loader
	elif "config_loader" in call_args_list[1]:
		assert call_args_list[1]["config_loader"] == mock_config_loader


@pytest.mark.asyncio
async def test_embed_chunks(mock_config_loader, mock_generate_embeddings_batch):
	"""Test embedding multiple chunks."""
	embedder = DiffEmbedder(config_loader=mock_config_loader)

	# Create test chunks
	chunks = [
		DiffChunk(files=["file1.py"], content="diff1"),
		DiffChunk(files=["file2.py"], content="diff2"),
	]

	# Configure mock to return multiple embeddings
	mock_generate_embeddings_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

	# Embed chunks
	results = await embedder.embed_chunks(chunks)

	# Check results
	assert len(results) == 2
	for chunk, embedding in results:
		assert chunk in chunks
		assert isinstance(embedding, np.ndarray)
		assert embedding.shape == (3,)  # Shape from our mock
