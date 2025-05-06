"""Tests for the semantic_grouping.embedder module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from codemap.git.diff_splitter import DiffChunk
from codemap.git.semantic_grouping.embedder import DiffEmbedder


@pytest.fixture
def mock_sentence_transformer():
	"""Mock for SentenceTransformer to avoid actual dependency."""
	with patch("sentence_transformers.SentenceTransformer") as mock_st:
		# Configure the mock model to return a fixed embedding
		mock_model = Mock()
		mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
		mock_st.return_value = mock_model
		yield mock_st


def test_preprocess_diff():
	"""Test diff preprocessing to clean up diff formatting."""
	# Create a mock SentenceTransformer to avoid actual dependency
	with patch("sentence_transformers.SentenceTransformer"):
		embedder = DiffEmbedder()

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


def test_embed_chunk(mock_sentence_transformer):
	"""Test embedding a diff chunk."""
	embedder = DiffEmbedder()

	# Create a test chunk
	chunk = DiffChunk(
		files=["file.py"], content=("diff --git a/file.py b/file.py\n+def new_function():\n+    return 42\n")
	)

	# Embed the chunk
	embedding = embedder.embed_chunk(chunk)

	# Check that the embedding has the expected shape
	assert isinstance(embedding, np.ndarray)
	assert embedding.shape == (3,)  # Shape from our mock

	# Verify that encode was called with preprocessed content
	mock_model = mock_sentence_transformer.return_value
	args, _ = mock_model.encode.call_args
	processed_text = args[0]

	# The processed text should have the +/- removed
	assert "def new_function():" in processed_text
	assert "return 42" in processed_text


def test_embed_chunk_empty_content(mock_sentence_transformer):
	"""Test embedding a chunk with empty content."""
	embedder = DiffEmbedder()

	# Create a test chunk with empty content
	chunk = DiffChunk(files=["file1.py", "file2.py"], content="")

	# Embed the chunk
	embedding = embedder.embed_chunk(chunk)

	# Should still return an embedding
	assert isinstance(embedding, np.ndarray)

	# It should use the filenames when content is empty
	mock_model = mock_sentence_transformer.return_value
	args, _ = mock_model.encode.call_args
	assert args[0] == "file1.py file2.py"


def test_embed_chunks(mock_sentence_transformer):
	"""Test embedding multiple chunks."""
	embedder = DiffEmbedder()

	# Create test chunks
	chunks = [
		DiffChunk(files=["file1.py"], content="diff1"),
		DiffChunk(files=["file2.py"], content="diff2"),
	]

	# Embed chunks
	results = embedder.embed_chunks(chunks)

	# Check results
	assert len(results) == 2
	for chunk, embedding in results:
		assert chunk in chunks
		assert isinstance(embedding, np.ndarray)
		assert embedding.shape == (3,)  # Shape from our mock
