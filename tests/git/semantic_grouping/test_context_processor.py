"""Tests for the context processor module."""

from unittest.mock import MagicMock, patch

from codemap.git.diff_splitter import DiffChunk
from codemap.git.semantic_grouping.context_processor import (
	estimate_tokens,
	format_chunk,
	format_lod_entity,
	prioritize_chunks,
	process_chunks_with_lod,
	truncate_context,
)
from codemap.processor.lod import EntityType, LODEntity, LODLevel


class TestContextProcessor:
	"""Test cases for the context processor module."""

	def test_format_chunk(self):
		"""Test formatting a chunk."""
		# Create a simple diff chunk
		chunk = DiffChunk(
			files=["test.py", "other.py"],
			content="@@ -1,3 +1,4 @@\n+def test():\n    pass\n",
			description="Add test function",
		)

		# Format the chunk
		result = format_chunk(chunk)

		# Verify the result
		assert "## Files" in result
		assert "- test.py" in result
		assert "- other.py" in result
		assert "### Changes" in result
		assert "```diff" in result
		assert "+def test():" in result

	def test_estimate_tokens(self):
		"""Test token estimation."""
		# Test with various string lengths
		assert estimate_tokens("") == 0
		assert estimate_tokens("This is a test") == 3  # 14 chars -> 3 tokens
		assert estimate_tokens("A" * 100) == 25  # 100 chars -> 25 tokens

	def test_truncate_context(self):
		"""Test context truncation."""
		# Create a long context
		context = "\n\n".join([f"Chunk {i}\n\nThis is content." for i in range(20)])

		# Truncate to various sizes
		result_small = truncate_context(context, 10)
		result_medium = truncate_context(context, 50)

		# Verify the results
		assert len(result_small) < len(context)
		assert "[...TRUNCATED...]" in result_small
		assert len(result_medium) < len(context)
		assert "[...TRUNCATED...]" in result_medium

		# Test with context that doesn't need truncation
		small_context = "This is a small context"
		result = truncate_context(small_context, 100)
		assert result == small_context

	def test_prioritize_chunks(self):
		"""Test chunk prioritization."""
		# Create a variety of chunks
		chunks = [
			DiffChunk(files=["file1.py"], content="Python content"),
			DiffChunk(files=["file2.txt"], content="Text content"),
			DiffChunk(files=["file3.py", "file4.py"], content="Multiple Python files"),
			DiffChunk(files=["file5.jpg"], content="Binary content"),
		]

		# Prioritize with a limit
		result = prioritize_chunks(chunks, 2)

		# Verify the results
		assert len(result) == 2
		# Python files should be prioritized
		assert any(chunk.files[0].endswith(".py") for chunk in result)

	def test_format_lod_entity(self):
		"""Test formatting an LOD entity."""
		# Create a simple LOD entity
		entity = LODEntity(
			name="TestClass",
			entity_type=EntityType.CLASS,
			start_line=1,
			end_line=10,
			signature="class TestClass:",
			docstring="A test class",
			language="python",
		)

		# Add a child entity
		child = LODEntity(
			name="test_method",
			entity_type=EntityType.METHOD,
			start_line=3,
			end_line=5,
			signature="def test_method(self):",
			docstring="A test method",
			language="python",
		)
		entity.children.append(child)

		# Format with STRUCTURE level
		structure_result = format_lod_entity(entity, "test.py", LODLevel.STRUCTURE)

		# Verify the structure result
		assert "## test.py" in structure_result
		assert "**CLASS**: `TestClass`" in structure_result
		assert "```" in structure_result
		assert "class TestClass:" in structure_result
		assert "**METHOD**: `test_method`" in structure_result

		# Format with SIGNATURES level
		signatures_result = format_lod_entity(entity, "test.py", LODLevel.SIGNATURES)

		# Verify the signatures result
		assert "## test.py" in signatures_result
		assert "**CLASS**: `TestClass` - `class TestClass:`" in signatures_result
		assert "**METHOD**: `test_method` - `def test_method(self):`" in signatures_result

	@patch("codemap.git.semantic_grouping.context_processor.LODGenerator")
	@patch("codemap.git.semantic_grouping.context_processor.Path")
	@patch("codemap.git.semantic_grouping.context_processor.MAX_SIMPLE_CHUNKS", 1)  # Force LOD processing path
	def test_process_chunks_with_lod(self, mock_path_cls, mock_lod_generator_cls):
		"""Test LOD processing of chunks."""
		# Create mock LOD generator
		mock_lod_generator = MagicMock()
		mock_lod_generator_cls.return_value = mock_lod_generator

		# Create a mock LOD entity
		mock_entity = MagicMock()
		mock_entity.name = "TestClass"
		mock_entity.entity_type = EntityType.CLASS
		mock_entity.signature = "class TestClass:"
		mock_entity.children = []

		# Set up the generate_lod mock
		mock_lod_generator.generate_lod.return_value = mock_entity

		# Set up Path mock
		mock_path_instance = MagicMock()
		mock_path_instance.exists.return_value = True
		mock_path_cls.return_value = mock_path_instance

		# Create chunks - more than MAX_SIMPLE_CHUNKS to trigger LOD processing
		chunks = [
			DiffChunk(files=["test.py"], content="Python content"),
			DiffChunk(files=["other.py"], content="More Python content"),
			DiffChunk(files=["third.py"], content="Even more Python content"),
			DiffChunk(files=["fourth.py"], content="Yet more Python content"),
		]

		# Mock format_lod_entity to return a known string
		with patch("codemap.git.semantic_grouping.context_processor.format_lod_entity") as mock_format_lod:
			mock_format_lod.return_value = "Formatted LOD entity"

			# Process chunks
			result = process_chunks_with_lod(chunks, 1000)

			# Verify result contains our mock formatting
			assert "Formatted LOD entity" in result

			# Verify LOD generator was called
			assert mock_lod_generator.generate_lod.called

			# Result should be a string
			assert isinstance(result, str)
			assert result  # Not empty

	def test_small_chunks_use_regular_formatting(self):
		"""Test that small chunk sets use regular formatting."""
		# Create a small set of chunks
		chunks = [
			DiffChunk(files=["test.py"], content="Python content"),
			DiffChunk(files=["other.py"], content="More Python content"),
		]

		# Mock the LOD processing to ensure it's not called
		with patch("codemap.git.semantic_grouping.context_processor.LODGenerator") as mock_generator:
			# Process chunks
			result = process_chunks_with_lod(chunks, 1000)

			# Result should be a string
			assert isinstance(result, str)
			assert "## Files" in result  # Regular formatting contains this header

			# LODGenerator should not have been instantiated
			mock_generator.assert_not_called()

	def test_process_chunks_with_no_existing_files(self):
		"""Test processing chunks where files don't exist."""
		# Create chunks with files that don't exist
		chunks = [
			DiffChunk(files=["nonexistent.py"], content="Python content"),
		]

		# Process chunks
		result = process_chunks_with_lod(chunks, 1000)

		# Result should be a string
		assert isinstance(result, str)
		assert "## Files" in result  # Regular formatting contains this header
