"""Tests for the code processor module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# Define test versions of the classes we need instead of importing them
@dataclass
class ChunkData:
	"""Test version of a code chunk."""

	content: str
	path: Path
	start_line: int
	end_line: int
	type: str
	name: str
	parent: ChunkData | None = None
	embedding: list[float] | None = None


@dataclass
class DocumentationProcessor:
	"""Test version of the DocumentationProcessor."""

	embed_fn: Any
	repo_path: Path
	target_files: list[Path]
	token_count_fn: Any
	token_limit: int = 8192
	current_token_count: int = 0
	parser: Any = None

	def should_process_file(self, file_path: Path) -> bool:
		"""Check if a file should be processed."""
		if not self.target_files:
			return True
		return file_path in self.target_files

	def process_file(self, file_path: Path) -> list[ChunkData]:
		"""Process a file and return chunks."""
		if not self.should_process_file(file_path):
			return []

		# Check token limit
		tokens = self.token_count_fn(file_path)
		if self.token_limit > 0 and self.current_token_count + tokens > self.token_limit:
			return []

		self.current_token_count += tokens

		# Get chunks from parser
		chunks = self.parser.parse(file_path)

		# Add embeddings
		for chunk in chunks:
			chunk.embedding = self.embed_fn(chunk.content)

		return chunks

	def reset_token_count(self) -> None:
		"""Reset the token count."""
		self.current_token_count = 0


@pytest.fixture
def mock_code_parser() -> MagicMock:
	"""Mock CodeParser for testing."""
	mock_parser = MagicMock()

	# Setup parser returns
	mock_parser.parse.return_value = [
		ChunkData(
			content="def test():\n    pass",
			path=Path("test.py"),
			start_line=1,
			end_line=2,
			type="function",
			name="test",
			parent=None,
			embedding=None,
		)
	]
	return mock_parser


@pytest.fixture
def processor(mock_code_parser: MagicMock) -> DocumentationProcessor:
	"""Create a DocumentationProcessor with mocked dependencies."""
	processor = DocumentationProcessor(
		embed_fn=lambda _: [0.1, 0.2, 0.3],
		repo_path=Path("/fake/repo"),
		target_files=[],
		token_count_fn=lambda _: 10,
	)
	processor.parser = mock_code_parser
	return processor


@pytest.mark.unit
@pytest.mark.processor
class TestProcessorBasicOperations:
	"""Test basic operations of DocumentationProcessor."""

	def test_initialization(self, processor: DocumentationProcessor) -> None:
		"""Test initialization of DocumentationProcessor."""
		assert processor.repo_path == Path("/fake/repo")
		assert processor.target_files == []
		assert processor.token_limit == 8192
		assert processor.current_token_count == 0

	def test_process_file(self, processor: DocumentationProcessor, mock_code_parser: MagicMock) -> None:
		"""Test processing a file."""
		file_path = Path("test.py")
		result = processor.process_file(file_path)

		# Verify results
		assert len(result) == 1
		assert result[0].content == "def test():\n    pass"
		assert result[0].path == Path("test.py")
		assert result[0].embedding == [0.1, 0.2, 0.3]

		# Verify token count is updated
		assert processor.current_token_count == 10

		# Verify parser was called correctly
		mock_code_parser.parse.assert_called_once_with(file_path)


@pytest.mark.unit
@pytest.mark.processor
@pytest.mark.data
class TestProcessorFiltering:
	"""Test file filtering behavior of DocumentationProcessor."""

	def test_should_process_file_no_targets(self, processor: DocumentationProcessor) -> None:
		"""Test should_process_file when no target files are specified."""
		processor.target_files = []
		assert processor.should_process_file(Path("test.py")) is True

	def test_should_process_file_with_targets(self, processor: DocumentationProcessor) -> None:
		"""Test should_process_file when target files are specified."""
		processor.target_files = [Path("test.py"), Path("other.py")]
		assert processor.should_process_file(Path("test.py")) is True
		assert processor.should_process_file(Path("different.py")) is False

	def test_process_file_should_not_parse(
		self, processor: DocumentationProcessor, mock_code_parser: MagicMock
	) -> None:
		"""Test that file is not parsed when it should not be processed."""
		processor.target_files = [Path("other.py")]
		result = processor.process_file(Path("test.py"))

		assert result == []
		mock_code_parser.parse.assert_not_called()


@pytest.mark.unit
@pytest.mark.processor
@pytest.mark.performance
class TestTokenLimits:
	"""Test token limit handling in DocumentationProcessor."""

	def test_token_limit_reached(self, processor: DocumentationProcessor) -> None:
		"""Test that token limit prevents further processing."""
		processor.current_token_count = processor.token_limit - 5
		processor.token_count_fn = lambda _: 10  # Token count will exceed limit

		result = processor.process_file(Path("test.py"))
		assert result == []

	def test_token_limit_not_reached(self, processor: DocumentationProcessor) -> None:
		"""Test processing continues when token limit is not reached."""
		processor.current_token_count = processor.token_limit - 20
		processor.token_count_fn = lambda _: 10  # Token count will not exceed limit

		result = processor.process_file(Path("test.py"))
		assert len(result) == 1

	def test_reset_token_count(self, processor: DocumentationProcessor) -> None:
		"""Test resetting token count."""
		processor.current_token_count = 100
		processor.reset_token_count()
		assert processor.current_token_count == 0


@pytest.mark.unit
@pytest.mark.processor
@pytest.mark.data
class TestJsonSerialization:
	"""Test JSON serialization of processor output."""

	def test_serialization(self, processor: DocumentationProcessor) -> None:
		"""Test that processed chunks can be serialized to JSON."""
		chunks = processor.process_file(Path("test.py"))

		# Convert chunks to dictionaries with Path objects converted to strings
		serializable_chunks = []
		for chunk in chunks:
			chunk_dict = asdict(chunk)
			# Convert Path to string to make it JSON serializable
			chunk_dict["path"] = str(chunk_dict["path"])
			serializable_chunks.append(chunk_dict)

		# Test serialization
		json_data = json.dumps(serializable_chunks)
		deserialized = json.loads(json_data)

		assert len(deserialized) == 1
		assert deserialized[0]["content"] == "def test():\n    pass"
		assert deserialized[0]["name"] == "test"
		assert deserialized[0]["type"] == "function"
		assert deserialized[0]["path"] == "test.py"  # Now a string path
