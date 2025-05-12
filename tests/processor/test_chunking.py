"""Tests for the code chunking module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from codemap.config import ConfigLoader
from codemap.processor.lod import LODEntity, LODGenerator
from codemap.processor.tree_sitter.base import EntityType
from codemap.processor.vector.chunking import TreeSitterChunker


@pytest.fixture
def mock_lod_generator() -> MagicMock:
	"""Mock LODGenerator for testing."""
	return MagicMock(spec=LODGenerator)


@pytest.fixture
def mock_config_loader() -> MagicMock:
	"""Mock ConfigLoader for testing."""
	from codemap.config.config_schema import AppConfigSchema, EmbeddingChunkingSchema, EmbeddingSchema

	mock_config = MagicMock(spec=ConfigLoader)

	# Create mock for app config
	mock_app_config = MagicMock(spec=AppConfigSchema)

	# Create mock for embedding config
	mock_embedding_config = MagicMock(spec=EmbeddingSchema)

	# Create mock for chunking config
	mock_chunking_config = MagicMock(spec=EmbeddingChunkingSchema)
	mock_chunking_config.max_hierarchy_depth = 2
	mock_chunking_config.max_file_lines = 1000

	# Set up the property chain
	mock_embedding_config.chunking = mock_chunking_config
	mock_app_config.embedding = mock_embedding_config

	# Set up the get property to return the app config
	type(mock_config).get = PropertyMock(return_value=mock_app_config)

	return mock_config


@pytest.fixture
def chunker(mock_lod_generator: MagicMock, mock_config_loader: MagicMock) -> TreeSitterChunker:
	"""Create a TreeSitterChunker with mocked dependencies."""
	return TreeSitterChunker(
		lod_generator=mock_lod_generator,
		config_loader=mock_config_loader,
	)


@pytest.fixture
def sample_python_content() -> str:
	"""Sample Python content for testing."""
	return """
def test_function():
    \"\"\"This is a test function.\"\"\"
    return True

class TestClass:
    \"\"\"A test class.\"\"\"

    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value
"""


@pytest.fixture
def sample_lod_entity(sample_python_content) -> LODEntity:
	"""Create a sample LOD entity for testing."""
	# Create a module entity
	module_entity = LODEntity(
		name="sample.py",
		entity_type=EntityType.MODULE,
		start_line=1,
		end_line=13,
		docstring="",
		signature="",
		content="",
		language="python",
		metadata={"full_content_str": sample_python_content, "language": "python"},
	)

	# Add a function entity
	function_entity = LODEntity(
		name="test_function",
		entity_type=EntityType.FUNCTION,
		start_line=2,
		end_line=4,
		docstring="This is a test function.",
		signature="def test_function():",
		content='def test_function():\n    """This is a test function."""\n    return True',
		language="python",
	)

	# Add a class entity
	class_entity = LODEntity(
		name="TestClass",
		entity_type=EntityType.CLASS,
		start_line=6,
		end_line=13,
		docstring="A test class.",
		signature="class TestClass:",
		content='class TestClass:\n    """A test class."""\n    \n    def __init__(self, value):\n        self.value = value\n        \n    def get_value(self):\n        return self.value',
		language="python",
	)

	# Add method entities to the class
	init_method = LODEntity(
		name="__init__",
		entity_type=EntityType.METHOD,
		start_line=9,
		end_line=10,
		docstring="",
		signature="def __init__(self, value):",
		content="def __init__(self, value):\n        self.value = value",
		language="python",
	)

	get_value_method = LODEntity(
		name="get_value",
		entity_type=EntityType.METHOD,
		start_line=12,
		end_line=13,
		docstring="",
		signature="def get_value(self):",
		content="def get_value(self):\n        return self.value",
		language="python",
	)

	# Set up hierarchy
	class_entity.children = [init_method, get_value_method]
	module_entity.children = [function_entity, class_entity]

	return module_entity


@pytest.mark.unit
@pytest.mark.processor
class TestTreeSitterChunker:
	"""Test the TreeSitterChunker class."""

	def test_initialization(self, chunker: TreeSitterChunker, mock_config_loader: MagicMock) -> None:
		"""Test initialization of TreeSitterChunker."""
		assert chunker.lod_generator is not None
		assert chunker.config_loader is mock_config_loader
		assert chunker.max_hierarchy_depth == 2
		assert chunker.max_file_lines == 1000

	def test_build_hierarchy_path(self, chunker: TreeSitterChunker) -> None:
		"""Test building hierarchy paths."""
		# Test with no parent path
		entity = LODEntity(name="test_function", entity_type=EntityType.FUNCTION, start_line=1, end_line=2)
		path = chunker._build_hierarchy_path(entity)
		assert path == "test_function"

		# Test with parent path
		parent_path = "module.TestClass"
		path = chunker._build_hierarchy_path(entity, parent_path)
		assert path == "module.TestClass.test_function"

		# Test with unnamed entity
		unnamed_entity = LODEntity(name="", entity_type=EntityType.FUNCTION, start_line=1, end_line=2)
		path = chunker._build_hierarchy_path(unnamed_entity)
		assert path == "<function>"

	def test_extract_nested_entities(self, chunker: TreeSitterChunker, sample_lod_entity: LODEntity) -> None:
		"""Test extracting nested entity information."""
		nested_info = chunker._extract_nested_entities(sample_lod_entity)

		# Should have 4 nested entities: function, class, and 2 methods
		assert len(nested_info) == 4

		# Verify structure
		entity_types = {info["type"] for info in nested_info}
		entity_names = {info["name"] for info in nested_info}

		assert "FUNCTION" in entity_types
		assert "CLASS" in entity_types
		assert "METHOD" in entity_types
		assert "test_function" in entity_names
		assert "TestClass" in entity_names

	def test_chunk_file(
		self,
		chunker: TreeSitterChunker,
		mock_lod_generator: MagicMock,
		sample_lod_entity: LODEntity,
		sample_python_content: str,
	) -> None:
		"""Test the chunk_file method."""
		# Create a mock file path with the required properties
		mock_file_path = MagicMock(spec=Path)
		mock_file_path.name = "sample.py"
		mock_file_path.is_absolute.return_value = True
		# Set up the resolve method to return the mock itself
		mock_resolve = MagicMock(return_value=mock_file_path)
		mock_file_path.resolve = mock_resolve
		mock_file_path.configure_mock(**{"__str__.return_value": "/test/sample.py"})

		# Create a stat result for the file_path
		mock_stat_result = MagicMock()
		mock_stat_result.st_mtime = 1234567890  # Mock timestamp
		mock_file_path.stat.return_value = mock_stat_result

		# Mock the file reading
		with patch("codemap.processor.vector.chunking.read_file_content") as mock_read:
			mock_read.return_value = sample_python_content

			# Mock the generate_lod method
			mock_lod_generator.generate_lod.return_value = sample_lod_entity

			# Call chunk_file
			chunks = list(chunker.chunk_file(mock_file_path))

			# Verify the chunks
			assert len(chunks) >= 3  # At least one for module, function, and class

			# Verify chunk properties
			chunk_types = set()
			for chunk in chunks:
				chunk_dict = chunk.model_dump()  # Convert Pydantic model to dictionary
				assert "content" in chunk_dict
				assert "metadata" in chunk_dict
				metadata = chunk_dict["metadata"]
				assert "chunk_id" in metadata
				assert "file_metadata" in metadata
				file_metadata = metadata["file_metadata"]
				assert "file_path" in file_metadata
				assert file_metadata["file_path"] == "sample.py"  # The file name, not the full path
				assert "entity_type" in metadata
				assert "hierarchy_path" in metadata

				chunk_types.add(metadata["entity_type"])

			# Should have FILE (instead of MODULE), FUNCTION, CLASS, METHOD
			assert "FILE" in chunk_types or "MODULE" in chunk_types  # Accept either FILE or MODULE
			assert "FUNCTION" in chunk_types or "CLASS" in chunk_types  # At least one of these should be present

	def test_chunk_file_empty_content(self, chunker: TreeSitterChunker, mock_lod_generator: MagicMock) -> None:
		"""Test chunk_file with empty file content."""
		test_file_path = Path("/test/empty.py")

		# Mock the file reading to return empty content
		with patch("codemap.processor.vector.chunking.read_file_content") as mock_read:
			mock_read.return_value = ""

			# Mock generate_lod to return None (analysis failed)
			mock_lod_generator.generate_lod.return_value = None

			# Call chunk_file
			chunks = list(chunker.chunk_file(test_file_path))

			# Should be empty
			assert len(chunks) == 0

	def test_serializable_chunks(
		self,
		chunker: TreeSitterChunker,
		mock_lod_generator: MagicMock,
		sample_lod_entity: LODEntity,
		sample_python_content: str,
	) -> None:
		"""Test that chunks can be serialized to JSON."""
		test_file_path = Path("/test/sample.py")

		# Mock the file reading
		with patch("codemap.processor.vector.chunking.read_file_content") as mock_read:
			mock_read.return_value = sample_python_content

			# Mock the generate_lod method
			mock_lod_generator.generate_lod.return_value = sample_lod_entity

			# Get chunks
			chunks = list(chunker.chunk_file(test_file_path))

			# Convert Pydantic models to dictionaries
			chunk_dicts = [chunk.dict() for chunk in chunks]

			# Try to serialize to JSON
			json_data = json.dumps(chunk_dicts)
			deserialized = json.loads(json_data)

			# Verify we got the same number of chunks back
			assert len(deserialized) == len(chunks)
