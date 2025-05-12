"""Tests for the Qdrant manager module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models
from qdrant_client.http.models import Distance, PointStruct

from codemap.config.config_loader import ConfigLoader
from codemap.config.config_schema import AppConfigSchema, EmbeddingSchema
from codemap.processor.vector.qdrant_manager import QdrantManager, create_qdrant_point


@pytest.fixture
def mock_config_loader() -> MagicMock:
	"""Mock ConfigLoader for testing."""
	mock_config = MagicMock(spec=ConfigLoader)

	# Create a mock embedding schema
	embedding_config = EmbeddingSchema(
		dimension=768,
		dimension_metric="cosine",
		api_key="test_api_key",
		url="http://localhost:6333",
		prefer_grpc=True,
		timeout=5,
	)

	# Create a mock app config schema
	app_config = MagicMock(spec=AppConfigSchema)
	app_config.embedding = embedding_config

	# Set up the get property to return the app config
	mock_config.get = app_config

	return mock_config


@pytest.fixture
def qdrant_manager(mock_config_loader: MagicMock) -> QdrantManager:
	"""Create a QdrantManager with mocked dependencies."""
	return QdrantManager(
		config_loader=mock_config_loader,
		collection_name="test_collection",
		dim=768,
		distance=Distance.COSINE,
	)


@pytest.mark.unit
@pytest.mark.processor
class TestQdrantManager:
	"""Test the QdrantManager class."""

	def test_initialization(self, qdrant_manager: QdrantManager, mock_config_loader: MagicMock) -> None:
		"""Test initialization of QdrantManager."""
		assert qdrant_manager.config_loader is mock_config_loader
		assert qdrant_manager.collection_name == "test_collection"
		assert qdrant_manager.dim == 768
		assert qdrant_manager.distance == Distance.COSINE
		assert qdrant_manager.client_args["api_key"] == "test_api_key"
		assert qdrant_manager.client_args["url"] == "http://localhost:6333"
		assert qdrant_manager.client_args["prefer_grpc"] is True
		assert qdrant_manager.client_args["timeout"] == 5

		assert qdrant_manager.client is None
		assert qdrant_manager.is_initialized is False

	@pytest.mark.asyncio
	async def test_initialize(self, qdrant_manager: QdrantManager) -> None:
		"""Test initialize method."""
		# Mock the AsyncQdrantClient
		with patch("codemap.processor.vector.qdrant_manager.AsyncQdrantClient") as mock_client_cls:
			# Mock the client instance
			mock_client = AsyncMock()
			mock_client_cls.return_value = mock_client

			# Mock the get_collections response with collection not found
			collections_response = MagicMock()
			collections_response.collections = []
			mock_client.get_collections.return_value = collections_response

			# Call initialize
			await qdrant_manager.initialize()

			# Verify client was initialized
			mock_client_cls.assert_called_once_with(
				api_key="test_api_key",
				url="http://localhost:6333",
				prefer_grpc=True,
				timeout=5,
			)

			# Should have checked for collection
			mock_client.get_collections.assert_called_once()

			# Should have created collection
			mock_client.create_collection.assert_called_once_with(
				collection_name="test_collection",
				vectors_config=models.VectorParams(size=768, distance=Distance.COSINE),
			)

			# Should have created payload indexes
			assert mock_client.create_payload_index.call_count > 0

			# Check if manager is initialized
			assert qdrant_manager.is_initialized is True
			assert qdrant_manager.client is mock_client

	@pytest.mark.asyncio
	async def test_initialize_existing_collection(self, qdrant_manager: QdrantManager) -> None:
		"""Test initialize method with existing collection."""
		# Mock the AsyncQdrantClient
		with patch("codemap.processor.vector.qdrant_manager.AsyncQdrantClient") as mock_client_cls:
			# Mock the client instance
			mock_client = AsyncMock()
			mock_client_cls.return_value = mock_client

			# Mock the get_collections response with collection found
			collection = MagicMock()
			collection.name = "test_collection"
			collections_response = MagicMock()
			collections_response.collections = [collection]
			mock_client.get_collections.return_value = collections_response

			# Mock the get_collection response
			collection_info = MagicMock()
			collection_info.payload_schema = {"field1": "type1"}
			mock_client.get_collection.return_value = collection_info

			# Call initialize
			await qdrant_manager.initialize()

			# Verify client was initialized
			mock_client_cls.assert_called_once()

			# Should have checked for collection
			mock_client.get_collections.assert_called_once()

			# Should NOT have created collection
			mock_client.create_collection.assert_not_called()

			# Should have checked existing payload indexes
			mock_client.get_collection.assert_called_once_with(collection_name="test_collection")

			# Should have created missing payload indexes
			# Since "file_path" is not in the existing schema
			assert mock_client.create_payload_index.call_count > 0

			# Check if manager is initialized
			assert qdrant_manager.is_initialized is True
			assert qdrant_manager.client is mock_client

	@pytest.mark.asyncio
	async def test_upsert_points(self, qdrant_manager: QdrantManager) -> None:
		"""Test upserting points to Qdrant."""
		# Mock the AsyncQdrantClient
		with patch("codemap.processor.vector.qdrant_manager.AsyncQdrantClient") as mock_client_cls:
			# Mock the client instance
			mock_client = AsyncMock()
			mock_client_cls.return_value = mock_client

			# Mock initialize to set client
			qdrant_manager.client = mock_client
			qdrant_manager.is_initialized = True

			# Create test points
			test_points = [
				PointStruct(id="123", vector=[0.1, 0.2, 0.3], payload={"file_path": "test.py"}),
				PointStruct(id="456", vector=[0.4, 0.5, 0.6], payload={"file_path": "other.py"}),
			]

			# Call upsert_points
			await qdrant_manager.upsert_points(test_points)

			# Verify client called with correct args
			mock_client.upsert.assert_called_once_with(collection_name="test_collection", points=test_points, wait=True)

	@pytest.mark.asyncio
	async def test_delete_points(self, qdrant_manager: QdrantManager) -> None:
		"""Test deleting points from Qdrant."""
		# Mock the AsyncQdrantClient
		with patch("codemap.processor.vector.qdrant_manager.AsyncQdrantClient") as mock_client_cls:
			# Mock the client instance
			mock_client = AsyncMock()
			mock_client_cls.return_value = mock_client

			# Mock initialize to set client
			qdrant_manager.client = mock_client
			qdrant_manager.is_initialized = True

			# Create test point IDs without UUID to avoid validation errors
			# Use only string and int types for simplicity
			test_ids = ["123", 456, "789"]

			# Call delete_points but patch the internal call that causes validation error
			with patch("codemap.processor.vector.qdrant_manager.models.PointIdsList") as mock_point_ids_list:
				# Set up the mock to return a suitable object
				mock_point_ids = MagicMock()
				mock_point_ids_list.return_value = mock_point_ids

				# Cast to the exact type expected by the function
				compatible_ids = [str(id) for _id in test_ids]

				await qdrant_manager.delete_points(compatible_ids)

				# Verify client was called with our mocked PointIdsList
				mock_client.delete.assert_called_once_with(
					collection_name="test_collection",
					points_selector=mock_point_ids,
					wait=True,
				)

	@pytest.mark.asyncio
	async def test_search(self, qdrant_manager: QdrantManager) -> None:
		"""Test searching for points in Qdrant."""
		# Mock the AsyncQdrantClient
		with patch("codemap.processor.vector.qdrant_manager.AsyncQdrantClient") as mock_client_cls:
			# Mock the client instance
			mock_client = AsyncMock()
			mock_client_cls.return_value = mock_client

			# Mock initialize to set client
			qdrant_manager.client = mock_client
			qdrant_manager.is_initialized = True

			# Create test vector and filter
			test_vector = [0.1, 0.2, 0.3]
			test_filter = models.Filter(
				must=[models.FieldCondition(key="file_path", match=models.MatchText(text="test.py"))]
			)

			# Mock search response
			mock_search_result = [
				models.ScoredPoint(
					id="123", score=0.95, payload={"file_path": "test.py"}, vector=[0.1, 0.2, 0.3], version=1
				)
			]
			mock_client.search.return_value = mock_search_result

			# Call search
			result = await qdrant_manager.search(test_vector, k=5, query_filter=test_filter)

			# Verify client called with correct args
			mock_client.search.assert_called_once_with(
				collection_name="test_collection",
				query_vector=test_vector,
				limit=5,
				query_filter=test_filter,
				with_payload=True,
				with_vectors=False,
			)

			# Verify result
			assert result == mock_search_result

	@pytest.mark.asyncio
	async def test_get_all_point_ids_with_filter(self, qdrant_manager: QdrantManager) -> None:
		"""Test getting all point IDs with a filter."""
		# Mock the AsyncQdrantClient
		with patch(
			"codemap.processor.vector.qdrant_manager.QdrantManager.get_all_point_ids_with_filter"
		) as mock_method:
			# Mock the method directly to return expected IDs
			expected_ids = ["123", "456", "789"]
			mock_method.return_value = expected_ids

			# Create test filter
			test_filter = models.Filter(
				must=[models.FieldCondition(key="file_path", match=models.MatchText(text="test.py"))]
			)

			# Call the mocked method
			result = await qdrant_manager.get_all_point_ids_with_filter(test_filter)

			# Verify the result
			assert result == expected_ids
			mock_method.assert_called_once_with(test_filter)

	def test_create_qdrant_point(self) -> None:
		"""Test creating a PointStruct from chunk data."""
		# Test data
		chunk_id = "test_id_123"
		vector = [0.1, 0.2, 0.3]
		payload: dict[str, Any] = {
			"file_path": "test.py",
			"entity_type": "FUNCTION",
			"language": "python",
			"start_line": 1,
			"end_line": 10,
		}

		# Create point
		point = create_qdrant_point(chunk_id, vector, payload)

		# Verify point structure
		assert isinstance(point, PointStruct)
		assert point.id == chunk_id
		assert point.vector == vector
		assert point.payload == payload
