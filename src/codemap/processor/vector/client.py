"""Handles Milvus Lite connection and collection management."""

import logging

from pymilvus import MilvusClient

# from milvus_lite.client import MilvusClient # Reverted change
from codemap.processor.utils.file_utils import ensure_directory_exists
from codemap.processor.utils.path_utils import get_cache_path
from codemap.processor.vector import config
from codemap.processor.vector.schema import create_collection_schema

logger = logging.getLogger(__name__)

# Module-level variable for the singleton client
_milvus_client: MilvusClient | None = None


def _initialize_milvus_client() -> MilvusClient | None:
	"""Initializes and returns a new MilvusClient instance."""
	global _milvus_client  # noqa: PLW0603
	try:
		vector_cache_dir = get_cache_path("vector")
		ensure_directory_exists(vector_cache_dir)
		db_file = vector_cache_dir / config.VECTOR_DB_FILE_NAME
		db_path = str(db_file)

		logger.info(f"Initializing Milvus Lite client at: {db_path}")
		client = MilvusClient(db_path)
		ensure_collection_exists(client)
		_milvus_client = client
		return _milvus_client
	except Exception:
		logger.exception("Failed to initialize Milvus client")
		if _milvus_client:
			try:
				_milvus_client.close()
			except Exception:
				logger.exception("Error closing Milvus client during cleanup")
			_milvus_client = None
		return None


def get_milvus_client() -> MilvusClient | None:
	"""Gets a singleton MilvusClient instance."""
	global _milvus_client  # noqa: PLW0603
	if _milvus_client is None:
		logger.debug("Milvus client is None, attempting initialization.")
		_milvus_client = _initialize_milvus_client()
		# Check if initialization succeeded
		if _milvus_client:
			logger.debug("Milvus client initialized successfully.")
		else:
			logger.error("Failed to initialize Milvus client.")
	return _milvus_client


def ensure_collection_exists(client: MilvusClient) -> None:
	"""Checks if the collection exists, creates it if not."""
	collection_name = config.COLLECTION_NAME
	try:
		if not client.has_collection(collection_name):
			logger.info(f"Collection '{collection_name}' not found. Creating...")
			schema = create_collection_schema()
			client.create_collection(collection_name, schema=schema)
			logger.info(f"Collection '{collection_name}' created successfully.")
		else:
			logger.debug(f"Collection '{collection_name}' already exists.")
	except Exception:
		logger.exception(f"Unexpected error checking/creating collection '{collection_name}'")
		raise


def close_milvus_connection() -> None:
	"""Closes the Milvus connection if open."""
	global _milvus_client  # noqa: PLW0603
	client_to_close = _milvus_client
	_milvus_client = None  # Set to None immediately

	if client_to_close:
		logger.info("Closing Milvus client connection.")
		try:
			client_to_close.close()
		except Exception:
			logger.exception("Unexpected error closing Milvus connection")
		finally:
			logger.info("Milvus client connection closed.")
	else:
		logger.debug("No active Milvus client connection to close.")


# Example Usage (can be removed later)
if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG)
	logger.info("Testing Milvus Client Initialization...")
	client_instance = get_milvus_client()
	if client_instance:
		logger.info("Client obtained successfully.")
		try:
			collections = client_instance.list_collections()
			logger.info(f"Available collections: {collections}")
		except Exception:
			# Use logger.exception here as well, but remove the redundant exception object
			# logger.error(f"Error listing collections: {e}")
			logger.exception("Error listing collections")
		close_milvus_connection()
	else:
		logger.error("Failed to obtain Milvus client.")
