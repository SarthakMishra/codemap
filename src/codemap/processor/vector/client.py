"""Handles Milvus Lite connection and collection management."""

import logging

from pymilvus import MilvusClient

from codemap.processor.vector import config
from codemap.processor.vector.schema import create_collection_schema

# from milvus_lite.client import MilvusClient # Reverted change
from codemap.utils.file_utils import ensure_directory_exists
from codemap.utils.path_utils import get_cache_path

logger = logging.getLogger(__name__)

# Module-level variable for the singleton client
_milvus_client: MilvusClient | None = None


def _initialize_milvus_client() -> MilvusClient | None:
	"""Initializes and returns a new MilvusClient instance."""
	global _milvus_client  # noqa: PLW0603
	try:
		vector_cache_dir = get_cache_path()
		ensure_directory_exists(vector_cache_dir)
		db_file = vector_cache_dir / config.VECTOR_DB_FILE_NAME
		db_path = str(db_file)

		logger.info(f"Initializing Milvus Lite client at: {db_path}")
		client = MilvusClient(db_path)
		ensure_collection_exists(client)
		_milvus_client = client
		logger.info("Milvus client initialized successfully.")
		return _milvus_client
	except Exception:
		logger.exception(
			"Failed to initialize Milvus client or ensure collection. Vector features will be unavailable."
		)
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
	return _milvus_client


def ensure_collection_exists(client: MilvusClient) -> None:
	"""Checks if the collection exists, creates it if not."""
	collection_name = config.COLLECTION_NAME
	try:
		has_col = client.has_collection(collection_name)

		if not has_col:
			logger.info(f"Collection '{collection_name}' not found. Creating...")
			schema = create_collection_schema()
			client.create_collection(collection_name, schema=schema)
			logger.info(f"Collection '{collection_name}' creation initiated.")
			# Add a verification step immediately after creation
			try:
				collection_info = client.describe_collection(collection_name)
				logger.info(
					(
						f"Successfully verified collection '{collection_name}' exists after creation.",
						f"Info: {collection_info}",
					)
				)
				# --- Create Index using prepare_index_params --- #
				logger.info(f"Creating index for field '{config.FIELD_EMBEDDING}' in collection '{collection_name}'...")

				# 1. Prepare index parameters object
				index_builder = client.prepare_index_params()

				# 2. Add index configuration
				index_builder.add_index(
					field_name=config.FIELD_EMBEDDING,
					index_type=config.INDEX_TYPE,  # Should be FLAT for Milvus Lite
					metric_type=config.METRIC_TYPE,
					params={},  # FLAT index usually doesn't require build params
				)

				# 3. Create the index using the configured builder
				client.create_index(collection_name=collection_name, index_params=index_builder)

				logger.info(f"Index created successfully for field '{config.FIELD_EMBEDDING}'.")
				# --- End Create Index --- #

			except Exception:
				logger.exception(
					f"Post-creation step (verification or indexing) failed for collection '{collection_name}'."
				)
				# Attempt cleanup if verification/indexing failed
				try:
					if client.has_collection(collection_name):
						logger.warning(f"Attempting to drop partially created/failed collection '{collection_name}'...")
						client.drop_collection(collection_name)
				except Exception:
					logger.exception(f"Failed to drop problematic collection '{collection_name}'")
				raise  # Re-raise the verification/indexing error
		else:
			logger.debug(f"Collection '{collection_name}' already exists.")
	except Exception:
		logger.exception(f"Error during Milvus collection check/create for '{collection_name}'")
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
