"""Handles Milvus Lite connection and collection management."""

import logging

from pymilvus import MilvusClient, exceptions

from . import config
from .schema import create_collection_schema

logger = logging.getLogger(__name__)

# Module-level variable for the singleton client
_milvus_client: MilvusClient | None = None


def _initialize_milvus_client() -> MilvusClient | None:
	"""Initializes and returns a new MilvusClient instance."""
	db_path = config.get_vector_db_path()
	try:
		# Using MilvusClient directly for Lite
		new_client = MilvusClient(uri=str(db_path))
		logger.info(f"Milvus client initialized with DB path: {db_path}")
		ensure_collection_exists(new_client)
		return new_client
	except exceptions.MilvusException:
		logger.exception(f"Milvus connection error for DB {db_path}")
		return None
	except Exception:
		logger.exception(f"Unexpected error initializing Milvus client for DB {db_path}")
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
			# For Milvus Lite, index_params in create_collection might not be strictly necessary
			# as it defaults to FLAT, but including it for clarity/future compatibility.
			index_params = client.prepare_index_params()
			index_params.add_index(
				field_name=config.FIELD_EMBEDDING,
				index_type=config.INDEX_TYPE,  # Should be FLAT for Lite
				metric_type=config.METRIC_TYPE,
				# params={} # FLAT index usually doesn't require params
			)
			client.create_collection(
				collection_name=collection_name,
				schema=schema,
				index_params=index_params,
				# Milvus Lite ignores consistency_level, num_shards etc.
			)
			logger.info(f"Collection '{collection_name}' created successfully.")
			# Note: Loading is implicitly handled by Milvus Lite
		else:
			logger.debug(f"Collection '{collection_name}' already exists.")
			# Optionally: Verify schema matches?
			# existing_schema = client.describe_collection(collection_name)
			# compare schemas...

	except exceptions.MilvusException:
		logger.exception(f"Milvus error checking/creating collection '{collection_name}'")
	except Exception:
		logger.exception(f"Unexpected error ensuring collection '{collection_name}' exists")


def close_milvus_connection() -> None:
	"""Closes the Milvus connection if open."""
	global _milvus_client  # noqa: PLW0603
	client_to_close = _milvus_client
	if client_to_close is not None:
		try:
			# Set global variable to None *before* closing
			_milvus_client = None
			client_to_close.close()
			logger.info("Milvus connection closed.")
		except exceptions.MilvusException:
			logger.exception("Error closing Milvus connection")
			# Should we try to restore _milvus_client? Probably not, it's likely unusable.
		except Exception:
			logger.exception("Unexpected error closing Milvus connection")
	else:
		logger.debug("Attempted to close Milvus connection, but it was already None.")


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
