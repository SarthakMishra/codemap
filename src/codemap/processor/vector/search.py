"""Handles semantic search queries against the vector database."""

import logging
from typing import Any

import numpy as np
from pymilvus import Hit, exceptions

from . import client, config, embedder

logger = logging.getLogger(__name__)

SearchResult = dict[str, Any]  # Includes distance, metadata fields


def semantic_search(query: str, k: int = 5) -> list[SearchResult] | None:
	"""
	Performs semantic search using the user query.

	1. Generates embedding for the query.
	2. Searches Milvus for the top-k similar vectors.
	3. Returns the results including metadata and distance.

	Args:
	    query (str): The user's search query.
	    k (int): The number of top similar results to retrieve.

	Returns:
	    List[SearchResult] | None: A list of search result dictionaries,
	                               or None if an error occurs.

	"""
	milvus_client = client.get_milvus_client()
	if not milvus_client:
		logger.error("Failed to get Milvus client. Search aborted.")
		return None

	# 1. Generate query embedding
	logger.info(f"Generating embedding for query: '{query[:50]}...'")
	query_embedding = embedder.generate_embeddings(query)

	if query_embedding is None:
		logger.error("Failed to generate query embedding.")
		return None

	# Ensure embedding is ndarray before calling tolist()
	if not isinstance(query_embedding, np.ndarray):
		logger.error(f"Query embedding is not a numpy array: {type(query_embedding)}")
		return None

	# Ensure query_embedding is a list of vectors (even if just one) for Milvus search
	search_vectors = [query_embedding.tolist()]  # Milvus expects list of lists/vectors

	# 2. Prepare search parameters
	search_params = {
		"metric_type": config.METRIC_TYPE,
		"params": {},  # Search params for FLAT index are usually empty or minimal
	}

	output_fields = [
		config.FIELD_FILE_PATH,
		config.FIELD_ENTITY_NAME,
		config.FIELD_CHUNK_TYPE,
		config.FIELD_CHUNK_TEXT,
		config.FIELD_START_LINE,
		config.FIELD_END_LINE,
		config.FIELD_GIT_HASH,  # Include hash for context
	]

	try:
		logger.info(f"Performing Milvus search for top {k} results...")
		results = milvus_client.search(
			collection_name=config.COLLECTION_NAME,
			data=search_vectors,
			anns_field=config.FIELD_EMBEDDING,  # The field containing vectors
			param=search_params,
			limit=k,
			output_fields=output_fields,
		)
		logger.info(f"Milvus search completed. Found {len(results[0]) if results else 0} potential results.")

		# 3. Format results
		# results is a list (one element per query vector) of lists (top k hits)
		# Each hit is a pymilvus.Hit object
		formatted_results: list[SearchResult] = []
		if results and results[0]:
			# Check type and access attributes safely
			for hit in results[0]:
				if isinstance(hit, Hit):
					entity_data = getattr(hit, "entity", None)
					search_result = {
						"distance": getattr(hit, "distance", float("inf")),
						"id": getattr(hit, "id", None),
						**(
							entity_data.to_dict() if entity_data and hasattr(entity_data, "to_dict") else {}
						),  # Unpack metadata
					}
					formatted_results.append(search_result)
				else:
					logger.warning(f"Unexpected item type in Milvus search results: {type(hit)}")

		logger.info(f"Formatted {len(formatted_results)} search results.")
		return formatted_results

	except exceptions.MilvusException:
		# logger.exception(f"Milvus search error: {e}")
		logger.exception("Milvus search error")
		return None
	except Exception:
		# logger.exception(f"Unexpected error during semantic search: {e}")
		logger.exception("Unexpected error during semantic search")
		return None


# Example Usage
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	test_query = "function to handle user login"

	search_results = semantic_search(test_query, k=3)

	if search_results is not None:
		if not search_results:
			pass
		else:
			for _i, _res in enumerate(search_results):
				pass  # Show preview
	else:
		pass
