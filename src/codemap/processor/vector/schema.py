"""Defines the schema for the Milvus collection."""

from pymilvus import CollectionSchema, DataType, FieldSchema

from . import config


def create_collection_schema() -> CollectionSchema:
	"""Creates and returns the Milvus CollectionSchema for CodeMap embeddings."""
	# Define fields based on config and todo/vectors.md
	id_field = FieldSchema(
		name=config.FIELD_ID,
		dtype=DataType.INT64,
		is_primary=True,
		auto_id=True,  # Let Milvus handle ID generation
	)

	embedding_field = FieldSchema(
		name=config.FIELD_EMBEDDING, dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIMENSION
	)

	file_path_field = FieldSchema(
		name=config.FIELD_FILE_PATH,
		dtype=DataType.VARCHAR,
		max_length=1024,  # Max file path length
		description="Relative path to the source file from repo root",
	)

	entity_name_field = FieldSchema(
		name=config.FIELD_ENTITY_NAME,
		dtype=DataType.VARCHAR,
		max_length=512,  # Max entity name length
		description="Name of the code entity (class, function, etc.) or filename for fallback",
		default_value="",  # Allow empty for fallback chunks
	)

	chunk_type_field = FieldSchema(
		name=config.FIELD_CHUNK_TYPE,
		dtype=DataType.VARCHAR,
		max_length=64,
		description="Type of chunk (module, class, function, regex_fallback)",
	)

	chunk_text_field = FieldSchema(
		name=config.FIELD_CHUNK_TEXT,
		dtype=DataType.VARCHAR,
		max_length=config.MAX_CHUNK_TEXT_LENGTH,  # Configurable max length
		description="The actual text content that was embedded (potentially truncated)",
	)

	git_hash_field = FieldSchema(
		name=config.FIELD_GIT_HASH,
		dtype=DataType.VARCHAR,
		max_length=40,  # SHA-1 hash length
		description="Git blob hash of the file version",
	)

	start_line_field = FieldSchema(
		name=config.FIELD_START_LINE,
		dtype=DataType.INT32,
		description="Starting line number of the chunk/entity in the file",
	)

	end_line_field = FieldSchema(
		name=config.FIELD_END_LINE,
		dtype=DataType.INT32,
		description="Ending line number of the chunk/entity in the file",
	)

	# Create the schema
	return CollectionSchema(
		fields=[
			id_field,
			embedding_field,
			file_path_field,
			entity_name_field,
			chunk_type_field,
			chunk_text_field,
			git_hash_field,
			start_line_field,
			end_line_field,
		],
		description="Collection of code chunk embeddings for CodeMap semantic search",
		enable_dynamic_field=False,  # Explicit schema is preferred
	)


# Example Usage
if __name__ == "__main__":
	import logging

	# Get a logger specific to this test block
	logger = logging.getLogger(__name__)
	logging.basicConfig(level=logging.INFO)
	schema_instance = create_collection_schema()
	logger.info("Schema created successfully:")
	logger.info(schema_instance)
