"""Configuration constants for the vector processing module."""

# --- Database Configuration ---
# Removed: VECTOR_DB_DIR_NAME, WORKSPACE_ROOT, VECTOR_DB_PATH
# These are now handled by path_utils.get_cache_path("vector")
VECTOR_DB_FILE_NAME = "milvus_vector_db.db"

# --- Milvus Collection Configuration ---
COLLECTION_NAME = "codemap_embeddings"
METRIC_TYPE = "COSINE"  # Or "IP"
INDEX_TYPE = "FLAT"  # Milvus Lite only supports FLAT

# --- Embedding Model Configuration ---
# Use the model already specified in the main config or commit logic
EMBEDDING_MODEL_NAME = "sarthak1/Qodo-Embed-M-1-1.5B-M2V-Distilled"
# Determine dimension from model spec (Qodo-Embed-M-1-1.5B-M2V-Distilled is likely 384)
EMBEDDING_DIMENSION = 384

# --- Schema Field Names (Consistent identifiers) ---
FIELD_ID = "id"
FIELD_EMBEDDING = "embedding"
FIELD_FILE_PATH = "file_path"
FIELD_ENTITY_NAME = "entity_name"
FIELD_CHUNK_TYPE = "chunk_type"
FIELD_CHUNK_TEXT = "chunk_text"
FIELD_GIT_HASH = "git_hash"
FIELD_START_LINE = "start_line"
FIELD_END_LINE = "end_line"

# --- Chunking Configuration ---
# Placeholder for potential chunking limits (e.g., max tokens/chars per chunk)
MAX_CHUNK_TEXT_LENGTH = 65535  # Max length for varchar in schema
MAX_CHUNK_OVERLAP = 50  # Example: if splitting large fallback chunks

# --- Fallback Chunk Types ---
CHUNK_TYPE_MODULE = "module"
CHUNK_TYPE_CLASS = "class"
CHUNK_TYPE_FUNCTION = "function"
CHUNK_TYPE_METHOD = "method"
CHUNK_TYPE_FALLBACK = "regex_fallback"
