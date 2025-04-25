"""Storage module for persisting code chunks, metadata, and embeddings.

This module provides an abstraction layer for storing and retrieving:
- Code chunks and their metadata
- Vector embeddings for semantic search
- Historical versions of code chunks

"""

from codemap.processor.storage.base import StorageBackend, StorageConfig
from codemap.processor.storage.lance import LanceDBStorage

__all__ = ["LanceDBStorage", "StorageBackend", "StorageConfig"]
