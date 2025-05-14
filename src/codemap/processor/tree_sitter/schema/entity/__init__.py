"""Entity schemas for Tree-sitter."""

from .base import EntitySchema, LocationSchema, MetadataSchema, ScopeSchema
from .types import EntityType, ScopeType

__all__ = [
	"EntitySchema",
	"EntityType",
	"LocationSchema",
	"MetadataSchema",
	"ScopeSchema",
	"ScopeType",
]
