"""Base entity schemas."""

from pydantic import BaseModel, Field

from .types import EntityType, ScopeType

# --- Scope Schema --- #


class LocationSchema(BaseModel):
	"""Schema for code location."""

	start_line: int
	end_line: int
	start_col: int
	end_col: int


class MetadataSchema(BaseModel):
	"""Schema for entity metadata."""

	file_path: str
	language: str
	node_kind: str


class ScopeSchema(BaseModel):
	"""
	Represents lexical scope across multiple programming languages.

	This schema is designed to be flexible and extensible, supporting a wide range of language-specific
	scoping mechanisms and visibility rules. It can be used for static analysis, code navigation,
	and language tooling.

	Usage Examples:

	1. **Basic Scope Structure**
	    - Global scope:
	        scope_id = 1
	        parent_scope_id = None
	        scope_type = ScopeType.GLOBAL

	    - Function scope:
	        scope_id = 2
	        parent_scope_id = 1
	        scope_type = ScopeType.FUNCTION

	2. **Entity Declarations**
	    - Track entities declared within the scope:
	        declarations = [101, 102, 103]  # IDs of entities in this scope

	3. **Import Tracking**
	    - Track imported namespaces/modules:
	        imports = ["numpy", "pandas", "typing"]

	Fields:
	    scope_id: Unique identifier for this scope.
	    parent_scope_id: ID of the parent scope (None indicates global scope).
	    scope_type: The kind of scope (e.g., GLOBAL, FUNCTION, CLASS, etc.).
	    declarations: List of entity IDs declared in this scope.
	    imports: List of imported namespaces/modules.
	"""

	scope_id: str
	parent_scope_id: str | None = Field(None, description="None indicates global scope")
	scope_type: ScopeType
	declarations: list[str] = Field(default_factory=list, description="IDs of entities declared in this scope")
	imports: list[str] = Field(default_factory=list, description="Imported namespaces/modules")


# --- Base Entity Schema --- #


class EntitySchema(BaseModel):
	"""Base schema for an extracted code entity."""

	id: str
	type: EntityType
	name: str | None = None
	parent_id: str | None = None
	children: list["EntitySchema"]
	location: LocationSchema
	metadata: MetadataSchema
	scope: ScopeSchema
