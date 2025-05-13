"""Scope schemas."""

from typing import Any

from pydantic import BaseModel, Field

from .types import ScopeType, VisibilityModifier

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
	scoping mechanisms, visibility rules, and temporal characteristics. It can be used for static analysis,
	code navigation, and language tooling.

	Usage Examples:

	1. **Visibility Controls**
	    - Java/C# access modifiers:
	        public class Foo { ... }
	        → visibility = VisibilityModifier.PUBLIC

	    - Python's underscore convention:
	        _internal_var = ...
	        → visibility = VisibilityModifier.PROTECTED

	2. **Temporal Management**
	    - Lifetime types:
	        | Lifetime Type    | Languages          | Characteristics          |
	        |------------------|--------------------|--------------------------|
	        | Automatic        | Python, Java, C#   | Garbage collected        |
	        | Static           | C/C++ static vars  | Program lifetime         |
	        | Manual           | Rust ownership     | Explicit memory control  |

	    - Example:
	        temporal_attributes = {
	            "lifetime": "automatic",
	            "hoisting": "none"
	        }

	3. **Cross-Language References**
	    - Track declarations using entity IDs from your existing schema.
	    - Handle imports/aliases through qualified namespace paths.

	4. **Language Adaptability**
	    - Store implementation-specific details in language_metadata:
	        # JavaScript hoisting
	        language_metadata = {"hoisting_rules": ["function", "var"]}

	        # Rust ownership
	        language_metadata = {"ownership_mode": "move", "borrow_checker": True}

	        # Python nonlocal
	        language_metadata = {"nonlocal_declarations": ["counter"]}

	5. **Scope Resolution**
	    - For ambiguous cases like JavaScript's var vs let, store resolution rules in language_metadata:
	        language_metadata = {"scope_resolution": "lexical"}

	6. **Pattern Matching**
	    - Handle modern pattern scoping (Python 3.10+, Rust) through ScopeType.PATTERN:
	        match value:
	            case Point(x, y):  # New scope for x, y

	7. **Templated Code**
	    - Support generics with ScopeType.TEMPLATE:
	        public class Stack<T> {
	            private T[] elements;
	        }
	        # T exists in template scope

	Fields:
	    scope_id: Unique identifier for this scope.
	    parent_scope_id: ID of the parent scope (None indicates global scope).
	    scope_type: The kind of scope (e.g., GLOBAL, FUNCTION, CLASS, etc.).
	    visibility: Access modifier for the scope.
	    namespace: Qualified namespace path (if applicable).
	    temporal_attributes: Dict for memory management and hoisting info.
	    declarations: List of entity IDs declared in this scope.
	    imports: List of imported namespaces or modules.
	    location: Positional tracking (see LocationSchema).
	    language_metadata: Dict for language-specific scope rules.
	"""

	scope_id: int
	parent_scope_id: int | None = Field(None, description="None indicates global scope")
	scope_type: ScopeType
	visibility: VisibilityModifier = VisibilityModifier.PRIVATE
	namespace: str | None = Field(None, description="Qualified namespace path")

	# Language-agnostic temporal characteristics
	temporal_attributes: dict[str, str] = Field(
		default_factory=lambda: {
			"lifetime": "automatic",  # static/dynamic/manual
			"hoisting": "none",  # declaration/function/none
		}
	)

	# Cross-language reference tracking
	declarations: list[int] = Field(default_factory=list, description="IDs of entities declared in this scope")

	imports: list[str] = Field(default_factory=list, description="Imported namespaces/modules")

	# Positional tracking using existing LocationSchema
	location: LocationSchema

	# Language-specific extensions
	language_metadata: dict[str, Any] = Field(
		default_factory=dict, description="Storage for language-specific scope rules"
	)

	# Metadata for the scope
	metadata: MetadataSchema
