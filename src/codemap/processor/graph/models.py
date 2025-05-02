"""Data models for graph elements (optional)."""

# Currently, data is passed as dictionaries or basic types.
# If more complex structures or validation are needed,
# Pydantic models or dataclasses could be defined here.

# Example (if needed later):
# from pydantic import BaseModel, Field
# from typing import Optional
# from codemap.processor.tree_sitter.base import EntityType

# class CodeEntityNode(BaseModel):
#     entity_id: str = Field(..., description="Unique entity identifier")
#     name: Optional[str] = None
#     entity_type: EntityType
#     start_line: int
#     end_line: int
#     # ... other fields
