"""Base entity schemas."""

from typing import Annotated

from pydantic import BaseModel, Field

from .scope import ScopeSchema
from .types import EntityType, ExpressionType, StatementType

# --- Base Entity Schema --- #


class BaseEntitySchema(BaseModel):
	"""Base schema for an extracted code entity."""

	id: int
	type: EntityType
	name: str | None = None
	scope: ScopeSchema


# --- Specific Entity Schemas --- #


class ModuleEntitySchema(BaseEntitySchema):
	"""Schema for a module entity."""

	docstring: str | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class ClassEntitySchema(BaseEntitySchema):
	"""Schema for a class entity."""

	docstring: str | None = None
	decorators: list[str] = Field(default_factory=list)
	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)
	base_classes: list[str] = Field(default_factory=list)


class FunctionEntitySchema(BaseEntitySchema):
	"""Schema for a function entity."""

	docstring: str | None = None
	decorators: list[str] = Field(default_factory=list)
	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)
	signature: str | None = None
	is_async: bool = False


class MethodEntitySchema(FunctionEntitySchema):
	"""Schema for a method entity."""

	is_static: bool = False
	is_classmethod: bool = False


class PropertyEntitySchema(BaseEntitySchema):
	"""Schema for a property entity."""

	docstring: str | None = None
	decorators: list[str] = Field(default_factory=list)
	parent_id: int | None = None
	signature: str | None = None


class VariableEntitySchema(BaseEntitySchema):
	"""Schema for a variable, constant, or class field entity."""

	parent_id: int | None = None
	assigned_type_str: str | None = None


class ConstantEntitySchema(VariableEntitySchema):
	"""Schema for a constant entity."""


class ClassFieldEntitySchema(VariableEntitySchema):
	"""Schema for a class field entity."""


class ImportEntitySchema(BaseEntitySchema):
	"""Schema for an import entity."""

	parent_id: int | None = None
	imported_from: str | None = None


class CommentEntitySchema(BaseEntitySchema):
	"""Schema for a comment or docstring entity."""

	parent_id: int | None = None


class DocstringEntitySchema(CommentEntitySchema):
	"""Schema for a docstring entity."""


class TypeAliasEntitySchema(BaseEntitySchema):
	"""Schema for a type alias."""

	parent_id: int | None = None
	alias_for_str: str | None = None


class InterfaceEntitySchema(BaseEntitySchema):
	"""Schema for an interface entity."""

	docstring: str | None = None
	parent_id: int | None = None
	base_classes: list[str] = Field(default_factory=list)


class ProtocolEntitySchema(BaseEntitySchema):
	"""Schema for a protocol entity."""

	docstring: str | None = None
	parent_id: int | None = None
	base_classes: list[str] = Field(default_factory=list)


class EnumEntitySchema(BaseEntitySchema):
	"""Schema for an enum entity."""

	docstring: str | None = None
	parent_id: int | None = None
	base_classes: list[str] = Field(default_factory=list)


class StructEntitySchema(BaseEntitySchema):
	"""Schema for a struct entity (common in languages like C, Go)."""

	parent_id: int | None = None
	docstring: str | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class NamespaceEntitySchema(BaseEntitySchema):
	"""Schema for a namespace entity."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class PackageEntitySchema(BaseEntitySchema):
	"""Schema for a package entity."""

	children: list["AnyEntitySchema"] = Field(default_factory=list)


class TestCaseEntitySchema(FunctionEntitySchema):
	"""Schema for a test case (typically a function)."""


class TestSuiteEntitySchema(ClassEntitySchema):
	"""Schema for a test suite (typically a class)."""


class ObjectEntitySchema(BaseEntitySchema):
	"""Schema for an object entity."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class ArrayEntitySchema(BaseEntitySchema):
	"""Schema for an array entity."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class TableEntitySchema(BaseEntitySchema):
	"""Schema for a table entity."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


# --- Base for Value-like entities ---
class ValueEntitySchema(BaseEntitySchema):
	"""Schema for a value entity."""

	parent_id: int | None = None
	value_str: str | None = None


class IntegerEntitySchema(ValueEntitySchema):
	"""Schema for an integer value."""


class FloatEntitySchema(ValueEntitySchema):
	"""Schema for a float value."""


class StringEntitySchema(ValueEntitySchema):
	"""Schema for a string value."""


class CharEntitySchema(ValueEntitySchema):
	"""Schema for a character value."""


class BooleanEntitySchema(ValueEntitySchema):
	"""Schema for a boolean value."""


class NullEntitySchema(ValueEntitySchema):
	"""Schema for a null/None value."""


class SymbolEntitySchema(ValueEntitySchema):
	"""Schema for a symbol value."""


# --- Numeric Extensions ---
class ComplexEntitySchema(ValueEntitySchema):
	"""Schema for a complex number value."""


class DecimalEntitySchema(ValueEntitySchema):
	"""Schema for a decimal number value."""


# --- Collections and Sequences ---
class ListEntitySchema(BaseEntitySchema):
	"""Schema for a list entity."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class TupleEntitySchema(BaseEntitySchema):
	"""Schema for a tuple entity."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class SetEntitySchema(BaseEntitySchema):
	"""Schema for a set entity."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class RangeEntitySchema(BaseEntitySchema):
	"""Schema for a range entity."""

	parent_id: int | None = None
	value_str: str | None = None


# --- Mapping/Associative Types ---
class MapEntitySchema(BaseEntitySchema):
	"""Schema for a map entity."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class DictEntitySchema(BaseEntitySchema):
	"""Schema for a dictionary entity."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


# --- Type Definitions ---
class GenericEntitySchema(BaseEntitySchema):
	"""Schema for a generic type or template."""

	docstring: str | None = None
	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class TraitEntitySchema(BaseEntitySchema):
	"""Schema for a trait entity."""

	docstring: str | None = None
	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class MixinEntitySchema(BaseEntitySchema):
	"""Schema for a mixin."""

	docstring: str | None = None
	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)
	base_classes: list[str] = Field(default_factory=list)


class UnionEntitySchema(BaseEntitySchema):
	"""Schema for a union type definition (e.g., type X = A | B)."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class IntersectionEntitySchema(BaseEntitySchema):
	"""Schema for an intersection type definition (e.g., type X = A & B)."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class PointerEntitySchema(BaseEntitySchema):
	"""Schema for a pointer type."""

	parent_id: int | None = None
	points_to_str: str | None = None


class ReferenceEntitySchema(BaseEntitySchema):
	"""Schema for a reference type."""

	parent_id: int | None = None
	refers_to_str: str | None = None


class DatetimeEntitySchema(ValueEntitySchema):
	"""Schema for a datetime value."""


class BinaryEntitySchema(ValueEntitySchema):
	"""Schema for a binary data entity."""


class RegexpEntitySchema(ValueEntitySchema):
	"""Schema for a regular expression entity."""


class OptionalEntitySchema(BaseEntitySchema):
	"""Schema for an optional type (e.g., Optional[int])."""

	parent_id: int | None = None
	wrapped_type_str: str | None = None


# --- Macro/Meta Programming ---
class MacroEntitySchema(BaseEntitySchema):
	"""Schema for a macro definition."""

	docstring: str | None = None
	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)
	signature: str | None = None


class AnnotationEntitySchema(BaseEntitySchema):
	"""Schema for an annotation entity."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


# --- Functions and Methods ---
class EventEntitySchema(BaseEntitySchema):
	"""Schema for an event definition."""

	docstring: str | None = None
	parent_id: int | None = None
	signature: str | None = None


# --- Variables and Constants ---


# --- Error/Exception Handling ---
class ExceptionEntitySchema(ClassEntitySchema):
	"""Schema for an exception or error class."""


# --- Code Organization/Meta ---
class DecoratorEntitySchema(BaseEntitySchema):
	"""Schema for a decorator usage."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class LabelEntitySchema(BaseEntitySchema):
	"""Schema for a label entity."""

	parent_id: int | None = None


# --- Documentation Entities ---
class KeyEntitySchema(ValueEntitySchema):
	"""Schema for a key in a key-value pair."""


# --- Data/Configuration Entities ---


# --- Special Cases ---
class UnknownEntitySchema(BaseEntitySchema):
	"""Schema for an unknown entity type."""

	parent_id: int | None = None


# --- Define the Union of ALL specific entity types --- #


class BlockEntitySchema(BaseEntitySchema):
	"""Schema for a code block entity (e.g., Python, JavaScript, SQL)."""

	parent_id: int | None = None
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class StatementEntitySchema(BaseEntitySchema):
	"""
	Represents a statement node in a program.

	- Covers all major statement types (control flow, declarations, assignments, etc.).
	- Most properties are shared with ScopeSchema (location, parent, metadata).
	- Supports nesting (e.g., blocks, compound statements).
	"""

	parent_id: int | None = None
	statement_type: StatementType
	children: list["AnyEntitySchema"] = Field(default_factory=list)


class ExpressionEntitySchema(BaseEntitySchema):
	"""
	Represents an expression node in a program.

	- Covers all major expression types (binary, unary, call, literal, etc.).
	- Can be nested (expressions within expressions).
	"""

	parent_id: int | None = None
	expression_type: ExpressionType
	children: list["AnyEntitySchema"] = Field(default_factory=list)


AnyEntitySchema = Annotated[
	ModuleEntitySchema
	| NamespaceEntitySchema
	| PackageEntitySchema
	| BlockEntitySchema
	| StatementEntitySchema
	| ExpressionEntitySchema
	| IntegerEntitySchema
	| FloatEntitySchema
	| StringEntitySchema
	| CharEntitySchema
	| BooleanEntitySchema
	| NullEntitySchema
	| SymbolEntitySchema
	| ComplexEntitySchema
	| DecimalEntitySchema
	| ArrayEntitySchema
	| ListEntitySchema
	| TupleEntitySchema
	| SetEntitySchema
	| RangeEntitySchema
	| MapEntitySchema
	| DictEntitySchema
	| ObjectEntitySchema
	| TableEntitySchema
	| ClassEntitySchema
	| InterfaceEntitySchema
	| ProtocolEntitySchema
	| StructEntitySchema
	| EnumEntitySchema
	| TypeAliasEntitySchema
	| GenericEntitySchema
	| TraitEntitySchema
	| MixinEntitySchema
	| UnionEntitySchema
	| IntersectionEntitySchema
	| PointerEntitySchema
	| ReferenceEntitySchema
	| DatetimeEntitySchema
	| BinaryEntitySchema
	| RegexpEntitySchema
	| OptionalEntitySchema
	| MacroEntitySchema
	| AnnotationEntitySchema
	| FunctionEntitySchema
	| MethodEntitySchema
	| PropertyEntitySchema
	| TestCaseEntitySchema
	| TestSuiteEntitySchema
	| EventEntitySchema
	| VariableEntitySchema
	| ConstantEntitySchema
	| ClassFieldEntitySchema
	| ExceptionEntitySchema
	| ImportEntitySchema
	| DecoratorEntitySchema
	| LabelEntitySchema
	| CommentEntitySchema
	| DocstringEntitySchema
	| KeyEntitySchema
	| ValueEntitySchema
	| UnknownEntitySchema,
	Field(discriminator="type"),
]

# --- Rebuild all models to resolve forward references --- #

BaseEntitySchema.model_rebuild(force=True)
ModuleEntitySchema.model_rebuild(force=True)
NamespaceEntitySchema.model_rebuild(force=True)
PackageEntitySchema.model_rebuild(force=True)
ValueEntitySchema.model_rebuild(force=True)
IntegerEntitySchema.model_rebuild(force=True)
FloatEntitySchema.model_rebuild(force=True)
StringEntitySchema.model_rebuild(force=True)
CharEntitySchema.model_rebuild(force=True)
BooleanEntitySchema.model_rebuild(force=True)
NullEntitySchema.model_rebuild(force=True)
SymbolEntitySchema.model_rebuild(force=True)
ComplexEntitySchema.model_rebuild(force=True)
DecimalEntitySchema.model_rebuild(force=True)
ArrayEntitySchema.model_rebuild(force=True)
ListEntitySchema.model_rebuild(force=True)
TupleEntitySchema.model_rebuild(force=True)
SetEntitySchema.model_rebuild(force=True)
RangeEntitySchema.model_rebuild(force=True)
MapEntitySchema.model_rebuild(force=True)
DictEntitySchema.model_rebuild(force=True)
ObjectEntitySchema.model_rebuild(force=True)
TableEntitySchema.model_rebuild(force=True)
ClassEntitySchema.model_rebuild(force=True)
InterfaceEntitySchema.model_rebuild(force=True)
ProtocolEntitySchema.model_rebuild(force=True)
StructEntitySchema.model_rebuild(force=True)
EnumEntitySchema.model_rebuild(force=True)
TypeAliasEntitySchema.model_rebuild(force=True)
GenericEntitySchema.model_rebuild(force=True)
TraitEntitySchema.model_rebuild(force=True)
MixinEntitySchema.model_rebuild(force=True)
UnionEntitySchema.model_rebuild(force=True)
IntersectionEntitySchema.model_rebuild(force=True)
PointerEntitySchema.model_rebuild(force=True)
ReferenceEntitySchema.model_rebuild(force=True)
DatetimeEntitySchema.model_rebuild(force=True)
BinaryEntitySchema.model_rebuild(force=True)
RegexpEntitySchema.model_rebuild(force=True)
OptionalEntitySchema.model_rebuild(force=True)
MacroEntitySchema.model_rebuild(force=True)
AnnotationEntitySchema.model_rebuild(force=True)
FunctionEntitySchema.model_rebuild(force=True)
MethodEntitySchema.model_rebuild(force=True)
PropertyEntitySchema.model_rebuild(force=True)
TestCaseEntitySchema.model_rebuild(force=True)
TestSuiteEntitySchema.model_rebuild(force=True)
EventEntitySchema.model_rebuild(force=True)
VariableEntitySchema.model_rebuild(force=True)
ConstantEntitySchema.model_rebuild(force=True)
ClassFieldEntitySchema.model_rebuild(force=True)
ExceptionEntitySchema.model_rebuild(force=True)
ImportEntitySchema.model_rebuild(force=True)
DecoratorEntitySchema.model_rebuild(force=True)
LabelEntitySchema.model_rebuild(force=True)
CommentEntitySchema.model_rebuild(force=True)
DocstringEntitySchema.model_rebuild(force=True)
KeyEntitySchema.model_rebuild(force=True)
UnknownEntitySchema.model_rebuild(force=True)
BlockEntitySchema.model_rebuild(force=True)
StatementEntitySchema.model_rebuild(force=True)
ExpressionEntitySchema.model_rebuild(force=True)
