"""Basic types."""

from enum import Enum, auto

# --- Detailed Entity Sub-Types --- #


class FileModuleLevelEntityType(Enum):
	"""File or module-level entity types."""

	MODULE = auto()  # Source file or logical module
	NAMESPACE = auto()  # Namespace (e.g., C++, C#, Python modules)
	PACKAGE = auto()  # Collection of modules (e.g., Python package, Java package)


class ScalarDataType(Enum):
	"""Scalar (primitive) data and numeric extension entity types."""

	INTEGER = auto()  # Whole numbers (e.g., 1, -42)
	FLOAT = auto()  # Floating-point numbers (e.g., 3.14, -0.001)
	STRING = auto()  # Sequence of characters (e.g., "hello")
	CHAR = auto()  # Single character (e.g., 'a')
	BOOLEAN = auto()  # Logical True/False
	NULL = auto()  # Absence of value (null, None)
	SYMBOL = auto()  # Symbol type (e.g., Ruby, Lisp, JavaScript ES6)
	COMPLEX = auto()  # Complex numbers (e.g., 1+2j)
	DECIMAL = auto()  # Arbitrary-precision decimal numbers


class CollectionEntityType(Enum):
	"""Collection and sequence entity types."""

	ARRAY = auto()  # Homogeneous list/array
	LIST = auto()  # Ordered collection (may allow mixed types)
	TUPLE = auto()  # Ordered, immutable sequence
	SET = auto()  # Unordered collection of unique elements
	RANGE = auto()  # Range type (e.g., Python's range, Ruby's 1..10)


class AssociativeEntityType(Enum):
	"""Mapping or associative structure entity types."""

	MAP = auto()  # Key-value pairs (general mapping)
	DICT = auto()  # Dictionary (Python-style mapping)
	OBJECT = auto()  # Structured object (fields/properties)
	TABLE = auto()  # Tabular data (e.g., database table, DataFrame)


class TypeDefinitionEntityType(Enum):
	"""Type definition and related construct entity types."""

	TYPE_ALIAS = auto()  # Type alias or typedef
	GENERIC = auto()  # Generic type or template (e.g., C++ template, Java generic)
	TRAIT = auto()  # Trait (e.g., Rust, Scala, Dart)
	MIXIN = auto()  # Mixin (multiple inheritance/behavior sharing)
	UNION = auto()  # Value can be one of several types
	INTERSECTION = auto()  # Intersection type (must satisfy multiple types)
	POINTER = auto()  # Memory address/reference (low-level)
	REFERENCE = auto()  # Reference type (as distinct from pointer)
	DATETIME = auto()  # Date and/or time value
	BINARY = auto()  # Raw binary data (bytes, buffers)
	REGEXP = auto()  # Regular expression type (e.g., JavaScript, Perl)
	OPTIONAL = auto()  # Optional/nullable type (e.g., Option, Optional)


class StorageEntityType(Enum):
	"""Variable, constant, and class field entity types."""

	VARIABLE = auto()  # Variable (local, global, or instance)
	CONSTANT = auto()  # Constant value
	CLASS_FIELD = auto()  # Class-level variable/field (static or member variable)


class DataPairEntityType(Enum):
	"""Key-value pair entity types for data structures."""

	KEY = auto()  # Key in a key-value pair (e.g., JSON, TOML)
	VALUE = auto()  # Value in a key-value pair


# --- Scope Types --- #


class ScopeType(Enum):
	"""Scope types."""

	GLOBAL = auto()
	MODULE = auto()
	CLASS = auto()
	FUNCTION = auto()
	METHOD = auto()
	BLOCK = auto()
	LAMBDA = auto()
	NAMESPACE = auto()
	TEMPLATE = auto()  # For generics in Java/C#
	COMPREHENSION = auto()  # Python-specific
	TRY_BLOCK = auto()  # Exception handling scope
	LOOP = auto()  # For loop/while scoping rules
	SWITCH = auto()  # C-style switch blocks
	PATTERN = auto()  # Modern pattern matching (Python/Rust/Swift)


# Not an entity type, but a property of ScopeType entities
class VisibilityModifier(Enum):
	"""Visibility modifiers."""

	PUBLIC = auto()
	PRIVATE = auto()
	PROTECTED = auto()
	INTERNAL = auto()
	PACKAGE = auto()


# --- Statement Types --- #


class StatementType(Enum):
	"""Statement types."""

	IF = auto()
	ELIF = auto()
	ELSE = auto()
	SWITCH = auto()
	CASE = auto()
	DEFAULT = auto()
	FOR = auto()
	FOR_IN = auto()
	FOR_OF = auto()
	WHILE = auto()
	DO_WHILE = auto()
	BREAK = auto()
	CONTINUE = auto()
	RETURN = auto()
	YIELD = auto()
	THROW = auto()
	TRY = auto()
	CATCH = auto()
	FINALLY = auto()
	WITH = auto()
	ASSERT = auto()
	PASS = auto()
	RAISE = auto()
	DEFER = auto()
	GOTO = auto()

	# Declarations
	VARIABLE_DECLARATION = auto()
	CONSTANT_DECLARATION = auto()
	FUNCTION_DECLARATION = auto()
	CLASS_DECLARATION = auto()
	INTERFACE_DECLARATION = auto()
	ENUM_DECLARATION = auto()
	IMPORT = auto()
	EXPORT = auto()
	TYPE_ALIAS_DECLARATION = auto()
	STRUCT_DECLARATION = auto()
	MODULE_DECLARATION = auto()
	NAMESPACE_DECLARATION = auto()
	PROPERTY_DECLARATION = auto()
	EXCEPTION_DECLARATION = auto()
	EVENT_DECLARATION = auto()
	ANNOTATION_DECLARATION = auto()
	MACRO_DECLARATION = auto()

	# Assignment & Expression
	ASSIGNMENT = auto()
	EXPRESSION_STATEMENT = auto()

	# Miscellaneous
	BLOCK = auto()
	EMPTY = auto()
	LABEL = auto()
	COMMENT = auto()
	DOCSTRING = auto()
	UNKNOWN = auto()  # fallback


# --- Expression Types --- #


class ExpressionType(Enum):
	"""Expression types."""

	BINARY_OPERATION = auto()
	UNARY_OPERATION = auto()
	CALL = auto()
	MEMBER_ACCESS = auto()
	SUBSCRIPT = auto()
	CONDITIONAL = auto()
	LITERAL = auto()
	IDENTIFIER = auto()
	ASSIGNMENT = auto()
	LAMBDA = auto()
	OBJECT = auto()
	ARRAY = auto()
	TUPLE = auto()
	MAP = auto()
	SET = auto()
	NEW = auto()
	CAST = auto()
	AWAIT = auto()
	YIELD = auto()
	COMPREHENSION = auto()
	PATTERN_MATCH = auto()
	DECORATOR = auto()
	ANNOTATION = auto()
	UNKNOWN = auto()  # fallback


# --- Main Entity Categories --- #


class EntityType(Enum):
	"""Broad categories of entities, each pointing at its detailed sub-enum."""

	FILE_MODULE_LEVEL = FileModuleLevelEntityType
	SCOPE = ScopeType
	STATEMENT = StatementType
	EXPRESSION = ExpressionType
	SCALAR_DATA = ScalarDataType
	COLLECTION = CollectionEntityType
	ASSOCIATIVE = AssociativeEntityType
	TYPE_DEFINITION = TypeDefinitionEntityType
	STORAGE = StorageEntityType
	DATA_PAIR = DataPairEntityType
	UNKNOWN = type("UnknownEnum", (Enum,), {})  # Fallback
