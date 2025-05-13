"""Map Tree-sitter supported languages to entity types."""

from enum import Enum

from .entity.base import (
	BooleanEntitySchema,
	CharEntitySchema,
	ClassEntitySchema,
	DecoratorEntitySchema,
	EntityType,
	FloatEntitySchema,
	FunctionEntitySchema,
	ImportEntitySchema,
	IntegerEntitySchema,
	InterfaceEntitySchema,
	MethodEntitySchema,
	ModuleEntitySchema,
	NullEntitySchema,
	RegexpEntitySchema,
	StringEntitySchema,
	TypeAliasEntitySchema,
	VariableEntitySchema,
)
from .languages import NodeTypes, SupportedLanguages

NodeMapping = dict[NodeTypes, tuple[Enum, type]]

COMMON_LITERAL_MAPPING: NodeMapping = {
	"integer": (EntityType.SCALAR_DATA.value.INTEGER, IntegerEntitySchema),
	"decimal_integer_literal": (EntityType.SCALAR_DATA.value.INTEGER, IntegerEntitySchema),
	"hex_integer_literal": (EntityType.SCALAR_DATA.value.INTEGER, IntegerEntitySchema),
	"float": (EntityType.SCALAR_DATA.value.FLOAT, FloatEntitySchema),
	"string": (EntityType.SCALAR_DATA.value.STRING, StringEntitySchema),
	"string_literal": (EntityType.SCALAR_DATA.value.STRING, StringEntitySchema),
	"character_literal": (EntityType.SCALAR_DATA.value.CHAR, CharEntitySchema),
	"true": (EntityType.SCALAR_DATA.value.BOOLEAN, BooleanEntitySchema),
	"false": (EntityType.SCALAR_DATA.value.BOOLEAN, BooleanEntitySchema),
	"null": (EntityType.SCALAR_DATA.value.NULL, NullEntitySchema),
	"none": (EntityType.SCALAR_DATA.value.NULL, NullEntitySchema),
	"regex": (EntityType.TYPE_DEFINITION.value.REGEXP, RegexpEntitySchema),
}

LANGUAGE_NODE_MAPPING: dict[SupportedLanguages, NodeMapping] = {
	"python": {
		"module": (EntityType.FILE_MODULE_LEVEL.value.MODULE, ModuleEntitySchema),
		"class_definition": (EntityType.STORAGE.value.CLASS_FIELD, ClassEntitySchema),
		"function_definition": (EntityType.STATEMENT.value.FUNCTION_DECLARATION, FunctionEntitySchema),
		"decorator": (EntityType.EXPRESSION.value.DECORATOR, DecoratorEntitySchema),
		"import_statement": (EntityType.STATEMENT.value.IMPORT, ImportEntitySchema),
		"assignment": (EntityType.STATEMENT.value.ASSIGNMENT, VariableEntitySchema),
		**COMMON_LITERAL_MAPPING,
	},
	"javascript": {
		"program": (EntityType.FILE_MODULE_LEVEL.value.MODULE, ModuleEntitySchema),
		"class_declaration": (EntityType.SCOPE.value.CLASS, ClassEntitySchema),
		"function_declaration": (EntityType.STATEMENT.value.FUNCTION_DECLARATION, FunctionEntitySchema),
		"method_definition": (EntityType.SCOPE.value.METHOD, MethodEntitySchema),
		"import_statement": (EntityType.STATEMENT.value.IMPORT, ImportEntitySchema),
		"variable_declaration": (EntityType.STATEMENT.value.VARIABLE_DECLARATION, VariableEntitySchema),
		**COMMON_LITERAL_MAPPING,
	},
	"typescript": {
		"program": (EntityType.FILE_MODULE_LEVEL.value.MODULE, ModuleEntitySchema),
		"class_declaration": (EntityType.SCOPE.value.CLASS, ClassEntitySchema),
		"function_declaration": (EntityType.STATEMENT.value.FUNCTION_DECLARATION, FunctionEntitySchema),
		"method_signature": (EntityType.SCOPE.value.METHOD, MethodEntitySchema),
		"import_statement": (EntityType.STATEMENT.value.IMPORT, ImportEntitySchema),
		"variable_declaration": (EntityType.STATEMENT.value.VARIABLE_DECLARATION, VariableEntitySchema),
		"type_alias_declaration": (EntityType.TYPE_DEFINITION.value.TYPE_ALIAS, TypeAliasEntitySchema),
		"interface_declaration": (EntityType.STATEMENT.value.INTERFACE_DECLARATION, InterfaceEntitySchema),
		**COMMON_LITERAL_MAPPING,
	},
	# extend for other SupportedLanguage entries...
}
