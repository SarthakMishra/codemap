"""Pattern-based code search tool using ast-grep."""

# ruff: noqa: E501
import ast
import json
import logging
from pathlib import Path

from pydantic_ai import RunContext
from pydantic_ai.tools import Tool

from codemap.config import ConfigLoader
from codemap.processor.semantic_search import AstGrepEngine

logger = logging.getLogger(__name__)


def validate_and_suggest_pattern(pattern: str, language: str | None = None) -> tuple[bool, str, str | None]:
	"""Validate an ast-grep pattern and suggest fixes if invalid.

	Args:
	    pattern: The pattern to validate
	    language: Optional language hint

	Returns:
	    Tuple of (is_valid, error_message, suggested_pattern)
	"""
	# Default to Python if no language specified
	lang = language or "python"

	# Common invalid patterns and their fixes
	common_fixes = {
		"python": {
			# Text search attempts
			r"codemap gen": "# For CLI commands, try: $FUNC('gen', $$$ARGS) or look for string literals",
			r"gen command": "# For command functions, try: def $NAME($$$PARAMS): $$$BODY with constraints",
			r"cli command": "# For CLI patterns, try: @click.command() or def $NAME($$$PARAMS): $$$BODY",
			# Common function patterns
			r"function ": "def $NAME($$$PARAMS): $$$BODY",
			r"class ": "class $NAME($$$BASES): $$$BODY",
			r"import ": "import $MODULE or from $MODULE import $$$ITEMS",
		}
	}

	# Quick syntax validation for Python
	if lang == "python":
		try:
			# Try to parse as Python - ast-grep patterns should be valid Python syntax
			ast.parse(pattern)
			return True, "", None
		except SyntaxError as e:
			error_msg = f"Invalid Python syntax in pattern: {e}"

			# Check for common fixes
			fixes = common_fixes.get("python", {})
			for bad_pattern, suggestion in fixes.items():
				if bad_pattern.lower() in pattern.lower():
					return False, error_msg, suggestion

			# Generic suggestions based on the error
			if "def " in pattern and "$$$" not in pattern:
				suggestion = "Try: def $NAME($$$PARAMS): $$$BODY"
			elif "class " in pattern and "$$$" not in pattern:
				suggestion = "Try: class $NAME($$$BASES): $$$BODY"
			elif "import " in pattern and "$$$" not in pattern:
				suggestion = "Try: import $MODULE or from $MODULE import $$$ITEMS"
			elif " " in pattern and not any(c in pattern for c in "()[]{}"):
				suggestion = (
					"Multi-word patterns need proper syntax. "
					"Try function calls: $FUNC($$$ARGS) or string literals: '$TEXT'"
				)
			else:
				suggestion = "Pattern must be valid Python syntax with meta variables ($VAR, $$VAR, $$$VAR)"

			return False, error_msg, suggestion

	# For other languages, basic validation
	if not pattern.strip():
		return False, "Empty pattern", "Provide a valid code pattern"

	return True, "", None


async def pattern_search(
	ctx: RunContext[ConfigLoader],  # noqa: ARG001
	pattern: str,
	constraints: str | None = None,
	file_pattern: str | None = None,
	language: str | None = None,
	limit: int = 10,
) -> str:
	"""Search code using ast-grep patterns with comprehensive pattern syntax support.

	ast-grep uses AST-based pattern matching with meta variables as placeholders.
	Patterns must be valid, parsable code with meta variable substitutions.

	Args:
	    ctx: The run context (unused but required for tool interface)
	    pattern: ast-grep pattern to search for. Must be valid code syntax.
	    constraints: Optional JSON string with constraints for pattern variables.
	    file_pattern: Optional glob pattern to limit files (e.g., "src/**/*.py")
	    language: Optional language override (python, javascript, typescript, etc.)
	    limit: Maximum number of results to return (default: 10)

	Meta Variable Syntax:
	    $VAR      - Matches single named AST nodes (identifiers, expressions, etc.)
	    $$VAR     - Matches single unnamed AST nodes (operators, punctuation, etc.)
	    $$$VAR    - Matches multiple consecutive nodes (parameters, arguments, statements)

	Python Pattern Examples:
	    Basic Patterns:
	    - "def $NAME($$$PARAMS): $$$BODY"           # Function definitions
	    - "class $NAME($$$BASES): $$$BODY"         # Class definitions
	    - "import $MODULE"                         # Import statements
	    - "from $MODULE import $$$ITEMS"           # From imports
	    - "if $CONDITION: $$$BODY"                 # If statements
	    - "for $VAR in $ITERABLE: $$$BODY"         # For loops
	    - "try: $$$BODY"                           # Try blocks
	    - "except $EXCEPTION: $$$BODY"             # Exception handlers
	    - "with $CONTEXT as $VAR: $$$BODY"         # Context managers
	    - "$VAR = $VALUE"                          # Variable assignments
	    - "$OBJ.$ATTR"                             # Attribute access
	    - "$FUNC($$$ARGS)"                         # Function calls
	    - "[$$$ITEMS]"                             # List literals
	    - "{$$$ITEMS}"                             # Dict literals
	    - "@$DECORATOR"                            # Decorators
	    - "assert $CONDITION"                      # Assertions
	    - "return $VALUE"                          # Return statements
	    - "yield $VALUE"                           # Yield statements
	    - "raise $EXCEPTION"                       # Raise statements
	    - "lambda $$$PARAMS: $BODY"                # Lambda functions

	JavaScript/TypeScript Pattern Examples:
	    Basic Patterns:
	    - "function $NAME($$$PARAMS) { $$$BODY }"  # Function declarations
	    - "const $NAME = $VALUE"                   # Const declarations
	    - "let $NAME = $VALUE"                     # Let declarations
	    - "var $NAME = $VALUE"                     # Var declarations
	    - "$OBJ.$PROP"                             # Property access
	    - "$OBJ[$KEY]"                             # Bracket notation
	    - "if ($CONDITION) { $$$BODY }"            # If statements
	    - "for ($INIT; $CONDITION; $UPDATE) { $$$BODY }" # For loops
	    - "while ($CONDITION) { $$$BODY }"         # While loops
	    - "try { $$$BODY } catch ($ERROR) { $$$HANDLER }" # Try-catch
	    - "new $CLASS($$$ARGS)"                    # Constructor calls
	    - "class $NAME extends $BASE { $$$BODY }"  # Class inheritance
	    - "import { $$$IMPORTS } from '$MODULE'"   # ES6 imports
	    - "export { $$$EXPORTS }"                  # ES6 exports
	    - "async function $NAME($$$PARAMS) { $$$BODY }" # Async functions
	    - "await $PROMISE"                         # Await expressions
	    - "($$$PARAMS) => $BODY"                   # Arrow functions
	    - "console.log($$$ARGS)"                   # Console logging
	    - "$VAR?.property"                         # Optional chaining
	    - "$VAR ?? $DEFAULT"                       # Nullish coalescing

	Advanced Pattern Examples:
	    Method Calls with Chaining:
	    - "$OBJ.$METHOD1().$METHOD2($$$ARGS)"      # Method chaining
	    - "$ARRAY.map($CALLBACK)"                  # Array methods
	    - "$PROMISE.then($CALLBACK)"               # Promise chains

	    Class and Object Patterns:
	    - "class $NAME { constructor($$$PARAMS) { $$$BODY } }" # Constructors
	    - "get $PROP() { $$$BODY }"                # Getter methods
	    - "set $PROP($VALUE) { $$$BODY }"          # Setter methods
	    - "static $METHOD($$$PARAMS) { $$$BODY }"  # Static methods

	    TypeScript Specific:
	    - "interface $NAME { $$$PROPS }"           # Interface definitions
	    - "type $NAME = $TYPE"                     # Type aliases
	    - "enum $NAME { $$$VALUES }"               # Enum definitions
	    - "function $NAME<$$$GENERICS>($$$PARAMS): $RETURN_TYPE { $$$BODY }" # Generic functions

	Constraint Examples (JSON format):
	    Find test functions:
	    pattern="def $NAME($$$PARAMS): $$$BODY"
	    constraints='{"NAME": {"regex": "^test_"}}'

	    Find private methods:
	    pattern="def $NAME($$$PARAMS): $$$BODY"
	    constraints='{"NAME": {"regex": "^_"}}'

	    Find specific imports:
	    pattern="import $MODULE"
	    constraints='{"MODULE": {"regex": "^(os|sys|json)$"}}'

	    Find error handling:
	    pattern="except $EXCEPTION: $$$BODY"
	    constraints='{"EXCEPTION": {"regex": "(Error|Exception)"}}'

	Common Pitfalls to Avoid:
	    ❌ Invalid: "$LEFT $OP $RIGHT"             # Operators need special handling
	    ✅ Valid:   Use constraints or kind matching for binary expressions

	    ❌ Invalid: "obj.$KIND foo() {}"           # Keywords can't be meta variables
	    ✅ Valid:   Use specific patterns for getters/setters

	    ❌ Invalid: "obj.on$EVENT"                 # Partial text in identifiers
	    ✅ Valid:   "$OBJ.on$EVENT" (if EVENT is a complete identifier)

	    ❌ Invalid: '"Hello $WORLD"'               # Meta vars inside strings
	    ✅ Valid:   Use regex constraints on string patterns

	    ❌ Invalid: "$var" (lowercase)             # Meta vars must be uppercase
	    ✅ Valid:   "$VAR", "$NAME", "$VALUE"

	Pattern Matching Notes:
	    - Patterns must be syntactically valid code
	    - $VAR matches single named nodes (identifiers, expressions)
	    - $$VAR matches single unnamed nodes (operators, punctuation)
	    - $$$VAR matches multiple consecutive nodes greedily
	    - Use constraints for complex filtering (regex, text matching)
	    - Incomplete code may work due to error recovery but isn't guaranteed

	Returns:
	    Formatted search results with code snippets, file locations, and context

	Example Usage Scenarios:
	    Find all async functions:
	    pattern="async def $NAME($$$PARAMS): $$$BODY"

	    Find logger usage:
	    pattern="logger.$LEVEL($$$ARGS)"
	    constraints='{"LEVEL": {"regex": "^(debug|info|warning|error|critical)$"}}'

	    Find class inheritance:
	    pattern="class $NAME($BASE): $$$BODY"

	    Find exception handling:
	    pattern="try: $$$BODY"

	    Find TODO comments (if language supports):
	    pattern="# TODO: $$$TEXT"

	    Find deprecated decorators:
	    pattern="@deprecated"

	    Find database queries:
	    pattern="$DB.execute($QUERY)"

	    Find security-sensitive patterns:
	    pattern="eval($CODE)"
	"""
	try:
		# Validate pattern first
		is_valid, error_msg, suggestion = validate_and_suggest_pattern(pattern, language)
		if not is_valid:
			result = f"❌ **Invalid Pattern:** `{pattern}`\n\n"
			result += f"**Error:** {error_msg}\n\n"
			if suggestion:
				result += f"**Suggestion:** {suggestion}\n\n"

			result += "**Pattern Requirements:**\n"
			result += "- Must be valid, parseable code syntax\n"
			result += "- Use meta variables: `$VAR` (single nodes), `$$VAR` (operators), `$$$VAR` (multiple nodes)\n"
			result += "- Examples: `def $NAME($$$PARAMS): $$$BODY`, `$OBJ.$METHOD($$$ARGS)`, `import $MODULE`\n\n"

			result += "**For CLI commands, try:**\n"
			result += '- `def $NAME($$$PARAMS): $$$BODY` with constraints `{"NAME": {"regex": "gen"}}`\n'
			result += "- `@click.command()` to find Click decorators\n"
			result += "- `$FUNC('gen', $$$ARGS)` for function calls with 'gen'\n"

			return result

		engine = AstGrepEngine()

		# Parse constraints if provided
		parsed_constraints = None
		if constraints:
			try:
				parsed_constraints = json.loads(constraints)
			except json.JSONDecodeError as e:
				return f"Invalid constraints JSON: {e}"

		# Get files to search
		file_paths = None
		if file_pattern:
			try:
				file_paths = list(Path.cwd().glob(file_pattern))
				if not file_paths:
					return f"No files found matching pattern: {file_pattern}"
			except (OSError, ValueError) as e:
				return f"Invalid file pattern: {e}"

		# Execute search
		try:
			results = engine.search_pattern(
				pattern=pattern,
				file_paths=file_paths,
				language=language,
				constraints=parsed_constraints,
				limit=limit,
			)
		except RuntimeError as e:
			if "cannot get matcher" in str(e):
				return (
					f"❌ **AST Pattern Error:** `{pattern}`\n\n"
					f"**Error:** {e}\n\n"
					f"**Common Issues:**\n"
					f"- Pattern contains invalid syntax that ast-grep cannot parse\n"
					f"- Multiple unconnected AST nodes (like `codemap gen` - two separate identifiers)\n"
					f"- Missing proper code structure around meta variables\n\n"
					f"**Solutions:**\n"
					f"- For function names: `def $NAME($$$PARAMS): $$$BODY` with constraints\n"
					f"- For command calls: `$FUNC('command_name', $$$ARGS)`\n"
					f"- For string literals: `'$TEXT'` or `\"$TEXT\"`\n"
					f"- For imports: `import $MODULE` or `from $MODULE import $$$ITEMS`\n\n"
					f"**For '{pattern}' specifically:**\n"
					f"- If looking for CLI commands: try `def $NAME($$$PARAMS): $$$BODY` "
					f'with constraints `{{"NAME": {{"regex": "gen"}}}}`\n'
					f"- If looking for function calls: try `$FUNC($$$ARGS)` with constraints\n"
					f"- If looking for string literals: try `'gen'` or `\"gen\"`"
				)
			return f"AST search engine error: {e}"

		if not results:
			return f"No results found for pattern: '{pattern}'"

		# Format results
		output = f"Found {len(results)} result(s) for pattern: `{pattern}`\n\n"

		if parsed_constraints:
			output += f"**Constraints:** {constraints}\n\n"

		for i, result in enumerate(results, 1):
			output += f"### Result {i}\n\n"
			output += result.to_formatted_string()
			if i < len(results):
				output += "---\n\n"

		return output.strip()

	except Exception as e:
		msg = f"Pattern search failed: {e}"
		logger.exception(msg)
		return msg


# Create the tool
pattern_search_tool = Tool(
	pattern_search,
	takes_ctx=True,
	name="pattern_search",
	description=(
		"Search code using ast-grep patterns with comprehensive syntax support. "
		"Provides semantic pattern matching across 20+ languages using AST-based patterns. "
		"Use $VAR for single nodes, $$VAR for operators, $$$VAR for multiple nodes. "
		"Supports constraints with regex patterns for advanced filtering. "
		"Much more precise and powerful than text-based search."
	),
)
