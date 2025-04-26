"""Validators for commit message components."""

from .constants import (
	ASCII_MAX_VALUE,
	BREAKING_CHANGE,
	BREAKING_CHANGE_HYPHEN,
	BREAKING_CHANGE_REGEX,
	CASE_FORMATS,
	VALID_FOOTER_TOKEN_REGEX,
	VALID_SCOPE_REGEX,
	VALID_TYPE_REGEX,
)


class CommitValidators:
	"""Collection of validator methods for different parts of commit messages."""

	@staticmethod
	def validate_footer_token(token: str) -> bool:
		"""
		Validate a footer token according to the Conventional Commits spec.

		According to the spec:
		1. Tokens MUST use hyphens instead of spaces
		2. BREAKING CHANGE must be uppercase
		3. Footer tokens should be ALL UPPERCASE
		4. Footer tokens should follow format with - for spaces
		5. No special characters or Unicode (non-ASCII) characters allowed

		Returns:
		    bool: True if token is valid, False otherwise

		"""
		# Check if token is a breaking change token in any case
		if BREAKING_CHANGE_REGEX.match(token.lower()):
			# If it's a breaking change token, it MUST be uppercase
			return token in (BREAKING_CHANGE, BREAKING_CHANGE_HYPHEN)

		# Check for special characters (except hyphens which are allowed)
		if any(c in token for c in "!@#$%^&*()+={}[]|\\:;\"'<>,./?"):
			return False

		# Check for non-ASCII characters
		if any(ord(c) > ASCII_MAX_VALUE for c in token):
			return False

		# Must match valid token pattern (uppercase, alphanumeric with hyphens)
		if not VALID_FOOTER_TOKEN_REGEX.match(token):
			return False

		# Check for spaces (must use hyphens instead, except for BREAKING CHANGE)
		return not (" " in token and token != BREAKING_CHANGE)

	@staticmethod
	def validate_type_and_scope(type_value: str, scope_value: str | None) -> list[str]:
		"""
		Validate type and scope values according to the spec.

		Type must contain only letters.
		Scope must contain only letters, numbers, hyphens, and slashes.
		Both must be ASCII-only.

		Args:
		    type_value (str): The commit message type
		    scope_value (str | None): The optional scope

		Returns:
		    list[str]: List of error messages, empty if valid

		"""
		errors = []

		# Check type (no special chars or unicode)
		if not VALID_TYPE_REGEX.match(type_value):
			errors.append(f"Invalid type '{type_value}'. Types must contain only letters (a-z, A-Z).")
		elif any(ord(c) > ASCII_MAX_VALUE for c in type_value):
			errors.append(f"Invalid type '{type_value}'. Types must contain only ASCII characters.")

		# Check scope (if present)
		if scope_value is not None:
			if scope_value == "":
				errors.append("Scope cannot be empty when parentheses are used.")
			elif not VALID_SCOPE_REGEX.match(scope_value):
				errors.append(
					f"Invalid scope '{scope_value}'. Scopes must contain only letters, numbers, hyphens, and slashes."
				)
			elif any(ord(c) > ASCII_MAX_VALUE for c in scope_value):
				errors.append(f"Invalid scope '{scope_value}'. Scopes must contain only ASCII characters.")
			elif any(c in scope_value for c in "!@#$%^&*()+={}[]|\\:;\"'<>,. "):
				errors.append(f"Invalid scope '{scope_value}'. Special characters are not allowed in scopes.")

		return errors

	@staticmethod
	def validate_case(text: str, case_format: str | list[str]) -> bool:
		"""
		Validate if the text follows the specified case format.

		Args:
		    text (str): The text to validate
		    case_format (str or list): The case format(s) to check

		Returns:
		    bool: True if text matches any of the specified case formats

		"""
		if isinstance(case_format, list):
			return any(CommitValidators.validate_case(text, fmt) for fmt in case_format)

		# Get the validator function for the specified case format
		validator = CASE_FORMATS.get(case_format)
		if not validator:
			# Default to allowing any case if invalid format specified
			return True

		return validator(text)

	@staticmethod
	def validate_length(text: str | None, min_length: int, max_length: float) -> bool:
		"""
		Validate if text length is between min and max length.

		Args:
		    text (str | None): The text to validate, or None
		    min_length (int): Minimum allowed length
		    max_length (int | float): Maximum allowed length

		Returns:
		    bool: True if text length is valid, False otherwise

		"""
		if text is None:
			return min_length == 0

		text_length = len(text)
		return min_length <= text_length < max_length

	@staticmethod
	def validate_enum(text: str, allowed_values: list[str]) -> bool:
		"""
		Validate if text is in the allowed values.

		Args:
		    text (str): The text to validate
		    allowed_values (list): The allowed values

		Returns:
		    bool: True if text is in allowed values, False otherwise

		"""
		# Allow any value if no allowed values are specified
		if not allowed_values:
			return True

		return text.lower() in (value.lower() for value in allowed_values)

	@staticmethod
	def validate_empty(text: str | None, should_be_empty: bool) -> bool:
		"""
		Validate if text is empty or not based on configuration.

		Args:
		    text (str | None): The text to validate
		    should_be_empty (bool): True if text should be empty, False if not

		Returns:
		    bool: True if text empty status matches should_be_empty

		"""
		is_empty = text is None or text.strip() == ""
		return is_empty == should_be_empty

	@staticmethod
	def validate_ends_with(text: str | None, suffix: str, should_end_with: bool) -> bool:
		"""
		Validate if text ends with a specific suffix.

		Args:
		    text (str | None): The text to validate
		    suffix (str): The suffix to check for
		    should_end_with (bool): True if text should end with suffix

		Returns:
		    bool: True if text ending matches expectation

		"""
		if text is None:
			return not should_end_with

		ends_with = text.endswith(suffix)
		return ends_with == should_end_with

	@staticmethod
	def validate_starts_with(text: str | None, prefix: str, should_start_with: bool) -> bool:
		"""
		Validate if text starts with a specific prefix.

		Args:
		    text (str | None): The text to validate
		    prefix (str): The prefix to check for
		    should_start_with (bool): True if text should start with prefix

		Returns:
		    bool: True if text starting matches expectation

		"""
		if text is None:
			return not should_start_with

		starts_with = text.startswith(prefix)
		return starts_with == should_start_with

	@staticmethod
	def validate_line_length(text: str | None, max_line_length: float) -> list[int]:
		"""
		Validate line lengths in multiline text.

		Args:
		    text (str | None): The text to validate
		    max_line_length (int | float): Maximum allowed line length

		Returns:
		    list: List of line numbers with errors (0-indexed)

		"""
		if text is None or max_line_length == float("inf"):
			return []

		lines = text.splitlines()
		return [i for i, line in enumerate(lines) if len(line) > max_line_length]

	@staticmethod
	def validate_leading_blank(text: str | None, required_blank: bool) -> bool:
		"""
		Validate if text starts with a blank line.

		Args:
		    text (str | None): The text to validate
		    required_blank (bool): True if text should start with blank line

		Returns:
		    bool: True if text leading blank matches expectation

		"""
		if text is None:
			return not required_blank

		lines = text.splitlines()
		has_leading_blank = len(lines) > 0 and (len(lines) == 1 or not lines[0].strip())
		return has_leading_blank == required_blank

	@staticmethod
	def validate_trim(text: str | None) -> bool:
		"""
		Validate if text has no leading/trailing whitespace.

		Args:
		    text (str | None): The text to validate

		Returns:
		    bool: True if text has no leading/trailing whitespace

		"""
		if text is None:
			return True

		return text == text.strip()

	@staticmethod
	def validate_contains(text: str | None, substring: str, should_contain: bool) -> bool:
		"""
		Validate if text contains a specific substring.

		Args:
		    text (str | None): The text to validate
		    substring (str): The substring to check for
		    should_contain (bool): True if text should contain substring

		Returns:
		    bool: True if text contains substring matches expectation

		"""
		if text is None:
			return not should_contain

		contains = substring in text
		return contains == should_contain
