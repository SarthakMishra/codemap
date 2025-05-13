"""Utility functions for scripts."""


def escape_string(s: str) -> str:
	"""Encode special characters in a string with proper escape sequences and surround with quotes.

	Returns string with double quotes by default, single quotes if a double quote is present.
	"""
	result = []
	has_double_quote = '"' in s
	for char in s:
		if char in "\\'\"\a\b\f\n\r\t\v":
			# Handle common escape sequences
			escape_map = {
				"\\": "\\\\",
				"'": "\\'",
				'"': '\\"',
				"\a": "\\a",
				"\b": "\\b",
				"\f": "\\f",
				"\n": "\\n",
				"\r": "\\r",
				"\t": "\\t",
				"\v": "\\v",
			}
			result.append(escape_map[char])
		elif ord(char) < 32 or ord(char) > 126:
			# Handle non-printable and non-ASCII characters
			if ord(char) <= 0xFF:
				result.append(f"\\x{ord(char):02x}")
			elif ord(char) <= 0xFFFF:
				result.append(f"\\u{ord(char):04x}")
			else:
				result.append(f"\\U{ord(char):08x}")
		else:
			result.append(char)

	escaped_string = "".join(result)
	# Add appropriate quotes based on content
	if has_double_quote:
		return f"'{escaped_string}'"

	return f'"{escaped_string}"'
