"""Constants for commit linting."""

import re

# Parsing constants
FOOTER_DETECTION_MIN_LINES = 2  # Minimum number of lines needed for footer detection
FOOTER_MIN_LINE_INDEX = 2  # Minimum line index for footers (after header and blank line)
MIN_BODY_LINE_INDEX = 2  # Minimum line index for body (after header and blank line)
ASCII_MAX_VALUE = 127  # Maximum ASCII character value

# Regex constants
# Regex to parse the commit message structure
# Groups: 1:type, 2:scope, 3:breaking(!), 4:description, 5:body_and_footers
COMMIT_REGEX = re.compile(
	r"^(?P<type>[a-zA-Z]+)"  # 1: Type (case-insensitive letters)
	# 2: Scope (optional, case-insensitive letters, numbers, hyphen, underscore, slash)
	r"(?:\((?P<scope>[a-zA-Z0-9\-_]*(?:/[a-zA-Z0-9\-_]*)?)\))?"
	r"(?P<breaking>!)?"  # 3: Breaking indicator (optional)
	r": (?P<description>.+?)"  # 4: Description (must exist, non-greedy)
	# 5: Everything after header (separated by one blank line)
	r"(?:\r?\n\r?\n(?P<body_and_footers>.*))?$",
	re.DOTALL | re.MULTILINE | re.IGNORECASE,  # Ignore case for type/scope
)

# Regex for identifying footer lines
# The regex handles both case-sensitive BREAKING CHANGE and standard token
# Updated to require uppercase tokens specifically
FOOTER_REGEX = re.compile(
	# Token: BREAKING[ -]CHANGE or uppercase standard token
	r"^(?P<token>(?:BREAKING[ -]CHANGE)|(?:[A-Z][A-Z0-9\-]+))"
	r"(?P<separator>: | #)"  # Separator: ': ' or ' #'
	r"(?P<value_part>.*)",  # Capture rest of the line
	re.MULTILINE | re.DOTALL,  # DOTALL for multi-line values
)

# Regex to detect any potential token format (used for identifying footer blocks)
# This is for identifying blocks that look like footers, regardless of case
POTENTIAL_FOOTER_TOKEN_REGEX = re.compile(
	r"^([A-Za-z][A-Za-z0-9\-]+|[Bb][Rr][Ee][Aa][Kk][Ii][Nn][Gg][ -][Cc][Hh][Aa][Nn][Gg][Ee])(: | #)", re.MULTILINE
)

# Constants for common strings
BREAKING_CHANGE = "BREAKING CHANGE"
BREAKING_CHANGE_HYPHEN = "BREAKING-CHANGE"

# Footer validation specific regex
# Enforce uppercase only token pattern
VALID_FOOTER_TOKEN_REGEX = re.compile(r"^(?:[A-Z][A-Z0-9\-]+|BREAKING[ -]CHANGE)$")

# Regex to validate type and scope (ASCII only, no special characters)
VALID_TYPE_REGEX = re.compile(r"^[a-zA-Z]+$")
VALID_SCOPE_REGEX = re.compile(r"^[a-zA-Z0-9\-_]*(?:/[a-zA-Z0-9\-_]*)*$")

# Regex to detect breaking change token in any case (for validation purposes)
BREAKING_CHANGE_REGEX = re.compile(r"^breaking[ -]change$", re.IGNORECASE)

# Case formats for type, scope, and subject
CASE_FORMATS = {
	"lower-case": lambda s: s.lower() == s,
	"upper-case": lambda s: s.upper() == s,
	"camel-case": lambda s: s and (s[0].islower() and " " not in s and "-" not in s and "_" not in s),
	"kebab-case": lambda s: s.lower() == s and "-" in s and " " not in s and "_" not in s,
	"pascal-case": lambda s: s and (s[0].isupper() and " " not in s and "-" not in s and "_" not in s),
	"sentence-case": lambda s: s and s[0].isupper() and s[1:].lower() == s[1:],
	"snake-case": lambda s: s.lower() == s and "_" in s and " " not in s and "-" not in s,
	"start-case": lambda s: all(w[0].isupper() for w in s.split() if w),
}
