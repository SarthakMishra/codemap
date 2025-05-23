"""Parsing utilities for commit messages."""

from __future__ import annotations

import re
from typing import Any, Protocol, cast

from .constants import (
	COMMIT_REGEX,
	FOOTER_DETECTION_MIN_LINES,
	FOOTER_REGEX,
	POTENTIAL_FOOTER_TOKEN_REGEX,
)


class MatchLike(Protocol):
	"""Protocol for objects that behave like re.Match."""

	def groupdict(self) -> dict[str, Any]:
		"""Return the dictionary mapping group names to the matched values."""
		...

	def group(self, group_id: int | str = 0) -> str | None:
		"""Return the match group by number or name."""
		...


class CommitParser:
	"""Parser for conventional commit messages.

	This parser handles parsing and validation of commit messages following the Conventional Commits
	specification. It supports extracting commit type, scope, description, body, and footers.
	"""

	def __init__(self) -> None:
		"""Initialize the commit parser."""
		self._commit_regex = COMMIT_REGEX
		self._footer_regex = FOOTER_REGEX
		self._potential_footer_token_regex = POTENTIAL_FOOTER_TOKEN_REGEX

	def parse_commit(self, message: str) -> MatchLike | None:
		"""Parse a commit message using the main regex pattern.

		This method parses the commit message according to the Conventional Commits specification,
		extracting the header, body, and footers. It handles cases where footers might not be
		immediately detected by the main regex pattern.

		Args:
		    message: The raw commit message string to parse.

		Returns:
		    A MatchLike object containing the parsed commit components (type, scope, description,
		    body, footers) if successful, or None if the message doesn't match the expected format.
		    The returned object provides access to match groups via group() and groupdict() methods,
		    with the addition of a 'footers' group that may be detected beyond the main regex match.
		"""
		match = self._commit_regex.match(message.strip())
		if match:
			# Shim for tests accessing match.group("footers") directly
			match_dict = match.groupdict()
			body_and_footers = match_dict.get("body_and_footers")
			# Always get the footers properly, even if we have to look beyond the regex
			_, footers_text = self.split_body_footers(body_and_footers)

			# If regex didn't capture footers but we detected potential footers in the message
			if not footers_text and len(message.strip().splitlines()) > FOOTER_DETECTION_MIN_LINES:
				message_lines = message.strip().splitlines()
				for i in range(len(message_lines) - 1):
					# Look for a line that looks like a footer (token: value or token #value)
					line = message_lines[i].strip()
					if self._potential_footer_token_regex.match(line):
						# This might be a footer
						footers_text = "\n".join(message_lines[i:])
						break

			class MatchWithFooters:
				"""Wrapper for regex match that adds footer text support.

				This class extends a regex match object to include footer text that may have been
				detected beyond the original regex match boundaries.

				Args:
				    original_match: The original regex match object.
				    footers_text: The detected footer text, if any.
				"""

				def __init__(self, original_match: re.Match[str], footers_text: str | None) -> None:
					"""Initialize the match wrapper with original match and footer text."""
					self._original_match = original_match
					self._footers_text = footers_text

				def groupdict(self) -> dict[str, Any]:
					"""Return a dictionary of all named subgroups of the match.

					The dictionary includes both the original match groups and the additional
					'footers' group if footer text was detected.

					Returns:
					    A dictionary containing all named match groups plus the 'footers' group.
					"""
					d = self._original_match.groupdict()
					d["footers"] = self._footers_text
					return d

				def group(self, group_id: int | str = 0) -> str | None:
					"""Return subgroup(s) of the match by group identifier.

					Args:
					    group_id: Either a group number (0 returns entire match) or group name.
					             Special case: 'footers' returns the detected footer text.

					Returns:
					    The matched subgroup or None if the group wasn't matched. Returns footer
					    text when group_id is 'footers'.
					"""
					if group_id == "footers":
						return self._footers_text
					return self._original_match.group(group_id)

			return cast("MatchLike", MatchWithFooters(match, footers_text))
		return None

	def parse_footers(self, footers_str: str | None) -> list[dict[str, Any]]:
		"""Parses commit footers from a string, handling multi-line values.

		Parses footer lines according to Conventional Commits specification, where each footer consists
		of a token, separator, and value. Handles both strict uppercase tokens and potential invalid
		footers for error reporting. Preserves multi-line values and blank lines within footer values.

		Args:
		    footers_str: The string containing footer lines to parse. May be None if no footers exist.

		Returns:
		    A list of dictionaries, where each dictionary represents a parsed footer with keys:
		    - 'token': The footer token (e.g., 'Signed-off-by')
		    - 'separator': The separator used (': ' or ' #')
		    - 'value': The footer value, which may span multiple lines

		Note:
		    For invalid footers (those not matching strict regex but looking like footers), the
		    dictionary will still be created but marked as invalid during validation.
		"""
		if not footers_str:
			return []

		lines = footers_str.strip().splitlines()
		footers: list[dict[str, Any]] = []
		current_footer: dict[str, Any] | None = None
		current_value_lines: list[str] = []

		def finalize_footer() -> None:
			"""Finalizes the current footer by joining its value lines and adding to footers list.

			This helper function:
			1. Joins all accumulated value lines for the current footer with newlines
			2. Strips whitespace from the resulting value
			3. Adds the completed footer to the footers list
			4. Resets the current_footer and current_value_lines for the next footer

			Only executes if there is a current_footer being processed.
			"""
			nonlocal current_footer, current_value_lines
			if current_footer is not None:
				footer_dict: dict[str, Any] = current_footer
				footer_dict["value"] = "\n".join(current_value_lines).strip()
				footers.append(footer_dict)
				current_footer = None
				current_value_lines = []

		i = 0
		while i < len(lines):
			line = lines[i]
			line_strip = line.strip()

			# Skip blank lines
			if not line_strip:
				if current_footer:
					# If we're in a footer value, preserve blank lines as part of the value
					current_value_lines.append("")
				i += 1
				continue

			# Check if line starts a new footer (using the strict uppercase pattern)
			footer_match = self._footer_regex.match(line_strip)

			# Check if line looks like a footer but doesn't match strict footer regex
			# This is for error reporting, not for accepting lowercase tokens
			potential_footer = False
			if not footer_match:
				# Check for patterns like "TOKEN: value" or "TOKEN # value"
				# even if the token has special characters or is not uppercase
				if ":" in line_strip:
					token_part, value_part = line_strip.split(":", 1)
					potential_footer = bool(token_part.strip() and not token_part.strip().startswith((" ", "\t")))
				elif " #" in line_strip:
					token_part, value_part = line_strip.split(" #", 1)
					potential_footer = bool(token_part.strip() and not token_part.strip().startswith((" ", "\t")))

			# Determine if line continues a footer or starts a new one
			if footer_match and (current_footer is None or not line.startswith((" ", "\t"))):
				# This is a new footer start
				finalize_footer()

				token = footer_match.group("token")
				separator = footer_match.group("separator")
				value_part = footer_match.group("value_part")

				# Create footer object
				current_footer = {
					"token": token,
					"separator": separator,
					"value": "",  # Will be set when finalized
				}

				current_value_lines.append(value_part)
			elif potential_footer:
				# This is a potential footer that doesn't match our strict regex
				# We'll finalize any current footer and keep track of this invalid one
				finalize_footer()

				# Extract token and value for error reporting
				if ":" in line_strip:
					token, value = line_strip.split(":", 1)
				else:
					token, value = line_strip.split(" #", 1)

				token = token.strip()

				# Add as an invalid footer for error reporting
				current_footer = {
					"token": token,
					"separator": ": " if ":" in line_strip else " #",
					"value": value.strip(),
				}
				current_value_lines = [value.strip()]
				finalize_footer()  # Immediately finalize for error reporting
			elif current_footer:
				# This is a continuation of the current footer value
				current_value_lines.append(line)
			else:
				# Not a recognized footer line and not in a footer value
				# This will be handled during validation
				pass

			i += 1

		# Finalize the last footer if any
		finalize_footer()

		return footers

	def split_body_footers(self, body_and_footers_str: str | None) -> tuple[str | None, str | None]:
		"""Splits the text after the header into body and footers.

		Args:
		    body_and_footers_str: The string containing both body and footers text, or None.

		Returns:
		    A tuple containing:
		        - First element: The body text as a string, or None if empty/not present
		        - Second element: The footers text as a string, or None if empty/not present
		"""
		if not body_and_footers_str:
			return None, None

		# Regular case
		blocks_with_separators = re.split(r"(?<=\S)(\r?\n\r?\n)(?=\S)", body_and_footers_str)
		processed_blocks = []
		temp_block = ""
		for part in blocks_with_separators:
			temp_block += part
			if temp_block.endswith(("\n\n", "\r\n\r\n")):
				if temp_block.strip():
					processed_blocks.append(temp_block)
				temp_block = ""
		if temp_block.strip():
			processed_blocks.append(temp_block)

		if not processed_blocks:
			return body_and_footers_str.strip() or None, None

		footer_blocks = []
		num_blocks = len(processed_blocks)

		for i in range(num_blocks - 1, -1, -1):
			potential_footer_block = processed_blocks[i]
			block_content_to_check = potential_footer_block.rstrip()
			lines = block_content_to_check.strip().splitlines()

			is_likely_footer_block = False
			has_any_footer_token = False
			if lines:
				is_likely_footer_block = True
				for _line_idx, line in enumerate(lines):
					line_strip = line.strip()
					if not line_strip:
						continue
					is_potential_footer = self._potential_footer_token_regex.match(line_strip)
					is_continuation = line.startswith((" ", "\t"))
					if is_potential_footer:
						has_any_footer_token = True
					elif is_continuation:
						pass
					else:
						is_likely_footer_block = False
						break
			is_likely_footer_block = is_likely_footer_block and has_any_footer_token

			if is_likely_footer_block:
				footer_blocks.insert(0, potential_footer_block)
			else:
				break

		if not footer_blocks:
			return body_and_footers_str.strip(), None

		footers_str = "".join(footer_blocks).strip()
		body_block_count = num_blocks - len(footer_blocks)
		body_str = "".join(processed_blocks[:body_block_count]).strip() if body_block_count > 0 else None

		return body_str, footers_str

	def _append_to_footer_value(self, footer: dict[str, str], text: str) -> dict[str, str]:
		"""Helper method to safely append text to a footer's value.

		Args:
		    footer: The footer dictionary to modify.
		    text: The text to append to the footer's value.

		Returns:
		    The modified footer dictionary with updated value.
		"""
		footer["value"] = footer.get("value", "") + text
		return footer
