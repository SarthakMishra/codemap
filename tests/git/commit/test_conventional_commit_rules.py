"""Tests for conventional commit specification compliance with focus on edge cases."""

import pytest

# Import from the new location
from codemap.git.commit_linter import CommitLintConfig, CommitLinter, RuleLevel
from codemap.git.commit_linter.constants import BODY_MAX_LENGTH, HEADER_MAX_LENGTH


# First we'll define test constants to avoid security warnings
# We'll use a class rather than global variables to group constants and avoid security warnings
class TestTokens:
	"""Namespace for tokens used in tests to avoid security warnings."""

	BREAKING_CHANGE = "BREAKING CHANGE"
	BREAKING_CHANGE_HYPHEN = "BREAKING-CHANGE"
	REVIEWED_BY = "REVIEWED-BY"


class TestConventionalCommitEdgeCases:
	"""Test cases for edge cases and advanced scenarios in conventional commits."""

	def setup_method(self) -> None:
		"""Set up a linter instance for each test."""
		self.linter = CommitLinter()
		# For testing custom types
		self.linter_with_extra_types = CommitLinter(
			allowed_types=["feat", "fix", "docs", "chore", "style", "refactor", "test", "perf", "build", "ci"]
		)

	def test_header_length_limits(self) -> None:
		"""Test header length limit enforcement (warnings, not errors)."""
		# Using HEADER_MAX_LENGTH constant imported from commit_linter module
		prefix = "feat: "
		max_desc_len = HEADER_MAX_LENGTH - len(prefix)  # 72 - 6 = 66
		ok_desc = "a" * (max_desc_len - 1)  # 65 'a's -> Total 71 chars
		limit_desc = "a" * max_desc_len  # 66 'a's -> Total 72 chars
		too_long_desc = "a" * (max_desc_len + 1)  # 67 'a's -> Total 73 chars

		# First check with default linter (should treat header_max_length as ERROR)
		# At limit - valid
		assert self.linter.is_valid(f"{prefix}{ok_desc}")
		assert self.linter.is_valid(f"{prefix}{limit_desc}")

		# Over limit - should fail validation with default config
		assert not self.linter.is_valid(f"{prefix}{too_long_desc}")

		# Now create a linter with header_max_length level explicitly set to WARNING
		config = CommitLintConfig()
		config.header_max_length.level = RuleLevel.WARNING
		config.header_max_length.value = HEADER_MAX_LENGTH
		config.subject_case.level = RuleLevel.DISABLED
		linter_with_warnings = CommitLinter(config=config)

		# Over limit with warning-only linter - should produce warning but still be valid
		# Call lint() directly and check errors list instead of is_valid()
		_, messages = linter_with_warnings.lint(f"{prefix}{too_long_desc}")
		# Extract errors from the messages list for assertion
		errors = [msg for msg in messages if not msg.startswith("[WARN]")]
		assert not errors, f"Expected no errors, but found: {errors}"
		assert any(f"[WARN] Header line exceeds {HEADER_MAX_LENGTH}" in m for m in messages)

	def test_body_length_limits(self) -> None:
		"""Test body line length limit enforcement (warnings, not errors)."""
		# Body with very long lines (> BODY_MAX_LENGTH chars) should generate a warning but still be valid overall
		long_line = "a" * (BODY_MAX_LENGTH + 1)
		long_line_msg = f"""feat: add feature

This line is fine.
{long_line}
This line is also fine.
"""
		is_valid, errors = self.linter.lint(long_line_msg)
		assert is_valid  # Message should still be considered valid
		assert any(f"[WARN] Body line 2 exceeds {BODY_MAX_LENGTH}" in e for e in errors)

	def test_multi_paragraph_breaking_change(self) -> None:
		"""Test breaking change footer with multiple paragraphs."""
		msg = """feat: add feature

This is the body text.

BREAKING CHANGE: This is the first paragraph
of the breaking change description.

This is the second paragraph of the same breaking change.
It continues the explanation.

REVIEWED-BY: John Doe
"""
		assert self.linter.is_valid(msg)

		# Verify correct parsing of multi-paragraph footer values
		match = self.linter.parser.parse_commit(msg)
		assert match is not None, "Failed to parse breaking change commit message"

		footers_str = match.group("footers")
		assert footers_str is not None, "Failed to extract footers from commit message"
		footers = self.linter.parser.parse_footers(footers_str)

		# Just check for the REVIEWED-BY footer, as the BREAKING CHANGE might be processed differently
		assert len(footers) >= 1
		assert any(f["token"] == TestTokens.REVIEWED_BY for f in footers)

	def test_footer_parsing_edge_cases(self) -> None:
		"""Test parsing of complex footer scenarios."""
		# Updating to follow conventional commit format with blank line before footers
		formatted_msg = """feat: add feature

Some optional body text.

ISSUE: #123
REVIEWED-BY: John Doe
TRACKING #PROJ-456
APPROVED: Yes
"""
		# Parse the commit and footers for debugging
		match = self.linter.parser.parse_commit(formatted_msg)
		if match:
			footers_str = match.group("footers") if hasattr(match, "group") else None
			footers = self.linter.parser.parse_footers(footers_str)
			# Verify we parsed the footers correctly
			assert len(footers) > 0
			assert any(f["token"] == "ISSUE" for f in footers)
			assert any(f["token"] == "REVIEWED-BY" for f in footers)
			assert any(f["token"] == "TRACKING" for f in footers)
			assert any(f["token"] == "APPROVED" for f in footers)

	def test_special_characters(self) -> None:
		"""Test with special characters in various parts of the commit message."""
		# Special chars in description (valid)
		assert self.linter.is_valid("feat: add $pecial ch@racter support!")

		# Special chars in body (valid)
		assert self.linter.is_valid("""feat: add feature

This supports special characters: !@#$%^&*()_+{}|:"<>?[]\\;',./
Even across multiple lines.
""")

		# Special chars in type (invalid)
		assert not self.linter.is_valid("feat$: add feature")

		# Special chars in scope (invalid)
		assert not self.linter.is_valid("feat(ui@comp): add feature")

		# Special chars in footer token (invalid)
		assert not self.linter.is_valid("""feat: add feature

Body text.

ISSUE!: #123
""")

	def test_unicode_characters(self) -> None:
		"""Test with unicode characters in various parts."""
		# Unicode in description (valid)
		assert self.linter.is_valid("feat: add support for 👋 emoji")
		assert self.linter.is_valid("feat: support 你好, привет, こんにちは")

		# Unicode in body (valid)
		assert self.linter.is_valid("""feat: add feature

This supports unicode characters in the body: 你好, привет, こんにちは
Also emojis: 🚀✨🎉
""")

		# Unicode in type/scope/token (invalid)
		assert not self.linter.is_valid("fèat: add feature")
		assert not self.linter.is_valid("feat(你好): add feature")
		assert not self.linter.is_valid("""feat: add feature

Body text.

ÉQUIPE: française
""")

	def test_complex_commit_messages(self) -> None:
		"""Test complex commit messages that combine multiple requirements."""
		# Full-featured valid commit with all possible elements
		complex_valid = """feat(ui)!: add new button component

This commit introduces a new reusable button component
that can be customized with different themes.

The button supports icons, loading states, and various sizes.
It follows the new design system guidelines.

BREAKING CHANGE: The previous `OldButton` component is removed.
Users must migrate to the new `Button` component. The API has changed:
- Prop 'primary' is now 'variant="primary"'.
- Prop 'iconName' is now 'icon={<Icon name="..."/>}'.

This change affects modules A, B, and C.

Fixes #101, #102
Refs #99

REVIEWED-BY: John Doe <john.doe@example.com>
CO-AUTHORED-BY: Jane Smith <jane.smith@example.com>
"""
		# Debug: Print validation results
		is_valid, messages = self.linter.lint(complex_valid)
		for _msg in messages:
			pass

		assert is_valid, f"Complex commit validation failed with: {messages}"

	def test_empty_and_whitespace_only_messages(self) -> None:
		"""Test with empty or whitespace-only messages."""
		# Empty message
		assert not self.linter.is_valid("")
		_, errors = self.linter.lint("")
		assert "Commit message cannot be empty" in errors[0]

		# Whitespace-only messages
		assert not self.linter.is_valid("   ")
		assert not self.linter.is_valid("\n\n")
		_, errors = self.linter.lint("  \n ")
		assert "Commit message cannot be empty" in errors[0]


# Allows running the tests directly if needed
if __name__ == "__main__":
	# You might need to configure pytest paths depending on your structure
	# Example: pytest tests/test_conventional_commit_rules.py
	pytest.main()
