"""
Configuration for commit message linting.

This module defines the configuration structures and rules for linting
commit messages according to Conventional Commits specifications.

"""

import enum
from dataclasses import dataclass, field
from typing import Any, Literal

from codemap.config import ConfigLoader

# Default values are now defined in src/codemap/config.py


class RuleLevel(enum.Enum):
	"""Enforcement level for a linting rule."""

	DISABLED = 0
	WARNING = 1
	ERROR = 2


@dataclass
class Rule:
	"""A rule configuration for commit linting."""

	name: str
	condition: str
	rule: Literal["always", "never"] = "always"
	level: RuleLevel = RuleLevel.ERROR
	value: Any = None


@dataclass
class CommitLintConfig:
	"""
	Configuration for commit message linting rules.

	Rather than providing default values here, this class now loads its
	configuration from the central config.py file via ConfigLoader.

	"""

	# Header rules
	header_max_length: Rule = field(
		default_factory=lambda: Rule(
			name="header-max-length",
			condition="header has value or less characters",
			rule="always",
			value=100,  # Default value, will be overridden by config
			level=RuleLevel.ERROR,
		)
	)

	# More rule definitions with minimal defaults...
	header_min_length: Rule = field(
		default_factory=lambda: Rule(
			name="header-min-length",
			condition="header has value or more characters",
			rule="always",
			value=0,
		)
	)

	header_case: Rule = field(
		default_factory=lambda: Rule(
			name="header-case",
			condition="header is in case value",
			rule="always",
			value="lower-case",
			level=RuleLevel.DISABLED,
		)
	)

	header_full_stop: Rule = field(
		default_factory=lambda: Rule(
			name="header-full-stop",
			condition="header ends with value",
			rule="never",
			value=".",
		)
	)

	header_trim: Rule = field(
		default_factory=lambda: Rule(
			name="header-trim",
			condition="header must not have initial and/or trailing whitespaces",
			rule="always",
		)
	)

	# Type rules
	type_enum: Rule = field(
		default_factory=lambda: Rule(
			name="type-enum",
			condition="type is found in value",
			rule="always",
			value=[],  # Will be populated from config
		)
	)

	type_case: Rule = field(
		default_factory=lambda: Rule(
			name="type-case",
			condition="type is in case value",
			rule="always",
			value="lower-case",
		)
	)

	type_empty: Rule = field(
		default_factory=lambda: Rule(
			name="type-empty",
			condition="type is empty",
			rule="never",
		)
	)

	# Other rules with minimal definitions...
	# Scope rules
	scope_enum: Rule = field(
		default_factory=lambda: Rule(
			name="scope-enum",
			condition="scope is found in value",
			rule="always",
			value=[],
			level=RuleLevel.DISABLED,
		)
	)

	scope_case: Rule = field(
		default_factory=lambda: Rule(
			name="scope-case",
			condition="scope is in case value",
			rule="always",
			value="lower-case",
		)
	)

	scope_empty: Rule = field(
		default_factory=lambda: Rule(
			name="scope-empty",
			condition="scope is empty",
			rule="never",
			level=RuleLevel.DISABLED,
		)
	)

	# Subject rules
	subject_case: Rule = field(
		default_factory=lambda: Rule(
			name="subject-case",
			condition="subject is in case value",
			rule="always",
			value=["sentence-case", "start-case", "pascal-case", "upper-case"],
		)
	)

	subject_empty: Rule = field(
		default_factory=lambda: Rule(
			name="subject-empty",
			condition="subject is empty",
			rule="never",
		)
	)

	subject_full_stop: Rule = field(
		default_factory=lambda: Rule(
			name="subject-full-stop",
			condition="subject ends with value",
			rule="never",
			value=".",
		)
	)

	subject_exclamation_mark: Rule = field(
		default_factory=lambda: Rule(
			name="subject-exclamation-mark",
			condition="subject has exclamation before the : marker",
			rule="never",
			level=RuleLevel.DISABLED,
		)
	)

	# Body rules
	body_leading_blank: Rule = field(
		default_factory=lambda: Rule(
			name="body-leading-blank",
			condition="body begins with blank line",
			rule="always",
			level=RuleLevel.WARNING,
		)
	)

	body_empty: Rule = field(
		default_factory=lambda: Rule(
			name="body-empty",
			condition="body is empty",
			rule="never",
			level=RuleLevel.DISABLED,
		)
	)

	body_max_line_length: Rule = field(
		default_factory=lambda: Rule(
			name="body-max-line-length",
			condition="body lines has value or less characters",
			rule="always",
			value=100,
		)
	)

	# Footer rules
	footer_leading_blank: Rule = field(
		default_factory=lambda: Rule(
			name="footer-leading-blank",
			condition="footer begins with blank line",
			rule="always",
			level=RuleLevel.WARNING,
		)
	)

	footer_empty: Rule = field(
		default_factory=lambda: Rule(
			name="footer-empty",
			condition="footer is empty",
			rule="never",
			level=RuleLevel.DISABLED,
		)
	)

	footer_max_line_length: Rule = field(
		default_factory=lambda: Rule(
			name="footer-max-line-length",
			condition="footer lines has value or less characters",
			rule="always",
			value=100,
		)
	)

	# Additional rules that are still referenced by the linter
	type_max_length: Rule = field(
		default_factory=lambda: Rule(
			name="type-max-length",
			condition="type has value or less characters",
			rule="always",
			value=float("inf"),
		)
	)

	type_min_length: Rule = field(
		default_factory=lambda: Rule(
			name="type-min-length",
			condition="type has value or more characters",
			rule="always",
			value=0,
		)
	)

	scope_max_length: Rule = field(
		default_factory=lambda: Rule(
			name="scope-max-length",
			condition="scope has value or less characters",
			rule="always",
			value=float("inf"),
		)
	)

	scope_min_length: Rule = field(
		default_factory=lambda: Rule(
			name="scope-min-length",
			condition="scope has value or more characters",
			rule="always",
			value=0,
		)
	)

	subject_max_length: Rule = field(
		default_factory=lambda: Rule(
			name="subject-max-length",
			condition="subject has value or less characters",
			rule="always",
			value=float("inf"),
		)
	)

	subject_min_length: Rule = field(
		default_factory=lambda: Rule(
			name="subject-min-length",
			condition="subject has value or more characters",
			rule="always",
			value=0,
		)
	)

	body_max_length: Rule = field(
		default_factory=lambda: Rule(
			name="body-max-length",
			condition="body has value or less characters",
			rule="always",
			value=float("inf"),
		)
	)

	body_min_length: Rule = field(
		default_factory=lambda: Rule(
			name="body-min-length",
			condition="body has value or more characters",
			rule="always",
			value=0,
		)
	)

	body_case: Rule = field(
		default_factory=lambda: Rule(
			name="body-case",
			condition="body is in case value",
			rule="always",
			value="lower-case",
			level=RuleLevel.DISABLED,
		)
	)

	body_full_stop: Rule = field(
		default_factory=lambda: Rule(
			name="body-full-stop",
			condition="body ends with value",
			rule="never",
			value=".",
			level=RuleLevel.DISABLED,
		)
	)

	# Reference rules
	references_empty: Rule = field(
		default_factory=lambda: Rule(
			name="references-empty",
			condition="references has at least one entry",
			rule="never",
			level=RuleLevel.DISABLED,
		)
	)

	# Signed-off rules
	signed_off_by: Rule = field(
		default_factory=lambda: Rule(
			name="signed-off-by",
			condition="message has value",
			rule="always",
			value="Signed-off-by:",
			level=RuleLevel.DISABLED,
		)
	)

	trailer_exists: Rule = field(
		default_factory=lambda: Rule(
			name="trailer-exists",
			condition="message has trailer value",
			rule="always",
			value="Signed-off-by:",
			level=RuleLevel.DISABLED,
		)
	)

	footer_max_length: Rule = field(
		default_factory=lambda: Rule(
			name="footer-max-length",
			condition="footer has value or less characters",
			rule="always",
			value=float("inf"),
		)
	)

	footer_min_length: Rule = field(
		default_factory=lambda: Rule(
			name="footer-min-length",
			condition="footer has value or more characters",
			rule="always",
			value=0,
		)
	)

	@classmethod
	def get_rules(cls, config_loader: ConfigLoader) -> "CommitLintConfig":
		"""
		Get the rules from the config.

		Args:
		    config_loader: ConfigLoader instance for retrieving additional configuration

		Returns:
		    CommitLintConfig: Configured instance
		"""
		config = cls()
		commit_config = config_loader.get.commit
		lint_config = commit_config.lint

		# Update all rules from lint config
		for rule_name in dir(config):
			if not rule_name.startswith("_") and isinstance(getattr(config, rule_name), Rule):
				rule_obj = getattr(config, rule_name)
				rule_config = getattr(lint_config, rule_name, None)

				if rule_config:
					rule_obj.rule = rule_config.rule
					rule_obj.value = rule_config.value
					rule_obj.level = getattr(RuleLevel, rule_config.level)

		# Handle special cases from commit convention
		if commit_config.convention.types:
			config.type_enum.value = commit_config.convention.types

		if commit_config.convention.scopes:
			config.scope_enum.value = commit_config.convention.scopes
			if config.scope_enum.value:
				config.scope_enum.level = RuleLevel.ERROR

		if commit_config.convention.max_length and not lint_config.header_max_length:
			config.header_max_length.value = commit_config.convention.max_length

		return config
