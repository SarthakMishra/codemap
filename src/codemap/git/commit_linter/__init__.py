"""
Commit linter package for validating git commit messages according to conventional commits.

This package provides modules for parsing, validating, and configuring
commit message linting.

"""

from codemap.config.config_loader import ConfigLoader

from .config import CommitLintConfig, Rule, RuleLevel
from .linter import CommitLinter

__all__ = ["CommitLintConfig", "CommitLinter", "Rule", "RuleLevel", "create_linter"]


def create_linter(
	allowed_types: list[str] | None = None,
	config: CommitLintConfig | None = None,
	config_loader: ConfigLoader | None = None,
) -> CommitLinter:
	"""
	Create a CommitLinter with proper dependency injection for configuration.

	This factory function follows the Chain of Responsibility pattern for configuration management,
	ensuring the linter uses the same ConfigLoader instance as the rest of the application.

	Args:
	    allowed_types: Override list of allowed commit types
	    config: Pre-configured CommitLintConfig object
	    config_loader: ConfigLoader instance for configuration (recommended)


	Returns:
	    CommitLinter: Configured commit linter instance

	"""
	# Create a ConfigLoader if not provided, but repo_root is
	if config_loader is None:
		config_loader = ConfigLoader.get_instance()

	# Create and return the linter with proper configuration injection
	return CommitLinter(
		allowed_types=allowed_types,
		config=config,
		config_loader=config_loader,
	)
