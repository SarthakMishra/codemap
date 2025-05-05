"""
Commit linter package for validating git commit messages according to conventional commits.

This package provides modules for parsing, validating, and configuring
commit message linting.

"""

from pathlib import Path

from codemap.utils.config_loader import ConfigLoader

from .config import CommitLintConfig, Rule, RuleLevel
from .constants import DEFAULT_TYPES
from .linter import CommitLinter

__all__ = ["DEFAULT_TYPES", "CommitLintConfig", "CommitLinter", "Rule", "RuleLevel", "create_linter"]


def create_linter(
	allowed_types: list[str] | None = None,
	config: CommitLintConfig | None = None,
	config_path: str | None = None,
	config_loader: ConfigLoader | None = None,
	repo_root: Path | None = None,
) -> CommitLinter:
	"""
	Create a CommitLinter with proper dependency injection for configuration.

	This factory function follows the Chain of Responsibility pattern for configuration management,
	ensuring the linter uses the same ConfigLoader instance as the rest of the application.

	Args:
	    allowed_types: Override list of allowed commit types
	    config: Pre-configured CommitLintConfig object
	    config_path: Path to a configuration file
	    config_loader: ConfigLoader instance for configuration (recommended)
	    repo_root: Repository root path

	Returns:
	    CommitLinter: Configured commit linter instance

	"""
	# Create a ConfigLoader if not provided, but repo_root is
	if config_loader is None and repo_root is not None:
		config_loader = ConfigLoader(repo_root=repo_root)

	# Create and return the linter with proper configuration injection
	return CommitLinter(
		allowed_types=allowed_types,
		config=config,
		config_path=config_path,
		config_loader=config_loader,
	)
