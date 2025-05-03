"""Command for generating conventional commit messages from Git diffs."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.markdown import Markdown
from rich.panel import Panel

try:
	from dotenv import load_dotenv
except ImportError:
	load_dotenv = None

if TYPE_CHECKING:
	from codemap.git.commit_generator import CommitMessageGenerator

from codemap.git import (
	DiffChunk,
)
from codemap.git.commit_generator.command import CommitCommand
from codemap.git.utils import (
	run_git_command,
	validate_repo_path,
)
from codemap.utils.cli_utils import (
	console,
	exit_with_error,
	setup_logging,
	show_warning,
)
from codemap.utils.config_loader import ConfigLoader

# Truncate to maximum of 10 lines
MAX_PREVIEW_LINES = 10

logger = logging.getLogger(__name__)

# Load environment variables from .env files
if load_dotenv:
	# Try to load from .env.local first, then fall back to .env
	env_local = Path(".env.local")
	if env_local.exists():
		load_dotenv(dotenv_path=env_local)
		logger.debug("Loaded environment variables from %s", env_local)
	else:
		env_file = Path(".env")
		if env_file.exists():
			load_dotenv(dotenv_path=env_file)
			logger.debug("Loaded environment variables from %s", env_file)


class GenerationMode(str, Enum):
	"""LLM message generation mode."""

	SMART = "smart"  # Use LLM-based generation
	SIMPLE = "simple"  # Use simple rule-based generation


@dataclass
class CommitOptions:
	"""Options for the commit command."""

	repo_path: Path
	generation_mode: GenerationMode = field(default=GenerationMode.SMART)
	model: str = field(default="openai/gpt-4o-mini")
	api_base: str | None = field(default=None)
	commit: bool = field(default=True)
	prompt_template: str | None = field(default=None)
	api_key: str | None = field(default=None)


def _load_prompt_template(template_path: str | None) -> str | None:
	"""
	Load custom prompt template from file.

	Args:
	    template_path: Path to prompt template file

	Returns:
	    Loaded template or None if loading failed

	"""
	if not template_path:
		return None

	try:
		template_file = Path(template_path)
		with template_file.open("r") as f:
			return f.read()
	except OSError:
		show_warning(f"Could not load prompt template: {template_path}")
		return None


def create_universal_generator(
	repo_path: Path,
	model: str | None = None,
	api_key: str | None = None,
	api_base: str | None = None,
	prompt_template: str | None = None,
) -> CommitMessageGenerator:
	"""
	Create a universal message generator with the provided options.

	Args:
	    repo_path: Repository path
	    model: Model name to use
	    api_key: API key to use
	    api_base: API base URL to use
	    prompt_template: Custom prompt template

	Returns:
	    Configured CommitMessageGenerator

	"""
	from codemap.git.commit_generator import CommitMessageGenerator
	from codemap.llm import create_client
	from codemap.utils.config_loader import ConfigLoader

	# Create LLM client
	llm_client = create_client(
		repo_path=repo_path,
		model=model,
		api_key=api_key,
		api_base=api_base,
	)

	# Create config loader
	config_loader = ConfigLoader(repo_root=repo_path)

	# Get default prompt template if not provided
	if not prompt_template:
		from codemap.git.commit_generator.prompts import DEFAULT_PROMPT_TEMPLATE

		prompt_template = DEFAULT_PROMPT_TEMPLATE

	# Create and return generator
	return CommitMessageGenerator(
		repo_root=repo_path,
		llm_client=llm_client,
		prompt_template=prompt_template,
		config_loader=config_loader,
	)


def setup_message_generator(options: CommitOptions) -> CommitMessageGenerator:
	"""
	Set up a message generator with the provided options.

	Args:
	    options: Command options

	Returns:
	    Configured message generator

	"""
	# Use the universal generator for simplified setup
	return create_universal_generator(
		repo_path=options.repo_path,
		model=options.model,
		api_key=options.api_key,
		api_base=options.api_base,
		prompt_template=_load_prompt_template(options.prompt_template),
	)


def print_chunk_summary(chunk: DiffChunk, index: int) -> None:
	"""
	Print a summary of the chunk.

	Args:
	    chunk: DiffChunk to summarize
	    index: Index of the chunk (1-based for display)

	"""
	# Print header
	console.print(f"\nCommit {index + 1} of {index + 1}")

	# Print chunk information in a panel

	# Create a content string with the files and changes
	content = "**Files:** " + ", ".join(chunk.files) + "\n"

	# Calculate line counts from the diff content
	added = len([line for line in chunk.content.splitlines() if line.startswith("+") and not line.startswith("+++")])
	removed = len([line for line in chunk.content.splitlines() if line.startswith("-") and not line.startswith("---")])

	# Add line counts
	content += "**Changes:** "
	if added > 0:
		content += f"{added} added"
	if removed > 0:
		if added > 0:
			content += ", "
		content += f"{removed} removed"
	content += "\n"

	# Add a preview of the diff content
	if chunk.content:
		content_lines = chunk.content.splitlines()
		if len(content_lines) > MAX_PREVIEW_LINES:
			content += (
				"\n```diff\n"
				+ "\n".join(content_lines[:MAX_PREVIEW_LINES])
				+ f"\n... ({len(content_lines) - MAX_PREVIEW_LINES} more lines)\n```"
			)
		else:
			content += "\n```diff\n" + chunk.content + "\n```"

	# Create the panel with the content
	panel = Panel(
		Markdown(content),
		title=f"Chunk {index + 1}",
		border_style="blue",
		expand=False,
		padding=(1, 2),
	)
	console.print(panel)


@dataclass
class RunConfig:
	"""Configuration options for running the commit command."""

	repo_path: Path | None = None
	force_simple: bool = False
	api_key: str | None = None
	model: str = "openai/gpt-4o-mini"
	api_base: str | None = None
	commit: bool = True
	prompt_template: str | None = None
	staged_only: bool = False  # Only process staged changes
	bypass_hooks: bool = False  # Whether to bypass git hooks (--no-verify)


DEFAULT_RUN_CONFIG = RunConfig()


def _raise_command_failed_error() -> None:
	"""Raise an error for failed command execution."""
	msg = "Command failed to run successfully"
	raise RuntimeError(msg)


def validate_and_process_commit(
	path: Path | None,
	all_files: bool = False,
	model: str = "gpt-4o-mini",
	bypass_hooks: bool = False,
) -> None:
	"""
	Validate repository path and process commit.

	Args:
	    path: Path to repository
	    all_files: Whether to commit all files
	    model: Model to use for generation
	    bypass_hooks: Whether to bypass git hooks with --no-verify

	"""
	try:
		# Load configuration from .codemap.yml if it exists
		repo_path = validate_repo_path(path)
		if repo_path:
			config_loader = ConfigLoader(repo_root=repo_path)
			# Get bypass_hooks from config if not explicitly set
			if not hasattr(bypass_hooks, "_set_explicitly"):
				bypass_hooks = config_loader.get_bypass_hooks()

		# Create the CommitCommand instance
		command = CommitCommand(
			path=path,
			model=model,
			bypass_hooks=bypass_hooks,
		)

		# Stage files if all_files flag is set
		if all_files:
			run_git_command(["git", "add", "."])

		# Run the command (message will be prompted during the interactive process)
		result = command.run()

		# If command completed but returned False and it wasn't an intentional abort,
		# raise an error
		if not result and command.error_state != "aborted":
			_raise_command_failed_error()

	except typer.Exit:
		# Let typer.Exit propagate for clean CLI exit
		raise
	except Exception as e:
		logger.exception("Error processing commit")
		exit_with_error(f"Error: {e}", exception=e)


def commit_command(
	path: Annotated[
		Path | None,
		typer.Argument(
			help="Path to repository or file to commit",
			exists=True,
		),
	] = None,
	message: Annotated[str | None, typer.Option("--message", "-m", help="Commit message")] = None,
	all_files: Annotated[bool, typer.Option("--all", "-a", help="Commit all changes")] = False,
	model: Annotated[
		str,
		typer.Option(
			"--model",
			help="LLM model to use for message generation",
		),
	] = "gpt-4o-mini",
	strategy: Annotated[str, typer.Option("--strategy", "-s", help="Strategy for splitting diffs")] = "semantic",
	non_interactive: Annotated[bool, typer.Option("--non-interactive", help="Run in non-interactive mode")] = False,
	bypass_hooks: Annotated[bool, typer.Option("--bypass-hooks", help="Bypass git hooks with --no-verify")] = False,
	is_verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
) -> None:
	"""
	Generate AI-assisted commit messages for staged changes.

	This command analyzes your staged changes and generates commit messages
	using an LLM.

	"""
	setup_logging(is_verbose=is_verbose)

	# Log environment setup and key configuration
	if is_verbose:
		# Log Python and environment details
		logger.debug("Python Path: %s", sys.executable)
		logger.debug("Python Version: %s", sys.version)

		# Log model information
		logger.debug("Using model: %s", model)

		# Log command parameters
		logger.debug("Message: %s", message)
		logger.debug("Strategy: %s", strategy)
		logger.debug("Non-interactive mode: %s", non_interactive)
		logger.debug("Bypass git hooks: %s", bypass_hooks)

		# Check sentence_transformers
		try:
			import sentence_transformers

			logger.debug("sentence_transformers version: %s", sentence_transformers.__version__)
		except ImportError:
			logger.debug("sentence_transformers is not installed or importable")
		except (AttributeError, RuntimeError) as e:
			logger.debug("Error checking sentence_transformers: %s", e)

		# Log important environment variables (without revealing API keys)
		provider_prefixes = ["OPENAI", "GROQ", "ANTHROPIC", "MISTRAL", "COHERE", "TOGETHER", "OPENROUTER"]
		for prefix in provider_prefixes:
			key_var = f"{prefix}_API_KEY"
			if key_var in os.environ:
				# Log presence but not the actual key
				logger.debug("%s is set in environment (length: %d)", key_var, len(os.environ[key_var]))

	# Continue with normal command execution - typer.Exit exceptions will propagate normally
	validate_and_process_commit(
		path=path,
		all_files=all_files,
		model=model,
		bypass_hooks=bypass_hooks,
	)
