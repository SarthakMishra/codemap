"""Command for generating conventional commit messages from Git diffs."""

import logging
from pathlib import Path
from typing import Annotated

import typer

logger = logging.getLogger(__name__)

# --- Command Argument Annotations (Keep these lightweight) ---

PathArg = Annotated[
	Path | None,
	typer.Argument(
		help="Path to repository or file to commit",
		exists=True,
	),
]

MessageOpt = Annotated[str | None, typer.Option("--message", "-m", help="Commit message")]

AllFilesFlag = Annotated[bool, typer.Option("--all", "-a", help="Commit all changes")]

ModelOpt = Annotated[
	str,
	typer.Option(
		"--model",
		"-m",  # Added alias back if it was intended
		help="LLM model to use for message generation",
	),
]

# StrategyOpt removed as strategy parameter is unused
# StrategyOpt = Annotated[str, typer.Option("--strategy", "-s", help="Strategy for splitting diffs")]

NonInteractiveFlag = Annotated[
	bool, typer.Option("--non-interactive", "-y", help="Run in non-interactive mode")
]  # Added alias back

BypassHooksFlag = Annotated[
	bool, typer.Option("--bypass-hooks", "--no-verify", help="Bypass git hooks with --no-verify")
]  # Added alias back

VerboseFlag = Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")]

# --- New options for semantic commit ---
EmbeddingModelOpt = Annotated[str, typer.Option("--embedding-model", help="Model to use for embedding generation")]

ClusteringMethodOpt = Annotated[
	str, typer.Option("--clustering-method", help="Method to use for clustering ('agglomerative' or 'dbscan')")
]

SimilarityThresholdOpt = Annotated[
	float, typer.Option("--similarity-threshold", help="Threshold for group similarity to trigger merging (0.0-1.0)")
]

# --- Registration Function ---


def register_command(app: typer.Typer) -> None:
	"""Register the commit commands with the CLI app."""

	@app.command(name="commit")
	def semantic_commit_command(
		path: PathArg = None,
		model: ModelOpt = "gpt-4o-mini",
		non_interactive: NonInteractiveFlag = False,
		bypass_hooks: BypassHooksFlag = False,
		embedding_model: EmbeddingModelOpt = "all-MiniLM-L6-v2",
		clustering_method: ClusteringMethodOpt = "agglomerative",
		similarity_threshold: SimilarityThresholdOpt = 0.6,
		is_verbose: VerboseFlag = False,
		pathspecs: list[str] | None = None,
	) -> None:
		"""
		Generate semantic commits by grouping related changes.

		This command analyzes your changes, groups them semantically, and
		creates multiple focused commits with AI-generated messages.

		"""
		# Defer heavy imports and logic to the implementation function
		_semantic_commit_command_impl(
			path=path,
			model=model,
			non_interactive=non_interactive,
			bypass_hooks=bypass_hooks,
			embedding_model=embedding_model,
			clustering_method=clustering_method,
			similarity_threshold=similarity_threshold,
			is_verbose=is_verbose,
			pathspecs=pathspecs,
		)


# --- Implementation Function (Heavy imports deferred here) ---


def _semantic_commit_command_impl(
	path: Path | None,
	model: str,
	non_interactive: bool,
	bypass_hooks: bool,
	embedding_model: str,
	clustering_method: str,
	similarity_threshold: float,
	is_verbose: bool,
	pathspecs: list[str] | None = None,
) -> None:
	"""Actual implementation of the semantic commit command."""
	# --- Heavy Imports ---

	try:
		from dotenv import load_dotenv
	except ImportError:
		load_dotenv = None

	from codemap.git.commit_generator.command import SemanticCommitCommand
	from codemap.git.utils import (
		validate_repo_path,
	)
	from codemap.utils.cli_utils import exit_with_error, handle_keyboard_interrupt, setup_logging
	from codemap.utils.config_loader import ConfigLoader

	# --- Environment Loading ---
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

	# --- Setup & Logic ---
	setup_logging(is_verbose=is_verbose)

	try:
		# Validate repo path (optional, defaults to cwd if None)
		repo_path = validate_repo_path(path) if path else Path.cwd()

		# Load config
		config_loader = ConfigLoader(repo_root=repo_path)
		config = config_loader.load_config()
		commit_config = config.get("commit", {})
		semantic_config = config.get("semantic_commit", {})

		# Determine parameters (CLI > Config > Default)
		final_model = model or commit_config.get("model", "gpt-4o-mini")
		is_non_interactive = non_interactive or semantic_config.get("non_interactive", False)
		should_bypass_hooks = bypass_hooks or semantic_config.get("bypass_hooks", False)
		final_embedding_model = embedding_model or semantic_config.get("embedding_model", "all-MiniLM-L6-v2")
		final_clustering_method = clustering_method or semantic_config.get("clustering_method", "agglomerative")
		final_similarity_threshold = (
			similarity_threshold
			if similarity_threshold is not None
			else semantic_config.get("similarity_threshold", 0.6)
		)

		# Create the semantic commit command
		semantic_workflow = SemanticCommitCommand(
			path=repo_path,
			model=final_model,
			bypass_hooks=should_bypass_hooks,
			embedding_model=final_embedding_model,
			clustering_method=final_clustering_method,
			similarity_threshold=final_similarity_threshold,
		)

		# Run the semantic commit process
		success = semantic_workflow.run(
			interactive=not is_non_interactive,
			pathspecs=pathspecs,
		)

		if not success:
			exit_with_error("Semantic commit process failed.")

	except KeyboardInterrupt:
		handle_keyboard_interrupt()
	except Exception as e:
		logger.exception("An unexpected error occurred during the semantic commit command.")
		exit_with_error(f"An unexpected error occurred: {e}", exception=e)
