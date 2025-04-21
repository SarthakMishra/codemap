"""Command for generating conventional commit messages from Git diffs."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Iterator

import questionary
import typer
from rich.console import Console
from rich.padding import Padding
from rich.spinner import Spinner

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from codemap.git import (
    DiffChunk,
    DiffSplitter,
    SplitStrategy,
)
from codemap.git.commit.message_generator import LLMError
from codemap.utils.cli_utils import console, setup_logging
from codemap.utils.git_utils import (
    GitError,
    commit_only_files,
    get_other_staged_files,
    get_staged_diff,
    get_unstaged_diff,
    get_untracked_files,
)
from codemap.utils.llm_utils import create_universal_generator, generate_message

from .cli_types import VerboseFlag

if TYPE_CHECKING:
    from codemap.git.commit.message_generator import MessageGenerator

# Truncate to maximum of 10 lines
MAX_PREVIEW_LINES = 10

# Try to import from utils, but fallback to defining locally if needed
try:
    from codemap.utils import loading_spinner, validate_repo_path
except ImportError:
    # Define loading_spinner locally as fallback
    @contextlib.contextmanager
    def loading_spinner(message: str = "Processing...") -> Iterator[None]:
        """Display a loading spinner while executing a task.

        Args:
            message: Message to display alongside the spinner

        Yields:
            None
        """
        console = Console()
        spinner = Spinner("dots", text=message)
        with console.status(spinner):
            yield

    # Also import validate_repo_path
    from codemap.utils import validate_repo_path

logger = logging.getLogger(__name__)

# Load environment variables from .env files
if load_dotenv:
    # Try to load from .env.local first, then fall back to .env
    if Path(".env.local").exists():
        load_dotenv(".env.local")
    load_dotenv()  # Load from .env if available


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
    """Load custom prompt template from file.

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
        console.print(f"[yellow]Warning:[/yellow] Could not load prompt template: {template_path}")
        return None


def _extract_provider_from_model(model: str) -> str | None:
    """Extract provider from model name if possible.

    Args:
        model: Model identifier like "provider/model_name" or "provider/org/model_name"

    Returns:
        Provider name or None if not in expected format
    """
    # Handle explicit provider prefixes (get first part before slash)
    if "/" in model:
        provider = model.split("/")[0]  # Take first part regardless of number of slashes
        return provider.lower()  # Normalize to lowercase

    # For models without provider prefix, return None
    # LiteLLM will handle these appropriately
    return None


def _get_api_key_for_provider(provider: str) -> str | None:
    """Get API key for the specified provider from environment.

    Args:
        provider: Provider name

    Returns:
        API key or None if not found
    """
    provider_env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "azure": "AZURE_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "together": "TOGETHER_API_KEY",
        "cohere": "COHERE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    # Get environment variable name for this provider
    env_var = provider_env_vars.get(provider)
    if not env_var:
        return None

    # Try to get key from environment
    api_key = os.environ.get(env_var)

    # Fallback to OpenAI if requested
    if not api_key and provider != "openai":
        api_key = os.environ.get("OPENAI_API_KEY")

    return api_key


def setup_message_generator(options: CommitOptions) -> MessageGenerator:
    """Set up a message generator with the provided options.

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


def generate_commit_message(
    chunk: DiffChunk,
    generator: MessageGenerator,
    mode: GenerationMode,
) -> tuple[str, bool]:
    """Generate a commit message for the given chunk.

    Args:
        chunk: Diff chunk to generate message for
        generator: Message generator to use
        mode: Generation mode

    Returns:
        Tuple of (message, whether LLM was used)
    """
    # Use the universal generate_message function
    use_simple_mode = mode == GenerationMode.SIMPLE

    try:
        message, used_llm = generate_message(chunk, generator, use_simple_mode)
        return message, used_llm
    except (ValueError, RuntimeError, LLMError) as e:
        console.print(f"[red]Error generating message:[/red] {e}")
        # Still try to generate a fallback message
        message = generator.fallback_generation(chunk)
        return message, False


def print_chunk_summary(chunk: DiffChunk, index: int) -> None:
    """Print a summary of a diff chunk.

    Args:
        chunk: The diff chunk to print
        index: Index of the chunk
    """
    console.print(f"Chunk {index + 1}:")
    console.print(f"  Files: {', '.join(chunk.files)}")

    # Calculate changes
    added = len([line for line in chunk.content.splitlines() if line.startswith("+") and not line.startswith("+++")])
    removed = len([line for line in chunk.content.splitlines() if line.startswith("-") and not line.startswith("---")])

    # Check if this is likely an untracked file chunk (no diff content)
    if not chunk.content and chunk.files:
        console.print("  [blue]New untracked files[/blue]")
    else:
        console.print(f"  Changes: {added} added, {removed} removed")

    # Preview of the diff
    if chunk.content:
        content_lines = chunk.content.splitlines()
        if len(content_lines) > MAX_PREVIEW_LINES:
            remaining_lines = len(content_lines) - MAX_PREVIEW_LINES
            preview = "\n".join(content_lines[:MAX_PREVIEW_LINES]) + f"\n... ({remaining_lines} more lines)"
        else:
            preview = chunk.content
        console.print(Padding(f"  [dim]{preview}[/dim]", (0, 0, 1, 2)))
    else:
        console.print(Padding("  [dim](New files - no diff content available)[/dim]", (0, 0, 1, 2)))


def _check_other_files(chunk_files: list[str]) -> tuple[list[str], list[str], bool]:
    """Check for other staged and untracked files.

    Args:
        chunk_files: Files in the current chunk

    Returns:
        Tuple of (other_staged, other_untracked, has_warnings)
    """
    other_staged = get_other_staged_files(chunk_files)

    # For untracked files, check if they are already included in the chunk
    all_untracked = get_untracked_files()
    other_untracked = [f for f in all_untracked if f not in chunk_files]

    has_warnings = bool(other_staged or other_untracked)

    # Display warnings
    if other_staged:
        console.print("[yellow]Warning:[/yellow] The following files are also staged but not part of this commit:")
        for file in other_staged:
            console.print(f"  - {file}")

    if other_untracked:
        console.print("[yellow]Warning:[/yellow] The following new files are not included in this commit:")
        for file in other_untracked:
            console.print(f"  - {file}")

    return other_staged, other_untracked, has_warnings


def _handle_other_files(chunk: DiffChunk, other_staged: list[str], other_untracked: list[str]) -> bool:
    """Handle other staged or untracked files.

    Args:
        chunk: The current chunk
        other_staged: Other staged files
        other_untracked: Other untracked files

    Returns:
        False if action is cancelled, True otherwise
    """
    # Prepare choices
    choices = [
        {"value": "continue", "name": "Continue with just the selected files"},
        {"value": "all_staged", "name": "Include all staged files in this commit"},
    ]

    # Only add untracked option if there are untracked files
    if other_untracked:
        choices.append({"value": "all_untracked", "name": "Include all untracked files in this commit"})

    if other_staged and other_untracked:
        choices.append({"value": "all", "name": "Include all staged and untracked files"})

    choices.append({"value": "cancel", "name": "Cancel this commit"})

    # Ask user what to do
    action = questionary.select("Other files found. What would you like to do?", choices=choices).ask()

    if action == "cancel":
        console.print("[yellow]Commit cancelled[/yellow]")
        return False

    # Update chunk files if needed
    if action in ("all_staged", "all"):
        chunk.files.extend(other_staged)
    if action in ("all_untracked", "all"):
        chunk.files.extend(other_untracked)

    return True


def _perform_commit(chunk: DiffChunk, message: str) -> None:
    """Perform the actual commit.

    Args:
        chunk: Diff chunk to commit
        message: Commit message
    """
    try:
        # Commit only the files in this chunk
        commit_only_files(chunk.files, message)
        console.print(f"[green]✓[/green] Committed {len(chunk.files)} files")
    except GitError as e:
        console.print(f"[red]Error:[/red] {e!s}")


def handle_commit_action(chunk: DiffChunk, message: str) -> None:
    """Handle commit action.

    Args:
        chunk: Diff chunk to commit
        message: Commit message
    """
    console.print("Committing changes...")
    _perform_commit(chunk, message)


def handle_edit_action(chunk: DiffChunk, message: str) -> None:
    """Handle edit action.

    Args:
        chunk: Diff chunk to edit and commit
        message: Initial commit message to edit
    """
    # Ask for a new commit message
    edited_message = questionary.text(
        "Edit commit message:",
        default=message,
        validate=lambda text: bool(text.strip()) or "Commit message cannot be empty",
    ).unsafe_ask()

    if edited_message:
        _perform_commit(chunk, edited_message)


@dataclass
class ChunkContext:
    """Context for processing a chunk."""

    chunk: DiffChunk
    index: int
    total: int
    generator: MessageGenerator
    mode: GenerationMode


def process_chunk_interactively(context: ChunkContext) -> str:
    """Process a diff chunk interactively.

    Args:
        context: Context for processing the chunk

    Returns:
        Action to take ("continue", "exit")
    """
    console.print(f"\n[bold]Commit {context.index + 1} of {context.total}[/bold]")
    print_chunk_summary(context.chunk, context.index)

    # Generate commit message
    message, used_llm = generate_commit_message(context.chunk, context.generator, context.mode)

    # Display proposed message
    tag = "[blue]AI[/blue]" if used_llm else "[yellow]Simple[/yellow]"
    console.print(f"\nProposed message ({tag}):")
    console.print(f"[green]{message}[/green]")

    # Ask user what to do
    choices = [
        {"value": "commit", "name": "Commit with this message"},
        {"value": "edit", "name": "Edit message and commit"},
        {"value": "regenerate", "name": "Regenerate message"},
        {"value": "skip", "name": "Skip this chunk"},
        {"value": "exit", "name": "Exit without committing"},
    ]

    action = questionary.select("What would you like to do?", choices=choices).ask()

    if action == "commit":
        handle_commit_action(context.chunk, message)
    elif action == "edit":
        handle_edit_action(context.chunk, message)
    elif action == "regenerate":
        # Just loop back for this chunk with smart generation
        return process_chunk_interactively(
            ChunkContext(
                chunk=context.chunk,
                index=context.index,
                total=context.total,
                generator=context.generator,
                mode=GenerationMode.SMART,
            ),
        )
    elif action == "skip":
        console.print("[yellow]Skipped![/yellow]")
    elif action == "exit":
        console.print("[yellow]Exiting commit process[/yellow]")
        return "exit"

    return "continue"


def display_suggested_messages(options: CommitOptions, chunks: list[DiffChunk], generator: MessageGenerator) -> None:
    """Display suggested commit messages without committing.

    Args:
        options: Commit options
        chunks: List of diff chunks
        generator: Message generator to use
    """
    console.print("Suggested commit messages (not committing):")

    for i, chunk in enumerate(chunks):
        print_chunk_summary(chunk, i)
        message, used_llm = generate_commit_message(chunk, generator, options.generation_mode)

        tag = "[blue]AI[/blue]" if used_llm else "[yellow]Simple[/yellow]"
        console.print(f"{tag} {message}")
        console.print()


def process_all_chunks(
    options: CommitOptions,
    chunks: list[DiffChunk],
    generator: MessageGenerator,
) -> int:
    """Process all chunks interactively.

    Args:
        options: Commit options
        chunks: List of diff chunks
        generator: Message generator to use

    Returns:
        Exit code (0 for success)
    """
    for i, chunk in enumerate(chunks):
        context = ChunkContext(
            chunk=chunk,
            index=i,
            total=len(chunks),
            generator=generator,
            mode=options.generation_mode,
        )

        if process_chunk_interactively(context) == "exit":
            return 0

    console.print("[green]✓[/green] All changes committed!")
    return 0


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


DEFAULT_RUN_CONFIG = RunConfig()


def _run_commit_command(config: RunConfig) -> int:
    """Run the commit command logic.

    Args:
        config: Run configuration

    Returns:
        Exit code
    """
    # Validate the repository path
    try:
        repo_path = validate_repo_path(config.repo_path)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e!s}")
        return 1

    # Show welcome message
    console.print(
        Padding("[bold]CodeMap Conventional Commit Generator[/]", (1, 0, 0, 0)),
    )

    # Determine generation mode
    mode = GenerationMode.SIMPLE if config.force_simple else GenerationMode.SMART

    # Configure options
    options = CommitOptions(
        repo_path=repo_path,
        generation_mode=mode,
        model=config.model,
        api_base=config.api_base,
        commit=config.commit,
        prompt_template=config.prompt_template,
        api_key=config.api_key,
    )

    try:
        # Check if there are any changes
        with loading_spinner("Checking for changes..."):
            try:
                staged = get_staged_diff()
                unstaged = get_unstaged_diff()
                untracked = get_untracked_files()
                has_changes = bool(staged.files or unstaged.files or untracked)
            except GitError:
                has_changes = False

        if not has_changes:
            console.print("[yellow]No changes to commit[/yellow]")
            return 0

        # Set up message generator
        with loading_spinner("Setting up message generator..."):
            generator = setup_message_generator(options)

        # Get staged and unstaged changes
        with loading_spinner("Analyzing repository changes..."):
            # Get changes from Git
            splitter = DiffSplitter(repo_path)
            chunks = []

            # Process staged changes
            staged_diff = get_staged_diff()
            if staged_diff.files:
                staged_chunks = splitter.split_diff(staged_diff, SplitStrategy.SEMANTIC)
                chunks.extend(staged_chunks)

            # Process unstaged changes
            if not chunks or not config.staged_only:
                unstaged_diff = get_unstaged_diff()
                if unstaged_diff.files:
                    unstaged_chunks = splitter.split_diff(unstaged_diff, SplitStrategy.SEMANTIC)
                    chunks.extend(unstaged_chunks)

        # Check if there are any chunks
        if not chunks:
            console.print("[yellow]No changes to commit[/yellow]")
            return 0

        # Process chunks
        return process_all_chunks(options, chunks, generator)
    except (ValueError, RuntimeError, TypeError) as e:
        console.print(f"[red]Error:[/red] {e!s}")
        return 1


def commit_command(
    path: Annotated[
        Path | None,
        typer.Argument(
            help="Path to repository or file to commit",
            exists=True,
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="LLM model to use for message generation",
        ),
    ] = "openai/gpt-4o-mini",
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="OpenAI API key (or set OPENAI_API_KEY env var)",
            envvar="OPENAI_API_KEY",
        ),
    ] = None,
    api_base: Annotated[
        str | None,
        typer.Option(
            "--api-base",
            help="Custom API base URL for the LLM provider",
        ),
    ] = None,
    force_simple: Annotated[
        bool,
        typer.Option(
            "--simple",
            "-s",
            help="Use simple message generation (no LLM)",
        ),
    ] = False,
    prompt_template: Annotated[
        str | None,
        typer.Option(
            "--template",
            "-t",
            help="Path to custom prompt template file",
        ),
    ] = None,
    staged_only: Annotated[
        bool,
        typer.Option(
            "--staged-only",
            help="Only process staged changes",
        ),
    ] = False,
    is_verbose: VerboseFlag = False,
) -> None:
    """Generate and apply conventional commits from changes in a Git repository."""
    setup_logging(is_verbose=is_verbose)

    try:
        # Create run configuration
        config = RunConfig(
            repo_path=path,
            force_simple=force_simple,
            api_key=api_key,
            model=model,
            api_base=api_base,
            prompt_template=prompt_template,
            staged_only=staged_only,
        )

        # Run commit command
        exit_code = _run_commit_command(config)
        if exit_code != 0:
            sys.exit(exit_code)

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logger.debug("Error running commit command", exc_info=True)
        raise typer.Exit(1) from e
