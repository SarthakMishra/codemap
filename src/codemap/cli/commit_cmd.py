"""Command for generating conventional commit messages from Git diffs."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import questionary
import typer
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if TYPE_CHECKING:
    from codemap.git.message_generator import MessageGenerator

from codemap.git import (
    DiffChunk,
    DiffSplitter,
    SplitStrategy,
)
from codemap.git.command import CommitCommand
from codemap.git.message_generator import LLMError
from codemap.utils import validate_repo_path
from codemap.utils.cli_utils import console, loading_spinner, setup_logging
from codemap.utils.git_utils import (
    GitError,
    commit_only_files,
    get_other_staged_files,
    get_staged_diff,
    get_unstaged_diff,
    get_untracked_files,
    run_git_command,
)
from codemap.utils.llm_utils import create_universal_generator, generate_message

from .cli_types import VerboseFlag

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
        from codemap.git.message_generator import DiffChunkData

        # Convert DiffChunk to DiffChunkData
        chunk_dict = DiffChunkData(files=chunk.files, content=chunk.content)

        # Add description if it exists
        if chunk.description is not None:
            chunk_dict["description"] = chunk.description

        message = generator.fallback_generation(chunk_dict)
        return message, False


def print_chunk_summary(chunk: DiffChunk, index: int) -> None:
    """Print a summary of the chunk.

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


def _commit_changes(
    message: str,
    files: list[str],
    ignore_hooks: bool = False,
) -> bool:
    """Commit the changes with the provided message.

    Args:
        message: The commit message to use
        files: The list of files to commit
        ignore_hooks: Whether to ignore Git hooks if they fail

    Returns:
        Success status (True if commit was created)
    """
    try:
        # Filter out files that don't exist or aren't tracked by Git
        valid_files = []
        tracked_files = set()

        try:
            # Get tracked files from Git
            tracked_output = run_git_command(["git", "ls-files"])
            tracked_files = set(tracked_output.splitlines())
        except (OSError, ImportError) as e:
            logger.warning("Failed to get tracked files, will rely on filesystem checks only: %s", e)

        # Verify each file exists or is tracked
        for file in files:
            if Path(file).exists() or file in tracked_files:
                valid_files.append(file)
            else:
                logger.warning("Skipping file that doesn't exist or isn't tracked: %s", file)

        if not valid_files:
            logger.error("No valid files to commit")
            return False

        # Commit the changes
        logger.info("Creating commit with message: %s", message)
        logger.info("Files to commit: %s", ", ".join(valid_files))

        # Call git_utils to create the commit
        other_staged = commit_only_files(valid_files, message, ignore_hooks=ignore_hooks)

        if other_staged:
            logger.warning("There are %d other staged files that weren't included in this commit", len(other_staged))

        return True
    except Exception:
        logger.exception("Failed to create commit")
        return False


def _perform_commit(chunk: DiffChunk, message: str) -> bool:
    """Perform the actual commit.

    Args:
        chunk: Diff chunk to commit
        message: Commit message

    Returns:
        True if commit was successful
    """
    success = _commit_changes(message, chunk.files)
    if success:
        console.print(f"[green]✓[/green] Committed {len(chunk.files)} files")
    return success


def _edit_commit_message(message: str, _unused_chunk: DiffChunk) -> str:
    """Let the user edit the commit message.

    Args:
        message: The initial commit message
        _unused_chunk: The diff chunk for context (unused but kept for API consistency)

    Returns:
        The edited message, or empty string if user cancels
    """
    # Ask for a new commit message
    edited_message = questionary.text(
        "Edit commit message:",
        default=message,
        validate=lambda text: bool(text.strip()) or "Commit message cannot be empty",
    ).unsafe_ask()

    return edited_message if edited_message else ""


def _commit_with_message(chunk: DiffChunk, message: str) -> None:
    """Commit the changes with the provided message.

    Args:
        chunk: The diff chunk to commit
        message: The commit message to use
    """
    console.print("Committing changes...")
    success = _perform_commit(chunk, message)
    if not success:
        console.print("[red]Failed to commit changes[/red]")


def _commit_with_user_input(chunk: DiffChunk, generated_message: str) -> None:
    """Commit the changes with user input for the message.

    Args:
        chunk: The diff chunk to commit
        generated_message: The initial generated message to edit
    """
    try:
        # Let user edit the message
        edited_message = _edit_commit_message(generated_message, chunk)

        if edited_message:
            success = _perform_commit(chunk, edited_message)
            if not success:
                console.print("[red]Failed to commit changes[/red]")
        else:
            console.print("[yellow]Commit canceled - empty message[/yellow]")
    except KeyboardInterrupt:
        console.print("[yellow]Commit canceled by user[/yellow]")
    except Exception:
        logger.exception("Error during commit process")
        console.print("[red]Error:[/red] An unexpected error occurred during the commit process")


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

    # Display proposed message in a panel
    tag = "AI" if used_llm else "Simple"
    message_panel = Panel(
        Text(message, style="green"),
        title=f"[bold blue]Proposed message ({tag})[/]",
        border_style="blue" if used_llm else "yellow",
        expand=False,
        padding=(1, 2),
    )
    console.print(message_panel)

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
        _commit_with_message(context.chunk, message)
    elif action == "edit":
        _commit_with_user_input(context.chunk, message)
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

        # Display the message in a panel
        tag = "AI" if used_llm else "Simple"
        message_panel = Panel(
            Text(message, style="green"),
            title=f"[bold blue]{tag}[/]",
            border_style="blue" if used_llm else "yellow",
            expand=False,
            padding=(1, 2),
        )
        console.print(message_panel)
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
        if repo_path is None:
            console.print("[red]Error:[/red] Repository path is None")
            return 1
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
        repo_path=repo_path,  # Now guaranteed to be a Path, not None
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
            splitter = DiffSplitter(repo_path)  # Now guaranteed to be a Path, not None
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


def _raise_command_failed_error() -> None:
    """Raise an error for failed command execution."""
    msg = "Command failed to run successfully"
    raise RuntimeError(msg)


def validate_and_process_commit(
    path: Path | None,
    all_files: bool = False,
    model: str = "gpt-4o-mini",
) -> None:
    """Validate repository path and process commit.

    Args:
        path: Path to repository
        all_files: Whether to commit all files
        model: Model to use for generation
    """
    try:
        # Create the CommitCommand instance
        command = CommitCommand(
            path=path,
            model=model,
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
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


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
    is_verbose: VerboseFlag = False,
) -> None:
    """Generate AI-assisted commit messages for staged changes.

    This command analyzes your staged changes and generates commit messages using an LLM.
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
    )
