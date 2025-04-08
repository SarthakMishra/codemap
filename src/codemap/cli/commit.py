"""Command for generating conventional commit messages from Git diffs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import questionary
import typer
from rich.console import Console
from rich.padding import Padding

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from codemap.commit.diff_splitter import DiffChunk, DiffSplitter, SplitStrategy
from codemap.commit.message_generator import LLMError, MessageGenerator
from codemap.git import GitWrapper
from codemap.utils import validate_repo_path

app = typer.Typer(help="Generate and apply conventional commits from Git diffs")
console = Console()
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
    split_strategy: SplitStrategy
    generation_mode: GenerationMode = field(default=GenerationMode.SMART)
    model: str = field(default="gpt-3.5-turbo")
    provider: str | None = field(default=None)
    api_base: str | None = field(default=None)
    commit: bool = field(default=True)
    prompt_template: str | None = field(default=None)
    api_key: str | None = field(default=None)


class ChunkAction(str, Enum):
    """Actions that can be taken on a diff chunk."""

    COMMIT = "commit"
    EDIT = "edit"
    REGENERATE = "regenerate"
    SKIP = "skip"
    EXIT = "exit"


@app.callback()
def callback() -> None:
    """Generate conventional commit messages."""


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
        model: Model identifier like "provider/model_name"

    Returns:
        Provider name or None if not in expected format
    """
    if "/" in model:
        provider, _ = model.split("/", 1)
        return provider
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
    }

    # Get environment variable name for this provider
    env_var = provider_env_vars.get(provider)
    if not env_var:
        return None

    # Try to get key from environment
    api_key = os.environ.get(env_var)

    # Special case for groq
    if not api_key and provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")

    # Fallback to OpenAI
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    return api_key


def setup_message_generator(options: CommitOptions) -> MessageGenerator:
    """Set up a message generator with the provided options.

    Args:
        options: Command options

    Returns:
        Configured message generator
    """
    # Try to load .env file if it exists
    if load_dotenv:
        load_dotenv()

    # Load custom prompt template if provided
    custom_prompt = _load_prompt_template(options.prompt_template)

    # Extract provider from model if not explicitly provided
    if not options.provider:
        options.provider = _extract_provider_from_model(options.model)

    # Set up API key if provided, otherwise try to get from environment
    api_key = options.api_key
    if not api_key and options.provider:
        api_key = _get_api_key_for_provider(options.provider)

    # Set up environment variables for API keys if provided
    if api_key and options.provider:
        set_provider_api_key(options.provider, api_key)

    # Create and return the message generator
    return MessageGenerator(
        options.repo_path,
        prompt_template=custom_prompt,
        model=options.model,
        provider=options.provider,
        api_base=options.api_base,
    )


def set_provider_api_key(provider: str | None, api_key: str) -> None:
    """Set the API key in the environment for the given provider.

    Args:
        provider: Provider name (or None for default)
        api_key: API key to set
    """
    if provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif provider == "cohere":
        os.environ["COHERE_API_KEY"] = api_key
    elif provider == "azure":
        os.environ["AZURE_API_KEY"] = api_key
    elif provider == "groq":
        os.environ["GROQ_API_KEY"] = api_key
    elif provider == "mistral":
        os.environ["MISTRAL_API_KEY"] = api_key
    elif provider == "together":
        os.environ["TOGETHER_API_KEY"] = api_key
    else:
        # Default to OpenAI
        os.environ["OPENAI_API_KEY"] = api_key


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
    if mode == GenerationMode.SIMPLE:
        message = generator.fallback_generation(chunk)
        used_llm = False
    else:
        try:
            message, used_llm = generator.generate_message(chunk)
        except LLMError as e:
            console.print(f"[red]Error generating message:[/red] {e!s}")
            message = generator.fallback_generation(chunk)
            used_llm = False

    return message, used_llm


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
        max_preview_length = 300
        preview = (
            chunk.content[:max_preview_length] + "..." if len(chunk.content) > max_preview_length else chunk.content
        )
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
    from codemap.utils.git_utils import get_other_staged_files, get_untracked_files

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


def _perform_commit(chunk: DiffChunk, message: str, git: GitWrapper) -> None:
    """Perform the actual commit operation.

    Args:
        chunk: Chunk to commit
        message: Commit message
        git: Git wrapper
    """
    try:
        git.commit_only_specified_files(chunk.files, message)
        console.print(f"[green]✓[/green] Committed {len(chunk.files)} file(s)")
    except (OSError, ValueError, RuntimeError) as e:
        # Check if this might be a git hook error
        if "hook" in str(e).lower():
            console.print(f"[yellow]Warning:[/yellow] Git hook failed: {e!s}")

            # Ask if user wants to bypass hooks
            if questionary.confirm("Would you like to try again bypassing git hooks?").ask():
                # Call with ignore_hooks=True
                from codemap.utils.git_utils import commit_only_files

                commit_only_files(chunk.files, message, ignore_hooks=True)
                console.print(f"[green]✓[/green] Committed {len(chunk.files)} file(s) (hooks bypassed)")
            else:
                console.print("[yellow]Commit cancelled due to hook failure[/yellow]")
        else:
            console.print(f"[red]Error:[/red] {e!s}")


def handle_commit_action(chunk: DiffChunk, message: str, git: GitWrapper) -> None:
    """Handle the commit action.

    Args:
        chunk: Diff chunk to commit
        message: Commit message
        git: Git wrapper
    """
    try:
        # Check for other files
        from codemap.utils.git_utils import GitError

        other_staged, other_untracked, has_warnings = _check_other_files(chunk.files)

        # If we have warnings, handle them
        if has_warnings and not _handle_other_files(chunk, other_staged, other_untracked):
            return

        # Perform the commit
        _perform_commit(chunk, message, git)

    except (OSError, ValueError, RuntimeError, GitError) as e:
        console.print(f"[red]Error:[/red] {e!s}")


def handle_edit_action(chunk: DiffChunk, message: str, git: GitWrapper) -> None:
    """Handle the edit action.

    Args:
        chunk: Diff chunk to commit
        message: Default commit message
        git: Git wrapper
    """
    # Let user edit the message
    new_message = questionary.text("Edit commit message:", default=message).ask()
    if not new_message:
        return

    try:
        # Check for other files
        from codemap.utils.git_utils import GitError

        other_staged, other_untracked, has_warnings = _check_other_files(chunk.files)

        # If we have warnings, handle them
        if has_warnings and not _handle_other_files(chunk, other_staged, other_untracked):
            return

        # Perform the commit with the edited message
        _perform_commit(chunk, new_message, git)

    except (OSError, ValueError, RuntimeError, GitError) as e:
        console.print(f"[red]Error:[/red] {e!s}")


@dataclass
class ChunkContext:
    """Context for processing a chunk."""

    chunk: DiffChunk
    index: int
    total: int
    generator: MessageGenerator
    git: GitWrapper
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
        {"value": ChunkAction.COMMIT, "name": "Commit with this message"},
        {"value": ChunkAction.EDIT, "name": "Edit message and commit"},
        {"value": ChunkAction.REGENERATE, "name": "Regenerate message"},
        {"value": ChunkAction.SKIP, "name": "Skip this chunk"},
        {"value": ChunkAction.EXIT, "name": "Exit without committing"},
    ]

    action = questionary.select("What would you like to do?", choices=choices).ask()

    if action == ChunkAction.COMMIT:
        handle_commit_action(context.chunk, message, context.git)
    elif action == ChunkAction.EDIT:
        handle_edit_action(context.chunk, message, context.git)
    elif action == ChunkAction.REGENERATE:
        # Just loop back for this chunk with smart generation
        return process_chunk_interactively(
            ChunkContext(
                chunk=context.chunk,
                index=context.index,
                total=context.total,
                generator=context.generator,
                git=context.git,
                mode=GenerationMode.SMART,
            ),
        )
    elif action == ChunkAction.SKIP:
        console.print("[yellow]Skipped![/yellow]")
    elif action == ChunkAction.EXIT:
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
    git: GitWrapper,
) -> int:
    """Process all chunks interactively.

    Args:
        options: Commit options
        chunks: List of diff chunks
        generator: Message generator to use
        git: Git wrapper

    Returns:
        Exit code (0 for success)
    """
    for i, chunk in enumerate(chunks):
        context = ChunkContext(
            chunk=chunk,
            index=i,
            total=len(chunks),
            generator=generator,
            git=git,
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
    split_strategy: SplitStrategy = SplitStrategy.FILE
    force_simple: bool = False
    api_key: str | None = None
    model: str = "gpt-3.5-turbo"
    provider: str | None = None
    api_base: str | None = None
    commit: bool = True
    prompt_template: str | None = None


DEFAULT_RUN_CONFIG = RunConfig()


@app.command(help="Generate and apply conventional commits from Git diffs")
def run(config: RunConfig = DEFAULT_RUN_CONFIG) -> int:
    """Run the commit command with the provided configuration.

    Args:
        config: Configuration options for the commit command.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    repo_path = validate_repo_path(config.repo_path)
    if not repo_path:
        console.print("[red]Error:[/red] Not a valid Git repository")
        return 1

    git = GitWrapper(repo_path)
    diff = git.get_uncommitted_changes()
    if not diff:
        console.print("No changes to commit")
        return 0

    splitter = DiffSplitter(repo_path)
    chunks = splitter.split_diff(diff, str(config.split_strategy))
    if not chunks:
        console.print("No changes to commit (after filtering)")
        return 0

    options = CommitOptions(
        repo_path=repo_path,
        split_strategy=config.split_strategy,
        generation_mode=GenerationMode.SIMPLE if config.force_simple else GenerationMode.SMART,
        model=config.model,
        provider=config.provider,
        api_base=config.api_base,
        commit=config.commit,
        prompt_template=config.prompt_template,
        api_key=config.api_key,
    )

    generator = setup_message_generator(options)

    if config.commit:
        return process_all_chunks(options, chunks, generator, git)
    display_suggested_messages(options, chunks, generator)
    return 0


if __name__ == "__main__":
    app()
