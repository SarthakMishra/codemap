"""Interactive commit interface for CodeMap."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import questionary
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

if TYPE_CHECKING:
    from pathlib import Path

    from codemap.git import GitWrapper

    from .diff_splitter import DiffChunk
    from .message_generator import MessageGenerator

logger = logging.getLogger(__name__)
console = Console()


def process_all_chunks(
    _repo_path: Path,  # Unused parameter but kept for API compatibility
    chunks: list[DiffChunk],
    generator: MessageGenerator,
    git: GitWrapper,
    interactive: bool = True,
) -> int:
    """Process all chunks interactively or automatically.

    Args:
        _repo_path: Repository path
        chunks: List of diff chunks
        generator: Message generator to use
        git: Git wrapper
        interactive: Whether to process chunks interactively

    Returns:
        Exit code (0 for success)
    """
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        if interactive:
            # Display chunk information
            console.print(f"\n[bold]Commit {i + 1} of {len(chunks)}[/bold]")
            _print_chunk_summary(chunk, i)

            # Generate commit message
            message, used_llm = _generate_commit_message(chunk, generator)

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
                _handle_commit_action(chunk, message, git)
                i += 1
            elif action == "edit":
                _handle_edit_action(chunk, message, git)
                i += 1
            elif action == "regenerate":
                # Stay on same index to regenerate
                continue
            elif action == "skip":
                console.print("[yellow]Skipped![/yellow]")
                i += 1
            elif action == "exit":
                console.print("[yellow]Exiting commit process[/yellow]")
                return 0
        else:
            # Non-interactive mode: commit all chunks automatically
            message, _ = _generate_commit_message(chunk, generator)
            _handle_commit_action(chunk, message, git)
            i += 1

    console.print("[green]✓[/green] All changes committed!")
    return 0


def _generate_commit_message(
    chunk: DiffChunk,
    generator: MessageGenerator,
) -> tuple[str, bool]:
    """Generate a commit message for the given chunk.

    Args:
        chunk: Diff chunk to generate message for
        generator: Message generator to use

    Returns:
        Tuple of (message, whether LLM was used)
    """
    try:
        message, used_llm = generator.generate_message(chunk)
    except (ValueError, RuntimeError, ConnectionError) as e:
        console.print(f"[red]Error generating message:[/red] {e!s}")
        message = generator.fallback_generation(chunk)
        used_llm = False

    return message, used_llm


def _print_chunk_summary(chunk: DiffChunk, index: int) -> None:
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
        console.print(f"  [dim]{preview}[/dim]")
    else:
        console.print("  [dim](New files - no diff content available)[/dim]")


def _handle_commit_action(chunk: DiffChunk, message: str, git: GitWrapper) -> None:
    """Handle the commit action.

    Args:
        chunk: Diff chunk to commit
        message: Commit message
        git: Git wrapper
    """
    try:
        # Perform the commit
        git.commit_only_specified_files(chunk.files, message)
        console.print(f"[green]✓[/green] Committed {len(chunk.files)} file(s)")
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]Error:[/red] {e!s}")


def _handle_edit_action(chunk: DiffChunk, message: str, git: GitWrapper) -> None:
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
        # Perform the commit with the edited message
        git.commit_only_specified_files(chunk.files, new_message)
        console.print(f"[green]✓[/green] Committed {len(chunk.files)} file(s)")
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]Error:[/red] {e!s}")


class ChunkAction(Enum):
    """Possible actions for a diff chunk."""

    ACCEPT = auto()
    EDIT = auto()
    SKIP = auto()
    ABORT = auto()


@dataclass
class ChunkResult:
    """Result of processing a diff chunk."""

    action: ChunkAction
    message: str | None = None


class CommitUI:
    """Interactive UI for the commit process."""

    def __init__(self) -> None:
        """Initialize the commit UI."""
        self.console = Console()

    def _display_chunk(self, chunk: DiffChunk) -> None:
        """Display a diff chunk to the user.

        Args:
            chunk: DiffChunk to display
        """
        # Show affected files
        self.console.print("\n[bold blue]Files changed:[/]")
        for file in chunk.files:
            self.console.print(f"  • {file}")

        # Show the diff
        self.console.print("\n[bold blue]Changes:[/]")
        syntax = Syntax(chunk.content, "diff", theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax))

        # Show generated message if available
        if chunk.description:
            self.console.print("\n[bold blue]Generated commit message:[/]")
            self.console.print(Panel(Markdown(chunk.description)))

    def _get_user_action(self) -> ChunkAction:
        """Get the user's desired action for the current chunk.

        Returns:
            ChunkAction indicating what to do with the chunk
        """
        # Define options with their display text and corresponding action
        options: list[tuple[str, ChunkAction]] = [
            ("Accept - Commit changes with current message", ChunkAction.ACCEPT),
            ("Edit - Edit commit message", ChunkAction.EDIT),
            ("Skip - Skip these changes", ChunkAction.SKIP),
            ("Exit - Abort the commit process", ChunkAction.ABORT),
        ]

        # Display the question using questionary
        self.console.print("\n[bold yellow]What would you like to do?[/]")
        result = questionary.select(
            "",
            choices=[option[0] for option in options],
            default=options[0][0],  # Set "Accept" as default
            qmark="",  # Remove the question mark
            use_indicator=True,
            use_arrow_keys=True,
        ).ask()

        # Map the result back to the ChunkAction
        for option, action in options:
            if option == result:
                return action

        # Fallback (should never happen)
        return ChunkAction.ABORT

    def _edit_message(self, current_message: str) -> str:
        """Get an edited commit message from the user.

        Args:
            current_message: Current commit message

        Returns:
            Edited commit message
        """
        self.console.print("\n[bold blue]Edit commit message:[/]")
        self.console.print("[dim]Press Enter to keep current message[/]")
        return Prompt.ask("Message", default=current_message)

    def process_chunk(self, chunk: DiffChunk) -> ChunkResult:
        """Process a single diff chunk interactively.

        Args:
            chunk: DiffChunk to process

        Returns:
            ChunkResult with the user's action and any modified message
        """
        self._display_chunk(chunk)
        action = self._get_user_action()

        if action == ChunkAction.EDIT:
            message = self._edit_message(chunk.description or "")
            return ChunkResult(ChunkAction.ACCEPT, message)

        if action == ChunkAction.ACCEPT:
            return ChunkResult(action, chunk.description)

        return ChunkResult(action)

    def confirm_abort(self) -> bool:
        """Ask the user to confirm aborting the commit process.

        Returns:
            True if the user confirms, False otherwise
        """
        return Confirm.ask(
            "\n[bold red]Are you sure you want to abort?[/]",
            default=False,
        )

    def show_success(self, message: str) -> None:
        """Show a success message.

        Args:
            message: Message to display
        """
        self.console.print(f"\n[bold green]✓[/] {message}")

    def show_error(self, message: str) -> None:
        """Show an error message.

        Args:
            message: Error message to display
        """
        self.console.print(f"\n[bold red]✗[/] {message}")

    def show_skipped(self, files: list[str]) -> None:
        """Show which files were skipped.

        Args:
            files: List of skipped files
        """
        if files:
            self.console.print("\n[yellow]Skipped changes in:[/]")
            for file in files:
                self.console.print(f"  • {file}")
