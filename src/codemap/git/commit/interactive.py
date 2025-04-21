"""Interactive commit interface for CodeMap."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Union, cast

import questionary
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.text import Text

if TYPE_CHECKING:
    from .diff_splitter import DiffChunk as DiffSplitterChunk
    from .message_generator import DiffChunk as MessageGeneratorDiffChunk
    from .message_generator import MessageGenerator

# Import the universal message generator function
try:
    from codemap.utils.llm_utils import generate_message
except ImportError:
    # Fallback if not available
    generate_message = None

# Import LLMError for exception handling
# Import MessageGeneratorDiffChunk only for runtime use
from codemap.git.commit.message_generator import DiffChunk as MessageGeneratorDiffChunk
from codemap.git.commit.message_generator import DiffChunkDict, LLMError
from codemap.utils.git_utils import GitError, commit_only_files

logger = logging.getLogger(__name__)
console = Console()

# Constants
MAX_PREVIEW_LENGTH = 200
MAX_PREVIEW_LINES = 10


def process_all_chunks(
    chunks: list[DiffSplitterChunk | MessageGeneratorDiffChunk],
    generator: MessageGenerator,
    interactive: bool = True,
) -> int:
    """Process all chunks interactively or automatically.

    Args:
        chunks: List of diff chunks
        generator: Message generator to use
        interactive: Whether to process chunks interactively

    Returns:
        Exit code (0 for success)
    """
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        if interactive:
            # Display chunk information with consistent terminology
            console.print(f"\n[bold]Commit {i + 1} of {len(chunks)}[/bold]")
            _print_chunk_summary(chunk, i)

            # Generate commit message
            message, used_llm = _generate_commit_message(chunk, generator)

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
                _handle_commit_action(chunk, message)
                i += 1
            elif action == "edit":
                _handle_edit_action(chunk, message)
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
            _handle_commit_action(chunk, message)
            i += 1

    console.print("[green]✓[/green] All changes committed!")
    return 0


def _generate_commit_message(
    chunk: DiffSplitterChunk | MessageGeneratorDiffChunk,
    generator: MessageGenerator,
) -> tuple[str, bool]:
    """Generate a commit message for the given chunk.

    Args:
        chunk: Diff chunk to generate message for
        generator: Message generator to use

    Returns:
        Tuple of (message, whether LLM was used)
    """
    # Use the universal generate_message function if available
    if generate_message:
        try:
            message, used_llm = generate_message(chunk, generator)
            return message, used_llm
        except (ValueError, RuntimeError, ConnectionError, LLMError) as e:
            console.print(f"[red]Error using universal message generator:[/red] {e!s}")
            # Fall through to legacy approach as fallback

    # Legacy approach as fallback
    try:
        # Cast chunk to the type expected by generator.generate_message
        message, used_llm = generator.generate_message(cast("Union[DiffChunkDict, MessageGeneratorDiffChunk]", chunk))
    except (ValueError, RuntimeError, ConnectionError) as e:
        console.print(f"[red]Error generating message:[/red] {e!s}")
        # Convert to DiffChunkDict before calling fallback_generation
        chunk_dict = DiffChunkDict(
            files=chunk.files,
            content=chunk.content,
        )
        message = generator.fallback_generation(chunk_dict)
        used_llm = False

    return message, used_llm


def _print_chunk_summary(chunk: DiffSplitterChunk | MessageGeneratorDiffChunk, index: int) -> None:
    """Print a summary of a diff chunk.

    Args:
        chunk: The diff chunk to print
        index: Index of the chunk
    """
    console.print(f"[bold]Commit {index + 1}[/bold]:")
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
        # Truncate to maximum of MAX_PREVIEW_LINES lines
        content_lines = chunk.content.splitlines()
        if len(content_lines) > MAX_PREVIEW_LINES:
            remaining_lines = len(content_lines) - MAX_PREVIEW_LINES
            preview = "\n".join(content_lines[:MAX_PREVIEW_LINES]) + f"\n... ({remaining_lines} more lines)"
        else:
            preview = chunk.content
        console.print(f"  [dim]{preview}[/dim]")
    else:
        console.print("  [dim](New files - no diff content available)[/dim]")


def _handle_commit_action(chunk: DiffSplitterChunk | MessageGeneratorDiffChunk, message: str) -> None:
    """Handle commit action for a chunk.

    Args:
        chunk: Diff chunk to commit
        message: Commit message
    """
    try:
        commit_only_files(chunk.files, message)
        console.print(f"[green]✓[/green] Committed {len(chunk.files)} files")
    except GitError as e:
        console.print(f"[red]Error:[/red] {e!s}")


def _handle_edit_action(chunk: DiffSplitterChunk | MessageGeneratorDiffChunk, message: str) -> None:
    """Handle edit action for a chunk.

    Args:
        chunk: Diff chunk to commit
        message: Initial commit message to edit
    """
    edited_message = Prompt.ask("Edit commit message", default=message)
    if edited_message:
        _handle_commit_action(chunk, edited_message)
    else:
        console.print("[yellow]Commit cancelled: empty message[/yellow]")


class ChunkAction(Enum):
    """Possible actions for a diff chunk."""

    ACCEPT = auto()
    EDIT = auto()
    SKIP = auto()
    ABORT = auto()
    REGENERATE = auto()


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

    def _display_chunk(
        self, chunk: DiffSplitterChunk | MessageGeneratorDiffChunk, index: int = 0, total: int = 1
    ) -> None:
        """Display a diff chunk to the user.

        Args:
            chunk: DiffChunk to display
            index: The 0-based index of the current chunk
            total: The total number of chunks
        """
        # Import rich components
        from rich.panel import Panel
        from rich.text import Text

        # Build file information
        file_info = Text("Files: ", style="blue")
        file_info.append(", ".join(chunk.files))

        # Calculate changes
        added = len(
            [line for line in chunk.content.splitlines() if line.startswith("+") and not line.startswith("+++")]
        )
        removed = len(
            [line for line in chunk.content.splitlines() if line.startswith("-") and not line.startswith("---")]
        )
        changes_info = Text("\nChanges: ", style="blue")
        changes_info.append(f"{added} added, {removed} removed")

        # Prepare diff content
        panel_content = chunk.content
        if not panel_content.strip():
            panel_content = "[dim]No content diff available (e.g., new file or mode change)[/dim]"

        # Truncate to maximum of MAX_PREVIEW_LINES lines
        content_lines = panel_content.splitlines()
        if len(content_lines) > MAX_PREVIEW_LINES:
            remaining_lines = len(content_lines) - MAX_PREVIEW_LINES
            panel_content = "\n".join(content_lines[:MAX_PREVIEW_LINES]) + f"\n... ({remaining_lines} more lines)"

        diff_content = Text("\n" + panel_content)

        # Determine title for the panel - use provided index and total
        panel_title = f"[bold]Commit {index + 1} of {total}[/bold]"

        # Create content for the panel conditionally
        if getattr(chunk, "description", None):
            # If there's a description, create a combined panel
            if getattr(chunk, "is_llm_generated", False):
                message_title = "[bold blue]Proposed message (AI)[/]"
                message_style = "blue"
            else:
                message_title = "[bold yellow]Proposed message (Simple)[/]"
                message_style = "yellow"

            # Create separate panels and print them
            # First, print the diff panel
            diff_panel = Panel(
                Group(file_info, changes_info, diff_content),
                title=panel_title,
                border_style="cyan",
                expand=True,
                width=self.console.width,
                padding=(1, 2),
            )
            self.console.print(diff_panel)

            # Print divider
            self.console.print(Rule(style="dim"))

            # Then print the message panel
            message_panel = Panel(
                Text(str(chunk.description), style="green"),
                title=message_title,
                border_style=message_style,
                expand=True,
                width=self.console.width,
                padding=(1, 2),
            )
            self.console.print(message_panel)
        else:
            # If no description, just print the diff panel
            panel = Panel(
                Group(file_info, changes_info, diff_content),
                title=panel_title,
                border_style="cyan",
                expand=True,
                width=self.console.width,
                padding=(1, 2),
            )
            self.console.print()
            self.console.print(panel)
            self.console.print()

    def _get_user_action(self) -> ChunkAction:
        """Get the user's desired action for the current chunk.

        Returns:
            ChunkAction indicating what to do with the chunk
        """
        # Define options with their display text and corresponding action
        options: list[tuple[str, ChunkAction]] = [
            ("Commit with this message", ChunkAction.ACCEPT),
            ("Edit message and commit", ChunkAction.EDIT),
            ("Regenerate message", ChunkAction.REGENERATE),
            ("Skip this chunk", ChunkAction.SKIP),
            ("Exit without committing", ChunkAction.ABORT),
        ]

        # Use questionary to get the user's choice
        result = questionary.select(
            "What would you like to do?",
            choices=[option[0] for option in options],
            default=options[0][0],  # Set "Commit with this message" as default
            qmark="»",
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

    def process_chunk(
        self, chunk: DiffSplitterChunk | MessageGeneratorDiffChunk, index: int = 0, total: int = 1
    ) -> ChunkResult:
        """Process a single diff chunk interactively.

        Args:
            chunk: DiffChunk to process
            index: The 0-based index of the current chunk
            total: The total number of chunks

        Returns:
            ChunkResult with the user's action and any modified message
        """
        # Display the combined diff and message panel
        self._display_chunk(chunk, index, total)

        # Now get the user's action through questionary (without displaying another message panel)
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

    def confirm_bypass_hooks(self) -> bool:
        """Ask the user to confirm bypassing git hooks.

        Returns:
            True if the user confirms, False otherwise
        """
        self.console.print("\n[bold yellow]Git hooks failed.[/]")
        self.console.print("[yellow]This may be due to linting or other pre-commit checks.[/]")
        return Confirm.ask(
            "\n[bold yellow]Do you want to bypass git hooks and commit anyway?[/]",
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
