"""Interactive commit interface for CodeMap."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

if TYPE_CHECKING:
    from .diff_splitter import DiffChunk

logger = logging.getLogger(__name__)
console = Console()


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
        choices = {
            "a": ChunkAction.ACCEPT,
            "e": ChunkAction.EDIT,
            "s": ChunkAction.SKIP,
            "x": ChunkAction.ABORT,
        }

        while True:
            self.console.print("\n[bold yellow]What would you like to do?[/]")
            self.console.print("  [a]ccept - Commit changes with current message")
            self.console.print("  [e]dit   - Edit commit message")
            self.console.print("  [s]kip   - Skip these changes")
            self.console.print("  e[x]it   - Abort the commit process")

            choice = Prompt.ask(
                "Choice",
                choices=list(choices.keys()),
                show_choices=False,
            ).lower()

            return choices[choice]

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
