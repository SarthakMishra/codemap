"""Interactive commit interface for CodeMap."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import questionary
import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.text import Text

if TYPE_CHECKING:
	from .diff_splitter import DiffChunk
	from .semantic_grouping import SemanticGroup

logger = logging.getLogger(__name__)

# Constants
MAX_PREVIEW_LENGTH = 200
MAX_PREVIEW_LINES = 10


class ChunkAction(Enum):
	"""Possible actions for a diff chunk."""

	COMMIT = auto()
	EDIT = auto()
	SKIP = auto()
	ABORT = auto()
	REGENERATE = auto()
	EXIT = auto()


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

	def display_chunk(self, chunk: DiffChunk, index: int = 0, total: int = 1) -> None:
		"""
		Display a diff chunk to the user.

		Args:
		    chunk: DiffChunk to display
		    index: The 0-based index of the current chunk
		    total: The total number of chunks

		"""
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
			panel_content = "No content diff available (e.g., new file or mode change)"

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

	def display_group(self, group: SemanticGroup, index: int = 0, total: int = 1) -> None:
		"""
		Display a semantic group to the user.

		Args:
		        group: SemanticGroup to display
		        index: The 0-based index of the current group
		        total: The total number of groups

		"""
		# Build file information
		file_list = "\n".join([f"  - {file}" for file in group.files])
		file_info = Text(f"Files ({len(group.files)}):\n", style="blue")
		file_info.append(file_list)

		# Prepare diff preview - show first few lines of diff content
		diff_preview = group.content
		content_lines = diff_preview.splitlines()
		if len(content_lines) > MAX_PREVIEW_LINES:
			remaining_lines = len(content_lines) - MAX_PREVIEW_LINES
			diff_preview = "\n".join(content_lines[:MAX_PREVIEW_LINES]) + f"\n... ({remaining_lines} more lines)"
		diff_content = Text("\n\nDiff Preview:\n", style="blue")
		diff_content.append(diff_preview)

		# Calculate changes
		added = len(
			[line for line in group.content.splitlines() if line.startswith("+") and not line.startswith("+++")]
		)
		removed = len(
			[line for line in group.content.splitlines() if line.startswith("-") and not line.startswith("---")]
		)
		changes_info = Text("\nChanges: ", style="blue")
		changes_info.append(f"{added} added, {removed} removed")

		# Determine title for the panel
		panel_title = f"[bold]Group {index + 1} of {total}[/bold]"

		# Create diff panel
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

		# Create message panel if message exists
		if hasattr(group, "message") and group.message:
			# Create message panel
			message_panel = Panel(
				Text(str(group.message), style="green"),
				title="[bold blue]Generated message[/]",
				border_style="green",
				expand=True,
				width=self.console.width,
				padding=(1, 2),
			)
			self.console.print(message_panel)
		else:
			self.console.print(
				Panel(
					Text("No message generated yet", style="dim"),
					title="[bold]Message[/]",
					border_style="yellow",
					expand=True,
					width=self.console.width,
					padding=(1, 2),
				)
			)

	def display_message(self, message: str, is_llm_generated: bool = False) -> None:
		"""
		Display a commit message to the user.

		Args:
		    message: The commit message to display
		    is_llm_generated: Whether the message was generated by an LLM

		"""
		tag = "AI" if is_llm_generated else "Simple"
		message_panel = Panel(
			Text(message, style="green"),
			title=f"[bold {'blue' if is_llm_generated else 'yellow'}]Proposed message ({tag})[/]",
			border_style="blue" if is_llm_generated else "yellow",
			expand=False,
			padding=(1, 2),
		)
		self.console.print(message_panel)

	def get_user_action(self) -> ChunkAction:
		"""
		Get the user's desired action for the current chunk.

		Returns:
		    ChunkAction indicating what to do with the chunk

		"""
		# Define options with their display text and corresponding action
		options: list[tuple[str, ChunkAction]] = [
			("Commit with this message", ChunkAction.COMMIT),
			("Edit message and commit", ChunkAction.EDIT),
			("Regenerate message", ChunkAction.REGENERATE),
			("Skip this chunk", ChunkAction.SKIP),
			("Exit without committing", ChunkAction.EXIT),
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
		return ChunkAction.EXIT

	def get_user_action_on_lint_failure(self) -> ChunkAction:
		"""
		Get the user's desired action when linting fails.

		Returns:
		    ChunkAction indicating what to do.

		"""
		options: list[tuple[str, ChunkAction]] = [
			("Regenerate message", ChunkAction.REGENERATE),
			("Bypass linter and commit with --no-verify", ChunkAction.COMMIT),
			("Edit message manually", ChunkAction.EDIT),
			("Skip this chunk", ChunkAction.SKIP),
			("Exit without committing", ChunkAction.EXIT),
		]
		result = questionary.select(
			"Linting failed. What would you like to do?",
			choices=[option[0] for option in options],
			qmark="?»",  # Use a different qmark to indicate failure state
			use_indicator=True,
			use_arrow_keys=True,
		).ask()
		for option, action in options:
			if option == result:
				return action
		return ChunkAction.EXIT  # Fallback

	def edit_message(self, current_message: str) -> str:
		"""
		Get an edited commit message from the user.

		Args:
		    current_message: Current commit message

		Returns:
		    Edited commit message

		"""
		self.console.print("\n[bold blue]Edit commit message:[/]")
		self.console.print("[dim]Press Enter to keep current message[/]")
		return Prompt.ask("Message", default=current_message)

	def process_chunk(self, chunk: DiffChunk, index: int = 0, total: int = 1) -> ChunkResult:
		"""
		Process a single diff chunk interactively.

		Args:
		    chunk: DiffChunk to process
		    index: The 0-based index of the current chunk
		    total: The total number of chunks

		Returns:
		    ChunkResult with the user's action and any modified message

		"""
		# Display the combined diff and message panel
		self.display_chunk(chunk, index, total)

		# Now get the user's action through questionary (without displaying another message panel)
		action = self.get_user_action()

		if action == ChunkAction.EDIT:
			message = self.edit_message(chunk.description or "")
			return ChunkResult(ChunkAction.COMMIT, message)

		if action == ChunkAction.COMMIT:
			return ChunkResult(action, chunk.description)

		return ChunkResult(action)

	def confirm_abort(self) -> bool:
		"""
		Ask the user to confirm aborting the commit process.

		Returns:
		    True if the user confirms, False otherwise

		Raises:
		    typer.Exit: When the user confirms exiting

		"""
		confirmed = Confirm.ask(
			"\n[bold yellow]Are you sure you want to exit without committing?[/]",
			default=False,
		)

		if confirmed:
			self.console.print("[yellow]Exiting commit process...[/yellow]")
			# Use a zero exit code to indicate a successful (intended) exit
			# This prevents error messages from showing when exiting
			raise typer.Exit(code=0)

		return False

	def confirm_bypass_hooks(self) -> ChunkAction:
		"""
		Ask the user what to do when git hooks fail.

		Returns:
		    ChunkAction indicating what to do next

		"""
		self.console.print("\n[bold yellow]Git hooks failed.[/]")
		self.console.print("[yellow]This may be due to linting or other pre-commit checks.[/]")

		options: list[tuple[str, ChunkAction]] = [
			("Force commit and bypass hooks", ChunkAction.COMMIT),
			("Regenerate message and try again", ChunkAction.REGENERATE),
			("Edit message manually", ChunkAction.EDIT),
			("Skip this group", ChunkAction.SKIP),
			("Exit without committing", ChunkAction.EXIT),
		]

		result = questionary.select(
			"What would you like to do?",
			choices=[option[0] for option in options],
			qmark="»",
			use_indicator=True,
			use_arrow_keys=True,
		).ask()

		for option, action in options:
			if option == result:
				return action

		# Fallback (should never happen)
		return ChunkAction.EXIT

	def show_success(self, message: str) -> None:
		"""
		Show a success message.

		Args:
		    message: Message to display

		"""
		self.console.print(f"\n[bold green]✓[/] {message}")

	def show_warning(self, message: str) -> None:
		"""
		Show a warning message to the user.

		Args:
		    message: Warning message to display

		"""
		self.console.print(f"\n[bold yellow]⚠[/] {message}")

	def show_error(self, message: str) -> None:
		"""
		Show an error message to the user.

		Args:
		    message: Error message to display

		"""
		if "No changes to commit" in message:
			# This is an informational message, not an error
			self.console.print(f"[yellow]{message}[/yellow]")
		else:
			# This is a real error
			self.console.print(f"[red]Error:[/red] {message}")

	def show_skipped(self, files: list[str]) -> None:
		"""
		Show which files were skipped.

		Args:
		    files: List of skipped files

		"""
		if files:
			self.console.print("\n[yellow]Skipped changes in:[/]")
			for file in files:
				self.console.print(f"  • {file}")

	def show_message(self, message: str) -> None:
		"""
		Show a general informational message.

		Args:
		    message: Message to display

		"""
		self.console.print(f"\n{message}")

	def show_regenerating(self) -> None:
		"""Show message indicating message regeneration."""
		self.console.print("\n[yellow]Regenerating commit message...[/yellow]")

	def show_all_committed(self) -> None:
		"""Show message indicating all changes are committed."""
		self.console.print("[green]✓[/green] All changes committed!")

	def show_all_done(self) -> None:
		"""
		Show a final success message when the process completes.

		This is an alias for show_all_committed for now, but could be
		customized.

		"""
		self.show_all_committed()

	def show_lint_errors(self, errors: list[str]) -> None:
		"""Display linting errors to the user."""
		self.console.print("[bold red]Commit message failed linting:[/bold red]")
		for error in errors:
			self.console.print(f"  - {error}")

	def confirm_commit_with_lint_errors(self) -> bool:
		"""Ask the user if they want to commit despite lint errors."""
		return questionary.confirm("Commit message has lint errors. Commit anyway?", default=False).ask()

	def confirm_exit(self) -> bool:
		"""Ask the user to confirm exiting without committing."""
		return questionary.confirm("Are you sure you want to exit without committing?", default=False).ask()

	def display_failed_lint_message(self, message: str, lint_errors: list[str], is_llm_generated: bool = False) -> None:
		"""
		Display a commit message that failed linting, along with the errors.

		Args:
		    message: The commit message to display.
		    lint_errors: List of linting error messages.
		    is_llm_generated: Whether the message was generated by an LLM.

		"""
		tag = "AI" if is_llm_generated else "Simple"
		message_panel = Panel(
			Text(message, style="yellow"),  # Use yellow style for the message text
			title=f"[bold yellow]Proposed message ({tag}) - LINTING FAILED[/]",
			border_style="yellow",  # Yellow border to indicate warning/failure
			expand=False,
			padding=(1, 2),
		)
		self.console.print(message_panel)

		# Display lint errors below
		if lint_errors:
			error_text = Text("\n".join([f"- {err}" for err in lint_errors]), style="red")
			error_panel = Panel(
				error_text,
				title="[bold red]Linting Errors[/]",
				border_style="red",
				expand=False,
				padding=(1, 2),
			)
			self.console.print(error_panel)

	def display_failed_json_message(
		self, raw_content: str, json_errors: list[str], is_llm_generated: bool = True
	) -> None:
		"""
		Display a raw response that failed JSON validation, along with the errors.

		Args:
		    raw_content: The raw string content that failed JSON validation.
		    json_errors: List of JSON validation/formatting error messages.
		    is_llm_generated: Whether the message was generated by an LLM (usually True here).
		"""
		tag = "AI" if is_llm_generated else "Manual"
		message_panel = Panel(
			Text(raw_content, style="dim yellow"),  # Use dim yellow for the raw content
			title=f"[bold yellow]Invalid JSON Response ({tag}) - VALIDATION FAILED[/]",
			border_style="yellow",  # Yellow border to indicate JSON warning
			expand=False,
			padding=(1, 2),
		)
		self.console.print(message_panel)

		# Display JSON errors below
		if json_errors:
			error_text = Text("\n".join([f"- {err}" for err in json_errors]), style="red")
			error_panel = Panel(
				error_text,
				title="[bold red]JSON Validation Errors[/]",
				border_style="red",
				expand=False,
				padding=(1, 2),
			)
			self.console.print(error_panel)

	def get_group_action(self) -> ChunkAction:
		"""
		Get the user's desired action for the current semantic group.

		Returns:
		        ChunkAction indicating what to do with the group

		"""
		# Define options with their display text and corresponding action
		options: list[tuple[str, ChunkAction]] = [
			("Commit this group", ChunkAction.COMMIT),
			("Edit message and commit", ChunkAction.EDIT),
			("Regenerate message", ChunkAction.REGENERATE),
			("Skip this group", ChunkAction.SKIP),
			("Exit without committing", ChunkAction.EXIT),
		]

		# Use questionary to get the user's choice
		result = questionary.select(
			"What would you like to do with this group?",
			choices=[option[0] for option in options],
			default=options[0][0],  # Set "Commit this group" as default
			qmark="»",
			use_indicator=True,
			use_arrow_keys=True,
		).ask()

		# Map the result back to the ChunkAction
		for option, action in options:
			if option == result:
				return action

		# Fallback (should never happen)
		return ChunkAction.EXIT
