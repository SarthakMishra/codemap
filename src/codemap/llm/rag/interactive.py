"""Interactive LLM interface for CodeMap."""

from __future__ import annotations

from typing import Any

from rich import print as rich_print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


class RagUI:
	"""Interactive UI for the RAG process."""

	def __init__(self) -> None:
		"""Initialize the RAG UI."""
		self.console = Console()

	def format_ask_response(self, response_text: str | None) -> Markdown:
		"""
		Formats the AI's response text using Rich Markdown.

		Args:
			response_text (Optional[str]): The text response from the AI.

		Returns:
			Markdown: A Rich Markdown object ready for printing.

		"""
		if response_text is None:
			response_text = "*No response generated.*"
		# Basic Markdown formatting. Can be enhanced later to detect code blocks,
		# file paths, etc., and apply specific styling or links.
		return Markdown(response_text)

	def print_ask_result(self, result: dict[str, Any]) -> None:
		"""
		Prints the structured result of the ask command using Rich.

		Args:
			result (Dict[str, Any]): The structured result containing 'answer' and 'context'.

		"""
		answer = result.get("answer")
		context = result.get("context", [])

		# Print the main answer
		rich_print(Panel(self.format_ask_response(answer), title="[bold green]Answer[/]", border_style="green"))

		# Print the context used if there are any items
		if context:
			# Build a single string with all context items as a numbered list
			context_list = []
			for i, item in enumerate(context, 1):
				file_path = item.get("file_path", "Unknown")
				start_line = item.get("start_line", -1)
				end_line = item.get("end_line", -1)
				distance = item.get("distance", -1.0)

				# Create the list item text
				location = f"{file_path}"
				if start_line > 0 and end_line > 0:
					location += f" (lines {start_line}-{end_line})"

				# Format with relevance info
				relevance = f"(similarity: {1 - distance:.2f})" if distance >= 0 else ""
				list_item = f"[bold cyan]{i}.[/bold cyan] {location} [dim]{relevance}[/dim]"
				context_list.append(list_item)

			# Join all list items into a single string
			context_content = "\n".join(context_list)

			# Print a single panel with all context items
			rich_print(
				Panel(
					context_content, title="[bold yellow]Context Used[/]", border_style="yellow", title_align="center"
				)
			)

			rich_print()

	def format_content_for_context(self, context_items: list[dict[str, Any]]) -> str:
		"""
		Format context items into a string suitable for inclusion in prompts.

		Args:
			context_items: List of context dictionaries with file_path, start_line, end_line, and content

		Returns:
			Formatted string with code snippets and file information

		"""
		if not context_items:
			return "No relevant code found in the repository."

		formatted_parts = []

		for i, item in enumerate(context_items, 1):
			# Extract file information
			file_path = item.get("file_path", "Unknown file")
			start_line = item.get("start_line", -1)
			end_line = item.get("end_line", -1)
			content = item.get("content", "")

			# Create a header with file info
			header = f"[{i}] {file_path}"
			if start_line > 0 and end_line > 0:
				header += f" (lines {start_line}-{end_line})"

			# Format the code snippet with the header
			formatted_parts.append(f"{header}\n{'-' * len(header)}\n{content}\n")

		return "\n\n".join(formatted_parts)
