"""Formatter for the ask command output."""

import json

from rich import print as rich_print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

# Assuming AskResult is defined in command.py or a shared types file
from .command import AskResult  # Import the specific type


def format_ask_response(response_text: str | None) -> Markdown:
	"""
	Formats the AI's response text using Rich Markdown.

	Args:
	    response_text (str | None): The text response from the AI.

	Returns:
	    Markdown: A Rich Markdown object ready for printing.

	"""
	if response_text is None:
		response_text = "*No response generated.*"
	# Basic Markdown formatting. Can be enhanced later to detect code blocks,
	# file paths, etc., and apply specific styling or links.
	return Markdown(response_text)


def print_ask_result(result: AskResult) -> None:
	"""
	Prints the structured result of the ask command using Rich.

	Args:
	    result (AskResult): The structured result containing 'answer' and 'context'.

	"""
	answer = result.get("answer")
	context = result.get("context", [])

	# Print the main answer
	rich_print(Panel(format_ask_response(answer), title="[bold green]Answer[/]", border_style="green"))

	# Print the context used
	if context:
		rich_print(Panel("[bold yellow]Context Used:[/]"))
		for i, tool_call_info in enumerate(context):
			func_name = tool_call_info.get("function_name", "Unknown Function")
			args = tool_call_info.get("arguments", "{}")
			response_str = tool_call_info.get("response", "No response")

			# Try to parse arguments and response for better formatting
			try:
				args_dict = json.loads(args)
				args_formatted = json.dumps(args_dict, indent=2)
			except json.JSONDecodeError:
				args_formatted = args

			content_panel = None
			try:
				response_data = json.loads(response_str)
				if isinstance(response_data, dict) and "error" in response_data:
					content_panel = Panel(f"[red]Error:[/red] {response_data['error']}", border_style="red")
				else:
					# Pretty print JSON response
					response_formatted = json.dumps(response_data, indent=2)
					content_panel = Syntax(response_formatted, "json", theme="default", line_numbers=False)
			except json.JSONDecodeError:
				# Handle non-JSON or truncated responses
				if response_str.endswith("... [truncated]"):
					content_panel = Panel(
						f"{response_str}", border_style="dim yellow", title="[dim yellow]Truncated Response[/]"
					)
				else:
					content_panel = Panel(response_str, border_style="dim")  # Plain text response

			tool_panel_title = f"[bold cyan]Tool Call {i + 1}:[/] [yellow]{func_name}[/]".strip()
			args_syntax = Syntax(args_formatted, "json", theme="default", line_numbers=False)

			rich_print(Panel(args_syntax, title=f"{tool_panel_title} - Arguments", border_style="cyan", padding=(1, 2)))
			if content_panel:
				rich_print(
					Panel(content_panel, title=f"{tool_panel_title} - Response", border_style="blue", padding=(1, 2))
				)
			rich_print()  # Add spacing between tool calls
