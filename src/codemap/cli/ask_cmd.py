"""CLI command for asking questions about the codebase using RAG."""

import logging
from typing import Annotated, Any, cast

import asyncer
import typer

logger = logging.getLogger(__name__)

# --- Command Argument Annotations (Keep these lightweight) ---

QuestionArg = Annotated[
	str | None, typer.Argument(help="Your question about the codebase (omit for interactive mode).")
]

InteractiveFlag = Annotated[bool, typer.Option("--interactive", "-i", help="Start an interactive chat session.")]


# --- Registration Function ---


def register_command(app: typer.Typer) -> None:
	"""Register the ask command with the CLI app."""

	@app.command(name="ask")
	@asyncer.runnify
	async def ask_command(
		question: QuestionArg = None,
		interactive: InteractiveFlag = False,
	) -> None:
		"""Ask questions about the codebase using Retrieval-Augmented Generation (RAG)."""
		# Defer heavy imports and logic to the implementation function
		await _ask_command_impl(
			question=question,
			interactive=interactive,
		)


# --- Implementation Function (Heavy imports deferred here) ---


async def _ask_command_impl(
	question: str | None = None,
	interactive: bool = False,
) -> None:
	"""Implementation of the ask command with heavy imports deferred."""
	# Import heavy dependencies here instead of at the top
	from rich.prompt import Prompt

	from codemap.config import ConfigLoader
	from codemap.llm.rag import RagUI
	from codemap.llm.rag.ask.command import AskCommand
	from codemap.utils.cli_utils import exit_with_error, handle_keyboard_interrupt

	# Determine if running in interactive mode (flag or config)
	config_loader = ConfigLoader.get_instance()
	is_interactive = interactive or config_loader.get.ask.interactive_chat

	if not is_interactive and question is None:
		exit_with_error("You must provide a question or use the --interactive flag.")

	try:
		# Initialize command once for potentially multiple runs (interactive)
		command = AskCommand()

		# Perform async initialization before running any commands
		await command.initialize()

		ui = RagUI()

		if is_interactive:
			typer.echo("Starting interactive chat session. Type 'exit' or 'quit' to end.")
			while True:
				user_input = Prompt.ask("\nAsk a question")
				user_input_lower = user_input.lower().strip()
				if user_input_lower in ("exit", "quit"):
					typer.echo("Exiting interactive session.")
					break
				if not user_input.strip():
					continue

				# Use await for the async run method
				result = await command.run(question=user_input)
				ui.print_ask_result(cast("dict[str, Any]", result))
		else:
			# Single question mode
			if question is None:
				exit_with_error("Internal error: Question is unexpectedly None in single-question mode.")
			# Use await for the async run method
			result = await command.run(question=cast("str", question))
			ui.print_ask_result(cast("dict[str, Any]", result))

	except KeyboardInterrupt:
		handle_keyboard_interrupt()
	except Exception as e:
		logger.exception("An error occurred during the ask command.")
		exit_with_error(f"Error executing ask command: {e}", exception=e)
