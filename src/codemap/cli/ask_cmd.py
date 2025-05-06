"""CLI command for asking questions about the codebase using RAG."""

import logging
from pathlib import Path
from typing import Annotated, Any, cast

import asyncer
import typer

logger = logging.getLogger(__name__)

# --- Command Argument Annotations (Keep these lightweight) ---

QuestionArg = Annotated[
	str | None, typer.Argument(help="Your question about the codebase (omit for interactive mode).")
]

PathOpt = Annotated[
	Path | None,
	typer.Option(
		"--path",
		"-p",
		help="Path to the repository root (defaults to current directory).",
		exists=True,
		file_okay=False,
		dir_okay=True,
		resolve_path=True,
	),
]

ModelOpt = Annotated[
	str | None,
	typer.Option(
		"--model", "-m", help="LLM model to use (e.g., 'openai/gpt-4o-mini'). Overrides config."
	),  # Added alias
]

ApiBaseOpt = Annotated[str | None, typer.Option("--api-base", help="Override the LLM API base URL.")]

ApiKeyOpt = Annotated[
	str | None, typer.Option("--api-key", help="Override the LLM API key (use environment variables for security).")
]

InteractiveFlag = Annotated[bool, typer.Option("--interactive", "-i", help="Start an interactive chat session.")]


# --- Registration Function ---


def register_command(app: typer.Typer) -> None:
	"""Register the ask command with the CLI app."""

	@app.command(name="ask")
	@asyncer.runnify  # Apply runnify directly to the command function
	async def ask_command(
		question: QuestionArg = None,
		path: PathOpt = None,
		model: ModelOpt = None,
		api_base: ApiBaseOpt = None,
		api_key: ApiKeyOpt = None,
		interactive: InteractiveFlag = False,
	) -> None:
		"""Ask questions about the codebase using Retrieval-Augmented Generation (RAG)."""
		# Defer heavy imports and logic to the implementation function
		await _ask_command_impl(
			question=question,
			path=path,
			model=model,
			api_base=api_base,
			api_key=api_key,
			interactive=interactive,
		)


# --- Implementation Function (Heavy imports deferred here) ---


async def _ask_command_impl(
	question: str | None = None,
	path: Path | None = None,
	model: str | None = None,
	api_base: str | None = None,
	api_key: str | None = None,
	interactive: bool = False,
) -> None:
	"""Implementation of the ask command with heavy imports deferred."""
	# Import heavy dependencies here instead of at the top
	from rich.prompt import Prompt

	from codemap.llm.rag.ask.command import AskCommand
	from codemap.llm.rag.ask.formatter import print_ask_result
	from codemap.utils.cli_utils import exit_with_error, handle_keyboard_interrupt
	from codemap.utils.config_loader import ConfigLoader

	repo_path = path or Path.cwd()
	logger.info(f"Received ask command for path: {repo_path}")

	# Determine if running in interactive mode (flag or config)
	config_loader = ConfigLoader.get_instance(repo_root=repo_path)
	config = config_loader.load_config()
	is_interactive = interactive or config.get("ask", {}).get("interactive_chat", False)

	if not is_interactive and question is None:
		exit_with_error("You must provide a question or use the --interactive flag.")

	try:
		# Initialize command once for potentially multiple runs (interactive)
		command = AskCommand(
			repo_path=repo_path,
			model=model,
			api_base=api_base,
			api_key=api_key,
		)

		# Perform async initialization before running any commands
		await command.initialize()

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
				print_ask_result(cast("dict[str, Any]", result))
		else:
			# Single question mode
			if question is None:
				exit_with_error("Internal error: Question is unexpectedly None in single-question mode.")
			# Use await for the async run method
			result = await command.run(question=cast("str", question))
			print_ask_result(cast("dict[str, Any]", result))

	except KeyboardInterrupt:
		handle_keyboard_interrupt()
	except Exception as e:
		logger.exception("An error occurred during the ask command.")
		exit_with_error(f"Error executing ask command: {e}", exception=e)
