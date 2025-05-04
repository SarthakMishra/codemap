"""CLI command for asking questions about the codebase using RAG."""

# Need asyncio for the async command
import logging
from pathlib import Path
from typing import Annotated, Any, cast

import asyncer
import typer
from rich.prompt import Prompt

from codemap.llm.rag.command import AskCommand
from codemap.llm.rag.formatter import print_ask_result
from codemap.utils.cli_utils import exit_with_error, setup_logging
from codemap.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


# Make the command function async
@asyncer.runnify
async def ask_command(
	question: Annotated[
		str | None, typer.Argument(help="Your question about the codebase (omit for interactive mode).")
	] = None,
	path: Annotated[
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
	] = None,
	model: Annotated[
		str | None, typer.Option("--model", help="LLM model to use (e.g., 'openai/gpt-4o-mini'). Overrides config.")
	] = None,
	api_base: Annotated[str | None, typer.Option("--api-base", help="Override the LLM API base URL.")] = None,
	api_key: Annotated[
		str | None, typer.Option("--api-key", help="Override the LLM API key (use environment variables for security).")
	] = None,
	interactive: Annotated[
		bool, typer.Option("--interactive", "-i", help="Start an interactive chat session.")
	] = False,
	is_verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging.")] = False,
) -> None:
	"""Ask questions about the codebase using Retrieval-Augmented Generation (RAG)."""
	setup_logging(is_verbose=is_verbose)
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
			# question=question, # Question is handled by run()
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

	except Exception as e:
		logger.exception("An error occurred during the ask command.")
		exit_with_error(f"Error executing ask command: {e}", exception=e)
