"""
Implementation of the gen command for code documentation generation.

This module implements the enhanced 'gen' command, which can generate:
1. LLM-optimized code context with semantic compression
2. Human-readable documentation in multiple formats

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from codemap.daemon.client import DaemonClient
from codemap.daemon.command import start_daemon, wait_for_daemon_api
from codemap.gen import (
	CompressionStrategy,
	DocFormat,
	GenCommand,
	GenConfig,
	GenerationMode,
)
from codemap.utils.cli_utils import exit_with_error, setup_logging
from codemap.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# Command line argument annotations
PathArg = Annotated[
	Path,
	typer.Argument(
		exists=True,
		help="Path to the codebase to analyze",
		show_default=True,
	),
]

OutputOpt = Annotated[
	Path | None,
	typer.Option(
		"--output",
		"-o",
		help="Output file path (overrides config)",
	),
]

ConfigOpt = Annotated[
	Path | None,
	typer.Option(
		"--config",
		"-c",
		help="Path to config file",
	),
]

MapTokensOpt = Annotated[
	int | None,
	typer.Option(
		"--map-tokens",
		help="Override token limit (set to 0 for unlimited)",
	),
]

MaxContentLengthOpt = Annotated[
	int | None,
	typer.Option(
		"--max-content-length",
		help="Maximum content length for file display (set to 0 for unlimited)",
	),
]

TreeFlag = Annotated[
	bool,
	typer.Option(
		"--tree",
		"-t",
		help="Include directory tree in output",
	),
]

VerboseFlag = Annotated[
	bool,
	typer.Option(
		"--verbose",
		"-v",
		help="Enable verbose logging",
	),
]

AutoStartDaemonOpt = Annotated[
	bool,
	typer.Option(
		"--auto-start-daemon/--no-auto-start-daemon",
		help="Automatically start the daemon if it's not running",
	),
]


def check_daemon_connection() -> bool:
	"""
	Check if the daemon is running and accessible.

	Returns:
	    bool: True if the daemon is running, False otherwise

	"""
	try:
		client = DaemonClient()
		client.check_status()
		return True
	except (RuntimeError, ConnectionError):
		return False


def gen_command(
	path: PathArg = Path(),
	output: OutputOpt = None,
	config: ConfigOpt = None,
	map_tokens: MapTokensOpt = None,
	max_content_length: MaxContentLengthOpt = None,
	compression: Annotated[
		CompressionStrategy,
		typer.Option(
			"--compression",
			"-c",
			help="Compression strategy to use",
			case_sensitive=False,
		),
	] = CompressionStrategy.SMART,
	mode: Annotated[
		GenerationMode,
		typer.Option(
			"--mode",
			"-m",
			help="Generation mode: llm (for AI context) or human (for human docs)",
			case_sensitive=False,
		),
	] = GenerationMode.LLM,
	format: Annotated[  # noqa: A002
		DocFormat,
		typer.Option(
			"--format",
			"-f",
			help="Output format for human-readable docs",
			case_sensitive=False,
		),
	] = DocFormat.MARKDOWN,
	semantic_analysis: Annotated[
		bool,
		typer.Option(
			"--semantic/--no-semantic",
			help="Enable/disable semantic analysis using LSP",
		),
	] = True,
	tree: Annotated[
		bool | None,
		typer.Option(
			"--tree/--no-tree",
			"-t",
			help="Include directory tree in output",
		),
	] = None,
	is_verbose: Annotated[
		bool,
		typer.Option(
			"--verbose",
			"-v",
			help="Enable verbose logging",
		),
	] = False,
	auto_start_daemon: AutoStartDaemonOpt = False,
) -> None:
	"""
	Generate code documentation for LLM context or human-readable docs.

	This command processes a codebase and generates either:
	- LLM-optimized context with semantic compression (default)
	- Human-readable documentation in various formats

	Examples:
	        codemap gen                      # Generate LLM context for current directory
	        codemap gen --mode human         # Generate human-readable docs
	        codemap gen --compression aggressive    # Use aggressive compression
	        codemap gen --format mkdocs      # Generate MkDocs format (for human mode)

	"""
	setup_logging(is_verbose=is_verbose)

	try:
		target_path = path.resolve()
		project_root = Path.cwd()

		# Load config
		config_loader = ConfigLoader(str(config) if config else None)
		config_data = config_loader.config

		# Get gen-specific config with defaults
		gen_config_data = config_data.get("gen", {})

		# Check if daemon is required and if it's running
		if not check_daemon_connection():
			# Get the auto_start_daemon setting from either the command line or config
			auto_start = auto_start_daemon or gen_config_data.get("auto_start_daemon", False)

			if auto_start:
				from codemap.utils.cli_utils import console

				console.print("[yellow]Daemon is not running. Attempting to start it...[/yellow]")

				result = start_daemon(config_path=config, foreground=False, timeout=30)
				if result != 0 or not wait_for_daemon_api(timeout=10):
					exit_with_error("Failed to start the daemon. Please start it manually with 'codemap daemon start'")

				console.print("[green]Daemon started successfully[/green]")
			else:
				exit_with_error(
					"The CodeMap daemon is not running. "
					"Please start it with 'codemap daemon start' or use --auto-start-daemon."
				)

		# Override config values from command line arguments
		token_limit = map_tokens if map_tokens is not None else gen_config_data.get("token_limit", 10000)
		content_length = (
			max_content_length if max_content_length is not None else gen_config_data.get("max_content_length", 5000)
		)

		# Get other gen settings with defaults
		config_compression = gen_config_data.get("compression", CompressionStrategy.SMART.value)
		config_mode = gen_config_data.get("mode", GenerationMode.LLM.value)
		config_doc_format = gen_config_data.get("doc_format", DocFormat.MARKDOWN.value)

		# Command line arguments override config file
		compression_strategy = compression or CompressionStrategy(config_compression)
		generation_mode = mode or GenerationMode(config_mode)
		doc_format = format or DocFormat(config_doc_format)

		# Handle boolean flags - default to config values if not provided
		include_tree = tree if tree is not None else gen_config_data.get("include_tree", False)
		enable_semantic = (
			semantic_analysis if semantic_analysis is not None else gen_config_data.get("semantic_analysis", True)
		)

		# Create generation config
		gen_config = GenConfig(
			mode=generation_mode,
			compression_strategy=compression_strategy,
			doc_format=doc_format,
			token_limit=token_limit,
			max_content_length=content_length,
			include_tree=include_tree,
			semantic_analysis=enable_semantic,
			auto_start_daemon=auto_start_daemon,
		)

		# Determine output path
		from codemap.gen.utils import determine_output_path

		output_path = determine_output_path(project_root, output, gen_config_data)

		# Create and execute the gen command
		command = GenCommand(gen_config)
		success = command.execute(target_path, output_path)

		if not success:
			exit_with_error("Generation failed")

	except (FileNotFoundError, PermissionError, OSError) as e:
		exit_with_error(f"File system error: {e!s}", exception=e)
	except ValueError as e:
		exit_with_error(f"Configuration error: {e!s}", exception=e)


# Alias for backward compatibility
generate_command = gen_command
