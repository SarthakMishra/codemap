"""
Implementation of the gen command for code documentation generation.

This module implements the enhanced 'gen' command, which can generate
human-readable documentation in Markdown format.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from codemap.gen import GenCommand, GenConfig
from codemap.processor import LODLevel, create_processor
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

ProcessingFlag = Annotated[
	bool,
	typer.Option(
		"--process/--no-process",
		help="Process the codebase before generation",
	),
]

LODLevelOpt = Annotated[
	LODLevel,
	typer.Option(
		"--lod",
		help="Level of Detail for code analysis",
		case_sensitive=False,
	),
]


def initialize_processor(repo_path: Path, config_data: dict) -> None:
	"""
	Initialize the processor for code analysis.

	Args:
	    repo_path: Path to the repository
	    config_data: Configuration data

	"""
	from codemap.utils.cli_utils import console

	# Extract processor configuration
	processor_config = config_data.get("processor", {})

	# Get ignored patterns
	ignored_patterns = set(processor_config.get("ignored_patterns", []))
	ignored_patterns.update(
		[
			"**/.git/**",
			"**/__pycache__/**",
			"**/.venv/**",
			"**/node_modules/**",
			"**/.codemap_cache/**",
			"**/*.pyc",
			"**/dist/**",
			"**/build/**",
		]
	)

	# Initialize processor with LOD support
	try:
		processor = create_processor(repo_path=repo_path)
		console.print("[green]Processor initialized successfully[/green]")
		processor.stop()
	except Exception as e:
		console.print(f"[red]Failed to initialize processor: {e}[/red]")
		raise


def gen_command(
	path: PathArg = Path(),
	output: OutputOpt = None,
	config: ConfigOpt = None,
	max_content_length: MaxContentLengthOpt = None,
	lod_level: LODLevelOpt = LODLevel.DOCS,
	semantic_analysis: Annotated[
		bool,
		typer.Option(
			"--semantic/--no-semantic",
			help="Enable/disable semantic analysis",
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
	process: ProcessingFlag = True,
) -> None:
	"""
	Generate code documentation.

	This command processes a codebase and generates Markdown documentation
	with configurable level of detail.

	Examples:
	        codemap gen                      # Generate docs for current directory
	        codemap gen --lod full           # Generate full implementation docs
	        codemap gen --lod signatures     # Generate docs with signatures only
	        codemap gen --no-semantic        # Generate without semantic analysis

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

		# Process the codebase if requested
		if process:
			from codemap.utils.cli_utils import console

			console.print("[yellow]Initializing processor...[/yellow]")
			initialize_processor(target_path, config_data)
			console.print("[green]Processor initialization completed successfully[/green]")

		# Command line arguments override config file
		content_length = (
			max_content_length if max_content_length is not None else gen_config_data.get("max_content_length", 5000)
		)

		# Handle boolean flags - default to config values if not provided
		include_tree = tree if tree is not None else gen_config_data.get("include_tree", False)
		enable_semantic = (
			semantic_analysis if semantic_analysis is not None else gen_config_data.get("semantic_analysis", True)
		)

		# Get LOD level from config if not specified
		config_lod = gen_config_data.get("lod_level", LODLevel.DOCS.value)
		lod_level = lod_level or LODLevel(config_lod)

		# Create generation config
		gen_config = GenConfig(
			lod_level=lod_level,
			max_content_length=content_length,
			include_tree=include_tree,
			semantic_analysis=enable_semantic,
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
