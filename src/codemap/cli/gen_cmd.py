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

from codemap.gen import (
	CompressionStrategy,
	DocFormat,
	GenCommand,
	GenConfig,
	GenerationMode,
)
from codemap.processor.pipeline import ProcessingPipeline
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

ProcessingFlag = Annotated[
	bool,
	typer.Option(
		"--process/--no-process",
		help="Process the codebase before generation",
	),
]


def initialize_pipeline(repo_path: Path, config_data: dict) -> ProcessingPipeline:
	"""
	Initialize the processing pipeline for code analysis.

	Args:
	    repo_path: Path to the repository
	    config_data: Configuration data

	Returns:
	    ProcessingPipeline: Initialized pipeline

	"""
	from codemap.processor.embedding.models import EmbeddingConfig
	from codemap.processor.storage.base import StorageConfig
	from codemap.utils.directory_manager import get_directory_manager

	# Extract processor configuration
	processor_config = config_data.get("processor", {})

	# Get directory manager for cache directories
	dir_manager = get_directory_manager()
	dir_manager.set_project_dir(repo_path)
	project_cache_dir = dir_manager.get_project_cache_dir(create=True)

	if not project_cache_dir:
		project_cache_dir = repo_path / ".codemap_cache"
		project_cache_dir.mkdir(exist_ok=True, parents=True)

	# Configure storage
	storage_dir = project_cache_dir / "storage"
	storage_dir.mkdir(exist_ok=True, parents=True)
	cache_dir = storage_dir / "cache"
	cache_dir.mkdir(exist_ok=True, parents=True)

	storage_config = StorageConfig(uri=str(storage_dir / "vector.lance"), create_if_missing=True, cache_dir=cache_dir)

	# Configure embeddings
	embedding_cache_dir = project_cache_dir / "embeddings"
	embedding_cache_dir.mkdir(exist_ok=True, parents=True)
	embedding_config = EmbeddingConfig(
		model=processor_config.get("embedding_model", "sarthak1/Qodo-Embed-M-1-1.5B-M2V-Distilled"),
		dimensions=processor_config.get("embedding_dimensions", 384),
		batch_size=processor_config.get("batch_size", 32),
	)

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

	# Initialize pipeline
	pipeline = ProcessingPipeline(
		repo_path=repo_path,
		storage_config=storage_config,
		embedding_config=embedding_config,
		ignored_patterns=ignored_patterns,
		max_workers=processor_config.get("max_workers", 4),
		enable_lsp=processor_config.get("enable_lsp", True),
	)

	logger.info("Processing pipeline initialized for repository: %s", repo_path)
	return pipeline


def process_codebase(repo_path: Path, config_data: dict) -> None:
	"""
	Process the codebase with the pipeline.

	Args:
	    repo_path: Path to the repository
	    config_data: Configuration data

	"""
	import os

	from codemap.utils.cli_utils import console, progress_indicator

	# Initialize the pipeline
	pipeline = initialize_pipeline(repo_path, config_data)

	# Create a list of all files to process
	all_files = []
	logger.info("Scanning repository at %s for files to process", repo_path)

	# Get ignored patterns as a list of strings
	ignored_patterns = list(pipeline.ignored_patterns)

	for root, _, files in os.walk(repo_path):
		root_path = Path(root)

		# Skip ignored directories
		if any(part.startswith(".") for part in root_path.parts):
			continue

		for file in files:
			# Skip hidden files
			if file.startswith("."):
				continue

			file_path = root_path / file

			# Check if file should be processed
			should_process = True
			for pattern in ignored_patterns:
				try:
					if file_path.match(pattern):
						should_process = False
						break
				except (TypeError, ValueError):
					# Skip invalid patterns
					continue

			if should_process:
				all_files.append(file_path)

	# Process files in batches with progress display
	total_files = len(all_files)
	if total_files == 0:
		console.print("[yellow]No files found to process in the repository.[/]")
		return

	console.print(f"Found {total_files} files to process.")

	# Use the progress_indicator utility
	with progress_indicator("Processing repository files", style="progress", total=total_files) as advance:
		# Process in batches for better performance
		batch_size = 100
		for i in range(0, total_files, batch_size):
			batch = all_files[i : min(i + batch_size, total_files)]

			# Process the batch
			pipeline.batch_process(batch)

			# Update progress
			advance(len(batch))

	logger.info("Repository scan completed. Processed %d files.", total_files)
	console.print(f"[green]Repository scan complete. Processed {total_files} files.[/]")

	# Clean up
	pipeline.stop()


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
	process: ProcessingFlag = True,
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

		# Process the codebase if requested
		if process:
			from codemap.utils.cli_utils import console

			console.print("[yellow]Processing codebase...[/yellow]")
			process_codebase(target_path, config_data)
			console.print("[green]Codebase processing completed successfully[/green]")

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
