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

EntityGraphFlag = Annotated[
	bool | None,
	typer.Option(
		"--entity-graph/--no-entity-graph",
		"-e",
		help="Include entity relationship graph in output",
	),
]

LODLevelOpt = Annotated[
	str,
	typer.Option(
		"--lod",
		help="Level of Detail for code analysis (e.g., 'full', 'docs', 'signatures')",
		case_sensitive=False,
	),
]

MermaidEntitiesOpt = Annotated[
	str | None,  # Comma-separated string
	typer.Option(
		"--mermaid-entities",
		help="Comma-separated list of entity types to include in Mermaid graph (e.g., 'module,class,function')",
	),
]

MermaidRelationshipsOpt = Annotated[
	str | None,  # Comma-separated string
	typer.Option(
		"--mermaid-relationships",
		help="Comma-separated list of relationship types to include in Mermaid graph (e.g., 'declares,imports,calls')",
	),
]

MermaidLegendFlag = Annotated[
	bool | None,
	typer.Option(
		"--mermaid-legend/--no-mermaid-legend",
		help="Show/hide the legend in the Mermaid diagram",
	),
]

MermaidUnconnectedFlag = Annotated[
	bool | None,
	typer.Option(
		"--mermaid-unconnected/--no-mermaid-unconnected",
		help="Remove/keep nodes with no connections in the Mermaid diagram",
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
	lod_level_str: LODLevelOpt = "docs",
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
	entity_graph: EntityGraphFlag = None,
	mermaid_entities_str: MermaidEntitiesOpt = None,
	mermaid_relationships_str: MermaidRelationshipsOpt = None,
	mermaid_show_legend_flag: MermaidLegendFlag = None,
	mermaid_remove_unconnected_flag: MermaidUnconnectedFlag = None,
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
		include_entity_graph = (
			entity_graph if entity_graph is not None else gen_config_data.get("include_entity_graph", True)
		)

		# Initialize lod_level to a default before the try block
		lod_level: LODLevel = LODLevel.DOCS  # Default if conversion fails somehow

		# Get LOD level from config if not specified
		config_lod_str = str(gen_config_data.get("lod_level", LODLevel.DOCS.name.lower()))  # Default to 'docs'

		# Determine the final LOD level string (CLI > Config > Default)
		final_lod_str = lod_level_str if lod_level_str != LODLevel.DOCS.name.lower() else config_lod_str

		# Convert the final string to the LODLevel enum
		try:
			# Look up enum member by name (uppercase) instead of value
			lod_level = LODLevel[final_lod_str.upper()]
		except (ValueError, KeyError) as e:  # Catch KeyError for invalid names
			# Provide a more helpful error message if conversion fails
			# Get valid names (lowercase) for the error message
			valid_names = [name.lower() for name in LODLevel.__members__]
			exit_with_error(
				f"Invalid LOD level '{final_lod_str}'. Valid levels are: {', '.join(valid_names)}", exception=e
			)

		# Handle Mermaid config (CLI > Config > Default)
		default_mermaid_entities = gen_config_data.get("mermaid_entities", [])
		mermaid_entities = (
			[e.strip().lower() for e in mermaid_entities_str.split(",")]
			if mermaid_entities_str
			else default_mermaid_entities
		)

		default_mermaid_relationships = gen_config_data.get("mermaid_relationships", [])
		mermaid_relationships = (
			[r.strip().lower() for r in mermaid_relationships_str.split(",")]
			if mermaid_relationships_str
			else default_mermaid_relationships
		)

		# Handle Mermaid legend visibility (CLI > Config > Default)
		mermaid_show_legend = (
			mermaid_show_legend_flag
			if mermaid_show_legend_flag is not None
			else gen_config_data.get("mermaid_show_legend", True)  # Default to True
		)

		# Handle Mermaid unconnected node removal (CLI > Config > Default)
		mermaid_remove_unconnected = (
			mermaid_remove_unconnected_flag
			if mermaid_remove_unconnected_flag is not None
			else gen_config_data.get("mermaid_remove_unconnected", False)  # Default to False
		)

		# Create generation config
		gen_config = GenConfig(
			lod_level=lod_level,
			max_content_length=content_length,
			include_tree=include_tree,
			semantic_analysis=enable_semantic,
			include_entity_graph=include_entity_graph,
			use_gitignore=gen_config_data.get("use_gitignore", True),
			output_dir=Path(gen_config_data.get("output_dir", "documentation")),
			mermaid_entities=mermaid_entities,
			mermaid_relationships=mermaid_relationships,
			mermaid_show_legend=mermaid_show_legend,
			mermaid_remove_unconnected=mermaid_remove_unconnected,
		)

		# Determine output path
		from codemap.gen.utils import determine_output_path

		# --- DIAGNOSTIC PRINT --- #
		logger.debug("Gen config data being passed to determine_output_path: %s", gen_config_data)
		# ---------------------- #

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
