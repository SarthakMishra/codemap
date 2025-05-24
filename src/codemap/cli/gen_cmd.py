"""CLI command for generating code documentation."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from codemap.utils.cli_utils import progress_indicator

logger = logging.getLogger(__name__)

# --- Command Argument Annotations (Keep these lightweight) ---

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

TreeOpt = Annotated[
	bool | None,
	typer.Option(
		"--tree/--no-tree",
		"-t",
		help="Include directory tree in output",
	),
]

EntityGraphOpt = Annotated[
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

MermaidLegendOpt = Annotated[
	bool | None,
	typer.Option(
		"--mermaid-legend/--no-mermaid-legend",
		help="Show/hide the legend in the Mermaid diagram",
	),
]

MermaidUnconnectedOpt = Annotated[
	bool | None,
	typer.Option(
		"--mermaid-unconnected/--no-mermaid-unconnected",
		help="Remove/keep nodes with no connections in the Mermaid diagram",
	),
]

SemanticAnalysisOpt = Annotated[
	bool,
	typer.Option(
		"--semantic/--no-semantic",
		help="Enable/disable semantic analysis",
	),
]


# --- Registration Function ---


def register_command(app: typer.Typer) -> None:
	"""Register the gen command with the CLI app."""

	@app.command(name="gen")
	def gen_command(
		path: PathArg = Path(),
		output: OutputOpt = None,
		max_content_length: MaxContentLengthOpt = None,
		lod_level_str: LODLevelOpt = "docs",
		semantic_analysis: SemanticAnalysisOpt = True,
		tree: TreeOpt = None,
		entity_graph: EntityGraphOpt = None,
		mermaid_entities_str: MermaidEntitiesOpt = None,
		mermaid_relationships_str: MermaidRelationshipsOpt = None,
		mermaid_show_legend_flag: MermaidLegendOpt = None,
		mermaid_remove_unconnected_flag: MermaidUnconnectedOpt = None,
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
		# Defer all heavy imports by calling implementation function
		_gen_command_impl(
			path=path,
			output=output,
			max_content_length=max_content_length,
			lod_level_str=lod_level_str,
			semantic_analysis=semantic_analysis,
			tree=tree,
			entity_graph=entity_graph,
			mermaid_entities_str=mermaid_entities_str,
			mermaid_relationships_str=mermaid_relationships_str,
			mermaid_show_legend_flag=mermaid_show_legend_flag,
			mermaid_remove_unconnected_flag=mermaid_remove_unconnected_flag,
		)


# --- Implementation Function (Heavy imports deferred here) ---


def _gen_command_impl(
	path: Path = Path(),
	output: Path | None = None,
	max_content_length: int | None = None,
	lod_level_str: str = "docs",
	semantic_analysis: bool = True,
	tree: bool | None = None,
	entity_graph: bool | None = None,
	mermaid_entities_str: str | None = None,
	mermaid_relationships_str: str | None = None,
	mermaid_show_legend_flag: bool | None = None,
	mermaid_remove_unconnected_flag: bool | None = None,
	mermaid_styled_flag: bool | None = None,
) -> None:
	"""Implementation of the gen command with heavy imports deferred."""
	# Import heavy dependencies here instead of at the top

	with progress_indicator("Setting up environment..."):
		from codemap.config import ConfigLoader
		from codemap.config.config_schema import GenSchema
		from codemap.gen import GenCommand
		from codemap.processor.lod import LODLevel
		from codemap.utils.cli_utils import exit_with_error, handle_keyboard_interrupt

	try:
		target_path = path.resolve()
		project_root = Path.cwd()

		# Load config
		config_loader = ConfigLoader.get_instance()

		# Get gen-specific config with defaults
		gen_config_data = config_loader.get.gen

		# Command line arguments override config file
		content_length = max_content_length if max_content_length is not None else gen_config_data.max_content_length

		# Handle boolean flags - default to config values if not provided
		include_tree = tree if tree is not None else gen_config_data.include_tree
		enable_semantic = semantic_analysis if semantic_analysis is not None else gen_config_data.semantic_analysis
		include_entity_graph = entity_graph if entity_graph is not None else gen_config_data.include_entity_graph

		# Initialize lod_level to a default before the try block
		lod_level: LODLevel = LODLevel.DOCS  # Default if conversion fails somehow

		# Handle LOD level (CLI > Config > Default)
		if lod_level_str != "docs":  # CLI argument was explicitly provided
			# Convert CLI string to LODLevel enum
			try:
				lod_level = getattr(LODLevel, lod_level_str.upper())
			except AttributeError:
				valid_names = [name.lower() for name in LODLevel.__members__]
				exit_with_error(f"Invalid LOD level '{lod_level_str}'. Valid levels are: {', '.join(valid_names)}")
		else:
			# Use config value (which is already a LODLevel enum thanks to our validator)
			lod_level = gen_config_data.lod_level

		# Handle Mermaid config (CLI > Config > Default)
		default_mermaid_entities = gen_config_data.mermaid_entities
		mermaid_entities = (
			[e.strip().lower() for e in mermaid_entities_str.split(",")]
			if mermaid_entities_str
			else default_mermaid_entities
		)

		default_mermaid_relationships = gen_config_data.mermaid_relationships
		mermaid_relationships = (
			[r.strip().lower() for r in mermaid_relationships_str.split(",")]
			if mermaid_relationships_str
			else default_mermaid_relationships
		)

		# Handle Mermaid legend visibility (CLI > Config > Default)
		mermaid_show_legend = (
			mermaid_show_legend_flag if mermaid_show_legend_flag is not None else gen_config_data.mermaid_show_legend
		)

		# Handle Mermaid unconnected node removal (CLI > Config > Default)
		mermaid_remove_unconnected = (
			mermaid_remove_unconnected_flag
			if mermaid_remove_unconnected_flag is not None
			else gen_config_data.mermaid_remove_unconnected
		)

		mermaid_styled = mermaid_styled_flag if mermaid_styled_flag is not None else gen_config_data.mermaid_styled

		# Create generation config
		gen_config = GenSchema(
			lod_level=lod_level,
			max_content_length=content_length,
			include_tree=include_tree,
			semantic_analysis=enable_semantic,
			include_entity_graph=include_entity_graph,
			use_gitignore=gen_config_data.use_gitignore,
			output_dir=str(gen_config_data.output_dir),  # Convert Path to string
			mermaid_entities=mermaid_entities,
			mermaid_relationships=mermaid_relationships,
			mermaid_show_legend=mermaid_show_legend,
			mermaid_remove_unconnected=mermaid_remove_unconnected,
			mermaid_styled=mermaid_styled,
		)

		# Determine output path
		from codemap.gen.utils import determine_output_path

		# --- DIAGNOSTIC PRINT --- #
		logger.debug("Gen config data being passed to determine_output_path: %s", gen_config_data)
		# ---------------------- #

		output_path = determine_output_path(project_root, config_loader, output)

		# Create and execute the gen command
		command = GenCommand(gen_config)
		success = command.execute(target_path, output_path)

		if not success:
			exit_with_error("Generation failed")

	except KeyboardInterrupt:
		handle_keyboard_interrupt()
	except (FileNotFoundError, PermissionError, OSError) as e:
		exit_with_error(f"File system error: {e!s}", exception=e)
	except ValueError as e:
		exit_with_error(f"Configuration error: {e!s}", exception=e)
