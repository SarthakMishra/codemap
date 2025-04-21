"""Implementation of the generate command."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.progress import Progress

from codemap.analyzer.processor import DocumentationProcessor
from codemap.analyzer.tree_parser import CodeParser
from codemap.generators.markdown_generator import MarkdownGenerator
from codemap.utils.cli_utils import console, create_spinner_progress, ensure_directory_exists, setup_logging
from codemap.utils.config_loader import ConfigLoader
from codemap.utils.file_utils import get_output_path

from .cli_types import (
    ConfigOpt,
    MapTokensOpt,
    MaxContentLengthOpt,
    OutputOpt,
    PathArg,
    TreeFlag,
    VerboseFlag,
)

logger = logging.getLogger(__name__)


def generate_tree_only(target_path: Path, output: Path | None, config_data: dict) -> None:
    """Generate only the directory tree.

    Args:
        target_path: Path to generate tree for
        output: Optional output path
        config_data: Configuration data
    """
    with Progress() as progress:
        task = progress.add_task("Generating directory tree...", total=1)
        generator = MarkdownGenerator(target_path, config_data)
        tree_content = generator.generate_tree(target_path)
        progress.update(task, advance=1)

    # Write tree to output file or print to console
    if output:
        ensure_directory_exists(output.parent)
        output.write_text(tree_content)
        console.print(f"[green]Tree written to {output}")
    else:
        console.print(tree_content)


def determine_output_path(project_root: Path, output: Path | None, config_data: dict) -> Path:
    """Determine the output path for documentation.

    Args:
        project_root: Root directory of the project
        output: Optional output path from command line
        config_data: Configuration data

    Returns:
        The determined output path
    """
    return get_output_path(project_root, output, config_data)


def write_documentation(output_path: Path, documentation: str) -> None:
    """Write documentation to the specified output path.

    Args:
        output_path: Path to write documentation to
        documentation: Documentation content to write
    """
    with create_spinner_progress() as progress:
        task = progress.add_task("Writing documentation...", total=1)
        try:
            # Ensure parent directory exists
            ensure_directory_exists(output_path.parent)
            output_path.write_text(documentation)
            progress.update(task, advance=1)
            console.print(f"[green]Documentation written to {output_path}")
        except (PermissionError, OSError) as e:
            progress.update(task, advance=1)
            console.print(f"[red]Error writing documentation to {output_path}: {e!s}")
            raise


def generate_command(
    path: PathArg = Path(),
    output: OutputOpt = None,
    config: ConfigOpt = None,
    map_tokens: MapTokensOpt = None,
    max_content_length: MaxContentLengthOpt = None,
    tree: TreeFlag = False,
    is_verbose: VerboseFlag = False,
) -> None:
    """Generate documentation for the specified codebase."""
    setup_logging(is_verbose=is_verbose)
    try:
        target_path = path.resolve()

        # Always use the current working directory as project root
        project_root = Path.cwd()

        # Load config
        config_loader = ConfigLoader(str(config) if config else None)
        config_data = config_loader.config

        # Override config values from command line arguments
        if map_tokens is not None:
            config_data["token_limit"] = map_tokens
        if max_content_length is not None:
            config_data["max_content_length"] = max_content_length

        # Handle tree-only mode
        if tree:
            generate_tree_only(target_path, output, config_data)
            return

        # Initialize parser
        parser = CodeParser(config_data)
        token_limit = config_data.get("token_limit", 10000)

        # Process files
        processor = DocumentationProcessor(parser, token_limit)
        parsed_files = processor.process(target_path)

        # Generate documentation
        with create_spinner_progress() as progress:
            task = progress.add_task("Generating documentation...", total=1)
            generator = MarkdownGenerator(target_path, config_data)
            documentation = generator.generate_documentation(parsed_files)
            progress.update(task, advance=1)

        # Determine output path and write documentation
        output_path = determine_output_path(project_root, output, config_data)
        write_documentation(output_path, documentation)

    except (FileNotFoundError, PermissionError, OSError) as e:
        console.print(f"[red]File system error: {e!s}")
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print(f"[red]Configuration error: {e!s}")
        raise typer.Exit(1) from e
