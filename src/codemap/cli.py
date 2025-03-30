"""Command-line interface for the codemap tool."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from typing_extensions import TypeAlias

from .analyzer.tree_parser import CodeParser
from .config import DEFAULT_CONFIG
from .generators.markdown_generator import MarkdownGenerator
from .utils.config_loader import ConfigLoader

# Configure logging
console = Console()
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="CodeMap - Generate optimized markdown documentation from your codebase.",
)

# Type aliases for CLI parameters
PathArg: TypeAlias = Annotated[
    Path,
    typer.Argument(
        exists=True,
        help="Path to the codebase to analyze",
        show_default=True,
    ),
]
OutputOpt: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--output",
        "-o",
        help="Output file path (overrides config)",
    ),
]
ConfigOpt: TypeAlias = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Path to config file",
    ),
]
MapTokensOpt: TypeAlias = Annotated[
    int | None,
    typer.Option(
        "--map-tokens",
        help="Override token limit",
    ),
]
TreeFlag: TypeAlias = Annotated[
    bool,
    typer.Option(
        "--tree",
        "-t",
        help="Generate only directory tree structure",
    ),
]
VerboseFlag: TypeAlias = Annotated[
    bool,
    typer.Option(
        "--verbose",
        "-v",
        help="Enable verbose output with debug logs",
    ),
]
ForceFlag: TypeAlias = Annotated[
    bool,
    typer.Option(
        "--force",
        "-f",
        help="Force overwrite existing files",
    ),
]


def _get_output_path(repo_root: Path, output_path: Path | None, config: dict) -> Path:
    """Get the output path for documentation.

    Args:
        repo_root: Root directory of the project
        output_path: Optional output path from command line
        config: Configuration dictionary

    Returns:
        Output path
    """
    if output_path:
        return output_path

    # Get output directory from config
    output_dir = config.get("output_dir", "documentation")

    # Create output directory if it doesn't exist
    output_dir_path = repo_root / output_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate a filename with timestamp
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"documentation_{timestamp}.md"

    return output_dir_path / filename


def setup_logging(*, is_verbose: bool) -> None:
    """Configure logging based on verbosity.

    Args:
        is_verbose: Whether to enable debug logging.
    """
    # Override LOG_LEVEL environment variable if verbose flag is set
    log_level = "DEBUG" if is_verbose else os.environ.get("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_path=True)],
    )


def _count_tokens(file_path: Path) -> int:
    """Rough estimation of tokens in a file.

    Args:
        file_path: Path to the file to count tokens in.

    Returns:
        Estimated number of tokens in the file.
    """
    try:
        with file_path.open(encoding="utf-8") as f:
            content = f.read()
            # Simple tokenization by whitespace
            return len(content.split())
    except OSError:
        return 0


@app.command()
def init(
    path: PathArg = Path(),
    force_flag: ForceFlag = False,
    is_verbose: VerboseFlag = None,
) -> None:
    """Initialize a new CodeMap project in the specified directory."""
    setup_logging(is_verbose=is_verbose)
    try:
        repo_root = path.resolve()
        config_file = repo_root / ".codemap.yml"
        docs_dir = repo_root / DEFAULT_CONFIG["output_dir"]

        # Check if files/directories already exist
        existing_files = []
        if config_file.exists():
            existing_files.append(config_file)
        if docs_dir.exists():
            existing_files.append(docs_dir)

        if not force_flag and existing_files:
            console.print("[yellow]CodeMap files already exist:")
            for f in existing_files:
                console.print(f"[yellow]  - {f}")
            console.print("[yellow]Use --force to overwrite.")
            raise typer.Exit(1)

        with Progress() as progress:
            task = progress.add_task("Initializing CodeMap...", total=3)

            # Create .codemap.yml
            config_file.write_text(yaml.dump(DEFAULT_CONFIG, sort_keys=False))
            progress.update(task, advance=1)

            # Create documentation directory
            if docs_dir.exists() and force_flag:
                docs_dir.rmdir()
            docs_dir.mkdir(exist_ok=True, parents=True)
            progress.update(task, advance=1)

            # Initialize parser to check it's working
            CodeParser()  # Just initialize without assigning to a variable
            progress.update(task, advance=1)

        console.print("\n✨ CodeMap initialized successfully!")
        console.print(f"[green]Created config file: {config_file}")
        console.print(f"[green]Created documentation directory: {docs_dir}")
        console.print("\nNext steps:")
        console.print("1. Review and customize .codemap.yml")
        console.print("2. Run 'codemap generate' to create documentation")

    except (FileNotFoundError, PermissionError, OSError) as e:
        console.print(f"[red]File system error: {e!s}")
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print(f"[red]Configuration error: {e!s}")
        raise typer.Exit(1) from e


@app.command()
def generate(  # noqa: PLR0913
    path: PathArg = Path(),
    output: OutputOpt = None,
    config: ConfigOpt = None,
    map_tokens: MapTokensOpt = None,
    tree: TreeFlag = None,
    is_verbose: VerboseFlag = None,
) -> None:
    """Generate documentation for the specified path."""
    setup_logging(is_verbose=is_verbose)
    try:
        repo_root = path.resolve()

        # If tree-only mode is requested, generate and output the tree
        if tree:
            # Load configuration
            config_loader = ConfigLoader(str(config) if config else None)
            config_data = config_loader.config

            # Generate tree
            with Progress() as progress:
                task = progress.add_task("Generating directory tree...", total=1)
                generator = MarkdownGenerator(repo_root, config_data)
                tree_content = generator.generate_tree(repo_root)
                progress.update(task, advance=1)

            # Write tree to output file or print to console
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(tree_content)
                console.print(f"[green]Tree written to {output}")
            else:
                console.print(tree_content)

            return

        # For full documentation generation, we need to load config, parse files, and generate docs
        config_data, parsed_files = _prepare_for_generation(repo_root, config, map_tokens)

        # Generate documentation
        documentation = _generate_documentation(repo_root, parsed_files, config_data)

        # Write documentation to file
        output_path = _write_documentation(repo_root, output, config_data, documentation)

        console.print(f"[green]Documentation generated at: {output_path}")

    except (OSError, ValueError) as e:
        console.print(f"[red]Error: {e!s}")
        logger.exception("Error generating documentation")
        raise typer.Exit(1) from e


def _prepare_for_generation(repo_root: Path, config: Path | None, map_tokens: int | None) -> tuple[dict, dict]:
    """Prepare for documentation generation by loading config and parsing files.

    Args:
        repo_root: Root directory of the repository
        config: Path to config file
        map_tokens: Optional token limit override

    Returns:
        Tuple of (config_data, parsed_files)
    """
    # Load configuration
    config_loader = ConfigLoader(str(config) if config else None)
    config_data = config_loader.config
    if map_tokens is not None:
        config_data["token_limit"] = map_tokens

    parsed_files = {}

    with Progress() as progress:
        # Count the number of files to parse
        parser = CodeParser(config_data)
        files_to_parse = []
        for root, _, files in os.walk(repo_root):
            for filename in files:
                file_path = Path(root) / filename
                if parser.should_parse(file_path):
                    files_to_parse.append(file_path)

        task = progress.add_task("Parsing files...", total=len(files_to_parse))

        # Parse each file
        for file_path in files_to_parse:
            file_info = parser.parse_file(file_path)
            parsed_files[file_path] = file_info
            progress.update(task, advance=1)

    return config_data, parsed_files


def _generate_documentation(repo_root: Path, parsed_files: dict, config_data: dict) -> str:
    """Generate documentation from parsed files.

    Args:
        repo_root: Root directory of the repository
        parsed_files: Dictionary of parsed files
        config_data: Configuration data

    Returns:
        Generated documentation as a string
    """
    token_limit = config_data.get("token_limit", 10000)

    with Progress() as progress:
        task = progress.add_task("Generating documentation...", total=2)

        # Apply token limit by selecting files based on size
        # Simple approach: sort files by size and include until we hit the token limit
        file_tokens = [(path, _count_tokens(path)) for path in parsed_files]
        sorted_files = sorted(file_tokens, key=lambda x: x[1], reverse=False)  # Small files first

        filtered_files = {}
        total_tokens = 0
        for path, tokens in sorted_files:
            if total_tokens + tokens <= token_limit:
                filtered_files[path] = parsed_files[path]
                total_tokens += tokens
            if total_tokens >= token_limit:
                break

        progress.update(task, advance=1)

        # Generate documentation
        generator = MarkdownGenerator(repo_root, config_data)
        documentation = generator.generate_documentation(filtered_files)
        progress.update(task, advance=1)

    return documentation


def _write_documentation(repo_root: Path, output: Path | None, config_data: dict, documentation: str) -> Path:
    """Write documentation to file.

    Args:
        repo_root: Root directory of the repository
        output: Optional output path override
        config_data: Configuration data
        documentation: Generated documentation string

    Returns:
        Path where documentation was written
    """
    with Progress() as progress:
        task = progress.add_task("Writing documentation...", total=1)

        # Determine output path
        output_path = _get_output_path(repo_root, output, config_data)

        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write documentation to file
        output_path.write_text(documentation)
        progress.update(task, advance=1)

    return output_path


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
