"""Command-line interface for the codemap tool."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer  # type: ignore[import]
import yaml  # type: ignore[import]
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing_extensions import TypeAlias

# Load environment variables from .env files
try:
    from dotenv import load_dotenv

    # Try to load from .env.local first, then fall back to .env
    if Path(".env.local").exists():
        load_dotenv(".env.local")
    load_dotenv()  # Load from .env if available
except ImportError:
    pass  # dotenv not installed, skip loading

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
        help="Override token limit (set to 0 for unlimited)",
    ),
]
MaxContentLengthOpt: TypeAlias = Annotated[
    int | None,
    typer.Option(
        "--max-content-length",
        help="Maximum content length for file display (set to 0 for unlimited)",
    ),
]
TreeFlag: TypeAlias = Annotated[
    bool | None,
    typer.Option(
        "--tree",
        "-t",
        help="Generate only directory tree structure",
    ),
]
VerboseFlag: TypeAlias = Annotated[
    bool | None,
    typer.Option(
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
]
ForceFlag: TypeAlias = Annotated[
    bool | None,
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

    # If output_dir is absolute, use it directly
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_absolute():
        # Otherwise, create the output directory in the project root
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
    except (OSError, UnicodeDecodeError):
        return 0


def _process_file(
    file_path: Path,
    parser: CodeParser,
    token_limit: int,
    total_tokens: int,
    progress: Progress | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """Process a single file and return its info and updated token count.

    Args:
        file_path: Path to the file to process
        parser: CodeParser instance
        token_limit: Maximum number of tokens allowed (0 means infinite)
        total_tokens: Current token count
        progress: Optional progress bar to update

    Returns:
        Tuple of (file_info, new_total_tokens)
    """
    try:
        if not parser.should_parse(file_path):
            return None, total_tokens

        file_info = parser.parse_file(file_path)
        tokens = _count_tokens(file_path)

        # Only check token limit if it's greater than 0 (not infinite)
        if token_limit > 0 and total_tokens + tokens > token_limit:
            logger.warning("Token limit reached, skipping remaining files")
            return None, total_tokens

        if progress:
            progress.update(progress.task_ids[0], advance=1)

        return file_info, total_tokens + tokens
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Failed to parse file %s: %s", file_path, e)
        return None, total_tokens


@app.command()
def init(
    path: PathArg = Path(),
    force_flag: ForceFlag = None,
    is_verbose: VerboseFlag = None,
) -> None:
    """Initialize a new CodeMap project in the specified directory."""
    setup_logging(is_verbose=bool(is_verbose))
    try:
        repo_root = path.resolve()
        config_file = repo_root / ".codemap.yml"
        docs_dir = repo_root / str(DEFAULT_CONFIG["output_dir"])

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


def _process_directory(
    target_path: Path,
    parser: CodeParser,
    token_limit: int,
) -> dict[Path, dict[str, Any]]:
    """Process a directory and return parsed files.

    Args:
        target_path: Path to process
        parser: CodeParser instance
        token_limit: Maximum number of tokens allowed (0 means infinite)

    Returns:
        Dictionary of parsed files
    """
    parsed_files: dict[Path, dict[str, Any]] = {}
    total_tokens = 0

    with Progress() as progress:
        _task = progress.add_task("Parsing files...", total=None)

        for root, _, files in os.walk(target_path):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                file_info, new_total = _process_file(file_path, parser, token_limit, total_tokens, progress)
                if file_info is not None:
                    parsed_files[file_path] = file_info
                    total_tokens = new_total
                # Only break if token_limit > 0 and we've exceeded it
                if token_limit > 0 and total_tokens >= token_limit:
                    break
            # Only break if token_limit > 0 and we've exceeded it
            if token_limit > 0 and total_tokens >= token_limit:
                break

    return parsed_files


@app.command()
def generate(
    path: PathArg = Path(),
    output: OutputOpt = None,
    config: ConfigOpt = None,
    map_tokens: MapTokensOpt = None,
    max_content_length: MaxContentLengthOpt = None,
    tree: TreeFlag = None,
    is_verbose: VerboseFlag = None,
) -> None:
    """Generate documentation for the specified path."""
    setup_logging(is_verbose=bool(is_verbose))
    try:
        target_path = path.resolve()

        # Always use the current working directory as project root
        project_root = Path.cwd()

        # Load config and respect the configured output directory
        config_loader = ConfigLoader(str(config) if config else None)
        config_data = config_loader.config

        # Override token limit if specified
        if map_tokens is not None:
            config_data["token_limit"] = map_tokens

        # Override max content length if specified
        if max_content_length is not None:
            config_data["max_content_length"] = max_content_length

        # If tree-only mode is requested, generate and output the tree
        if tree:
            # Generate tree
            with Progress() as progress:
                task = progress.add_task("Generating directory tree...", total=1)
                generator = MarkdownGenerator(target_path, config_data)
                tree_content = generator.generate_tree(target_path)
                progress.update(task, advance=1)

            # Write tree to output file or print to console
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(tree_content)
                console.print(f"[green]Tree written to {output}")
            else:
                console.print(tree_content)

            return

        # Initialize parser and parse files
        parser = CodeParser(config_data)
        token_limit = config_data.get("token_limit", 10000)

        # Process files based on whether target is a file or directory
        if target_path.is_file():
            parsed_files = {}
            file_info, _ = _process_file(target_path, parser, token_limit, 0)
            if file_info is not None:
                parsed_files[target_path] = file_info
        else:
            parsed_files = _process_directory(target_path, parser, token_limit)

        # Generate documentation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("Generating documentation...", total=1)
            generator = MarkdownGenerator(target_path, config_data)
            documentation = generator.generate_documentation(parsed_files)
            progress.update(task, advance=1)

        # Determine output path
        if output:
            # Use explicit output path if provided
            output_path = output
        else:
            # Get output directory from config
            output_dir = config_data.get("output_dir", "documentation")

            # If output_dir is absolute, use it directly
            output_dir_path = Path(output_dir)
            if not output_dir_path.is_absolute():
                # Otherwise, create the output directory relative to the project root
                output_dir_path = project_root / output_dir

            try:
                output_dir_path.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                console.print(f"[red]Unable to create output directory {output_dir_path}: {e!s}")
                raise typer.Exit(1) from e

            # Generate a timestamp for the filename
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"documentation_{timestamp}.md"
            output_path = output_dir_path / filename

        # Write documentation to file
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("Writing documentation...", total=1)
            try:
                # Ensure parent directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(documentation)
                progress.update(task, advance=1)
                console.print(f"[green]Documentation written to {output_path}")
            except (PermissionError, OSError) as e:
                progress.update(task, advance=1)
                console.print(f"[red]Error writing documentation to {output_path}: {e!s}")
                raise typer.Exit(1) from e

    except (FileNotFoundError, PermissionError, OSError) as e:
        console.print(f"[red]File system error: {e!s}")
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print(f"[red]Configuration error: {e!s}")
        raise typer.Exit(1) from e


@app.command()
def commit(
    path: Annotated[
        Path | None,
        typer.Argument(
            help="Path to repository or file to commit",
            exists=True,
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="LLM model to use for message generation",
        ),
    ] = "gpt-4o-mini",
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="OpenAI API key (or set OPENAI_API_KEY env var)",
            envvar="OPENAI_API_KEY",
        ),
    ] = None,
    is_verbose: VerboseFlag = None,
) -> None:
    """Generate and apply conventional commits from changes in a Git repository."""
    setup_logging(is_verbose=is_verbose is True)

    try:
        from .git.commit.command import CommitCommand

        # Set API key in environment if provided
        if api_key:
            import os

            os.environ["OPENAI_API_KEY"] = api_key

        command = CommitCommand(path, model=model)
        if not command.run():
            # Use try/finally pattern instead of raise
            console.print("[red]Error: Command failed[/]")
            sys.exit(1)

    except ImportError as e:
        console.print("[red]Failed to import commit feature. Please check your installation.[/]")
        logger.debug("Import error: %s", e)
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        logger.debug("Error running commit command", exc_info=True)
        raise typer.Exit(1) from e


# Import and add the PR command group
from .cli.pr import app as pr_app  # noqa: E402

# Add the PR command group to the main app
app.add_typer(pr_app, name="pr")

# Import and add the commit command group
from .cli.commit import app as commit_app  # noqa: E402

# Add the commit command group to the main app
app.add_typer(commit_app, name="commit")


def main() -> int:
    """Run the CLI application."""
    return app()


if __name__ == "__main__":
    sys.exit(main())
