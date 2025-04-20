"""Main CLI module for CodeMap that duplicates functionality for tests."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console

# Initialize typer app - this will be used in tests
app = typer.Typer(
    help="CodeMap - Generate optimized markdown documentation from your codebase.",
)

# Create a PR command group for testing
pr_app = typer.Typer(help="Generate and manage pull requests")
app.add_typer(pr_app, name="pr")

# Setup logging and console
console = Console()
logger = logging.getLogger(__name__)

# Define parameter dependencies to avoid B008 error
PATH_ARGUMENT = Annotated[
    Path,
    typer.Argument(exists=True, file_okay=False, dir_okay=True, help="Path to initialize CodeMap in"),
]
FORCE_OPTION = Annotated[bool, typer.Option("--force", "-f", help="Force overwrite existing files")]

CODEBASE_PATH_ARGUMENT = Annotated[Path, typer.Argument(exists=True, help="Path to the codebase to analyze")]
OUTPUT_OPTION = Annotated[Path | None, typer.Option("--output", "-o", help="Output file path")]
CONFIG_OPTION = Annotated[Path | None, typer.Option("--config", "-c", help="Path to config file")]
MAP_TOKENS_OPTION = Annotated[
    int | None,
    typer.Option("--map-tokens", help="Override token limit (set to 0 for unlimited)"),
]
TREE_OPTION = Annotated[bool, typer.Option("--tree", "-t", help="Generate only directory tree structure")]

# Default path for the generate command
DEFAULT_PATH = Path.cwd()


def _get_output_path(repo_root: Path, output_path: Path | None, config: dict[str, Any]) -> Path:
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


@app.command(help="Initialize CodeMap in a directory.")
def init(
    path: PATH_ARGUMENT,
    force: FORCE_OPTION = False,  # FBT002: Boolean default is required by typer
) -> None:
    """Initialize CodeMap in a directory."""
    # Check if files already exist
    config_path = path / ".codemap.yml"
    docs_dir = path / "documentation"

    if config_path.exists() and not force:
        console.print("[red]Error:[/red] CodeMap files already exist. Use --force to overwrite.")
        raise typer.Exit(code=1)

    # Create config file
    default_config = {
        "token_limit": 0,
        "use_gitignore": True,
        "output_dir": "documentation",
        "max_content_length": 0,
    }

    # Ensure docs directory exists
    docs_dir.mkdir(exist_ok=True, parents=True)

    # Write config file
    with config_path.open("w") as f:
        yaml.dump(default_config, f, default_flow_style=False)

    console.print("[green]✓[/green] CodeMap initialized successfully!")


@app.command(help="Generate documentation from codebase.")
def generate(
    path: CODEBASE_PATH_ARGUMENT = DEFAULT_PATH,  # Using module-level singleton instead of function call
    output: OUTPUT_OPTION = None,
    config: CONFIG_OPTION = None,
    map_tokens: MAP_TOKENS_OPTION = None,  # ARG001: Kept for API compatibility
    tree: TREE_OPTION = False,  # FBT002: Boolean default is required by typer
) -> None:
    """Generate documentation from codebase."""
    # Load config from file if available
    config_dict = {"output_dir": "documentation"}
    config_path = config or path / ".codemap.yml"
    if config_path.exists() and config_path.is_file():
        try:
            with config_path.open("r") as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config and isinstance(loaded_config, dict):
                    config_dict.update(loaded_config)
        except (OSError, yaml.YAMLError):
            pass

    # Implementation is mocked for tests - the real impl is in the main cli.py
    if tree:
        # Generate simple directory tree for testing
        tree_content = generate_simple_tree(path)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(tree_content)
        else:
            console.print(tree_content)
    elif output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("Test documentation content")
    else:
        doc_path = _get_output_path(path, output, config_dict)
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text("Test documentation content")


def generate_simple_tree(path: Path) -> str:
    """Generate a simple text representation of the directory tree.

    Args:
        path: Root directory to generate tree for

    Returns:
        Text representation of the directory tree
    """
    tree_lines = []

    def walk_dir(current_path: Path, prefix: str = "", is_last: bool = False) -> None:
        """Recursively walk directory and build tree representation.

        FBT001/FBT002: is_last is a required boolean parameter for the tree structure
        """
        # Add the current directory/file
        if current_path.is_dir():
            # Skip if it's not a directory or is hidden
            if current_path.name.startswith(".") and current_path != path:
                return

            if current_path == path:
                tree_lines.append(f"{current_path.name}")
            else:
                tree_lines.append(f"{prefix}{'└── ' if is_last else '├── '}{current_path.name}")

            # Get all non-hidden items
            items = sorted(
                [p for p in current_path.iterdir() if not p.name.startswith(".")],
                key=lambda p: (p.is_file(), p.name),
            )

            # Process subdirectories and files
            for i, item in enumerate(items):
                is_last_item = i == len(items) - 1
                new_prefix = prefix + ("    " if is_last else "│   ")
                walk_dir(item, new_prefix, is_last_item)
        else:
            # It's a file
            tree_lines.append(f"{prefix}{'└── ' if is_last else '├── '}{current_path.name}")

    # Start the recursion
    walk_dir(path)

    return "\n".join(tree_lines)
