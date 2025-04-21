"""Main CLI module for CodeMap (legacy entry point)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer

# Create a new app instance for backward compatibility
app = typer.Typer(
    help="CodeMap - Generate optimized markdown documentation from your codebase.",
)

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

# Add PR command group
from .pr import app as pr_app  # noqa: E402

app.add_typer(pr_app, name="pr")

# This module is kept for backward compatibility
# All functionality has been moved to codemap.cli_app
