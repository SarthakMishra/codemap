"""Type definitions for CLI parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from typing_extensions import TypeAlias

# Type aliases for common CLI parameters
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
        help="Enable verbose logging",
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
