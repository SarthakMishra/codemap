"""Type definitions for CLI parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

# Type aliases for common CLI parameters
type PathArg = Annotated[
	Path,
	typer.Argument(
		exists=True,
		help="Path to the codebase to analyze",
		show_default=True,
	),
]

type OutputOpt = Annotated[
	Path | None,
	typer.Option(
		"--output",
		"-o",
		help="Output file path (overrides config)",
	),
]

type ConfigOpt = Annotated[
	Path | None,
	typer.Option(
		"--config",
		"-c",
		help="Path to config file",
	),
]

type MapTokensOpt = Annotated[
	int | None,
	typer.Option(
		"--map-tokens",
		help="Override token limit (set to 0 for unlimited)",
	),
]

type MaxContentLengthOpt = Annotated[
	int | None,
	typer.Option(
		"--max-content-length",
		help="Maximum content length for file display (set to 0 for unlimited)",
	),
]

type TreeFlag = Annotated[
	bool,
	typer.Option(
		"--tree",
		"-t",
		help="Generate only directory tree structure",
	),
]

type VerboseFlag = Annotated[
	bool,
	typer.Option(
		"--verbose",
		"-v",
		help="Enable verbose logging",
	),
]

type ForceFlag = Annotated[
	bool,
	typer.Option(
		"--force",
		"-f",
		help="Force overwrite existing files",
	),
]
