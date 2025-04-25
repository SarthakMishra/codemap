"""Implementation of the init command."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

import typer
import yaml

from codemap.analyzer.tree_parser import CodeParser
from codemap.config import DEFAULT_CONFIG
from codemap.utils.cli_utils import console, setup_logging

logger = logging.getLogger(__name__)


def init_command(
	path: Annotated[
		Path,
		typer.Argument(
			exists=True,
			help="Path to the codebase to analyze",
			show_default=True,
		),
	] = Path(),
	force_flag: Annotated[
		bool,
		typer.Option(
			"--force",
			"-f",
			help="Force overwrite existing files",
		),
	] = False,
	is_verbose: Annotated[
		bool,
		typer.Option(
			"--verbose",
			"-v",
			help="Enable verbose logging",
		),
	] = False,
) -> None:
	"""Initialize a new CodeMap project in the specified directory."""
	setup_logging(is_verbose=is_verbose)
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

		with typer.progressbar(
			range(3),
			label="Initializing CodeMap...",
			show_pos=True,
		) as progress:
			# Create .codemap.yml
			config_file.write_text(yaml.dump(DEFAULT_CONFIG, sort_keys=False))
			next(progress)

			# Create documentation directory
			if docs_dir.exists() and force_flag:
				shutil.rmtree(docs_dir)
			docs_dir.mkdir(exist_ok=True, parents=True)
			next(progress)

			# Initialize parser to check it's working
			CodeParser()  # Just initialize without assigning to a variable
			next(progress)

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
