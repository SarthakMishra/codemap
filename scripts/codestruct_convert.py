#!/usr/bin/env python3
"""CodeStruct conversion tool for files and directories."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from codemap.processor.tree_sitter.code_struct import entities_to_codestruct, minify_codestruct
from codemap.processor.tree_sitter.schema.extractor import extract_entities
from codemap.processor.tree_sitter.schema.languages import LANGUAGES


def parse_arguments():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description="Convert source code to CodeStruct notation")
	parser.add_argument("path", help="Path to the source file or directory to convert")
	parser.add_argument(
		"--minify",
		"-m",
		action="store_true",
		help="Output minified CodeStruct notation",
	)
	parser.add_argument(
		"--output",
		"-o",
		help="Output file path or directory (default: print to stdout)",
	)
	parser.add_argument(
		"--recursive",
		"-r",
		action="store_true",
		help="Process directories recursively",
	)
	parser.add_argument("--include", "-i", nargs="+", help="File extensions to include (e.g. -i py js)")
	parser.add_argument("--debug", "-d", action="store_true", help="Include debug information in the output")

	return parser.parse_args()


def get_supported_extensions():
	"""Get all supported file extensions from LANGUAGES."""
	extensions = []
	for lang in LANGUAGES:
		extensions.extend(lang.get("extensions", []))
	return extensions


def process_file(file_path, output_path=None, minify=False, console=None):
	"""Process a single file and convert to CodeStruct notation.

	Args:
		file_path: Path to the source file
		output_path: Optional output file path
		minify: Whether to minify output
		console: Rich console for output

	Returns:
		0 on success, 1 on error
	"""
	if console is None:
		console = Console()

	try:
		# Read the source file
		with Path(file_path).open("rb") as f:
			source = f.read()

		# Extract entities
		entities = extract_entities(source, str(file_path))

		# Convert to CodeStruct notation
		codestruct = entities_to_codestruct(entities)

		# Minify if requested
		if minify:
			codestruct = minify_codestruct(codestruct)

		# Output
		if output_path:
			# Ensure directory exists
			output_path.parent.mkdir(parents=True, exist_ok=True)

			with output_path.open("w") as f:
				f.write(codestruct)
			console.print(f"[green]CodeStruct notation written to {output_path}[/green]")
		else:
			console.print(f"[blue]CodeStruct for {file_path}:[/blue]")
			console.print(codestruct)

		return 0

	except Exception as e:
		console.print(f"[red]Error converting {file_path} to CodeStruct: {e}[/red]")
		return 1


def process_directory(dir_path, output_dir=None, minify=False, recursive=False, include=None, console=None):
	"""Process all supported files in a directory.

	Args:
		dir_path: Path to the directory
		output_dir: Optional output directory
		minify: Whether to minify output
		recursive: Whether to process subdirectories
		include: List of file extensions to include
		console: Rich console for output

	Returns:
		0 on success, non-zero on partial or complete failure
	"""
	if console is None:
		console = Console()

	# Get all supported extensions if not specified
	if not include:
		include = get_supported_extensions()
	else:
		# Ensure extensions start with a dot
		include = [ext if ext.startswith(".") else f".{ext}" for ext in include]

	pattern = "**/*" if recursive else "*"
	failed = 0
	processed = 0

	# Find all matching files
	for ext in include:
		for file_path in dir_path.glob(f"{pattern}{ext}"):
			if not file_path.is_file():
				continue

			processed += 1

			# Determine output path if necessary
			output_path = None
			if output_dir:
				# Maintain directory structure relative to input directory
				rel_path = file_path.relative_to(dir_path)
				output_path = output_dir / f"{rel_path}.cs"

			# Process the file
			result = process_file(file_path, output_path, minify, console)
			if result != 0:
				failed += 1

	# Report summary
	if processed > 0:
		status = "green" if failed == 0 else "yellow" if failed < processed else "red"
		console.print(f"[{status}]Processed {processed} files with {failed} failures[/{status}]")
	else:
		console.print(f"[yellow]No matching files found in {dir_path}[/yellow]")

	return 1 if processed > 0 and failed == processed else 0


def main():
	"""Run the CodeStruct converter."""
	args = parse_arguments()
	path = Path(args.path)
	console = Console()

	if not path.exists():
		console.print(f"[red]Path not found: {path}[/red]")
		return 1

	# Determine include extensions
	include = args.include if args.include else None

	# Process path based on type
	if path.is_file():
		# Single file mode
		output_path = Path(args.output) if args.output else None
		return process_file(path, output_path, args.minify, console)

	if path.is_dir():
		# Directory mode
		output_dir = Path(args.output) if args.output else None
		return process_directory(path, output_dir, args.minify, args.recursive, include, console)

	# Neither file nor directory
	console.print(f"[red]Path is neither a file nor a directory: {path}[/red]")
	return 1


if __name__ == "__main__":
	sys.exit(main())
