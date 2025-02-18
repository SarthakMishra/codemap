"""Command-line interface for the codemap tool."""

import shutil
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.progress import Progress

from .analyzer.dependency_graph import DependencyGraph
from .analyzer.tree_parser import CodeParser
from .config import DEFAULT_CONFIG
from .generators.markdown_generator import MarkdownGenerator
from .utils.config_loader import ConfigLoader

console = Console()
app = typer.Typer(
    help="CodeMap - Generate optimized markdown documentation from your codebase.",
)

PATH_ARG = typer.Argument(
    ".",
    exists=True,
    help="Path to the codebase to analyze",
    show_default=True,
)
OUTPUT_OPT = typer.Option(
    "documentation.md",
    "--output",
    "-o",
    help="Output file path",
    show_default=True,
)
CONFIG_OPT = typer.Option(
    None,
    "--config",
    "-c",
    help="Path to config file",
)
MAP_TOKENS_OPT = typer.Option(
    None,
    "--map-tokens",
    help="Override token limit",
)


@app.command()
def init(
    force_flag: int = typer.Option(
        0,
        "--force",
        "-f",
        count=True,
        help="Force overwrite existing files",
    ),
    path: Path = PATH_ARG,
) -> None:
    """Initialize a new CodeMap project in the specified directory."""
    try:
        repo_root = path.resolve()
        config_file = repo_root / ".codemap.yml"
        cache_dir = repo_root / ".codemap_cache"

        # Check if files/directories already exist
        if not force_flag and (config_file.exists() or cache_dir.exists()):
            console.print("[yellow]CodeMap files already exist. Use --force to overwrite.")
            raise typer.Exit(1)

        with Progress() as progress:
            task = progress.add_task("Initializing CodeMap...", total=3)

            # Create .codemap.yml
            config_file.write_text(yaml.dump(DEFAULT_CONFIG, sort_keys=False))
            progress.update(task, advance=1)

            # Create and setup cache directory
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(exist_ok=True)
            (cache_dir / ".gitignore").write_text("*\n!.gitignore\n")
            progress.update(task, advance=1)

            # Initialize parser and cache basic repository info
            parser = CodeParser()
            file_count = sum(1 for _ in repo_root.rglob("*") if parser.should_parse(_))
            cache_info = {
                "version": "0.1.0",
                "last_update": None,
                "file_count": file_count,
                "languages": list(parser.parsers.keys()),
            }
            (cache_dir / "info.json").write_text(yaml.dump(cache_info))
            progress.update(task, advance=1)

        console.print("\n✨ CodeMap initialized successfully!")
        console.print(f"[green]Created config file: {config_file}")
        console.print(f"[green]Created cache directory: {cache_dir}")
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
def generate(
    path: Path = PATH_ARG,
    output: Path = OUTPUT_OPT,
    config: Path = CONFIG_OPT,
    map_tokens: int = MAP_TOKENS_OPT,
) -> None:
    """Generate documentation for the specified path."""
    try:
        repo_root = path.resolve()
        config_loader = ConfigLoader(str(config) if config else None)
        config_data = config_loader.config

        if map_tokens:
            config_data["token_limit"] = map_tokens

        with Progress() as progress:
            # Parse files
            task1 = progress.add_task("Parsing files...", total=100)
            parser = CodeParser()
            parsed_files = {}

            for file_path in repo_root.rglob("*"):
                if parser.should_parse(file_path):
                    parsed_files[file_path] = parser.parse_file(file_path)
            progress.update(task1, completed=100)

            # Build dependency graph
            task2 = progress.add_task("Analyzing dependencies...", total=100)
            graph = DependencyGraph(repo_root)
            graph.build_graph(parsed_files)
            important_files = graph.get_important_files(config_data["token_limit"])
            progress.update(task2, completed=100)

            # Generate documentation
            task3 = progress.add_task("Generating documentation...", total=100)
            generator = MarkdownGenerator(repo_root, config_data)
            documentation = generator.generate_documentation(
                {k: parsed_files[k] for k in important_files},
            )
            progress.update(task3, completed=100)

        # Write output
        output_path = Path(output)
        output_path.write_text(documentation)
        console.print(f"\n✨ Documentation generated successfully: {output_path}")

    except (FileNotFoundError, PermissionError, OSError) as e:
        console.print(f"[red]File system error: {e!s}")
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print(f"[red]Configuration error: {e!s}")
        raise typer.Exit(1) from e


def main() -> None:
    """Entry point for the CodeMap CLI application."""
    app()


if __name__ == "__main__":
    main()
