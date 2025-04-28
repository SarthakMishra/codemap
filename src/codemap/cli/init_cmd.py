"""Implementation of the init command with integrated configuration wizard."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated, Any

import questionary
import typer
from rich.panel import Panel

from codemap.analyzer.tree_parser import CodeParser
from codemap.utils.cli_utils import console, progress_indicator
from codemap.utils.config_manager import get_config_manager
from codemap.utils.directory_manager import get_directory_manager
from codemap.utils.log_setup import setup_logging

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
	full_setup: Annotated[
		bool,
		typer.Option(
			"--full-setup",
			help="Run full setup including global configurations",
		),
	] = False,
	wizard_only: Annotated[
		bool,
		typer.Option(
			"--wizard-only",
			help="Run only the configuration wizard without creating files",
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
	"""
	Initialize a new CodeMap project with optional wizard-guided configuration.

	On first run, this will set up global configurations (daemon, LLM API,
	storage). Subsequent runs will only configure repository-specific
	settings unless --full-setup is used.

	"""
	setup_logging(is_verbose=is_verbose)
	try:
		# Get directory and config managers
		dir_manager = get_directory_manager()
		config_manager = get_config_manager()

		# Ensure global directories exist
		dir_manager.ensure_directories()

		# Set up the repository
		repo_root = path.resolve()
		dir_manager.set_project_dir(repo_root)
		config_manager.set_project(repo_root)

		# Check for project files
		config_file = repo_root / ".codemap.yml"
		docs_dir = repo_root / "documentation"
		project_cache_dir = repo_root / ".codemap_cache"

		# Determine if this is first run or if full setup is requested
		global_config_path = dir_manager.config_dir / "settings.yml"
		is_first_run = not global_config_path.exists()
		needs_full_setup = is_first_run or full_setup

		# Build configuration
		config = config_manager.get_config(scope="default").copy()

		# Check if files/directories already exist
		existing_files = []
		if config_file.exists():
			existing_files.append(config_file)
		if docs_dir.exists() and (list(docs_dir.iterdir()) if docs_dir.is_dir() else True):
			existing_files.append(docs_dir)
		if project_cache_dir.exists():
			existing_files.append(project_cache_dir)

		if not force_flag and existing_files and not wizard_only:
			console.print("[yellow]CodeMap files already exist:")
			for f in existing_files:
				console.print(f"[yellow]  - {f}")
			console.print("[yellow]Use --force to overwrite.")
			raise typer.Exit(1)

		# Run appropriate wizard based on context
		if needs_full_setup:
			console.print(
				Panel.fit(
					"[bold]Welcome to CodeMap Setup![/bold]\n\n"
					"We'll guide you through configuring CodeMap for your environment and project.",
					title="CodeMap Setup",
					border_style="blue",
				)
			)

			# Global configurations
			if run_global_config_wizard(global_config_path, force_flag):
				console.print("[green]Global configuration completed.[/]")

			# Repository configuration is always needed
			repo_config = run_repo_config_wizard(repo_root)
			if repo_config:
				config.update(repo_config)
		else:
			# Only run repository-specific configuration
			console.print("[bold blue]Repository Configuration[/]")
			repo_config = run_repo_config_wizard(repo_root)
			if repo_config:
				config.update(repo_config)

		# Skip file creation if wizard-only mode
		if wizard_only:
			console.print("[green]Configuration wizard completed. No files were modified.[/]")
			return

		with progress_indicator("Initializing CodeMap", style="step", total=4) as advance:
			# Create .codemap.yml
			config_manager.initialize_project_config(config)
			advance(1)

			# Create documentation directory
			if docs_dir.exists() and force_flag:
				shutil.rmtree(docs_dir)
			docs_dir.mkdir(exist_ok=True, parents=True)
			advance(1)

			# Ensure project cache directory
			dir_manager.get_project_cache_dir(create=True)
			advance(1)

			# Initialize parser to check it's working
			CodeParser()
			advance(1)

		console.print("\n✨ CodeMap initialized successfully!")
		console.print(f"[green]Created config file: {config_file}")
		console.print(f"[green]Created documentation directory: {docs_dir}")
		console.print(f"[green]Set up cache directory: {project_cache_dir}")
		console.print("\nNext steps:")
		console.print("1. Review and customize .codemap.yml if needed")
		console.print("2. Run 'codemap generate' to create documentation")
		console.print("3. Run 'codemap daemon start' to start the background service")

	except (FileNotFoundError, PermissionError, OSError) as e:
		console.print(f"[red]File system error: {e!s}")
		raise typer.Exit(1) from e
	except ValueError as e:
		console.print(f"[red]Configuration error: {e!s}")
		raise typer.Exit(1) from e


def run_global_config_wizard(config_path: Path, force: bool = False) -> bool:
	"""
	Run wizard for global configurations.

	Args:
	        config_path: Path to the global config file
	        force: Force overwrite existing config

	Returns:
	        True if configuration was completed, False otherwise

	"""
	# Handle existing config
	if config_path.exists() and not force:
		should_reconfigure = questionary.confirm(
			"Global configuration already exists. Reconfigure?",
			default=False,
		).ask()

		if not should_reconfigure:
			return False

	# Get directory manager for correct paths
	dir_manager = get_directory_manager()

	# Begin configuration
	console.print("[bold blue]Global CodeMap Configuration[/]")

	# Get API configuration
	provider = questionary.select(
		"Select primary LLM provider:",
		choices=[
			"OpenAI",
			"Anthropic",
			"Groq",
			"Mistral",
			"Cohere",
			"Together AI",
			"OpenRouter",
			"Other/Custom",
		],
		default="OpenAI",
	).ask()

	# Update global config
	global_config: dict[str, Any] = {
		"llm": {
			"provider": provider.lower(),
		},
		"directories": {
			"data": str(dir_manager.user_data_dir),
			"config": str(dir_manager.user_config_dir),
			"cache": str(dir_manager.user_cache_dir),
			"logs": str(dir_manager.user_log_dir),
		},
	}

	# Save configuration
	config_manager = get_config_manager()
	config_manager.update_config("global", global_config)

	return True


def run_repo_config_wizard(repo_path: Path) -> dict[str, Any]:
	"""
	Run wizard for repository-specific configurations.

	Args:
	        repo_path: Path to the repository

	Returns:
	        Dictionary of repository configuration settings

	"""
	return configure_repository_settings(repo_path)


def configure_repository_settings(repo_path: Path | None = None) -> dict[str, Any]:
	"""
	Configure repository-related settings.

	Args:
	        repo_path: Path to the repository

	Returns:
	        Dictionary of repository configuration settings

	"""
	# Get repository path
	repo_path_obj = repo_path or Path.cwd()

	# Get repository name
	repo_name = questionary.text(
		"Enter a name for this repository:",
		default=repo_path_obj.name,
	).ask()

	# Get default file extensions to analyze
	extensions = questionary.text(
		"Enter file extensions to analyze (comma-separated, e.g., py,js,ts):",
		default="py,js,ts,jsx,tsx,cpp,c,h,hpp,java,go,rb",
	).ask()

	use_gitignore = questionary.confirm(
		"Respect .gitignore patterns when scanning files?",
		default=True,
	).ask()

	token_limit = questionary.text(
		"Maximum token limit for documentation (0 for unlimited):",
		default="10000",
		validate=lambda text: text.isdigit(),
	).ask()

	# Processor options
	enable_processor = questionary.confirm(
		"Enable code processing pipeline?",
		default=True,
	).ask()

	max_workers = (
		questionary.select(
			"Number of worker threads for processing:",
			choices=["1", "2", "4", "8", "16"],
			default="4",
		).ask()
		if enable_processor
		else "4"
	)

	# Build configuration
	config = {
		"repo_path": str(repo_path_obj),
		"repo_name": repo_name,
		"file_extensions": extensions.split(","),
		"use_gitignore": use_gitignore,
		"token_limit": int(token_limit),
		"output_dir": "documentation",
	}

	# Add processor configuration
	if enable_processor:
		config["processor"] = {
			"enabled": True,
			"max_workers": int(max_workers),
			"cache_dir": ".codemap_cache",
		}

	return config
