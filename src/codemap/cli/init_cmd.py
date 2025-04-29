"""Implementation of the init command with integrated configuration wizard."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Annotated, Any

import questionary
import typer
from rich.panel import Panel

from codemap.analyzer.tree_parser import CodeParser
from codemap.daemon.command import start_daemon
from codemap.processor import ProcessingPipeline
from codemap.processor.embedding.models import EmbeddingConfig
from codemap.processor.storage.base import StorageConfig
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
	scan: Annotated[
		bool,
		typer.Option(
			"--scan",
			help="Perform initial repository scan after initialization",
		),
	] = True,
	background: Annotated[
		bool,
		typer.Option(
			"--background",
			help="Start daemon service after initialization",
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

		with progress_indicator("Initializing CodeMap", style="step", total=5) as advance:
			# Create .codemap.yml
			config_manager.initialize_project_config(config)
			advance(1)

			# Update .gitignore
			update_gitignore(repo_root)
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

		# Initialize processor if enabled in config
		if config.get("processor", {}).get("enabled", True):
			try:
				console.print("\n[bold]Initializing processor pipeline...[/bold]")
				pipeline = initialize_processor(repo_root, config)
				console.print("[green]Processor pipeline initialized successfully[/green]")

				# Run initial scan if requested
				if scan:
					console.print("\n[bold]Running initial repository scan...[/bold]")
					run_initial_scan(pipeline, repo_root)
					pipeline.stop()  # Clean shutdown after scanning

				# Start daemon if requested
				if background:
					console.print("\n[bold]Starting CodeMap daemon service...[/bold]")
					try:
						# Use the existing daemon start command
						result = start_daemon(config_path=None, foreground=False, timeout=30)
						if result == 0:
							console.print("[green]Started CodeMap daemon service[/green]")
						else:
							console.print(
								"[yellow]Failed to start daemon service. "
								"Try running 'codemap daemon start' manually.[/yellow]"
							)
					except Exception as e:
						console.print(f"[yellow]Error starting daemon: {e}[/yellow]")
						logger.exception("Failed to start daemon")
			except Exception as e:
				console.print(f"[yellow]Warning: Could not initialize processor: {e}[/yellow]")
				logger.exception("Failed to initialize processor")

		console.print("\nNext steps:")
		console.print("1. Review and customize .codemap.yml for your project")
		console.print("2. Run 'codemap generate' to create documentation")
		console.print("3. Run 'codemap daemon start' to start the background service")
		console.print("4. Run 'codemap commit' to use AI-powered commit messages")

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

	# Get API key
	api_key = questionary.text(
		f"Enter API key for {provider}:",
		default="",
		hide_input=True,
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

	# Set up API keys
	setup_api_keys(dir_manager.user_config_dir, provider, api_key)

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


def get_models_for_provider(provider: str) -> list[str]:
	"""
	Get a list of models available for the given provider.

	Args:
	        provider: Provider name

	Returns:
	        List of model names

	"""
	# Define models for each provider
	models = {
		"openai": [
			"gpt-4o",
			"gpt-4o-mini",
			"gpt-4-turbo",
			"gpt-4",
			"gpt-3.5-turbo",
		],
		"anthropic": [
			"claude-3-opus",
			"claude-3-sonnet",
			"claude-3-haiku",
			"claude-2.1",
			"claude-2.0",
		],
		"groq": [
			"llama3-70b-8192",
			"llama3-8b-8192",
			"mixtral-8x7b-32768",
			"gemma-7b-it",
		],
		"mistral": [
			"mistral-large-latest",
			"mistral-medium-latest",
			"mistral-small-latest",
		],
		"cohere": [
			"command-r-plus",
			"command-r",
			"command",
		],
		"together": [
			"together-ai/llama-3-70b-instruct",
			"togethercomputer/RedPajama-INCITE-7B-Instruct",
		],
		"openrouter": [
			"openai/gpt-4o",
			"anthropic/claude-3-opus",
			"meta-llama/llama-3-70b-instruct",
			"mistralai/mistral-large",
		],
	}

	return models.get(provider.lower(), ["custom-model"])


def format_model_name(provider: str, model: str) -> str:
	"""
	Format a model name according to provider conventions.

	Args:
	        provider: Provider name
	        model: Model name

	Returns:
	        Formatted model name

	"""
	# Add provider prefix if not already present
	if "/" not in model and provider.lower() not in ["custom"]:
		return f"{provider.lower()}/{model}"

	return model


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

	# LLM configuration section
	console.print("\n[bold blue]LLM Configuration[/bold blue]")

	llm_provider = questionary.select(
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

	# Get appropriate model based on provider
	model_choices = get_models_for_provider(llm_provider)
	llm_model = questionary.select(
		f"Select {llm_provider} model:",
		choices=model_choices,
		default=model_choices[0] if model_choices else "",
	).ask()

	# Ask about API key management
	api_key_source = questionary.select(
		"How would you like to provide your API key?",
		choices=[
			"Use environment variable (recommended)",
			"Enter now (will be stored in .env.local)",
			"I'll configure it later",
		],
		default="Use environment variable (recommended)",
	).ask()

	api_key = None
	if api_key_source == "Enter now (will be stored in .env.local)":
		api_key = questionary.text(
			f"Enter your {llm_provider} API key:",
			default="",
			hide_input=True,
		).ask()

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
			"embedding_model": "Qodo/Qodo-Embed-1-1.5B",
			"batch_size": 32,
		}

	# Add LLM configuration
	config["llm"] = {
		"provider": llm_provider.lower(),
		"model": format_model_name(llm_provider, llm_model),
	}

	# Add commit configuration
	config["commit"] = {
		"llm": {
			"model": format_model_name(llm_provider, llm_model),
		},
		"strategy": "semantic",
	}

	# Set up API key if provided
	if api_key:
		setup_api_keys(repo_path_obj, llm_provider, api_key)

	return config


def initialize_processor(repo_path: Path, config: dict) -> ProcessingPipeline:
	"""
	Initialize the processor pipeline.

	Args:
	        repo_path: Repository path
	        config: Configuration dictionary

	Returns:
	        Configured processing pipeline

	"""
	# Extract processor configuration
	processor_config = config.get("processor", {})

	# Use directory manager to get properly set up cache directories
	dir_manager = get_directory_manager()
	project_cache_dir = dir_manager.get_project_cache_dir(create=True)

	if not project_cache_dir:
		# If project_cache_dir is None for some reason, fall back to default path
		project_cache_dir = repo_path / ".codemap_cache"
		project_cache_dir.mkdir(exist_ok=True, parents=True)
		logger.warning("Failed to get project cache directory from directory manager, using default path")

	# Configure storage using directory manager paths
	storage_dir = project_cache_dir / "storage"
	storage_dir.mkdir(exist_ok=True, parents=True)

	storage_config = StorageConfig(
		uri=str(storage_dir / "vector.lance"), create_if_missing=True, cache_dir=storage_dir / "cache"
	)

	# Configure embedding with proper cache directory
	embedding_cache_dir = project_cache_dir / "embeddings"
	embedding_cache_dir.mkdir(exist_ok=True, parents=True)

	embedding_config = EmbeddingConfig(
		model=processor_config.get("embedding_model", "Qodo/Qodo-Embed-1-1.5B"),
		dimensions=processor_config.get("embedding_dimensions", 384),
		batch_size=processor_config.get("batch_size", 32),
	)

	# Get ignored patterns
	ignored_patterns = set(processor_config.get("ignored_patterns", []))
	# Always include common patterns
	ignored_patterns.update(
		[
			"**/.git/**",
			"**/__pycache__/**",
			"**/.venv/**",
			"**/node_modules/**",
			"**/.codemap_cache/**",
			"**/*.pyc",
			"**/dist/**",
			"**/build/**",
		]
	)

	# Initialize pipeline
	pipeline = ProcessingPipeline(
		repo_path=repo_path,
		storage_config=storage_config,
		embedding_config=embedding_config,
		ignored_patterns=ignored_patterns,
		max_workers=processor_config.get("max_workers", 4),
		enable_lsp=processor_config.get("enable_lsp", True),
	)

	logger.info("Processor pipeline initialized for repository: %s", repo_path)
	return pipeline


def run_initial_scan(pipeline: ProcessingPipeline, repo_path: Path) -> None:
	"""
	Run an initial scan of the repository.

	Args:
	        pipeline: Processing pipeline
	        repo_path: Repository path

	"""
	# Create a list of all files to process
	all_files = []
	logger.info("Scanning repository at %s for files to process", repo_path)

	for root, _, files in os.walk(repo_path):
		root_path = Path(root)

		# Skip ignored directories
		if any(part.startswith(".") for part in root_path.parts):
			continue

		for file in files:
			# Skip hidden files
			if file.startswith("."):
				continue

			file_path = root_path / file

			# Check if file should be processed (not in ignored patterns)
			should_process = True
			for pattern in pipeline.ignored_patterns:
				if file_path.match(pattern):
					should_process = False
					break

			if should_process:
				all_files.append(file_path)

	# Process files in batches with progress display
	total_files = len(all_files)
	if total_files == 0:
		console.print("[yellow]No files found to process in the repository.[/]")
		return

	console.print(f"Found {total_files} files to process.")

	# Use the progress_indicator utility from cli_utils
	with progress_indicator("Processing repository files", style="progress", total=total_files) as advance:
		# Process in batches for better performance
		batch_size = 100
		for i in range(0, total_files, batch_size):
			batch = all_files[i : min(i + batch_size, total_files)]

			# Process the batch
			pipeline.batch_process(batch)

			# Update progress
			advance(len(batch))

	logger.info("Initial repository scan completed. Processed %d files.", total_files)
	console.print(f"[green]Repository scan complete. Processed {total_files} files.[/]")


def update_gitignore(repo_path: Path) -> None:
	"""
	Update the project's .gitignore file to include CodeMap entries.

	Args:
	        repo_path: Repository path

	"""
	gitignore_path = repo_path / ".gitignore"

	# Define CodeMap entries to add
	codemap_entries = [
		"\n# CodeMap",
		".codemap_cache/",
		".env.local",
		"# CodeMap documentation output",
		"documentation/",
	]

	# Check if .gitignore exists
	if gitignore_path.exists():
		# Read existing content
		with Path.open(gitignore_path) as f:
			content = f.read()

		# Check if CodeMap entries already exist
		if ".codemap_cache/" in content:
			logger.debug("CodeMap entries already in .gitignore")
			return

		# Add entries
		with Path.open(gitignore_path, "a") as f:
			f.write("\n".join(codemap_entries) + "\n")
			logger.info("Added CodeMap entries to existing .gitignore file")
	else:
		# Create new .gitignore
		with Path.open(gitignore_path, "w") as f:
			f.write("\n".join(codemap_entries) + "\n")
			logger.info("Created new .gitignore file with CodeMap entries")

	console.print("[green]Updated .gitignore file with CodeMap entries[/green]")


def setup_api_keys(repo_path: Path, provider: str, api_key: str | None) -> None:
	"""
	Set up API keys for the selected provider.

	Args:
	        repo_path: Repository path
	        provider: Provider name
	        api_key: API key (if provided)

	"""
	if not api_key:
		return

	# Create .env.local file
	env_file = repo_path / ".env.local"

	# Map provider to environment variable
	env_var_map = {
		"openai": "OPENAI_API_KEY",
		"anthropic": "ANTHROPIC_API_KEY",
		"groq": "GROQ_API_KEY",
		"mistral": "MISTRAL_API_KEY",
		"cohere": "COHERE_API_KEY",
		"together": "TOGETHER_API_KEY",
		"openrouter": "OPENROUTER_API_KEY",
	}

	env_var = env_var_map.get(provider.lower(), f"{provider.upper()}_API_KEY")

	# Read existing content
	existing_content = {}
	if env_file.exists():
		with Path.open(env_file) as f:
			for line in f:
				if "=" in line and not line.strip().startswith("#"):
					key, value = line.strip().split("=", 1)
					existing_content[key] = value

	# Update with new API key
	existing_content[env_var] = api_key

	# Write updated content
	with Path.open(env_file, "w") as f:
		f.write("# CodeMap API Keys - KEEP THIS FILE SECRET\n")
		f.write("# Do not commit this file to version control\n\n")
		for key, value in existing_content.items():
			f.write(f"{key}={value}\n")

	# Set permissions to restrict access (POSIX systems only)
	try:
		env_file.chmod(0o600)  # Owner read/write only
	except (OSError, PermissionError) as e:
		logger.warning("Could not set restrictive permissions on .env.local: %s", e)

	console.print(f"[green]Added {env_var} to .env.local[/green]")
	console.print("[yellow]Important: Keep .env.local secure and don't commit it to version control[/yellow]")
