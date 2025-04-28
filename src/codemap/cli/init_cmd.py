"""Implementation of the init command with integrated configuration wizard."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Annotated, Any

import questionary
import typer
import yaml
from rich.panel import Panel

from codemap.analyzer.tree_parser import CodeParser
from codemap.config import DEFAULT_CONFIG
from codemap.utils.cli_utils import console, progress_indicator, setup_logging

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
		repo_root = path.resolve()
		config_file = repo_root / ".codemap.yml"
		docs_dir = repo_root / str(DEFAULT_CONFIG["output_dir"])
		global_config_file = Path.home() / ".codemap" / "config.yml"

		# Determine if this is first run or if full setup is requested
		is_first_run = not global_config_file.exists()
		needs_full_setup = is_first_run or full_setup

		# Build configuration
		config = DEFAULT_CONFIG.copy()

		# Check if files/directories already exist
		existing_files = []
		if config_file.exists():
			existing_files.append(config_file)
		if docs_dir.exists():
			existing_files.append(docs_dir)

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
			if run_global_config_wizard(global_config_file, force_flag):
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

		with progress_indicator("Initializing CodeMap", style="step", total=3) as advance:
			# Create .codemap.yml
			config_file.write_text(yaml.dump(config, sort_keys=False))
			advance(1)

			# Create documentation directory
			if docs_dir.exists() and force_flag:
				shutil.rmtree(docs_dir)
			docs_dir.mkdir(exist_ok=True, parents=True)
			advance(1)

			# Initialize parser to check it's working
			CodeParser()
			advance(1)

		console.print("\n✨ CodeMap initialized successfully!")
		console.print(f"[green]Created config file: {config_file}")
		console.print(f"[green]Created documentation directory: {docs_dir}")
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


def run_global_config_wizard(global_config_path: Path, force: bool = False) -> bool:
	"""
	Run wizard for global configurations (daemon, LLM, storage).

	Args:
	        global_config_path: Path to the global configuration file
	        force: Whether to force overwrite existing configuration

	Returns:
	        True if configuration was successful, False otherwise

	"""
	if global_config_path.exists() and not force:
		use_existing = questionary.confirm(
			f"Global configuration already exists at {global_config_path}. Reconfigure?",
			default=False,
		).ask()

		if not use_existing:
			return False

	# Create empty or starter config
	global_config = {}

	# Configure daemon
	console.print("\n[bold blue]Daemon Configuration[/]")
	daemon_config = configure_daemon_settings({})
	if daemon_config:
		global_config["server"] = daemon_config

	# Configure LLM API
	console.print("\n[bold blue]LLM API Configuration[/]")
	llm_config = configure_llm_settings()
	if llm_config:
		global_config["llm"] = llm_config

	# Configure storage
	console.print("\n[bold blue]Storage Configuration[/]")
	storage_config = configure_storage_settings({})
	if storage_config:
		global_config["storage"] = storage_config

	# Save global configuration
	try:
		global_config_path.parent.mkdir(parents=True, exist_ok=True)
		with global_config_path.open("w", encoding="utf-8") as f:
			yaml.dump(global_config, f, default_flow_style=False)
		console.print(f"[green]Global configuration saved to {global_config_path}[/]")
		return True
	except Exception as e:
		console.print(f"[red]Error saving global configuration: {e!s}[/]")
		logger.exception("Error saving global configuration")
		return False


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

	return {
		"repo_path": str(repo_path_obj),
		"repo_name": repo_name,
		"file_extensions": extensions.split(","),
		"use_gitignore": use_gitignore,
	}


def configure_daemon_settings(current_config: dict[str, Any]) -> dict[str, Any]:
	"""
	Configure daemon-related settings.

	Args:
	        current_config: Current daemon configuration

	Returns:
	        Dictionary of updated daemon configuration settings

	"""
	# Start with current config
	daemon_config = current_config.copy()

	# Get host and port
	host = questionary.text(
		"Enter daemon host address:",
		default=daemon_config.get("host", "127.0.0.1"),
	).ask()

	port = questionary.text(
		"Enter daemon port:",
		default=str(daemon_config.get("port", 8765)),
	).ask()

	try:
		port = int(port)
	except ValueError:
		console.print("[yellow]Invalid port number, using default (8765)[/]")
		port = 8765

	# Get log level
	log_level_choices = ["debug", "info", "warning", "error"]
	log_level = questionary.select(
		"Select log level:",
		choices=log_level_choices,
		default=daemon_config.get("log_level", "info"),
	).ask()

	# Get daemon settings
	auto_start = questionary.confirm(
		"Start daemon automatically when needed?",
		default=daemon_config.get("auto_start", False),
	).ask()

	# Update config
	daemon_config.update(
		{
			"host": host,
			"port": port,
			"log_level": log_level,
			"auto_start": auto_start,
		}
	)

	return daemon_config


def configure_llm_settings() -> dict[str, Any]:
	"""
	Configure LLM API settings.

	Returns:
	        Dictionary of LLM configuration settings

	"""
	# Get provider
	provider_choices = [
		{
			"name": "OpenAI (ChatGPT, GPT-4)",
			"value": "openai",
		},
		{
			"name": "Anthropic (Claude)",
			"value": "anthropic",
		},
		{
			"name": "Azure OpenAI",
			"value": "azure",
		},
		{
			"name": "Mistral AI",
			"value": "mistral",
		},
		{
			"name": "Cohere",
			"value": "cohere",
		},
		{
			"name": "Ollama (local)",
			"value": "ollama",
		},
	]

	provider = questionary.select(
		"Select your preferred LLM provider:",
		choices=provider_choices,
	).ask()

	# Get model
	default_model = get_default_model(provider)
	model = questionary.text(
		"Enter the model name:",
		default=default_model,
	).ask()

	# Get API key
	api_key_env_var = get_api_key_env_var(provider)
	current_key = os.environ.get(api_key_env_var, "")

	if current_key:
		use_existing_key = questionary.confirm(
			f"Use existing {api_key_env_var} from environment?",
			default=True,
		).ask()

		if not use_existing_key:
			api_key = questionary.password(
				f"Enter your {provider} API key (will not be displayed):",
			).ask()
		else:
			api_key = current_key
	else:
		api_key = questionary.password(
			f"Enter your {provider} API key (will not be displayed):",
		).ask()

	# Get API base URL for custom endpoints
	custom_api_base = questionary.confirm(
		"Do you use a custom API endpoint URL?",
		default=False,
	).ask()

	api_base = None
	if custom_api_base:
		api_base = questionary.text(
			"Enter the API base URL:",
			default=get_default_api_base(provider),
		).ask()

	return {
		"provider": provider,
		"model": model,
		"api_key": api_key,
		"api_base": api_base,
	}


def configure_storage_settings(current_config: dict[str, Any]) -> dict[str, Any]:
	"""
	Configure storage-related settings.

	Args:
	        current_config: Current storage configuration

	Returns:
	        Dictionary of updated storage configuration settings

	"""
	# Start with current config
	storage_config = current_config.copy()

	# Get data directory
	data_dir = questionary.text(
		"Enter directory to store CodeMap data:",
		default=str(Path(storage_config.get("data_dir", "~/.codemap/data")).expanduser()),
	).ask()

	# Get cache settings
	use_cache = questionary.confirm(
		"Enable caching for better performance?",
		default=storage_config.get("use_cache", True),
	).ask()

	cache_size_mb = storage_config.get("cache_size_mb", 512)
	if use_cache:
		cache_size = questionary.text(
			"Enter cache size (MB):",
			default=str(cache_size_mb),
		).ask()

		try:
			cache_size_mb = int(cache_size)
		except ValueError:
			console.print("[yellow]Invalid cache size, using default (512 MB)[/]")
			cache_size_mb = 512

	# Update config
	storage_config.update(
		{
			"data_dir": data_dir,
			"use_cache": use_cache,
			"cache_size_mb": cache_size_mb,
		}
	)

	return storage_config


def get_default_model(provider: str) -> str:
	"""Get the default model name for a given provider."""
	defaults = {
		"openai": "gpt-4o-mini",
		"anthropic": "claude-3-haiku-20240307",
		"azure": "gpt-4",
		"mistral": "mistral-large-latest",
		"cohere": "command-r-plus",
		"ollama": "llama3",
	}
	return defaults.get(provider, "gpt-4o-mini")


def get_api_key_env_var(provider: str) -> str:
	"""Get the environment variable name for a provider's API key."""
	env_vars = {
		"openai": "OPENAI_API_KEY",
		"anthropic": "ANTHROPIC_API_KEY",
		"azure": "AZURE_API_KEY",
		"mistral": "MISTRAL_API_KEY",
		"cohere": "COHERE_API_KEY",
		"ollama": "OLLAMA_API_KEY",
	}
	return env_vars.get(provider, "OPENAI_API_KEY")


def get_default_api_base(provider: str) -> str:
	"""Get the default API base URL for a given provider."""
	defaults = {
		"openai": "https://api.openai.com/v1",
		"anthropic": "https://api.anthropic.com/v1",
		"azure": "https://<your-resource-name>.openai.azure.com",
		"mistral": "https://api.mistral.ai/v1",
		"cohere": "https://api.cohere.ai/v1",
		"ollama": "http://localhost:11434",
	}
	return defaults.get(provider, "")
