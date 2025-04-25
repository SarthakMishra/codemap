# Init Command Enhancement Plan

## Current Implementation Analysis

The current init command implementation provides minimal functionality:
- Creates a basic `.codemap.yml` file with default configuration
- Creates a documentation directory
- Initializes a `CodeParser` instance to validate functionality

The command is non-interactive and lacks integration with advanced CodeMap features, particularly the processor module.

## Enhancement Goals

1. **Processor Pipeline Integration**
   - Initialize and configure the processor module
   - Run initial repository scan
   - Set up database and storage infrastructure
   - Implement background processing capability

2. **Interactive Configuration**
   - Implement an interactive wizard for setup
   - Provide smart defaults based on repository analysis
   - Allow customization of various settings

3. **Caching and Storage Management**
   - Create `.codemap_cache` directory structure
   - Configure LanceDB storage
   - Set up embedding cache

4. **AI Provider Configuration**
   - Interactive selection of LLM providers
   - Secure API key management
   - Robust provider detection

## Enhancement Plan

### 1. Processor Pipeline Integration

The enhanced init command will properly initialize the processor module:

```python
def initialize_processor(repo_path: Path, config: dict) -> ProcessingPipeline:
    """Initialize the processor pipeline.
    
    Args:
        repo_path: Repository path
        config: Configuration dictionary
        
    Returns:
        Configured processing pipeline
    """
    # Extract processor configuration
    processor_config = config.get("processor", {})
    
    # Configure storage
    cache_dir = repo_path / ".codemap_cache"
    storage_dir = cache_dir / "storage"
    storage_config = StorageConfig(
        uri=str(storage_dir),
        table_name=processor_config.get("table_name", "codemap"),
        create_if_missing=True,
    )
    
    # Configure embedding
    embedding_config = EmbeddingConfig(
        model=processor_config.get("embedding_model", "Qodo/Qodo-Embed-1-1.5B"),
        dimensions=processor_config.get("embedding_dimensions", 384),
        batch_size=processor_config.get("batch_size", 32),
    )
    
    # Get ignored patterns
    ignored_patterns = set(processor_config.get("ignored_patterns", []))
    # Always include common patterns
    ignored_patterns.update([
        "**/.git/**",
        "**/__pycache__/**",
        "**/.venv/**",
        "**/node_modules/**",
        "**/.codemap_cache/**",
    ])
    
    # Initialize pipeline
    pipeline = ProcessingPipeline(
        repo_path=repo_path,
        storage_config=storage_config,
        embedding_config=embedding_config,
        ignored_patterns=ignored_patterns,
        max_workers=processor_config.get("max_workers", 4),
        enable_lsp=processor_config.get("enable_lsp", True),
    )
    
    return pipeline
```

**Initial Repository Scan:**

```python
def run_initial_scan(pipeline: ProcessingPipeline, repo_path: Path) -> None:
    """Run an initial scan of the repository.
    
    Args:
        pipeline: Processing pipeline
        repo_path: Repository path
    """
    # Get all files to process
    all_files = []
    for root, _, files in os.walk(repo_path):
        root_path = Path(root)
        # Skip ignored directories
        if any(part.startswith(".") for part in root_path.parts):
            continue
            
        for file in files:
            if file.startswith("."):
                continue
                
            file_path = root_path / file
            if should_process_file(file_path, pipeline.ignored_patterns):
                all_files.append(file_path)
    
    # Show progress bar for initial scan
    with typer.progressbar(
        all_files,
        label="Scanning repository files...",
        show_pos=True,
    ) as progress:
        # Process files in batches
        batch_size = 100
        for i in range(0, len(all_files), batch_size):
            batch = all_files[i:i+batch_size]
            pipeline.batch_process(batch)
            # Update progress
            for _ in batch:
                next(progress)
```

**Background Processing:**

```python
def start_background_watcher(pipeline: ProcessingPipeline) -> None:
    """Start the pipeline in background watch mode.
    
    Args:
        pipeline: Processing pipeline
    """
    # Start the pipeline
    pipeline.start()
    
    # Create a PID file to track the background process
    pid_file = pipeline.repo_path / ".codemap_cache" / "watcher.pid"
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
        
    # Log startup
    logger.info("Started background watcher with PID %d", os.getpid())
    console.print("[green]Started background file watcher[/green]")
    
    # Set up signal handlers for graceful shutdown
    def handle_exit(signum, frame):
        logger.info("Received signal %d, shutting down", signum)
        pipeline.stop()
        if pid_file.exists():
            pid_file.unlink()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # Keep the process running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        handle_exit(signal.SIGINT, None)
```

### 2. Interactive Configuration Wizard

Create an interactive setup wizard:

```python
def interactive_setup(repo_path: Path) -> dict:
    """Run interactive setup to generate configuration.
    
    Args:
        repo_path: Repository path
        
    Returns:
        Generated configuration dictionary
    """
    console.print("[bold]CodeMap Interactive Setup[/bold]")
    console.print("Let's configure CodeMap for your repository.\n")
    
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # 1. Basic configuration
    console.print("[bold]Basic Configuration[/bold]")
    use_gitignore = questionary.confirm(
        "Respect .gitignore patterns?",
        default=True
    ).ask()
    
    token_limit = questionary.text(
        "Maximum token limit for documentation (0 for unlimited):",
        default="10000",
        validate=lambda text: text.isdigit(),
    ).ask()
    
    # 2. Processor configuration
    console.print("\n[bold]Processor Configuration[/bold]")
    enable_processor = questionary.confirm(
        "Enable code processing pipeline?",
        default=True
    ).ask()
    
    enable_lsp = questionary.confirm(
        "Enable Language Server Protocol (LSP) for enhanced code analysis?",
        default=True
    ).ask()
    
    max_workers = questionary.select(
        "Number of worker threads for processing:",
        choices=["1", "2", "4", "8", "16"],
        default="4"
    ).ask()
    
    # 3. LLM Provider configuration
    console.print("\n[bold]AI Provider Configuration[/bold]")
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
        default="OpenAI"
    ).ask()
    
    # Get appropriate model based on provider
    model_choices = get_models_for_provider(llm_provider)
    llm_model = questionary.select(
        f"Select {llm_provider} model:",
        choices=model_choices,
        default=model_choices[0]
    ).ask()
    
    # Ask for API key if needed
    api_key_source = questionary.select(
        "How would you like to provide your API key?",
        choices=[
            "Environment variable (recommended)",
            "Enter now (will be stored in .env.local)",
            "I'll configure it later",
        ],
        default="Environment variable (recommended)"
    ).ask()
    
    api_key = None
    if api_key_source == "Enter now (will be stored in .env.local)":
        api_key = questionary.password(
            f"Enter your {llm_provider} API key:"
        ).ask()
    
    # Update configuration with user choices
    config["use_gitignore"] = use_gitignore
    config["token_limit"] = int(token_limit)
    
    # Add processor section
    config["processor"] = {
        "enabled": enable_processor,
        "enable_lsp": enable_lsp,
        "max_workers": int(max_workers),
        "cache_dir": ".codemap_cache",
        "embedding_model": "Qodo/Qodo-Embed-1-1.5B",
        "batch_size": 32,
    }
    
    # Update LLM configuration
    config["llm"] = {
        "provider": llm_provider.lower(),
        "model": format_model_name(llm_provider, llm_model),
    }
    
    # If commit section doesn't exist, create it
    if "commit" not in config:
        config["commit"] = {}
        
    # Update commit LLM configuration
    config["commit"]["llm"] = {
        "model": format_model_name(llm_provider, llm_model),
    }
    
    return config
```

### 3. Caching and Storage Management

Set up the cache directory structure:

```python
def setup_cache_directory(repo_path: Path) -> Path:
    """Set up the cache directory structure.
    
    Args:
        repo_path: Repository path
        
    Returns:
        Path to cache directory
    """
    cache_dir = repo_path / ".codemap_cache"
    
    # Create main cache directory
    cache_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (cache_dir / "storage").mkdir(exist_ok=True)  # LanceDB storage
    (cache_dir / "embeddings").mkdir(exist_ok=True)  # Embedding cache
    (cache_dir / "lsp").mkdir(exist_ok=True)  # LSP cache
    (cache_dir / "logs").mkdir(exist_ok=True)  # Logs
    
    # Create a .gitignore file in the cache directory
    with open(cache_dir / ".gitignore", "w") as f:
        f.write("# Automatically generated by CodeMap\n")
        f.write("# Cache files should not be committed\n")
        f.write("*\n")
        f.write("!.gitignore\n")
    
    return cache_dir
```

Update the project's `.gitignore` file:

```python
def update_gitignore(repo_path: Path) -> None:
    """Update the project's .gitignore file to include CodeMap cache.
    
    Args:
        repo_path: Repository path
    """
    gitignore_path = repo_path / ".gitignore"
    
    # Define CodeMap entries
    codemap_entries = [
        "\n# CodeMap",
        ".codemap_cache/",
        ".env.local",
    ]
    
    # Check if .gitignore exists
    if gitignore_path.exists():
        # Read existing content
        with open(gitignore_path, "r") as f:
            content = f.read()
            
        # Check if CodeMap entries already exist
        if ".codemap_cache/" in content:
            return
            
        # Add entries
        with open(gitignore_path, "a") as f:
            f.write("\n".join(codemap_entries) + "\n")
    else:
        # Create new .gitignore
        with open(gitignore_path, "w") as f:
            f.write("\n".join(codemap_entries) + "\n")
```

### 4. API Key Management

Securely manage API keys:

```python
def setup_api_keys(repo_path: Path, provider: str, api_key: str | None) -> None:
    """Set up API keys for the selected provider.
    
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
        with open(env_file, "r") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    key, value = line.strip().split("=", 1)
                    existing_content[key] = value
    
    # Update with new API key
    existing_content[env_var] = api_key
    
    # Write updated content
    with open(env_file, "w") as f:
        f.write("# CodeMap API Keys - KEEP THIS FILE SECRET\n")
        f.write("# Do not commit this file to version control\n\n")
        for key, value in existing_content.items():
            f.write(f"{key}={value}\n")
    
    # Set permissions to restrict access
    try:
        os.chmod(env_file, 0o600)  # Owner read/write only
    except Exception:
        logger.warning("Could not set restrictive permissions on .env.local")
        
    console.print(f"[green]Added {env_var} to .env.local[/green]")
    console.print("[yellow]Important: Keep .env.local secure and don't commit it to version control[/yellow]")
```

## Enhanced Command Implementation

```python
def init_command(
    path: PathArg = Path(),
    force_flag: ForceFlag = False,
    non_interactive: Annotated[
        bool, 
        typer.Option("--non-interactive", help="Run in non-interactive mode")
    ] = False,
    scan: Annotated[
        bool,
        typer.Option("--scan", help="Perform initial repository scan")
    ] = True,
    background: Annotated[
        bool,
        typer.Option("--background", help="Start background watcher after initialization")
    ] = False,
    is_verbose: VerboseFlag = False,
) -> None:
    """Initialize a new CodeMap project with interactive setup."""
    setup_logging(is_verbose=is_verbose)
    try:
        repo_path = path.resolve()
        config_file = repo_path / ".codemap.yml"
        
        # Check if files already exist
        existing_files = []
        if config_file.exists():
            existing_files.append(config_file)
            
        cache_dir = repo_path / ".codemap_cache"
        if cache_dir.exists():
            existing_files.append(cache_dir)
            
        if not force_flag and existing_files:
            console.print("[yellow]CodeMap files already exist:")
            for f in existing_files:
                console.print(f"[yellow]  - {f}")
            console.print("[yellow]Use --force to overwrite.")
            raise typer.Exit(1)
            
        # Interactive or default configuration
        if non_interactive:
            config = DEFAULT_CONFIG.copy()
            # Add processor section with defaults
            config["processor"] = {
                "enabled": True,
                "enable_lsp": True,
                "max_workers": 4,
                "cache_dir": ".codemap_cache",
                "embedding_model": "Qodo/Qodo-Embed-1-1.5B",
                "batch_size": 32,
            }
        else:
            config = interactive_setup(repo_path)
            
        # Set up directories and files
        with console.status("Setting up CodeMap...") as status:
            # 1. Create .codemap.yml
            config_file.write_text(yaml.dump(config, sort_keys=False))
            status.update("Created configuration file")
            
            # 2. Set up cache directory
            cache_dir = setup_cache_directory(repo_path)
            status.update("Set up cache directory")
            
            # 3. Update .gitignore
            update_gitignore(repo_path)
            status.update("Updated .gitignore")
            
            # 4. Set up API keys if provided through interactive setup
            if not non_interactive and "llm" in config:
                provider = config["llm"].get("provider", "openai")
                api_key = getattr(config["llm"], "_api_key", None)
                if api_key:
                    setup_api_keys(repo_path, provider, api_key)
                    # Remove the key from config after storing it
                    delattr(config["llm"], "_api_key")
                    status.update("Configured API keys")
        
        console.print("\n✨ [bold green]CodeMap initialized successfully![/bold green]")
        console.print(f"[green]Created config file: {config_file}[/green]")
        console.print(f"[green]Set up cache directory: {cache_dir}[/green]")
        
        # Initialize processor if enabled
        if config.get("processor", {}).get("enabled", True):
            try:
                pipeline = initialize_processor(repo_path, config)
                console.print("[green]Initialized processing pipeline[/green]")
                
                # Run initial scan if requested
                if scan:
                    console.print("\n[bold]Running initial repository scan...[/bold]")
                    run_initial_scan(pipeline, repo_path)
                    console.print("[green]Repository scan complete[/green]")
                
                # Start background watcher if requested
                if background:
                    if os.fork() == 0:  # Child process
                        # Detach from parent
                        os.setsid()
                        # Close standard file descriptors
                        os.close(0)
                        os.close(1)
                        os.close(2)
                        # Start watcher
                        start_background_watcher(pipeline)
                    else:  # Parent process
                        console.print("[green]Started background watcher process[/green]")
                else:
                    pipeline.stop()  # Clean shutdown if not running in background
            except Exception as e:
                console.print(f"[yellow]Warning: Could not initialize processor: {e}[/yellow]")
                logger.exception("Failed to initialize processor")
                
        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Review and customize .codemap.yml for your project")
        console.print("2. Run 'codemap generate' to create documentation")
        console.print("3. Run 'codemap commit' to use AI-powered commit messages")
        console.print("4. Run 'codemap pr' to create pull requests")

    except (FileNotFoundError, PermissionError, OSError) as e:
        console.print(f"[red]File system error: {e!s}[/red]")
        raise typer.Exit(1) from e
    except ValueError as e:
        console.print(f"[red]Configuration error: {e!s}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Unexpected error: {e!s}[/red]")
        logger.exception("Unexpected error during initialization")
        raise typer.Exit(1) from e
```

## Extended Configuration Structure

The enhanced `.codemap.yml` will include new sections to support the processor module:

```yaml
# Basic settings
token_limit: 10000
use_gitignore: true
output_dir: documentation
max_content_length: 5000

# LLM configuration (global)
llm:
  provider: openai
  model: openai/gpt-4o-mini
  
# Processor configuration
processor:
  enabled: true
  enable_lsp: true
  max_workers: 4
  cache_dir: .codemap_cache
  embedding_model: Qodo/Qodo-Embed-1-1.5B
  batch_size: 32
  ignored_patterns:
    - "**/.git/**"
    - "**/__pycache__/**"
    - "**/.venv/**"
    - "**/node_modules/**"
    - "**/.codemap_cache/**"
    - "**/*.pyc"
    - "**/dist/**"
    - "**/build/**"
  
# Commit feature configuration
commit:
  # Strategy for splitting diffs: file, hunk, semantic
  strategy: semantic
  # LLM configuration
  llm:
    model: openai/gpt-4o-mini
  # Commit convention settings
  convention:
    types:
      - feat     # New feature
      - fix      # Bug fix
      - docs     # Documentation
      - style    # Formatting, missing semicolons, etc.
      - refactor # Code change that neither fixes a bug nor adds a feature
      - perf     # Performance improvement
      - test     # Adding or updating tests
      - build    # Build system or external dependencies
      - ci       # CI configuration
      - chore    # Other changes that don't modify src or test files
    scopes: []
    max_length: 72
```

## Implementation Roadmap

### Phase 1: Core Initialization
1. Enhance init command with non-interactive defaults
2. Set up proper directory structure
3. Update `.gitignore` handling
4. Implement basic configuration generation

### Phase 2: Interactive Setup
1. Create interactive wizard with questionary
2. Implement provider detection and model selection
3. Add secure API key management
4. Create smart defaults based on repository analysis

### Phase 3: Processor Integration
1. Implement processor initialization
2. Add initial repository scanning
3. Configure storage and embedding backend
4. Test with different repository types

### Phase 4: Background Processing
1. Implement background watcher process
2. Add proper process management and cleanup
3. Create status commands to check background processes
4. Implement logging and monitoring

## Expected Outcomes

The enhanced init command will:

1. **Improve User Experience**
   - Interactive setup reduces configuration complexity
   - Smart defaults speed up initialization
   - Clear next steps guide users through the workflow

2. **Enable Advanced Features**
   - Proper processor setup enables semantic code search
   - Background processing keeps code analysis up-to-date
   - Centralized configuration improves feature cohesion

3. **Provide Better Performance**
   - Structured caching reduces redundant operations
   - Initial scan prepares repository for immediate use
   - Optimized storage improves query performance

4. **Enhance Security**
   - Secure API key management prevents credential leakage
   - Proper `.gitignore` integration prevents accidental commits
   - Restricted permissions on sensitive files
