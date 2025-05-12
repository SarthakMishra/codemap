# Configuration

The `conf` command helps you create and manage your CodeMap configuration file (`.codemap.yml`). Use it to quickly generate a default config or overwrite an existing one.

## Command Options

```bash
codemap conf [OPTIONS]
# Or using the alias:
cm conf [OPTIONS]
```

**Options:**

- `--force`, `-f`: Overwrite the existing configuration file if it already exists
- `--verbose`, `-v`: Enable verbose logging

## Examples

```bash
# Create a default .codemap.yml in the project root
codemap conf

# Overwrite the config file if it already exists
cm conf --force
```

---

## All Available Options

Below are all available configuration options with their default values. Commented options are advanced and can be enabled as needed.

```yaml
# LLM Configuration (applies globally unless overridden by command-specific LLM config)
llm:
  model: "openai/gpt-4o-mini"   # Default LLM model (provider:model-name)
  # temperature: 0.5             # Lower for more deterministic outputs, higher for creativity
  # max_output_tokens: 1024      # Maximum tokens in responses
  # api_base: null               # Custom API base URL (e.g., for local LLMs or proxies)

# Embedding Configuration
embedding:
  model_name: "minishlab/potion-base-8M"  # Only Model2Vec static models are supported
  # dimension_metric: "cosine"            # Metric for dimension calculation
  # max_retries: 3                        # Maximum retries for embedding requests
  # retry_delay: 5                        # Delay in seconds between retries
  # max_content_length: 5000               # Maximum characters per file chunk
  # qdrant_batch_size: 100                 # Batch size for Qdrant uploads
  # url: "http://localhost:6333"          # Qdrant server URL
  # timeout: 30                            # Qdrant client timeout in seconds
  # prefer_grpc: true                      # Prefer gRPC for Qdrant communication
  # chunking:
  #   max_hierarchy_depth: 2               # Maximum depth of code hierarchy to consider
  #   max_file_lines: 1000                 # Maximum lines per file before splitting
  # clustering:
  #   method: "agglomerative"             # Clustering method: "agglomerative", "dbscan"
  #   agglomerative:
  #     metric: "precomputed"
  #     distance_threshold: 0.3
  #     linkage: "complete"
  #   dbscan:
  #     eps: 0.3
  #     min_samples: 2
  #     algorithm: "auto"
  #     metric: "precomputed"

# RAG (Retrieval Augmented Generation) Configuration
rag:
  max_context_length: 8000      # Maximum context length for the LLM
  max_context_results: 10       # Maximum number of context results to return
  similarity_threshold: 0.75    # Minimum similarity score (0-1) for relevance
  # system_prompt: null         # Optional system prompt to guide the RAG model
  include_file_content: true    # Include file content in context
  include_metadata: true        # Include file metadata in context

# Sync Configuration
sync:
  exclude_patterns:
    - "^node_modules/"
    - "^.venv/"
    - "^venv/"
    - "^env/"
    - "^__pycache__/"
    - "^.mypy_cache/"
    - "^.pytest_cache/"
    - "^.ruff_cache/"
    - "^dist/"
    - "^build/"
    - "^.git/"
    - ".pyc$"
    - ".pyo$"
    - ".so$"
    - ".dll$"

# Documentation Generation Settings ('gen' command)
gen:
  max_content_length: 5000       # Max content length per file (0 = unlimited)
  use_gitignore: true            # Respect .gitignore patterns
  output_dir: documentation      # Directory for generated docs
  include_tree: true             # Include directory tree in output
  include_entity_graph: true     # Include Mermaid entity relationship graph
  semantic_analysis: true        # Enable semantic analysis using LSP
  lod_level: docs                # Level of Detail: signatures, structure, docs, full
  # mermaid_entities:
  #   - module
  #   - class
  #   - function
  #   - method
  #   - constant
  #   - variable
  #   - import
  # mermaid_relationships:
  #   - declares
  #   - imports
  #   - calls
  # mermaid_show_legend: true
  # mermaid_remove_unconnected: false

# Processor configuration
processor:
  enabled: true
  max_workers: 4
  ignored_patterns:
    - "**/.git/**"
    - "**/__pycache__/**"
    - "**/.venv/**"
    - "**/node_modules/**"
    - "**/*.pyc"
    - "**/dist/**"
    - "**/build/**"
  default_lod_level: signatures
  # watcher:
  #   enabled: true
  #   debounce_delay: 1.0

# Commit Feature Configuration ('commit' command)
commit:
  strategy: semantic             # Diff splitting strategy: file, hunk, semantic
  bypass_hooks: false            # Default for --bypass-hooks flag (--no-verify)
  use_lod_context: true          # Use level of detail context
  is_non_interactive: false      # Run in non-interactive mode
  # diff_splitter:
  #   similarity_threshold: 0.6
  #   directory_similarity_threshold: 0.3
  #   file_move_similarity_threshold: 0.85
  #   min_chunks_for_consolidation: 2
  #   max_chunks_before_consolidation: 20
  #   max_file_size_for_llm: 50000
  #   max_log_diff_size: 1000
  #   default_code_extensions:
  #     - js
  #     - py
  #     - ...
  convention:
    types:
      - feat
      - fix
      - docs
      - style
      - refactor
      - perf
      - test
      - build
      - ci
      - chore
    scopes: []                   # Optional scopes (can be auto-derived if empty)
    max_length: 72               # Max length for commit subject line
  # lint:
  #   header_max_length: { level: ERROR, rule: always, value: 100 }
  #   type_enum: { level: ERROR, rule: always }
  #   type_case: { level: ERROR, rule: always, value: lower-case }
  #   subject_empty: { level: ERROR, rule: never }
  #   subject_full_stop: { level: ERROR, rule: never, value: . }

# Pull Request Configuration ('pr' command)
pr:
  defaults:
    base_branch: null            # Default base branch (null = repo default)
    feature_prefix: "feature/"   # Default prefix for feature branches
  strategy: github-flow          # Git workflow: github-flow, gitflow, trunk-based
  # branch_mapping:
  #   feature: { base: develop, prefix: "feature/" }
  #   release: { base: main, prefix: "release/" }
  #   hotfix: { base: main, prefix: "hotfix/" }
  #   bugfix: { base: develop, prefix: "bugfix/" }
  generate:
    title_strategy: llm         # How to generate title: commits, llm, template
    description_strategy: llm   # How to generate description: commits, llm, template
    use_workflow_templates: true # Use built-in templates based on workflow/branch type?
    # description_template: |
    #   ## Changes
    #   {changes}
    #
    #   ## Testing
    #   {testing_instructions}
    #
    #   ## Screenshots
    #   {screenshots}

# Ask Command Configuration
ask:
  interactive_chat: false        # Enable interactive chat mode for the 'ask' command
```

## Configuration Priority

The configuration is loaded in the following order (later sources override earlier ones):

1. Default configuration from the package
2. `.codemap.yml` in the project root
3. Custom config file specified with `--config`
4. Command-line arguments

## Configuration Tips

Refer to the main README section for detailed tips on configuring:

- Token Limits & Content Length
- Git Integration (`use_gitignore`, `convention.scopes`, `bypass_hooks`)
- LLM Settings (`llm.model`, `llm.api_base`, `--model` flag)
- Commit Conventions & Linting (`commit.convention`, `commit.lint`)
- PR Workflow Settings (`pr.strategy`, `pr.defaults`, `pr.branch_mapping`, `pr.generate`)
- Documentation Generation (`gen.*` settings and flags)
- Embedding and RAG settings for advanced semantic search