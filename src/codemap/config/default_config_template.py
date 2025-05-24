"""Default configuration template for CodeMap."""

# ruff: noqa: E501

DEFAULT_CONFIG_TEMPLATE = """# CodeMap Configuration File
# -------------------------
# This file configures CodeMap's behavior. Uncomment and modify settings as needed.

# LLM Configuration - Controls which model is used for AI operations
llm:
  # Format: "provider:model-name", e.g., "openai:gpt-4o", "anthropic:claude-3-opus"
  model: "openai:gpt-4o-mini"
  temperature: 0.5  # Lower for more deterministic outputs, higher for creativity
  max_output_tokens: 1024  # Maximum tokens in responses

# Embedding Configuration - Controls vector embedding behavior
embedding:
  # Recommended models: "minishlab/potion-base-8M3", Only Model2Vec static models are supported
  model_name: "minishlab/potion-base-8M"
  # dimension: 256
  # dimension_metric: "cosine" # Metric for dimension calculation (e.g., "cosine", "euclidean")
  # max_retries: 3 # Maximum retries for embedding requests
  # retry_delay: 5 # Delay in seconds between retries
  # max_content_length: 5000  # Maximum characters per file chunk
  # Qdrant (Vector DB) settings
  # qdrant_batch_size: 100 # Batch size for Qdrant uploads
  # url: "http://localhost:6333" # Qdrant server URL
  # timeout: 30 # Qdrant client timeout in seconds
  # prefer_grpc: true # Prefer gRPC for Qdrant communication

  # Advanced chunking settings - controls how code is split
  # chunking:
  #   max_hierarchy_depth: 2  # Maximum depth of code hierarchy to consider
  #   max_file_lines: 1000  # Maximum lines per file before splitting

  # Clustering settings for embeddings
  # clustering:
  #   method: "agglomerative"  # Clustering method: "agglomerative", "dbscan"
  #   agglomerative: # Settings for Agglomerative Clustering
  #     metric: "precomputed" # Metric: "cosine", "euclidean", "manhattan", "l1", "l2", "precomputed"
  #     distance_threshold: 0.3 # Distance threshold for forming clusters
  #     linkage: "complete" # Linkage criterion: "ward", "complete", "average", "single"
  #   dbscan: # Settings for DBSCAN Clustering
  #     eps: 0.3 # The maximum distance between two samples for one to be considered as in the neighborhood of the other
  #     min_samples: 2 # The number of samples in a neighborhood for a point to be considered as a core point
  #     algorithm: "auto" # Algorithm to compute pointwise distances: "auto", "ball_tree", "kd_tree", "brute"
  #     metric: "precomputed" # Metric for distance computation: "cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "precomputed"

# RAG (Retrieval Augmented Generation) Configuration
rag:
  max_context_length: 8000  # Maximum context length for the LLM
  max_context_results: 10  # Maximum number of context results to return
  similarity_threshold: 0.75  # Minimum similarity score (0-1) for relevance
  # system_prompt: null # Optional system prompt to guide the RAG model (leave commented or set if needed)
  include_file_content: true  # Include file content in context
  include_metadata: true  # Include file metadata in context

# Sync Configuration - Controls which files are excluded from processing
sync:
  exclude_patterns:
    - "^node_modules/"
    - "^\\\\.venv/"
    - "^venv/"
    - "^env/"
    - "^__pycache__/"
    - "^\\\\.mypy_cache/"
    - "^\\\\.pytest_cache/"
    - "^\\\\.ruff_cache/"
    - "^dist/"
    - "^build/"
    - "^\\\\.git/"
    - "\\\\\\\\.pyc$"
    - "\\\\\\\\.pyo$"
    - "\\\\\\\\.so$"
    - "\\\\\\\\.dll$"

# Generation Configuration - Controls documentation generation
gen:
  max_content_length: 5000  # Maximum content length per file for generation
  use_gitignore: true  # Use .gitignore patterns to exclude files
  output_dir: "documentation"  # Directory to store generated documentation
  include_tree: true  # Include directory tree in output
  include_entity_graph: true  # Include entity relationship graph
  semantic_analysis: true  # Enable semantic analysis
  lod_level: "docs"  # Level of detail: "signatures", "structure", "docs", "skeleton", "full"

  # Mermaid diagram configuration for entity graphs
  # mermaid_entities:
  #   - "module"
  #   - "class"
  #   - "function"
  #   - "method"
  #   - "constant"
  #   - "variable"
  #   - "import"
  # mermaid_relationships:
  #   - "declares"
  #   - "imports"
  #   - "calls"
  # mermaid_show_legend: true
  # mermaid_remove_unconnected: false  # Show isolated nodes

# Processor Configuration - Controls code processing behavior
processor:
  enabled: true  # Enable the processor
  max_workers: 4  # Maximum number of parallel workers
  ignored_patterns:  # Patterns to ignore during processing
    - "**/.git/**"
    - "**/__pycache__/**"
    - "**/.venv/**"
    - "**/node_modules/**"
    - "**/*.pyc"
    - "**/dist/**"
    - "**/build/**"
  default_lod_level: "signatures"  # Default level of detail: "signatures", "structure", "docs", "full"

  # File watcher configuration
  # watcher:
  #   enabled: true  # Enable file watching
  #   debounce_delay: 1.0  # Delay in seconds before processing changes

# Commit Command Configuration
commit:
  strategy: "semantic"  # Strategy for splitting diffs: "file", "hunk", "semantic"
  bypass_hooks: false  # Whether to bypass git hooks
  use_lod_context: true  # Use level of detail context
  is_non_interactive: false  # Run in non-interactive mode

  # Diff splitter configuration
  # diff_splitter:
  #   similarity_threshold: 0.6  # Similarity threshold for grouping related changes
  #   directory_similarity_threshold: 0.3 # Threshold for considering directories similar (e.g., for renames)
  #   file_move_similarity_threshold: 0.85 # Threshold for detecting file moves/renames based on content
  #   min_chunks_for_consolidation: 2 # Minimum number of small chunks to consider for consolidation
  #   max_chunks_before_consolidation: 20 # Maximum number of chunks before forcing consolidation
  #   max_file_size_for_llm: 50000  # Maximum file size (bytes) for LLM processing of individual files
  #   max_log_diff_size: 1000 # Maximum size (lines) of diff log to pass to LLM for context
  #   default_code_extensions: # File extensions considered as code for semantic splitting
  #     - "js"
  #     - "jsx"
  #     - "ts"
  #     - "tsx"
  #     - "py"
  #     - "java"
  #     - "c"
  #     - "cpp"
  #     - "h"
  #     - "hpp"
  #     - "cc"
  #     - "cs"
  #     - "go"
  #     - "rb"
  #     - "php"
  #     - "rs"
  #     - "swift"
  #     - "scala"
  #     - "kt"
  #     - "sh"
  #     - "pl"
  #     - "pm"

  # Commit convention configuration (Conventional Commits)
  convention:
    types: # Allowed commit types
      - "feat"
      - "fix"
      - "docs"
      - "style"
      - "refactor"
      - "perf"
      - "test"
      - "build"
      - "ci"
      - "chore"
    scopes: []  # Add project-specific scopes here, e.g., ["api", "ui", "db"]
    max_length: 72  # Maximum length of commit message header

  # Commit linting configuration (based on conventional-changelog-lint rules)
  # lint:
  #   # Rules are defined as: {level: "ERROR"|"WARNING"|"DISABLED", rule: "always"|"never", value: <specific_value_if_any>}
  #   header_max_length:
  #     level: "ERROR"
  #     rule: "always"
  #     value: 100
  #   header_case: # e.g., 'lower-case', 'upper-case', 'camel-case', etc.
  #     level: "DISABLED"
  #     rule: "always"
  #     value: "lower-case"
  #   header_full_stop:
  #     level: "ERROR"
  #     rule: "never"
  #     value: "."
  #   type_enum: # Types must be from the 'convention.types' list
  #     level: "ERROR"
  #     rule: "always"
  #   type_case:
  #     level: "ERROR"
  #     rule: "always"
  #     value: "lower-case"
  #   type_empty:
  #     level: "ERROR"
  #     rule: "never"
  #   scope_case:
  #     level: "ERROR"
  #     rule: "always"
  #     value: "lower-case"
  #   scope_empty: # Set to "ERROR" if scopes are mandatory
  #     level: "DISABLED"
  #     rule: "never"
  #   scope_enum: # Scopes must be from the 'convention.scopes' list if enabled
  #     level: "DISABLED"
  #     rule: "always"
  #     # value: [] # Add allowed scopes here if rule is "always" and level is not DISABLED
  #   subject_case: # Forbids specific cases in the subject
  #     level: "ERROR"
  #     rule: "never"
  #     value: ["sentence-case", "start-case", "pascal-case", "upper-case"]
  #   subject_empty:
  #     level: "ERROR"
  #     rule: "never"
  #   subject_full_stop:
  #     level: "ERROR"
  #     rule: "never"
  #     value: "."
  #   subject_exclamation_mark:
  #     level: "DISABLED"
  #     rule: "never"
  #   body_leading_blank: # Body must start with a blank line after subject
  #     level: "WARNING"
  #     rule: "always"
  #   body_empty:
  #     level: "DISABLED"
  #     rule: "never"
  #   body_max_line_length:
  #     level: "ERROR"
  #     rule: "always"
  #     value: 100
  #   footer_leading_blank: # Footer must start with a blank line after body
  #     level: "WARNING"
  #     rule: "always"
  #   footer_empty:
  #     level: "DISABLED"
  #     rule: "never"
  #   footer_max_line_length:
  #     level: "ERROR"
  #     rule: "always"
  #     value: 100

# Pull Request Configuration
pr:
  defaults:
    base_branch: null  # Default base branch (null = auto-detect, e.g., main, master, develop)
    feature_prefix: "feature/"  # Default feature branch prefix

  strategy: "github-flow"  # Git workflow: "github-flow", "gitflow", "trunk-based"

  # Branch mapping for different PR types (primarily used in gitflow strategy)
  # branch_mapping:
  #   feature:
  #     base: "develop"
  #     prefix: "feature/"
  #   release:
  #     base: "main"
  #     prefix: "release/"
  #   hotfix:
  #     base: "main"
  #     prefix: "hotfix/"
  #   bugfix:
  #     base: "develop"
  #     prefix: "bugfix/"

  # PR generation configuration
  generate:
    title_strategy: "llm"  # Strategy for generating PR titles: "commits" (from commit messages), "llm" (AI generated)
    description_strategy: "llm"  # Strategy for descriptions: "commits", "llm"
    # description_template: | # Template for PR description when using 'llm' strategy. Placeholders: {changes}, {testing_instructions}, {screenshots}
    #   ## Changes
    #   {changes}
    #
    #   ## Testing
    #   {testing_instructions}
    #
    #   ## Screenshots
    #   {screenshots}
    use_workflow_templates: true  # Use workflow-specific templates if available (e.g., for GitHub PR templates)

# Ask Command Configuration
ask:
  interactive_chat: false  # Enable interactive chat mode for the 'ask' command
"""
