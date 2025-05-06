# Configuration

Create a `.codemap.yml` file in your project root to customize the behavior. Below are all available configuration options with their default values:

```yaml
# LLM configuration (applies globally unless overridden by command-specific LLM config)
llm:
  model: openai/gpt-4o-mini  # Default LLM model (provider/model_name format)
  api_base: null             # Custom API base URL (e.g., for local LLMs or proxies)

# Documentation Generation Settings ('gen' command)
gen:
  max_content_length: 5000       # Max content length per file (0 = unlimited)
  use_gitignore: true            # Respect .gitignore patterns
  output_dir: documentation       # Directory for generated docs (Note: mkdocs uses 'docs/')
  include_tree: true             # Include directory tree in output
  include_entity_graph: true     # Include Mermaid entity relationship graph
  semantic_analysis: true        # Enable semantic analysis using LSP
  lod_level: docs                # Level of Detail: signatures, structure, docs, full
  mermaid_entities:              # Entity types for Mermaid graph
    - module
    - class
    - function
    - method
    - constant
    - variable
    - import
  mermaid_relationships:         # Relationship types for Mermaid graph
    - declares
    - imports
    - calls
  mermaid_show_legend: true      # Show legend in Mermaid diagram
  mermaid_remove_unconnected: false # Remove unconnected nodes in Mermaid diagram

# Processor configuration (background analysis - currently unused)
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

# Commit Feature Configuration ('commit' command)
commit:
  strategy: semantic             # Diff splitting strategy: file, hunk, semantic
  bypass_hooks: false            # Default for --bypass-hooks flag (--no-verify)

  convention:                    # Commit convention settings (based on Conventional Commits)
    types:                       # Allowed commit types
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

  lint:                          # Commitlint rule configuration (see https://commitlint.js.org/#/reference-rules)
    # Example rules (full list in README)
    header_max_length: { level: ERROR, rule: always, value: 100 }
    type_enum: { level: ERROR, rule: always } # Uses types from commit.convention.types
    type_case: { level: ERROR, rule: always, value: lower-case }
    subject_empty: { level: ERROR, rule: never }
    subject_full_stop: { level: ERROR, rule: never, value: . }

# Pull Request Configuration ('pr' command)
pr:
  defaults:
    base_branch: null            # Default base branch (null = repo default)
    feature_prefix: "feature/"   # Default prefix for feature branches

  strategy: github-flow          # Git workflow: github-flow, gitflow, trunk-based

  branch_mapping:                # Branch base/prefix mapping (primarily for GitFlow)
    feature: { base: develop, prefix: "feature/" }
    release: { base: main, prefix: "release/" }
    hotfix: { base: main, prefix: "hotfix/" }
    bugfix: { base: develop, prefix: "bugfix/" }

  generate:                      # Content generation settings
    title_strategy: commits      # How to generate title: commits, llm, template
    description_strategy: commits # How to generate description: commits, llm, template
    use_workflow_templates: true # Use built-in templates based on workflow/branch type?
    # Template used if description_strategy is 'template' AND use_workflow_templates is false
    description_template: |
      ## Changes
      {changes}

      ## Testing
      {testing_instructions}

      ## Screenshots
      {screenshots}
```

## Configuration Priority

The configuration is loaded in the following order (later sources override earlier ones):

1. Default configuration from the package
2. `.codemap.yml` in the project root
3. Custom config file specified with `--config`
4. Command-line arguments

## Configuration Tips

Refer to the main README section for detailed tips on configuring:

- Token Limits (Deprecated) & Content Length
- Git Integration (`use_gitignore`, `convention.scopes`, `bypass_hooks`)
- LLM Settings (`llm.model`, `llm.api_base`, `--model` flag)
- Commit Conventions & Linting (`commit.convention`, `commit.lint`)
- PR Workflow Settings (`pr.strategy`, `pr.defaults`, `pr.branch_mapping`, `pr.generate`)
- Documentation Generation (`gen.*` settings and flags)

## Environment Variables

LLM API keys and optional base URLs can be set via environment variables. See the [LLM Support](llm-support.md) page for details. 