# CodeMap

[![PyPI](https://img.shields.io/pypi/v/codemap)](https://pypi.org/project/codemap/)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/SarthakMishra/codemap/actions/workflows/tests.yml/badge.svg)](https://github.com/SarthakMishra/code-map/actions/workflows/tests.yml)
[![Lint](https://github.com/SarthakMishra/codemap/actions/workflows/lint.yml/badge.svg)](https://github.com/SarthakMishra/codemap/actions/workflows/lint.yml)
[![CodeQL](https://github.com/SarthakMishra/codemap/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/SarthakMishra/codemap/actions/workflows/github-code-scanning/codeql)
[![codecov](https://codecov.io/gh/SarthakMishra/codemap/branch/main/graph/badge.svg)](https://codecov.io/gh/SarthakMishra/codemap)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/95d85720e3a14494abf27b5d2070d92f)](https://app.codacy.com/gh/SarthakMishra/codemap/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> [!Caution]
> CodeMap is currently in active development. Use with caution and expect breaking changes.

## Overview

CodeMap is an AI-powered developer toolkit. Generate optimized docs, analyze code semantically, and streamline Git workflows (AI commits, PRs) with multi-LLM support via an interactive CLI.

## Features

- 🎯 Token-optimized documentation generation
- 📝 Rich markdown output with code structure
- 🌳 Repository structure visualization
- 🔄 Smart Git commit assistance with AI-generated messages
- 🔃 AI-powered PR creation and management

## Installation

> [!Important]
> CodeMap currently only supports Unix-based platforms (macOS, Linux). For Windows users, we recommend using Windows Subsystem for Linux (WSL).

> [!Tip]
> After installation, you can use either `codemap` or the shorter alias `cm` to run the commands.

### Installation using pipx (Recommended)

Using `pipx` is recommended as it installs the package in an isolated environment and automatically manages the PATH.

```bash
# Ensure pipx is installed (install it if you haven't)
# python3 -m pip install --user pipx
# python3 -m pipx ensurepath

# Install codemap from PyPI
pipx install codemap
```

### Alternative: Manual Installation using pip

If you prefer not to use `pipx`, you can install using `pip` directly:

```bash
# Install with pip (user installation)
pip install --user codemap

# Make sure your PATH includes the user bin directory
# Add the following to your shell profile (e.g., ~/.bashrc, ~/.zshrc):
# export PATH="$HOME/.local/bin:$PATH"
# Or find the correct path using: python3 -m site --user-base
```

### Development Version (Latest GitHub)

If you want to try the latest development version with unreleased features:

```bash
# Using pipx
pipx install git+https://github.com/SarthakMishra/codemap.git

# Or using pip
pip install --user git+https://github.com/SarthakMishra/codemap.git
```

### Updating CodeMap

To update CodeMap to the latest version:

```bash
# If installed with pipx from PyPI
pipx upgrade codemap

# If installed with pip from PyPI
pip install --user --upgrade codemap

# If installed from GitHub
pipx upgrade codemap  # or
pip install --user --upgrade git+https://github.com/SarthakMishra/codemap.git
```

### Uninstalling

```bash
# If installed with pipx
pipx uninstall codemap

# If installed with pip
pip uninstall codemap
```

## Generate Markdown Docs

Generate optimized markdown documentation and directory structures for your project:

### Command Options

```bash
codemap gen [PATH] [OPTIONS]
```

**Arguments:**
- `PATH`: Path to the codebase to analyze (defaults to current directory)

**Options:**
- `--output`, `-o`: Output file path for the documentation (overrides config)
- `--config`, `-c`: Path to custom configuration file
- `--max-content-length`: Maximum content length for file display (set to 0 for unlimited, overrides config)
- `--lod`: Level of Detail for code analysis (signatures, structure, docs, full). Default: `docs`. Overrides config.
- `--semantic`/`--no-semantic`: Enable/disable semantic analysis using LSP. Default: enabled. Overrides config.
- `--tree`/`--no-tree`: Include/exclude directory tree in output. Overrides config (`gen.include_tree`).
- `--verbose`, `-v`: Enable verbose logging
- `--process`/`--no-process`: Process the codebase before generation. Default: enabled.
- `--entity-graph`/`--no-entity-graph`: Include/exclude entity relationship graph (Mermaid) in output. Overrides config (`gen.include_entity_graph`).
- `--mermaid-entities`: Comma-separated list of entity types (e.g., 'module,class,function'). Overrides config (`gen.mermaid_entities`).
- `--mermaid-relationships`: Comma-separated list of relationship types (e.g., 'declares,imports,calls'). Overrides config (`gen.mermaid_relationships`).
- `--mermaid-legend`/`--no-mermaid-legend`: Show/hide the legend in the Mermaid diagram. Overrides config (`gen.mermaid_show_legend`).
- `--mermaid-unconnected`/`--no-mermaid-unconnected`: Remove/keep nodes with no connections in the Mermaid diagram. Overrides config (`gen.mermaid_remove_unconnected`).

### Examples

```bash
# Generate documentation for current directory using defaults
codemap gen
# Or using the alias:
cm gen

# Generate for a specific path with full detail and no semantic analysis
codemap gen /path/to/project --lod full --no-semantic

# Generate docs with signatures only and custom Mermaid settings
cm gen --lod signatures --mermaid-entities "class,function" --mermaid-relationships "calls"

# Generate only directory tree (implicitly disables entity graph)
codemap gen --tree --no-entity-graph

# Custom output location and content length
codemap gen -o ./docs/codebase.md --max-content-length 1500

# Use custom configuration file
codemap gen --config custom-config.yml

# Verbose mode for debugging
codemap gen -v
```

## Smart Commit Feature

Create intelligent Git commits with AI-assisted message generation. The tool analyzes your changes, splits them into logical chunks, and generates meaningful commit messages using LLMs.

### Basic Usage

```bash
# Basic usage with default settings (interactive, semantic splitting)
codemap commit
# Or using the alias:
cm commit

# Commit with a specific message (skips AI generation)
codemap commit -m "feat: add new feature"

# Commit all changes (including untracked files)
codemap commit -a

# Use a specific LLM model
codemap commit --model groq/llama-3.1-8b-instant

# Bypass git hooks (e.g., pre-commit)
codemap commit --bypass-hooks
```

### Command Options

```bash
codemap commit [PATH] [OPTIONS]
```

**Arguments:**
- `PATH`: Path to repository or specific file to commit (defaults to current directory)

**Options:**
- `--message`, `-m`: Specify a commit message directly (skips AI generation)
- `--all`, `-a`: Commit all changes (stages untracked files)
- `--model`: LLM model to use for message generation (default: `openai/gpt-4o-mini`). Overrides config (`commit.llm.model`).
- `--strategy`, `-s`: Strategy for splitting diffs (default: `semantic`). Options: `file`, `hunk`, `semantic`. Overrides config (`commit.strategy`).
- `--non-interactive`: Run in non-interactive mode (accepts all generated messages)
- `--bypass-hooks`: Bypass git hooks with `--no-verify` (overrides config `commit.bypass_hooks`).
- `--verbose`, `-v`: Enable verbose logging

### Interactive Workflow

The commit command provides an interactive workflow that:
1. Analyzes your changes and splits them into logical chunks
2. Generates AI-powered commit messages for each chunk
3. Allows you to:
   - Accept the generated message
   - Edit the message before committing
   - Regenerate the message
   - Skip the chunk
   - Exit the process

### Commit Linting Feature

CodeMap includes automatic commit message linting to ensure your commit messages follow conventions:

1. **Automatic Validation**: Generated commit messages are automatically validated against conventional commit standards.
2. **Linting Rules**:
   - Type must be one of the allowed types (configurable in `.codemap.yml`)
   - Type must be lowercase
   - Subject must not end with a period
   - Subject must be at least 10 characters long
   - Header line should not exceed the configured maximum length (default: 72 characters)
   - Scope must be in lowercase (if provided)
   - Header must have a space after the colon
   - Description must start with an imperative verb

3. **Auto-remediation**: If a generated message fails linting, CodeMap will:
   - Identify the specific issues with the message
   - Automatically attempt to regenerate a compliant message (up to 3 attempts)
   - Provide feedback during regeneration with a loading spinner

4. **Fallback Mechanism**: If all regeneration attempts fail, the last message will be used, with linting status indicated.

### Commit Strategy

The tool uses semantic analysis to group related changes together based on:
- File relationships (e.g., implementation files with their tests)
- Code content similarity
- Directory structure
- Common file patterns

> [!Note]
> The semantic strategy utilizes a custom, distilled version of the `Qodo/Qodo-Embed-1-1.5B` model, named `Qodo-Embed-M-1-1.5B-M2V-Distilled`.
> This [Model2Vec](https://github.com/MinishLab/model2vec) distilled model is significantly smaller (233MB vs 5.9GB) and faster (~112x) than the original while retaining ~85% of its performance, making semantic analysis efficient.
> You can find more details [here](https://huggingface.co/sarthak1/Qodo-Embed-M-1-1.5B-M2V-Distilled).

### Environment Variables

The following environment variables can be used to configure the commit command:
- `OPENAI_API_KEY`: OpenAI API key (default LLM provider)
- `ANTHROPIC_API_KEY`: Anthropic API key
- `GROQ_API_KEY`: Groq API key
- `MISTRAL_API_KEY`: Mistral API key
- `COHERE_API_KEY`: Cohere API key
- `TOGETHER_API_KEY`: Together API key
- `OPENROUTER_API_KEY`: OpenRouter API key

### Examples

```bash
# Basic interactive commit
codemap commit

# Commit specific files
codemap commit path/to/file.py

# Use a specific model with custom strategy
codemap commit --model anthropic/claude-3-sonnet --strategy semantic

# Non-interactive commit with all changes
codemap commit -a --non-interactive

# Commit with verbose logging
codemap commit -v

# Demonstrate automatic linting and regeneration
codemap commit --verbose  # Will show linting feedback and regeneration attempts
```

## PR Command Feature

The `codemap pr` command helps you create and manage pull requests with ease. It integrates with the existing `codemap commit` command to provide a seamless workflow from code changes to pull request creation.

### PR Command Features

- Create branches with intelligent naming based on your current changes
- Support for multiple Git workflow strategies (GitHub Flow, GitFlow, Trunk-Based)
- Rich branch visualization with metadata and relationships
- Smart base branch selection based on branch type
- Automatic content generation for different PR types (feature, release, hotfix)
- **Workflow-specific PR templates based on branch type**
- Interactive PR content editing with previews
- Update existing PRs with new commits
- Configurable via `.codemap.yml` for team-wide settings

### PR Command Requirements

- Git repository with a remote named `origin`
- GitHub CLI (`gh`) installed for PR creation and management
- Valid GitHub authentication for the `gh` CLI

### Creating a PR

```bash
codemap pr create [PATH] [OPTIONS]
# Or using the alias:
cm pr create [PATH] [OPTIONS]
```

**Arguments:**
- `PATH`: Path to the codebase to analyze (defaults to current directory)

**Options:**
- `--branch`, `-b`: Target branch name
- `--type`, `-t`: Branch type (e.g., feature, release, hotfix, bugfix). Valid types depend on workflow strategy.
- `--base`: Base branch for the PR (defaults to repo default or workflow-defined default)
- `--title`: Pull request title
- `--desc`, `-d`: Pull request description (file path or text)
- `--no-commit`: Skip the commit process before creating PR
- `--force-push`, `-f`: Force push the branch
- `--workflow`, `-w`: Git workflow strategy (github-flow, gitflow, trunk-based). Overrides config (`pr.strategy`).
- `--non-interactive`: Run in non-interactive mode
- `--model`, `-m`: LLM model for content generation (overrides config `llm.model`).
- `--verbose`, `-v`: Enable verbose logging

### Updating a PR

```bash
codemap pr update [PATH] [OPTIONS]
# Or using the alias:
cm pr update [PATH] [OPTIONS]
```

**Arguments:**
- `PATH`: Path to the codebase to analyze (defaults to current directory)

**Options:**
- `--pr`: PR number to update (required if not updating PR for current branch)
- `--title`: New PR title
- `--desc`, `-d`: New PR description (file path or text)
> [!Note]
> --no-commit is NOT an option for 'update'
- `--force-push`, `-f`: Force push the branch (use with caution)
- `--non-interactive`: Run in non-interactive mode
- `--verbose`, `-v`: Enable verbose logging

### Git Workflow Strategies

The PR command supports multiple Git workflow strategies:

1. **GitHub Flow** (default)
   - Simple, linear workflow
   - Feature branches merge directly to main
   
2. **GitFlow**
   - Feature branches → develop
   - Release branches → main
   - Hotfix branches → main (with back-merge to develop)
   
3. **Trunk-Based Development**
   - Short-lived feature branches
   - Emphasizes small, frequent PRs

### PR Template System

CodeMap includes a robust PR template system that automatically generates appropriate titles and descriptions based on:
1. The selected workflow strategy (GitHub Flow, GitFlow, Trunk-Based)
2. The branch type (feature, release, hotfix, bugfix)
3. The changes being made

#### Workflow-Specific Templates

Each Git workflow strategy provides specialized templates:

**GitHub Flow Templates**
- Simple, general-purpose templates
- Focus on changes and testing
- Example format: `{description}` for title, structured sections for description

**GitFlow Templates**
- Specialized templates for each branch type:
  - **Feature**: Focus on new functionality with implementation details
  - **Release**: Structured release notes with features, bug fixes, and breaking changes
  - **Hotfix**: Emergency fix templates with impact analysis
  - **Bugfix**: Templates focused on bug description, root cause, and testing

**Trunk-Based Templates**
- Concise templates for short-lived branches
- Focus on quick implementation and rollout plans
- Emphasis on testing and deployment strategies

#### Template Configuration

In your `.codemap.yml`, you can configure how templates are used:

```yaml
pr:
  # Content generation settings
  generate:
    title_strategy: "template"  # Options: commits, llm, template
    description_strategy: "template"  # Options: commits, llm, template
    use_workflow_templates: true  # Use workflow-specific templates (default: true)
    
    # Custom template (used when use_workflow_templates is false)
    description_template: |
      ## Changes
      {description}
      
      ## Testing
      - [ ] Unit tests
      - [ ] Integration tests
      
      ## Additional Notes
      
      ## Related Issues
      Closes #
```

**Configuration Options:**
- `title_strategy`: How PR titles are generated
  - `commits`: Generate from commit messages
  - `llm`: Use AI to generate titles
  - `template`: Use workflow-specific templates
  
- `description_strategy`: How PR descriptions are generated
  - `commits`: Generate structured content from commit messages
  - `llm`: Use AI to generate descriptions
  - `template`: Use workflow-specific templates
  
- `use_workflow_templates`: Whether to use built-in templates for each workflow strategy
  - When `true`: Uses the appropriate template based on workflow and branch type
  - When `false`: Uses the custom template defined in `description_template`

- `description_template`: Custom template with placeholder variables
  - `{description}`: Brief description of changes
  - `{changes}`: List of changes from commits
  - `{user}`: Current Git user
  - Supports any Markdown formatting

### Examples

```bash
# Create PR using workflow-specific templates (GitFlow)
codemap pr create --workflow gitflow --type feature

# Create PR with custom title but workflow-based description
codemap pr create --title "My Custom Title" --workflow trunk-based

# Override both the workflow template and use custom description
codemap pr create --desc "Custom description with **markdown** support"

# Non-interactive PR creation with defined template usage
codemap pr create --non-interactive --workflow gitflow --type release
```

## LLM Provider Support

CodeMap supports multiple LLM providers through LiteLLM:

```bash
# Using OpenAI (default)
codemap commit --model openai/gpt-4o-mini
# Or using the alias:
cm commit --model openai/gpt-4o-mini

# Using Anthropic
codemap commit --model anthropic/claude-3-sonnet-20240229

# Using Groq (recommended for speed)
codemap commit --model groq/llama-3.1-8b-instant

# Using OpenRouter
codemap commit --model openrouter/meta-llama/llama-3-8b-instruct
```

## Configuration

Create a `.codemap.yml` file in your project root to customize the behavior. Below are all available configuration options with their default values from `config.py`:

```yaml
# LLM configuration (applies globally unless overridden by command-specific LLM config)
llm:
  model: openai/gpt-4o-mini  # Default LLM model (provider/model_name format)
  api_base: null             # Custom API base URL (e.g., for local LLMs or proxies)

# Documentation Generation Settings ('gen' command)
gen:
  max_content_length: 5000       # Max content length per file (0 = unlimited)
  use_gitignore: true            # Respect .gitignore patterns
  output_dir: documentation       # Directory for generated docs
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

# Processor configuration (background analysis)
processor:
  enabled: true                  # Enable background processing (currently unused)
  max_workers: 4                 # Max parallel workers for analysis
  ignored_patterns:              # Patterns to ignore during analysis
    - "**/.git/**"
    - "**/__pycache__/**"
    - "**/.venv/**"
    - "**/node_modules/**"
    - "**/*.pyc"
    - "**/dist/**"
    - "**/build/**"
  default_lod_level: signatures  # Default LOD for background processing

# Commit Feature Configuration ('commit' command)
commit:
  strategy: semantic             # Diff splitting strategy: file, hunk, semantic
  bypass_hooks: false            # Default for --bypass-hooks flag (--no-verify)
  # Note: 'commit.llm' config is deprecated; use global 'llm' section

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
    # Header rules
    header_max_length: { level: ERROR, rule: always, value: 100 }
    header_case: { level: DISABLED, rule: always, value: lower-case }
    header_full_stop: { level: ERROR, rule: never, value: . }
    # Type rules
    type_enum: { level: ERROR, rule: always } # Uses types from commit.convention.types
    type_case: { level: ERROR, rule: always, value: lower-case }
    type_empty: { level: ERROR, rule: never }
    # Scope rules
    scope_case: { level: ERROR, rule: always, value: lower-case }
    scope_empty: { level: DISABLED, rule: never }
    scope_enum: { level: DISABLED, rule: always } # Uses scopes from commit.convention.scopes
    # Subject rules
    subject_case: { level: ERROR, rule: never, value: [sentence-case, start-case, pascal-case, upper-case] }
    subject_empty: { level: ERROR, rule: never }
    subject_full_stop: { level: ERROR, rule: never, value: . }
    subject_exclamation_mark: { level: DISABLED, rule: never }
    # Body rules
    body_leading_blank: { level: WARNING, rule: always }
    body_empty: { level: DISABLED, rule: never }
    body_max_line_length: { level: ERROR, rule: always, value: 100 }
    # Footer rules
    footer_leading_blank: { level: WARNING, rule: always }
    footer_empty: { level: DISABLED, rule: never }
    footer_max_line_length: { level: ERROR, rule: always, value: 100 }

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
    # Template used if description_strategy is 'template' AND use_workflow_templates is false
    description_template: |
      ## Changes
      {changes}

      ## Testing
      {testing_instructions}

      ## Screenshots
      {screenshots}
    use_workflow_templates: true # Use built-in templates based on workflow/branch type?
```

### Configuration Priority

The configuration is loaded in the following order (later sources override earlier ones):
1. Default configuration from the package
2. `.codemap.yml` in the project root
3. Custom config file specified with `--config`
4. Command-line arguments

### Environment Variables

Configuration can also be influenced by environment variables. Create a `.env` or `.env.local` file:

```env
# LLM Provider API Keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here

# Optional: Custom API Base URLs
OPENAI_API_BASE=your_custom_url
ANTHROPIC_API_BASE=your_custom_url
```

### Configuration Tips

1. **Token Limits (Deprecated)**: The `token_limit` setting is no longer used. Use `max_content_length` to control file content size.

2. **Git Integration**
    *   `gen.use_gitignore: true` respects your `.gitignore`.
    *   Configure `commit.convention.scopes` to match your project structure or leave empty for auto-derivation.
    *   Use `commit.bypass_hooks` or the `--bypass-hooks` flag to skip Git hooks.

3. **LLM Settings**
    *   Use the global `llm` section for default model and API base.
    *   Override with `--model` in `commit` or `pr` commands.
    *   Set custom API bases (e.g., `llm.api_base`) for self-hosted or proxy services.

4. **Commit Conventions & Linting**
    *   Customize `commit.convention.types` and `commit.convention.scopes`.
    *   Adjust `commit.convention.max_length`.
    *   Configure `commit.lint` rules (based on commitlint standard). The `linting` section mentioned previously is now under `commit.lint`.

5. **PR Workflow Settings**
    *   Choose a `pr.strategy` (`github-flow`, `gitflow`, `trunk-based`).
    *   Configure `pr.defaults` and `pr.branch_mapping` (especially for GitFlow).
    *   Customize PR content generation (`pr.generate.title_strategy`, `pr.generate.description_strategy`).
    *   Use `pr.generate.use_workflow_templates` to control template usage. Define a fallback in `pr.generate.description_template`.

6. **Documentation Generation (`gen`)**
    *   Control detail level with `gen.lod_level` or `--lod`.
    *   Enable/disable semantic analysis (`gen.semantic_analysis`, `--semantic`).
    *   Toggle the directory tree (`gen.include_tree`, `--tree`).
    *   Configure the Mermaid entity graph (`gen.include_entity_graph`, `gen.mermaid_*` options, and corresponding flags).

### Output Structure

The generated documentation includes:
1. Project overview and structure
2. Directory tree visualization
3. Token-optimized code summaries
4. File relationships and dependencies
5. Rich markdown formatting with syntax highlighting

### File Processing

The generator:
- Respects `.gitignore` patterns by default
- Intelligently analyzes code structure
- Optimizes content for token limits
- Generates well-structured markdown
- Handles various file types and languages

## Development Setup

Before contributing, please read our [Code of Conduct](.github/CODE_OF_CONDUCT.md) and [Contributing Guidelines](.github/CONTRIBUTING.md).

1.  **Clone the repository:**
     ```bash
     git clone https://github.com/SarthakMishra/codemap.git
     cd codemap
     ```

2.  **Install Prerequisites:**
     *   **Task:** Follow the official installation guide: [https://taskfile.dev/installation/](https://taskfile.dev/installation/)
     *   **uv:** Install the `uv` package manager. We recommend using `pipx`:
         ```bash
         # Using pipx (recommended)
         pipx install uv

         # Or using pip
         # pip install uv
         ```
     *   **Python:** Ensure you have Python 3.12 or later installed.

3.  **Set up the Virtual Environment:**
     ```bash
     # Create a virtual environment using uv (creates .venv directory)
     uv venv

     # Activate the virtual environment
     # On Linux/macOS (bash/zsh):
     source .venv/bin/activate
     # On Windows (Command Prompt):
     # .venv\Scripts\activate.bat
     # On Windows (PowerShell):
     # .venv\Scripts\Activate.ps1
     ```

4.  **Install Dependencies:**
     Install project dependencies, including development tools, using `uv`:
     ```bash
     # Installs dependencies from pyproject.toml including the 'dev' group
     uv sync --dev
     ```

5.  **Verify Setup:**
     You can list available development tasks using Task:
     ```bash
     task -l
     ```
     To run all checks and tests (similar to CI):
     ```bash
     task ci
     ```

For detailed contribution guidelines, branching strategy, and coding standards, please refer to our [Contributing Guide](.github/CONTRIBUTING.md).

## Acknowledgments

CodeMap relies on these excellent open-source libraries and models:

### Core Dependencies
* [LiteLLM](https://github.com/BerriAI/litellm) (>=1.67.0) - Unified interface for LLM providers
* [NumPy](https://numpy.org/) (>=2.2.5) - Numerical computing for vector operations
* [Pygments](https://pygments.org/) (>=2.19.1) - Syntax highlighting for code snippets
* [Python-dotenv](https://github.com/theskumar/python-dotenv) (>=1.1.0) - Environment variable management
* [PyYAML](https://pyyaml.org/) (>=6.0.2) - YAML parsing and configuration management
* [Questionary](https://github.com/tmbo/questionary) (>=2.1.0) - Interactive user prompts
* [Requests](https://requests.readthedocs.io/) (>=2.32.3) - HTTP library for API interactions
* [Rich](https://rich.readthedocs.io/) (>=14.0.0) - Beautiful terminal formatting and output
* [Typer](https://typer.tiangolo.com/) (>=0.15.2) - Modern CLI framework for Python
* [Typing Extensions](https://github.com/python/typing_extensions) (>=4.13.2) - Backported typing features
* [Sentence-Transformers](https://www.sbert.net/) (>=4.1.0) - Text embeddings for semantic code analysis

### Development Tools
* [isort](https://pycqa.github.io/isort/) (>=6.0.1) - Import sorting
* [pylint](https://pylint.readthedocs.io/) (>=3.3.6) - Code analysis
* [pyright](https://github.com/microsoft/pyright) (>=1.1.399) - Static type checking
* [pytest](https://docs.pytest.org/) (>=8.3.5) - Testing framework
* [pytest-cov](https://pytest-cov.readthedocs.io/) (>=6.1.1) - Test coverage reporting
* [ruff](https://github.com/astral-sh/ruff) (>=0.11.6) - Fast Python linter

### Models
* **Code Embeddings**: [Qodo/Qodo-Embed-1-1.5B](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B) - State-of-the-art embedding model optimized for code retrieval tasks.
* **LLM Support**: Compatible with various providers through LiteLLM including:
  - OpenAI models
  - Anthropic Claude models
  - Groq models
  - Mistral models
  - Cohere models
  - Together AI models
  - OpenRouter providers

### Special Thanks
* [Cursor](https://www.cursor.com/)
* [OpenHands](https://github.com/All-Hands-AI/OpenHands)
* [GitHub Actions](https://github.com/features/actions)
* [Img Shields](https://shields.io)
* [Codecov](https://about.codecov.io/)

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
