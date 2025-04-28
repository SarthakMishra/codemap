# CodeMap

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/SarthakMishra/codemap/actions/workflows/tests.yml/badge.svg)](https://github.com/SarthakMishra/code-map/actions/workflows/tests.yml)
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

> [!Note]
> This package will be published to PyPI upon reaching a stable version.

> [!Important]
> CodeMap currently only supports Unix-based platforms (macOS, Linux). For Windows users, we recommend using Windows Subsystem for Linux (WSL).

### Easy Install (Recommended)

```bash
# One-line installer (downloads and runs install.sh)
curl -sSL https://raw.githubusercontent.com/SarthakMishra/codemap/main/install.sh | bash

# Or download and run manually
curl -sSL -o install.sh https://raw.githubusercontent.com/SarthakMishra/codemap/main/install.sh
chmod +x install.sh
./install.sh
```

### Manual Installation

If you prefer to install manually:

```bash
# Install with pip (user installation)
pip install --user git+https://github.com/SarthakMishra/codemap.git

# Make sure your PATH includes the user bin directory
# Usually this is ~/.local/bin on Linux or ~/Library/Python/X.Y/bin on macOS
```

### Updating CodeMap

CodeMap automatically checks for updates and will notify you when a new version is available.

```bash
# Check for updates
codemap version

# Update to the latest version
codemap update

# Alternatively, use the package management commands
codemap pkg update
```

### Uninstalling

```bash
# Uninstall CodeMap
codemap pkg uninstall

# Or manually with pip
pip uninstall codemap
```

## Generate Markdown Docs

Generate optimized markdown documentation and directory structures for your project:

### Command Options

```bash
codemap generate [PATH] [OPTIONS]
```

**Arguments:**
- `PATH`: Path to the codebase to document (defaults to current directory)

**Options:**
- `--output`, `-o`: Output file path for the documentation
- `--config`, `-c`: Path to custom configuration file
- `--map-tokens`: Override token limit for documentation (default: 10000)
- `--max-content-length`: Maximum content length for each file (default: 2000)
- `--tree`: Generate only the directory tree structure
- `--verbose`, `-v`: Enable verbose logging

### Examples

```bash
# Generate documentation for current directory
codemap generate

# Generate documentation for a specific path
codemap generate /path/to/your/project

# Generate only directory tree
codemap generate --tree /path/to/project

# Custom output location
codemap generate -o ./docs/codebase.md

# Override token limits
codemap generate --map-tokens 5000 --max-content-length 1500

# Use custom configuration
codemap generate --config custom-config.yml

# Verbose mode for debugging
codemap generate -v
```

## Smart Commit Feature

Create intelligent Git commits with AI-assisted message generation. The tool analyzes your changes, splits them into logical chunks, and generates meaningful commit messages using LLMs.

> [!Warning]
> **Known Issue:** The commit command may sometimes incorrectly identify test files containing diff code as actual diff files, causing Git commit operations to fail. If you encounter this error, simply re-run the command or use standard Git commit as a workaround. This issue will be fixed in an upcoming release.

### Basic Usage

```bash
# Basic usage with default settings
codemap commit

# Commit with a specific message
codemap commit -m "feat: add new feature"

# Commit all changes (including untracked files)
codemap commit -a

# Use a specific LLM model
codemap commit --model groq/llama-3-8b-instruct
```

### Command Options

```bash
codemap commit [PATH] [OPTIONS]
```

**Arguments:**
- `PATH`: Path to repository or specific file to commit (defaults to current directory)

**Options:**
- `--message`, `-m`: Specify a commit message directly
- `--all`, `-a`: Commit all changes (including untracked files)
- `--model`: LLM model to use for message generation (default: gpt-4o-mini)
- `--strategy`, `-s`: Strategy for splitting diffs (default: semantic)
- `--non-interactive`: Run in non-interactive mode
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
```

**Options:**
- `--branch`, `-b`: Target branch name
- `--type`, `-t`: Branch type (feature, release, hotfix, bugfix)
- `--base`: Base branch for the PR (defaults to repo default)
- `--title`: Pull request title
- `--desc`, `-d`: Pull request description (file path or text)
- `--no-commit`: Skip the commit process before creating PR
- `--force-push`, `-f`: Force push the branch
- `--workflow`, `-w`: Git workflow strategy (github-flow, gitflow, trunk-based)
- `--non-interactive`: Run in non-interactive mode
- `--model`, `-m`: LLM model for content generation
- `--verbose`, `-v`: Enable verbose logging

### Updating a PR

```bash
codemap pr update [PATH] [OPTIONS]
```

**Options:**
- `--pr`: PR number to update
- `--title`: New PR title
- `--desc`, `-d`: New PR description
- `--no-commit`: Skip the commit process before updating PR
- `--force-push`, `-f`: Force push the branch
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

# Using Anthropic
codemap commit --model anthropic/claude-3-sonnet-20240229

# Using Groq (recommended for speed)
codemap commit --model groq/llama-3.1-8b-instant

# Using OpenRouter
codemap commit --model openrouter/meta-llama/llama-3-8b-instruct
```

## Configuration

Create a `.codemap.yml` file in your project root to customize the behavior. Below are all available configuration options with their default values:

```yaml
# Documentation Generation Settings
token_limit: 10000              # Maximum tokens for entire documentation (0 for unlimited)
use_gitignore: true            # Whether to respect .gitignore patterns
output_dir: documentation       # Directory to store documentation files
max_content_length: 5000       # Maximum content length per file (0 for unlimited)

# Commit Feature Configuration
commit:
  # Strategy for splitting diffs: file, hunk, semantic
  strategy: file

  # LLM Configuration
  llm:
    model: openai/gpt-4o-mini  # Default LLM model
    api_base: null             # Custom API base URL (optional)

  # Commit Convention Settings
  convention:
    # Available commit types
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

    # Optional scopes for your project
    scopes: []   # Define custom scopes or leave empty to derive from directory structure

    # Maximum length for commit message subject line
    max_length: 72
    
    # Commit linting configuration (optional)
    linting:
      enabled: true                # Enable/disable commit message linting (default: true)
      auto_fix: true               # Attempt to fix non-compliant messages (default: true)
      max_regeneration_attempts: 3 # Maximum attempts to regenerate a compliant message (default: 3)
      rules:
        subject_min_length: 10     # Minimum length for commit subject (default: 10)
        imperative_subject: true   # Enforce imperative mood in subject (default: true)
        no_subject_period: true    # No period at the end of subject (default: true)
        scope_lowercase: true      # Enforce lowercase scope (default: true)
        type_in_allowed: true      # Type must be in allowed types list (default: true)

# Pull Request Configuration
pr:
  # Default branch settings
  defaults:
    base_branch: main
    feature_prefix: "feature/"
    
  # Git workflow strategy
  strategy: "github-flow"  # Options: github-flow, gitflow, trunk-based
  
  # Branch mapping for different PR types
  branch_mapping:
    feature:
      base: develop
      prefix: "feature/"
    bugfix:
      base: develop
      prefix: "bugfix/"
    release:
      base: main
      prefix: "release/"
    hotfix:
      base: main
      prefix: "hotfix/"
      
  # Content generation
  generate:
    title_strategy: "commits"  # Options: commits, llm, branch-name
    description_strategy: "llm"  # Options: commits, llm, template
    template:
      sections:
        - title: "Changes"
          content: "{changes}"
        - title: "Testing"
          content: "Tested by: {user}"
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

1. **Token Limits**
   - Set `token_limit: 0` for unlimited tokens (careful with large codebases)
   - Adjust `max_content_length` based on your file sizes

2. **Git Integration**
   - `use_gitignore: true` respects your `.gitignore` patterns
   - Configure commit scopes to match your project structure

3. **LLM Settings**
   - Choose provider-specific models (e.g., `openai/gpt-4`, `anthropic/claude-3`)
   - Set custom API bases for self-hosted or proxy services

4. **Commit Conventions**
   - Add custom commit types to match your workflow
   - Adjust `max_length` to enforce commit message style
   - Define scopes to categorize changes
   - Commit linter automatically validates messages against convention rules
   - Configure custom linting rules in the `.codemap.yml` convention section
   - Control linting behavior with the `linting` configuration:
     - Enable/disable linting with `enabled: true|false`
     - Set auto-fixing of non-compliant messages with `auto_fix: true|false`
     - Control regeneration attempts with `max_regeneration_attempts`
     - Fine-tune specific rules in the `rules` subsection

5. **PR Workflow Settings**
   - Choose from different Git workflow strategies (`github-flow`, `gitflow`, `trunk-based`)
   - Configure default base branches for different PR types
   - Set branch prefixes for consistent naming (`feature/`, `bugfix/`, etc.)
   - Customize PR content generation with different strategies
   - Define PR description templates with custom sections
   - Control PR creation and update behavior

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
