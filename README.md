# CodeMap

[![Python Version](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-unstable-red.svg)](https://github.com/SarthakMishra/code-map)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/SarthakMishra/codemap/actions/workflows/tests.yml/badge.svg)](https://github.com/SarthakMishra/code-map/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/SarthakMishra/codemap/branch/main/graph/badge.svg)](https://codecov.io/gh/SarthakMishra/codemap)

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

```bash
# Install with pipx
pipx install git+https://github.com/SarthakMishra/codemap.git

# Upgrade
pipx upgrade codemap
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
```

## PR Command Feature

The `codemap pr` command helps you create and manage pull requests with ease. It integrates with the existing `codemap commit` command to provide a seamless workflow from code changes to pull request creation.

### PR Command Features

- Create a new branch by analyzing your changes
- Commit all changes using the pre-existing commit tools
- Push changes to origin
- Generate a PR with appropriate messages by analyzing commits
- Update existing PRs with new commits
- Interactive workflow with helpful prompts

### PR Command Requirements

- Git repository with a remote named `origin`
- GitHub CLI (`gh`) installed for PR creation and management
- Valid GitHub authentication for the `gh` CLI

### Creating a PR

```bash
codemap pr create [PATH] [OPTIONS]
```

**Arguments:**
- `PATH`: Path to the repository (defaults to current directory)

**Options:**
- `--branch`, `-b`: Branch name to use (will be created if it doesn't exist)
- `--base`: Base branch for the PR (default: main or master)
- `--title`, `-t`: PR title (generated from commits if not provided)
- `--description`, `-d`: PR description (generated from commits if not provided)
- `--no-commit`: Don't commit changes before creating PR
- `--force-push`, `-f`: Force push branch to remote
- `--non-interactive`: Run in non-interactive mode
- `--model`, `-m`: LLM model to use for commit message generation
- `--api-key`: API key for LLM provider

### Updating a PR

```bash
codemap pr update [PR_NUMBER] [OPTIONS]
```

**Arguments:**
- `PR_NUMBER`: PR number to update (if not provided, will try to find PR for current branch)

**Options:**
- `--path`, `-p`: Path to repository
- `--title`, `-t`: New PR title
- `--description`, `-d`: New PR description
- `--no-commit`: Don't commit changes before updating PR
- `--force-push`, `-f`: Force push branch to remote
- `--non-interactive`: Run in non-interactive mode
- `--model`, `-m`: LLM model to use for commit message generation
- `--api-key`: API key for LLM provider

### PR Examples

```bash
# Interactive mode (recommended)
codemap pr create

# Specify a branch name
codemap pr create --branch feature-branch

# Create PR with custom title and description
codemap pr create --title "My Feature" --description "This PR adds a new feature"

# Create PR without committing changes
codemap pr create --no-commit

# Update PR by number
codemap pr update 123

# Update PR for current branch
codemap pr update

# Update PR with new title
codemap pr update --title "Updated Feature"
```

### PR Workflow

The typical workflow with the `codemap pr` command is:

1. Make changes to your code
2. Run `codemap pr create`
3. Follow the interactive prompts to:
   - Create or select a branch
   - Commit your changes
   - Push to remote
   - Create a PR with generated title and description
4. Make additional changes
5. Run `codemap pr update` to add new commits and update the PR

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
