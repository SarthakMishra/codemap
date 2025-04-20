# CodeMap

⚠️ **DEVELOPMENT STATUS**: CodeMap is currently in active development. Use with caution in production environments.

CodeMap is a CLI tool that generates optimized markdown documentation from your Python codebase. It analyzes source code, creates repository maps, and produces markdown files that can be used as context for LLMs.

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

## Development Setup

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. To set up pre-commit:

```bash
uv run pre-commit install
```

The pre-commit hooks will:
- Run linting checks with Ruff
- Format your code with Ruff
- Run pytest to ensure tests pass
- Perform common file checks (trailing whitespace, YAML validation, etc.)

To manually run all pre-commit hooks:
```bash
uv run pre-commit run --all-files
```

## Basic Usage

Generate documentation for your project:

```bash
# Basic usage
codemap generate /path/to/your/project

# Generate only directory tree
codemap generate --tree /path/to/your/project

# Verbose mode for debugging
codemap generate -v /path/to/your/project
```

## Smart Commit Feature

Create intelligent Git commits with AI-assisted message generation:

```bash
# Basic usage
codemap commit

# Specify a different LLM model
codemap commit --model gpt-4
```

### Commit Strategy

CodeMap uses a semantic commit strategy that intelligently groups related files together based on:
- Directory structure
- File relationships (e.g., test files with implementation files)
- Code content similarity
- Common file patterns (e.g., frontend components with their styles)

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
codemap commit --model gpt-4.0-mini

# Using Anthropic
codemap commit --model anthropic/claude-3-sonnet-20240229

# Using Groq (recommended for speed)
codemap commit --model groq/meta-llama/llama-4-scout-17b-16e-instruct
```

API keys should be provided via environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Configuration

Create a `.codemap.yml` file in your project root:

```yaml
# Documentation settings
token_limit: 10000
use_gitignore: true
output_dir: documentation
max_content_length: 5000

# Commit feature configuration
commit:
  llm:
    model: gpt-4o-mini
    provider: openai

  # Semantic chunking settings
  semantic:
    similarity_threshold: 0.7
    embedding_model: "flax-sentence-embeddings/st-codesearch-distroberta-base"
```

## Command Options

```bash
# Set custom token limit and output file
codemap generate --map-tokens 5000 -o docs.md /path/to/project

# Use a specific config file
codemap generate --config my-config.yml /path/to/project
```

## Acknowledgments

CodeMap relies on these excellent open-source libraries:

- [LiteLLM](https://github.com/BerriAI/litellm) - Unified interface for LLM providers
- [Typer](https://typer.tiangolo.com/) - Modern CLI framework for Python
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal formatting and output
- [Questionary](https://github.com/tmbo/questionary) - Interactive user prompts
- [PyYAML](https://pyyaml.org/) - YAML parsing and configuration management
- [Pygments](https://pygments.org/) - Syntax highlighting for code snippets
- [Sentence-Transformers](https://www.sbert.net/) - Text embeddings for semantic code analysis
- [Python-dotenv](https://github.com/theskumar/python-dotenv) - Environment variable management
- [NumPy](https://numpy.org/) - Numerical computing for vector operations
- [Requests](https://requests.readthedocs.io/) - HTTP library for API interactions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
