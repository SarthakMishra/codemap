# CodeMap

⚠️ **DEVELOPMENT STATUS**: CodeMap is currently in active development. Use with caution in production environments.

CodeMap is a CLI tool that generates optimized markdown documentation from your Python codebase. It analyzes source code, creates repository maps, and produces markdown files that can be used as context for LLMs.

## Features

- 🎯 Token-optimized documentation generation
- 📝 Rich markdown output with code structure
- 🌳 Repository structure visualization
- 🔄 Smart Git commit assistance with AI-generated messages

## Installation

```bash
# Install with pipx
pipx install git+https://github.com/SarthakMishra/codemap.git

# Upgrade
pipx upgrade codemap
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
# Basic usage (file-based strategy)
codemap commit

# With semantic strategy for code-aware chunking
codemap commit --strategy semantic

# Specify a different LLM model
codemap commit --model gpt-4
```

### Commit Strategies

- **File Strategy** (`--strategy file`, default): One commit per changed file
- **Hunk Strategy** (`--strategy hunk`): Splits by code hunks within files
- **Semantic Strategy** (`--strategy semantic`): Groups related files using AI code understanding

### LLM Provider Support

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
  strategy: file
  llm:
    model: gpt-4o-mini
    provider: openai
  
  # Semantic chunking settings (optional)
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