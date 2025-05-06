# CodeMap

[![PyPI](https://img.shields.io/pypi/v/codemap)](https://pypi.org/project/codemap/)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Tests](https://github.com/SarthakMishra/codemap/actions/workflows/tests.yml/badge.svg)](https://github.com/SarthakMishra/code-map/actions/workflows/tests.yml)
[![Lint](https://github.com/SarthakMishra/codemap/actions/workflows/lint.yml/badge.svg)](https://github.com/SarthakMishra/codemap/actions/workflows/lint.yml)
[![CodeQL](https://github.com/SarthakMishra/codemap/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/SarthakMishra/codemap/actions/workflows/github-code-scanning/codeql)
[![codecov](https://codecov.io/gh/SarthakMishra/codemap/branch/main/graph/badge.svg)](https://codecov.io/gh/SarthakMishra/codemap)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/95d85720e3a14494abf27b5d2070d92f)](https://app.codacy.com/gh/SarthakMishra/codemap/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Interrogate](docs/assets/interrogate_badge.svg)](https://interrogate.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> [!Caution]
> CodeMap is currently in active development. Use with caution and expect breaking changes.

## Overview

CodeMap is an AI-powered developer toolkit designed to enhance your coding workflow. It offers features for code analysis, documentation generation, and Git process streamlining, all accessible through an interactive CLI with multi-LLM support.

> [!Important]
> For detailed information on all features and commands, please visit our documentation site: **[codemap.run](https://codemap.run)**

## Features

- 🎯 Token-optimized documentation generation
- 📝 Rich markdown output with code structure
- 🌳 Repository structure visualization
- 🔄 Smart Git commit assistance with AI-generated messages
- 🔃 AI-powered PR creation and management
- 💬 Ask questions about your codebase using RAG
- 🔍 Index and search your repository for semantic understanding




## Installation

> [!Important]
> CodeMap currently only supports Unix-based platforms (macOS, Linux). Windows users should use WSL.

> [!Tip]
> After installation, use `codemap` or the alias `cm`.

>[!Note]
> Ensure uv is installed: `curl -LsSf https://astral.sh/uv/install.sh | sh `
> To learn more aboyt uv [Click here](https://docs.astral.sh/uv/getting-started/installation/)

### Global Installation
```bash
# Stable version:
uv tool install codemap
```

```bash
# Development Version:
uv tool install codemap --prerelease allow
```

### Updating

```bash
uv tool upgrade codemap
```

### Uninstalling

```bash
uv tool uninstall codemap
```

**For more detailed installation options and troubleshooting, see the [Installation Guide](https://codemap.run/installation/) on our documentation site.**

## Configuration

CodeMap can be configured using a `.codemap.yml` file in your project root.

**For a full list of configuration options and examples, refer to the [Configuration Guide](https://codemap.run/usage/configuration/) on our documentation site.**

## Development Setup

Interested in contributing? Please read our [Code of Conduct](.github/CODE_OF_CONDUCT.md) and [Contributing Guidelines](.github/CONTRIBUTING.md).

1.  **Clone:** `git clone https://github.com/SarthakMishra/codemap.git && cd codemap`
2.  **Prerequisites:** Install [Task](https://taskfile.dev/installation/), [uv](https://github.com/astral-sh/uv#installation), and Python 3.12+.
3.  **Setup Env:** `uv venv && source .venv/bin/activate` (or appropriate activation command for your shell)
4.  **Install Deps:** `uv sync --dev`
5.  **Verify:** `task -l` lists available tasks. `task ci` runs checks and tests.

**Detailed contribution instructions are in the [Contributing Guide](https://codemap.run/contributing/guidelines/).**

## Acknowledgments

CodeMap relies on these excellent open-source libraries and models:

*   [LiteLLM](https://github.com/BerriAI/litellm)
*   [NumPy](https://numpy.org/)
*   [Pygments](https://pygments.org/)
*   [Python-dotenv](https://github.com/theskumar/python-dotenv)
*   [PyYAML](https://pyyaml.org/)
*   [Questionary](https://github.com/tmbo/questionary)
*   [Requests](https://requests.readthedocs.io/)
*   [Rich](https://rich.readthedocs.io/)
*   [Typer](https://typer.tiangolo.com/)
*   [Sentence-Transformers](https://www.sbert.net/)
*   And many development tools like `ruff`, `pytest`, `isort`, `pyright`.

### Models
*   **Code Embeddings**: Leverages models like [Qodo/Qodo-Embed-1-1.5B](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B).
*   **LLM Support**: Compatible with OpenAI, Anthropic, Groq, Mistral, Cohere, Together AI, OpenRouter via LiteLLM.

### Special Thanks
*   [Cursor](https://www.cursor.com/)
*   [OpenHands](https://github.com/All-Hands-AI/OpenHands)
*   [GitHub Actions](https://github.com/features/actions)
*   [Img Shields](https://shields.io)
*   [Codecov](https://about.codecov.io/)

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
