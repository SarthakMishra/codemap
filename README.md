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

CodeMap is an AI-powered developer toolkit designed to enhance your coding workflow. It offers features for code analysis, documentation generation, semantic search, and Git process streamlining—all accessible through an interactive CLI with multi-LLM support.

> [!Important]
> For detailed information on all features and commands, please visit our documentation site: **[codemap.run](https://codemap.run)**

## Features

- 📄 **Generate Documentation:** Create optimized markdown documentation and visualize repository structures.
- 📝 **Smart Commits:** Get AI-generated commit messages based on semantic analysis of your changes.
- 🔃 **AI-Powered PRs:** Streamline pull request creation and management with intelligent suggestions.
- 💬 **AI Chat:** Ask questions about your codebase using RAG and LLMs.
- 🔍 **Index & Search:** Build a semantic vector index and search your repository for deep code understanding.
- 🤖 **LLM Support:** Integrate with various LLM providers via [PydanticAI](https://ai.pydantic.dev/models/).

## Quick Start

> [!Important]
> CodeMap currently only supports Unix-based platforms (macOS, Linux). Windows users should use WSL.

> [!Tip]
> After installation, use `codemap` or the alias `cm`.

### Installation

Ensure [uv](https://docs.astral.sh/uv/getting-started/installation/) is installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install CodeMap globally:
```bash
uv tool install codemap
```

### Key Commands

- **Generate Documentation:**
  ```bash
  cm gen path/to/source
  ```
- **Smart Commits:**
  ```bash
  cm commit
  ```
- **AI-Powered PRs:**
  ```bash
  cm pr
  ```
- **AI Chat:**
  ```bash
  cm ask "Which module is responsible for managing auth tokens?"
  ```
- **Index & Search:**
  ```bash
  cm index
  ```

### Configuration

CodeMap can be configured using a `.codemap.yml` file in your project root. Generate a default config with:
```bash
cm conf
```

**For a full list of configuration options and examples, refer to the [Configuration Guide](https://codemap.run/usage/configuration/) on our documentation site.**

### Environment Variables

Add your LLM API keys to `.env` or `.env.local` in your project root. See [LLM Support](https://codemap.run/usage/llm-support/) for details.

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

### Core Dependencies
* [PydanticAI](https://ai.pydantic.dev/) - Unified interface for LLM providers
* [Pydantic](https://docs.pydantic.dev/latest/) - Data validation library for Python
* [Questionary](https://github.com/tmbo/questionary) - Interactive user prompts
* [Rich](https://rich.readthedocs.io/) - Beautiful terminal formatting and output
* [Typer](https://typer.tiangolo.com/) - Modern CLI framework for Python
* [Model2Vec](https://github.com/MinishLab/model2vec) - Text embeddings for semantic code analysis
* [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) - Robust parsing system for code analysis
* [SQLModel](https://sqlmodel.tiangolo.com/) - SQL database integration with Python
* [Qdrant](https://qdrant.tech/) - Vector search engine for semantic analysis
* [PyGit2](https://www.pygit2.org/) - Git repository manipulation
* [Scikit-learn](https://scikit-learn.org/) - Machine learning utilities
* [PyGithub](https://pygithub.readthedocs.io/) - GitHub API integration
* [Docker SDK](https://docker-py.readthedocs.io/) - Docker container management
* [Watchdog](https://python-watchdog.readthedocs.io/) - Filesystem event monitoring

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
