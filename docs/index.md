# CodeMap: AI-Powered Developer Toolkit

CodeMap is an AI-powered developer toolkit designed to enhance your development workflow. It offers features like token-optimized documentation generation, semantic code analysis, and streamlined Git operations with AI assistance.

## Quick Start

/// tip
After installation, you can use either `codemap` or the shorter alias `cm` to run the commands.
///

/// warning
CodeMap currently only supports Unix-based platforms (macOS, Linux). For Windows users, we recommend using Windows Subsystem for Linux (WSL).
///

### Installation

Using `uv` is recommended as it installs the package in an isolated environment and automatically manages the PATH.

```bash
# Stable version:
uv tool install codemap
```

### Key Commands

- **Generate Documentation:** Create optimized markdown documentation and visualize repository structures.
	```bash
	cm gen path/to/source
	```
- **Smart Commits:** Get AI-generated commit messages based on semantic analysis of your changes.
	```bash
	cm commit
	```
- **AI-Powered PRs:** Streamline pull request creation and management with intelligent suggestions.
	```bash
	cm pr
	```
- **AI Chat:** Ask questions about your codebase.
	```bash
	cm ask "Which module is responsible for managing auth tokens?"
	```
- **LLM Support:** Integrate with various LLM providers supported by [PydanticAi](https://ai.pydantic.dev/models/).
	```env
	# CodeMap Environment Variables Example
	# Copy this file to .env or .env.local and add your API keys
	# IMPORTANT: Make sure .env and .env.local are in your .gitignore file!

	# LLM Provider API Keys - Uncomment and add your actual keys
	# OPENAI_API_KEY=sk-...
	# ANTHROPIC_API_KEY=sk-ant-...
	# GROQ_API_KEY=gsk_...
	# AZURE_API_KEY=...
	# MISTRAL_API_KEY=...
	# TOGETHER_API_KEY=...
	# GOOGLE_API_KEY=...
	# OPENROUTER_API_KEY=...
	```