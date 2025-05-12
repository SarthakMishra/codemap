# Ask Questions (`ask`)

Ask questions about your codebase using Retrieval-Augmented Generation (RAG) and AI chat. This command lets you query your codebase for explanations, architecture, usage, and more, either in single-question or interactive chat mode.

## Command Options

```bash
codemap ask [QUESTION] [OPTIONS]
# Or using the alias:
cm ask [QUESTION] [OPTIONS]
```

**Arguments:**

- `QUESTION`: Your question about the codebase (optional if using interactive mode)

**Options:**

- `--interactive`, `-i`: Start an interactive chat session (multi-turn Q&A)
- `--verbose`, `-v`: Enable verbose logging

## Examples

```bash
# Ask a single question about the codebase
codemap ask "Which module manages authentication?"

# Start an interactive chat session
cm ask --interactive

# Use the alias for a quick question
cm ask "How does the vector index work?"
```

## Features

- Answers questions using semantic search and LLMs (RAG)
- Interactive chat mode for multi-turn conversations
- Supports both single-question and chat workflows
- Respects project configuration and environment variables