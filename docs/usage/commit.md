# Smart Commit (`commit`)

Create intelligent Git commits with AI-assisted message generation. The tool analyzes your changes, splits them into logical chunks, and generates meaningful commit messages using LLMs.

## Basic Usage

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

## Command Options

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

## Interactive Workflow

The commit command provides an interactive workflow that:
1. Analyzes your changes and splits them into logical chunks
2. Generates AI-powered commit messages for each chunk
3. Allows you to:
   - Accept the generated message
   - Edit the message before committing
   - Regenerate the message
   - Skip the chunk
   - Exit the process

## Commit Linting Feature

CodeMap includes automatic commit message linting to ensure your commit messages follow conventions:

1. **Automatic Validation**: Generated commit messages are automatically validated against conventional commit standards.
2. **Linting Rules**: Configurable in `.codemap.yml` (see [Configuration](configuration.md)).
3. **Auto-remediation**: If a generated message fails linting, CodeMap attempts to regenerate a compliant message.
4. **Fallback Mechanism**: If regeneration fails, the last message is used with linting status indicated.

## Commit Strategy

The tool uses semantic analysis to group related changes together based on:
- File relationships
- Code content similarity
- Directory structure
- Common file patterns

/// note
The semantic strategy utilizes a custom, distilled version of the `Qodo/Qodo-Embed-1-1.5B` model, named `Qodo-Embed-M-1-1.5B-M2V-Distilled`.
This [Model2Vec](https://github.com/MinishLab/model2vec) distilled model is significantly smaller (233MB vs 5.9GB) and faster (~112x) than the original while retaining ~85% of its performance.
Find more details [here](https://huggingface.co/sarthak1/Qodo-Embed-M-1-1.5B-M2V-Distilled).
///

## Environment Variables

Refer to the [LLM Support](llm-support.md) page for relevant environment variables.

## Examples

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