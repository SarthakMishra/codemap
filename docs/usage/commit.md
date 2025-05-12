# Smart Commit (`commit`)

Create intelligent Git commits with AI-assisted message generation. The tool analyzes your changes, splits them into logical chunks, and generates meaningful commit messages using LLMs.

## Basic Usage

```bash
# Basic usage with default settings (interactive, semantic splitting)
codemap commit
# Or using the alias:
cm commit

# Commit specific files or directories
codemap commit path/to/file.py path/to/dir/

# Run in non-interactive mode (accepts all generated messages)
codemap commit --non-interactive

# Bypass git hooks (e.g., pre-commit)
codemap commit --bypass-hooks
```

## Command Options

```bash
codemap commit [PATHS] [OPTIONS]
```

**Arguments:**

- `PATHS`: Optional. One or more files or directories to include in the commit (defaults to all staged changes).

**Options:**

- `--non-interactive`, `-y`: Run in non-interactive mode (accepts all generated messages)
- `--bypass-hooks`, `--no-verify`: Bypass git hooks with `--no-verify`
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

## Examples

```bash
# Basic interactive commit
codemap commit

# Commit specific files
codemap commit path/to/file.py

# Non-interactive commit with all changes
codemap commit --non-interactive

# Commit with verbose logging
codemap commit -v

# Bypass git hooks
codemap commit --bypass-hooks
```

## Notes
- Direct commit message input (`--message`), model selection (`--model`), and diff strategy (`--strategy`) are not available as CLI options. These can be configured in `.codemap.yml`.
- The command is designed for semantic, AI-powered commit workflows.