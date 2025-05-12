# Index Codebase (`index`)

The `index` command processes your repository, generates code embeddings, and stores them in a vector database for semantic search and Retrieval-Augmented Generation (RAG). It can also watch for file changes and keep the index up to date.

## Command Options

```bash
codemap index [PATH] [OPTIONS]
# Or using the alias:
cm index [PATH] [OPTIONS]
```

**Arguments:**

- `PATH`: Path to the repository root directory (defaults to current directory)

**Options:**

- `--sync/--no-sync`: Synchronize the vector database with the current Git state on startup (default: sync enabled)
- `--watch`, `-w`: Keep running and watch for file changes, automatically syncing the index
- `--verbose`, `-v`: Enable verbose logging

## Examples

```bash
# Index the current repository and sync with Git state
codemap index

# Index a specific directory and watch for changes
cm index /path/to/repo --watch

# Index without syncing to Git state (faster, but may miss changes)
codemap index --no-sync
```

## Features

- Processes your codebase and builds a semantic vector index
- Supports background file watching for live updates
- Integrates with CodeMap's RAG and AI chat features
- Respects configuration and .gitignore patterns

Keeping your index up to date ensures the best results for AI-powered search and question answering. 