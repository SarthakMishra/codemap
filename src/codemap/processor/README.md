# CodeMap Processor Module

## Overview

This module is a core component of the CodeMap project, responsible for processing source code repositories. It orchestrates a pipeline that involves watching files, chunking code into meaningful semantic units, analyzing these chunks using various techniques (Git history, Language Server Protocol, Tree-sitter syntax trees), generating vector embeddings (inferred functionality), and persisting the processed data for later retrieval and analysis.

**Total Files:** 17
**Languages:** Python (17)

## Features

*   **Automated File Watching:** Monitors repository files for changes (creations, modifications, deletions) using the `watcher` component.
*   **Syntactic Code Chunking:** Uses Tree-sitter (`chunking.tree_sitter`) to parse code and divide it into hierarchical, semantically relevant chunks (functions, classes, methods, etc.).
*   **Git Metadata Analysis:** Extracts version control information (`analysis.git`) for each chunk, including commit history, authorship, and modification timestamps.
*   **LSP-based Semantic Analysis:** Leverages the Language Server Protocol via `multilspy` (`analysis.lsp`) to enrich chunks with semantic information like type details, symbol references, definitions, and hover text.
*   **Persistent Storage:** Stores processed chunks, metadata, and embeddings using a configurable storage backend. Includes an implementation for LanceDB (`storage.lance`).
*   **Orchestration Pipeline:** Manages the flow of processing jobs (`pipeline.py`), handling file events, coordinating analysis steps, and managing concurrency.
*   **Configurable:** Allows customization of storage backends, embedding models (via `EmbeddingConfig`), LSP enablement, and worker concurrency.

## Project Structure

```
├── __init__.py                 # Main module entry point, initialization
├── pipeline.py                 # Orchestrates the processing workflow
├── analysis/                   # Code analysis tools
│   ├── git/                    # Git history and metadata analysis
│   │   ├── __init__.py
│   │   ├── analyzer.py         # GitMetadataAnalyzer implementation
│   │   └── models.py           # GitMetadata data model
│   ├── lsp/                    # Language Server Protocol analysis
│   │   ├── __init__.py
│   │   ├── analyzer.py         # LSPAnalyzer implementation (using multilspy)
│   │   └── models.py           # LSPMetadata, LSPReference, LSPTypeInfo models
│   └── tree_sitter/            # Tree-sitter based syntax analysis
│       ├── __init__.py
│       ├── analyzer.py         # TreeSitterAnalyzer implementation (code not shown)
│       ├── base.py             # Base definitions (e.g., EntityType enum)
│       └── languages/          # Language-specific Tree-sitter configurations
│           └── typescript.py   # Example: TypeScript configuration
├── chunking/                   # Code chunking strategies
│   ├── __init__.py
│   ├── base.py                 # Base Chunk, ChunkMetadata, Location models (code not shown)
│   └── tree_sitter.py          # TreeSitterChunker implementation (code not shown)
├── embedding/                  # Vector embedding generation (code not shown)
│   ├── __init__.py             # (Inferred structure)
│   ├── generator.py            # (Inferred: EmbeddingGenerator)
│   └── models.py               # (Inferred: EmbeddingConfig, EmbeddingResult)
├── storage/                    # Data persistence layer
│   ├── __init__.py
│   ├── base.py                 # StorageBackend interface, StorageConfig model
│   ├── lance.py                # LanceDB storage implementation
│   └── utils.py                # Serialization/deserialization helpers
└── watcher/                    # File system monitoring
    ├── __init__.py
    └── base.py                 # FileWatcher, FileEventHandler base (code not shown)
```

## Core Components

*   **`pipeline.ProcessingPipeline`**: The central orchestrator. It initializes all necessary components (watcher, chunker, analyzers, storage, embedder), manages a thread pool for concurrent processing, and handles file events received from the watcher.
*   **`chunking.TreeSitterChunker`**: Parses source code using Tree-sitter grammars and language-specific configurations (`analysis.tree_sitter.languages`) to extract hierarchical `Chunk` objects.
*   **`analysis.git.GitMetadataAnalyzer`**: Interacts with the `git` command-line tool to retrieve blame information and commit details relevant to code chunks.
*   **`analysis.lsp.LSPAnalyzer`**: Communicates with language servers (via `multilspy`) to perform semantic analysis, finding references, definitions, type information, etc.
*   **`embedding.EmbeddingGenerator`** (Inferred): Responsible for taking code chunks (potentially with their metadata) and generating vector embeddings using a specified model (configured via `EmbeddingConfig`).
*   **`storage.StorageBackend`**: An abstract base class defining the interface for storing and retrieving chunks, embeddings, and metadata.
*   **`storage.LanceDBStorage`**: A concrete implementation of `StorageBackend` using LanceDB for efficient vector storage and retrieval.
*   **`watcher.FileWatcher`**: Monitors the file system for changes within the target repository and triggers processing jobs in the pipeline.

## Core Concepts

*   **`Chunk`**: The fundamental unit of processed code. Represents a semantic block like a function, class, or method, containing its code content, location, type (`EntityType`), and associated metadata.
*   **`ChunkMetadata`**: Contains information about a chunk, such as its name, type, language, location within the file, dependencies, and other attributes.
*   **`EntityType`**: An enumeration (`analysis.tree_sitter.base.EntityType`) defining the different kinds of code structures that can be identified (e.g., `CLASS`, `FUNCTION`, `METHOD`, `IMPORT`).
*   **`GitMetadata`**: Stores Git-related information for a chunk (commit ID, author, timestamp, etc.).
*   **`LSPMetadata`**: Stores semantic information obtained from the LSP (references, type info, hover text, definition location).
*   **`StorageConfig`**: Configuration object for setting up storage backends (e.g., database URI, cache directory).
*   **`EmbeddingConfig`** (Inferred): Configuration object for setting up the embedding model and its parameters.
*   **`ProcessingJob`**: Represents a unit of work within the pipeline, typically corresponding to processing a single file event (creation, modification, deletion).

## Usage

The primary entry point for using this module is the `initialize_processor` function in the top-level `__init__.py`.

```python
from pathlib import Path
from codemap.processor import initialize_processor
from codemap.processor.storage.base import StorageConfig
# from codemap.processor.embedding.models import EmbeddingConfig # Assuming this exists

# Define the path to the repository you want to process
repo_path = Path("/path/to/your/repository")

# Optional: Configure storage (defaults to LanceDB in project data dir)
# storage_cfg = StorageConfig(uri="path/to/lancedb_storage")
storage_cfg = None

# Optional: Configure embedding model (replace with actual config)
# embedding_cfg = EmbeddingConfig(model_name="your-model-name")
embedding_cfg = None

# Initialize the processing pipeline
pipeline = initialize_processor(
    repo_path=repo_path,
    storage_config=storage_cfg,
    embedding_config=embedding_cfg,
    enable_lsp=True,  # Enable Language Server Protocol analysis
    max_workers=4     # Number of parallel processing threads
)

# The pipeline typically starts processing automatically
# (e.g., via an internal call to start the watcher and process existing files)
# Or you might need to explicitly start it (depending on implementation details not shown):
# pipeline.start()

# Keep the main thread alive if the pipeline runs in background threads
# import time
# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     pipeline.stop() # Assuming a stop method exists

print(f"CodeMap Processor initialized for repository: {repo_path}")

```

## Configuration

*   **Repository Path**: Set via the `repo_path` argument in `initialize_processor`.
*   **Storage**: Configure using a `StorageConfig` object passed to `initialize_processor`. If `None`, a default `LanceDBStorage` configuration is used, typically storing data within the CodeMap application's data directory.
*   **Embeddings**: Configure using an `EmbeddingConfig` object passed to `initialize_processor`. (Details depend on the `embedding` module implementation).
*   **LSP Analysis**: Enable or disable using the `enable_lsp` boolean flag (default is `True`).
*   **Concurrency**: Control the number of parallel processing threads using the `max_workers` argument.

## Extensibility

*   **Adding Languages**: Requires adding a Tree-sitter grammar, implementing a language configuration class (like `TypeScriptConfig`) in `analysis.tree_sitter.languages/`, and potentially updating language mappings (e.g., in `LSPAnalyzer`).
*   **Adding Storage Backends**: Implement the `StorageBackend` interface (`storage.base.py`) and potentially update `initialize_processor` or configuration handling to allow selecting the new backend.
*   **Adding Analysis Tools**: Create new classes within the `analysis/` directory, potentially following the pattern of existing analyzers, and integrate them into the `ProcessingPipeline`.

## Dependencies

*   LanceDB (`lancedb`): For vector storage.
*   MultiLSPy (`multilspy`): For interacting with Language Servers.
*   Tree-sitter (`tree_sitter`): For parsing source code.
*   Pandas (`pandas`): Used internally by LanceDB storage.
*   Python Standard Library (`logging`, `concurrent.futures`, `pathlib`, `datetime`, etc.)

*(This list might not be exhaustive and depends on the full implementation details)*