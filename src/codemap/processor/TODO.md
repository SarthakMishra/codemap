**Phase 1: Core Chunking & Metadata Enhancement**

1.  **Complete `TreeSitterChunker` Implementation (`processor/chunking/tree_sitter.py`):**
    *   [X] Implement the core parsing loop within `TreeSitterChunker.chunk`.
    *   [X] Implement the recursive logic to traverse the `tree-sitter` `Node` tree.
    *   [X] Map `tree-sitter` node types (using language configs) to your `EntityType` enum (`processor/chunking/base.py`).
    *   [X] Extract code content (`chunk.content`) based on node start/end bytes.
    *   [X] Populate `ChunkMetadata` (`processor/chunking/base.py`):
        *   [X] `entity_type`: Map from node type.
        *   [X] `name`: Extract identifier (function name, class name, variable name) from the relevant child nodes. This requires language-specific logic within the chunker or config.
        *   [X] `location`: Calculate `start_line`, `end_line`, `start_col`, `end_col` from node properties.
        *   [X] `language`: Determined before parsing.
        *   [X] `description`: Initial pass - extract docstrings (needs logic to identify docstring nodes specifically).
    *   [X] Build the `parent`/`children` hierarchy for `Chunk` objects based on the syntax tree structure.
    *   [X] Handle `UNKNOWN` entity types gracefully.
    *   [X] Add robust error handling for parsing failures.
    *   [X] Implement fallback chunking (now using `RegExpChunker`) when tree-sitter fails.

2.  **Refine Language Configurations (`processor/analysis/tree_sitter/languages/*.py`):**
    *   [X] Review and expand node types in `PYTHON_CONFIG` and `JAVASCRIPT_CONFIG`.
    *   [X] Add configurations for other languages you intend to support.
    *   [X] Standardize how entity names are extracted for different node types across languages.

3.  **Basic Dependency Extraction (Syntax-based in `processor/chunking/tree_sitter.py`):**
    *   [X] Implement logic to identify import statements (`import_` types).
    *   [X] Extract imported names and populate `ChunkMetadata.dependencies`.
    *   [X] Mapping dependencies, inheritance, function calls/usage, different entity usage/relationships.

**Phase 2: Git Integration**

4.  **Implement Git Metadata Fetching (`processor/analysis/git/`):**
    *   [X] Create a dedicated module `processor/analysis/git/` for Git metadata collection.
    *   [X] Implement `GitMetadataAnalyzer` class to handle Git operations.
    *   [X] Function to get current `commit_id` and `branch`.
    *   [X] Function to get author/timestamp for file/line range (e.g., using `git blame`).
        *   [X] **Challenge:** Map `git blame` output accurately to multi-line chunks using the porcelain format.
    *   [X] Function to get `last_modified_by`/`last_modified_at` for chunk range.
    *   [X] Integrate these functions into the chunking/metadata enrichment process to populate `ChunkMetadata.git`.
    *   [X] Implement recursive enrichment of chunks with git metadata.

5.  **Handling Branches:**
    *   [X] Design the pipeline entry point to accept a branch/commit.
    *   [X] Ensure Git operations in `processor/analysis/git/` respect the target branch/commit.
    *   [X] Define strategy for storing/managing chunks from different branches in the chosen storage solution.

**Phase 3: Semantic Analysis with LSP**

6.  **Integrate Language Server Protocol (`processor/analysis/lsp/`, using `multilspy`):**
    *   [X] Set up `multilspy` for supported languages.
    *   [X] Implement LSP interaction logic, likely within `processor/analysis/lsp/`.
    *   [X] Modify the pipeline to invoke LSP analysis *after* syntax chunking.
    *   [X] Use LSP results to enhance `ChunkMetadata`:
        *   [X] Refined Dependencies (function calls, variable usage).
        *   [X] Symbol Resolution/Verification.
        *   [X] Scope Analysis.
        *   [X] Richer Docstring/Hover Info.
    *   [X] **Challenge:** Correlate LSP information with `Chunk` objects / `tree-sitter` nodes.

**Phase 4: Pipeline Orchestration & Data Management**

7.  **Implement File Watcher Integration:**
    *   [X] Core watcher implementation exists (`watcher/watcher.py`).
    *   [X] Integrate `FileWatcher` into the main application flow.
    *   [X] Define and implement the callback functions (`on_created`, `on_modified`, `on_deleted`) passed to `FileEventHandler`. These callbacks should trigger the appropriate pipeline steps.
    *   [X] Implement debouncing/batching for file events if needed (potentially within the callbacks or the orchestrator).
    *   [X] Load ignored patterns (e.g., from `.gitignore` or `config.py`) and pass them to `FileEventHandler`. Use `utils/file_filters.py` if applicable.

8.  **Design & Implement Storage Strategy:**
    *   [X] Choose storage solution(s) (Relational DB, Document DB, Vector DB, Combination).
    *   [X] Define schema/structure for storing `Chunk`, `ChunkMetadata`, and embeddings.
    *   [X] Implement data access layer (functions/classes) to save, update, delete, and retrieve chunk/embedding data. 
    *   [X] Implement `LanceDBStorage` as the primary storage backend.
    *   [X] Create table schemas for chunks, embeddings, and file history.
    *   [X] Implement CRUD operations for chunks and embeddings.
    *   [X] Optimize storage performance with indexing.

9.  **Implement Chunk Versioning (Time-based Changes):**
    *   [X] Decide on versioning strategy (per-commit copies, history tracking, diffs).
    *   [X] Integrate versioning with storage updates and Git metadata (`commit_id`). Ensure updates correctly handle modifications vs. creations/deletions based on file changes and commit history.
    *   [X] Implement file history tracking in storage backend.

10. **Implement Embedding Generation (`processor/embedding/`):**
    *   [X] Choose embedding model.
    *   [X] Implement embedding generation logic with `EmbeddingGenerator` in `processor/embedding/`.
    *   [X] Define data models for embeddings with `EmbeddingResult` and `EmbeddingConfig`.
    *   [X] Support multiple embedding providers through `EmbeddingProvider` enum.
    *   [X] Implement batched embedding generation for efficiency.
    *   [X] Integrate into the pipeline after chunking/enrichment.
    *   [X] Store embeddings in the vector storage (LanceDB).
    *   [X] Implement error handling and fallbacks for embedding generation.

11. **Define and Implement the Update Pipeline Logic:**
    *   [X] Orchestrate the flow: File Event -> Identify Changes -> Fetch Git Info -> Chunk File (Syntax) -> Enrich (Git Meta, LSP) -> Generate Embeddings -> Update Storage.
    *   [X] Implement the `ProcessingPipeline` class with multi-threaded processing.
    *   [X] Implement a job tracking system with `ProcessingJob` to monitor file processing status.
    *   [X] Connect chunking, embedding, and storage components in a coherent flow.
    *   [X] Add robust error handling throughout the pipeline.
    *   [X] Implement search capabilities (vector and text-based) in the pipeline.
    *   [X] Implement optimization logic (process only changed files/chunks). This might involve diffing or comparing against stored versions.

**Phase 5: Testing & Robustness**

12. **Develop Comprehensive Tests:**
    *   [X] Unit tests for modules (`processor/chunking`, `processor/analysis/git`, `watcher`, etc.).
    *   [X] Integration tests for `TreeSitterChunker` and `RegExpChunker`.
    *   [X] Integration tests for Git metadata fetching.
    *   [ ] End-to-end tests triggered by file events via the watcher, verifying storage updates.
    *   [ ] Test edge cases.

13. **Logging and Monitoring:**
    *   [X] Ensure consistent, structured logging throughout all modules.
    *   [X] Add performance monitoring for critical operations like embedding generation and storage.
    *   [ ] Implement more detailed monitoring points (processing times, queue lengths if applicable).

**Phase 6: Code Quality & Maintenance**

14. **Code Quality:**
    *   [X] Enforce static typing with proper type hints across all modules.
    *   [X] Fix linting issues and ensure code follows best practices.
    *   [X] Add comprehensive docstrings to all public functions and classes.
    *   [X] Resolve exception handling patterns (avoid blind exception catching).
    *   [X] Optimize performance-critical code sections.

**Phase 7: Background Processing & Deployment**

15. **Background Processing Implementation:**
    *   [X] Design and implement background processing architecture for the processor component.
    *   [X] Create a daemon service that runs the processor in the background.
    *   [X] Implement proper signal handling for graceful shutdown.
    *   [X] Add startup/shutdown scripts for service management.
    *   [X] Implement status reporting and health checks.
    *   [ ] Add support for processing optimization (batch jobs, prioritization).
    *   [ ] Improve error recovery and resilience for long-running jobs.

16. **Daemon Integration:**
    *   [X] Create `daemon/` module within the existing codebase structure.
    *   [X] Add HTTP API server for client-daemon communication.
    *   [X] Extend `ProcessingPipeline` with methods to support daemon operations.
    *   [X] Implement `CodeMapDaemon` class for daemon lifecycle management.
    *   [X] Implement job tracking and querying functionality.
    *   [ ] Add WebSocket support for real-time updates and notifications.
    *   [X] Implement authentication and authorization for API access.

17. **CLI Integration:**
    *   [X] Add daemon-specific commands to the CLI (`daemon start`, `daemon stop`, etc.).
    *   [X] Implement job management commands (`daemon jobs`).
    *   [X] Add status reporting and monitoring.
    *   [X] Create user-friendly output formatting for daemon operations.
    *   [X] Add support for daemon configuration through CLI.

18. **Configuration & Deployment:**
    *   [X] Extend configuration system to support daemon settings.
    *   [X] Implement logging configuration with proper rotation.
    *   [X] Create systemd service unit files for Linux platforms.
    *   [X] Add supervisor configuration templates.
    *   [X] Add installation guides for different platforms.

19. **User Experience:**
    *   [X] Implement clear status output for background services.
    *   [X] Add color-coded terminal output for error/warning/info messages.
    *   [X] Implement job status reporting for long-running operations.
    *   [X] Create interactive configuration wizard for first-time setup.
    *   [X] Add progress indicators for repository processing.

**Future Work & Enhancements:**

20. **Performance Optimizations:**
    *   [ ] Implement parallel processing for LSP analysis across multiple files.
    *   [ ] Add caching for frequently accessed embeddings.
    *   [ ] Optimize memory usage for large repositories.
    *   [ ] Implement incremental updating based on git diff.
    *   [ ] Add resource limiting and throttling for daemon operations.

21. **Additional Daemon Features:**
    *   [ ] Implement scheduled processing tasks.
    *   [ ] Add multi-repository support in daemon mode.
    *   [ ] Create centralized logging and monitoring dashboard.
    *   [ ] Implement remote API access for distributed setups.
    *   [ ] Add support for clustering and load balancing.

22. **Web Interface:**
    *   [ ] Design and implement React-based web interface.
    *   [ ] Create dashboard for monitoring daemon status and jobs.
    *   [ ] Add visual codebase explorer with syntax highlighting.
    *   [ ] Implement LLM chat interface.
    *   [ ] Add vector database visualization tools.
    *   [ ] Create user management and authentication system.

**Future Considerations (Keep in Mind During Design):**

*   **Scalability:** Parallel processing, asynchronous operations (especially for IO-bound tasks like LSP, embedding APIs, storage).
*   **Extensibility:** Adding languages, metadata extractors, embedding models.
*   **Downstream API/Query Interface:** How will `cli/`, `generators/`, etc., access the processed data?
*   **Daemon Architecture:** Plan for eventual migration to full modular architecture with separate packages.
*   **Integration Points:** Ensure clean interfaces for adding web UI and other clients in the future.