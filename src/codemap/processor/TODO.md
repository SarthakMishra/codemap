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
    *   [ ] Extract imported names and populate `ChunkMetadata.dependencies`.
    *   [ ] Mapping dependencies, function calls/usage, different entity usage/relationships.

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
    *   [ ] Define strategy for storing/managing chunks from different branches in the chosen storage solution.

**Phase 3: Semantic Analysis with LSP**

6.  **Integrate Language Server Protocol (`processor/analysis/lsp/`, using `multilspy`):**
    *   [ ] Set up `multilspy` for supported languages.
    *   [ ] Implement LSP interaction logic, likely within `processor/analysis/lsp/`.
    *   [ ] Modify the pipeline to invoke LSP analysis *after* syntax chunking.
    *   [ ] Use LSP results to enhance `ChunkMetadata`:
        *   [ ] Refined Dependencies (function calls, variable usage).
        *   [ ] Symbol Resolution/Verification.
        *   [ ] Scope Analysis.
        *   [ ] Richer Docstring/Hover Info.
    *   **Challenge:** Correlate LSP information with `Chunk` objects / `tree-sitter` nodes.

**Phase 4: Pipeline Orchestration & Data Management**

7.  **Implement File Watcher Integration:**
    *   [X] Core watcher implementation exists (`watcher/watcher.py`).
    *   [ ] Integrate `FileWatcher` into the main application flow.
    *   [ ] Define and implement the callback functions (`on_created`, `on_modified`, `on_deleted`) passed to `FileEventHandler`. These callbacks should trigger the appropriate pipeline steps.
    *   [ ] Implement debouncing/batching for file events if needed (potentially within the callbacks or the orchestrator).
    *   [ ] Load ignored patterns (e.g., from `.gitignore` or `config.py`) and pass them to `FileEventHandler`. Use `utils/file_filters.py` if applicable.

8.  **Design & Implement Storage Strategy:**
    *   [ ] Choose storage solution(s) (Relational DB, Document DB, Vector DB, Combination).
    *   [ ] Define schema/structure for storing `Chunk`, `ChunkMetadata`, and embeddings.
    *   [ ] Implement data access layer (functions/classes) to save, update, delete, and retrieve chunk/embedding data. (Location TBD - could be `utils/storage_utils.py` or dedicated `storage/` module).

9.  **Implement Chunk Versioning (Time-based Changes):**
    *   [ ] Decide on versioning strategy (per-commit copies, history tracking, diffs).
    *   [ ] Integrate versioning with storage updates and Git metadata (`commit_id`). Ensure updates correctly handle modifications vs. creations/deletions based on file changes and commit history.

10. **Implement Embedding Generation (`processor/embedding/`):**
    *   [ ] Choose embedding model.
    *   [ ] Implement embedding generation logic (likely in `processor/embedding/`).
    *   [ ] Decide *what* content to embed.
    *   [ ] Integrate into the pipeline after chunking/enrichment.
    *   [ ] Store embeddings in the chosen (vector) storage.

11. **Define and Implement the Update Pipeline Logic:**
    *   [ ] Orchestrate the flow: File Event -> Identify Changes -> Fetch Git Info -> Chunk File (Syntax) -> Enrich (Git Meta, LSP) -> Generate Embeddings -> Update Storage.
    *   [ ] Implement optimization logic (process only changed files/chunks). This might involve diffing or comparing against stored versions.

**Phase 5: Testing & Robustness**

12. **Develop Comprehensive Tests:**
    *   [X] Unit tests for modules (`processor/chunking`, `processor/analysis/git`, `watcher`, etc.).
    *   [X] Integration tests for `TreeSitterChunker` and `RegExpChunker`.
    *   [X] Integration tests for Git metadata fetching.
    *   [ ] End-to-end tests triggered by file events via the watcher, verifying storage updates.
    *   [ ] Test edge cases.

13. **Logging and Monitoring:**
    *   [X] Ensure consistent, structured logging throughout all modules.
    *   [ ] Add monitoring points (processing times, queue lengths if applicable).

**Future Considerations (Keep in Mind During Design):**

*   **Scalability:** Parallel processing, asynchronous operations (especially for IO-bound tasks like LSP, embedding APIs, storage).
*   **Extensibility:** Adding languages, metadata extractors, embedding models.
*   **Downstream API/Query Interface:** How will `cli/`, `generators/`, etc., access the processed data?