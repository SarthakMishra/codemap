# TODO: Automated Mapping of Tree-sitter Nodes to Entity Schema and Generation of language_map.py

## Goal
Automate the process of mapping Tree-sitter node types (from `literals.py`) to the entity schemas defined in `entity/base.py`, and generate `language_map.py` (currently managed manually). This should leverage code-aware semantic similarity models to improve mapping accuracy and reduce manual effort.

---

## Implementation Plan

### 1. **Data Preparation**
- **Extract Node Types:**
  - Use the generated `NodeTypes` from `literals.py` for each language.
- **Extract Entity Schemas:**
  - Parse or introspect the classes in `entity/base.py` to get all available entity schema types and their docstrings/fields.
- **Collect Example Node Data:**
  - Optionally, gather example code snippets or AST fragments for ambiguous node types to improve mapping.

### 2. **Semantic Similarity Model Selection**
- **Choose a Model:**
  - Use a code-aware embedding model (e.g., OpenAI's `code-search-ada`, HuggingFace's `CodeBERT`, `StarCoder`, or similar) that can embed both node type names and schema class names/descriptions.
- **Set Up Embedding Pipeline:**
  - Implement a utility to embed both node type names (and optionally, their context/examples) and entity schema names/descriptions.

### 3. **Automated Mapping Process**
- **Compute Embeddings:**
  - For each node type, compute its embedding.
  - For each entity schema, compute its embedding (using class name, docstring, and field names).
- **Similarity Search:**
  - For each node type, compute multiple similarity metrics:
    - Cosine similarity between node type and schema embeddings
    - Levenshtein distance for string similarity
    - Jaccard similarity on tokenized names
    - Semantic similarity using code-aware model embeddings
  - Combine scores using weighted ensemble:
    - Primary weight on semantic similarity
    - Secondary weights on string-based metrics
  - Set confidence thresholds for each metric
  - Flag matches for manual review if:
    - Any metric falls below its threshold
    - Significant disagreement between metrics
    - Ensemble score is below overall threshold
- **Manual Overrides:**
  - Allow for a manual mapping file or override mechanism for edge cases or ambiguous nodes.

### 4. **Mapping Output and Generation**
- **Generate Mapping Table:**
  - Produce a mapping from `(language, node_type)` to `(EntityType, EntitySchemaClass)`.
- **Generate `language_map.py`:**
  - Write a script to output the `LANGUAGE_NODE_MAPPING` dictionary in the format currently used in `language_map.py`, using the generated mapping.
  - Ensure the output is readable and includes comments for low-confidence or manual mappings.

### 5. **Testing and Validation**
- **Unit Tests:**
  - Test that all node types for each language are mapped to a valid entity schema.
  - Test that the generated `language_map.py` is importable and usable by downstream code.
- **Manual Review:**
  - Review mappings with low similarity scores or flagged by the model.

### 6. **Integration and Automation**
- **Script Integration:**
  - Integrate the mapping and generation scripts into the codebase (e.g., as a CLI tool or as part of the build process).
- **Documentation:**
  - Document the mapping process, how to update/override mappings, and how to retrain or swap out the embedding model if needed.

---

## Considerations
- **Model Licensing:** Ensure the chosen embedding model is compatible with the project's license and can be used for this purpose.
- **Performance:** For large numbers of node types and schemas, consider batching or caching embeddings.
- **Extensibility:** Design the system to support new languages, node types, or entity schemas with minimal manual intervention.
- **Fallbacks:** Always provide a fallback or 'UnknownEntitySchema' for unmapped or ambiguous nodes.

---

## References
- `src/codemap/processor/tree_sitter/schema/languages/literals.py` (NodeTypes)
- `src/codemap/processor/tree_sitter/schema/entity/base.py` (Entity Schemas)
- `src/codemap/processor/tree_sitter/schema/language_map.py` (Target Output)

---

*This plan aims to reduce manual mapping effort, improve consistency, and leverage modern code understanding models for robust schema mapping.*
