# Tree-sitter Module Refactoring: Query-Based Entity Recognition

This document outlines the steps to migrate from the current manual, language-specific node type mapping to a more scalable approach using Tree-sitter queries and standard tags.

## Background

The current implementation (`LanguageSyntaxHandler` subclasses like `PythonSyntaxHandler`) requires writing detailed mapping logic for each supported language, checking `node.type` and context. This is difficult to scale and maintain.

The proposed approach leverages Tree-sitter's query engine (`.scm` files, often `queries/tags.scm`) which use standard tags (e.g., `@definition.function`, `@reference.import`) to identify code constructs. We will map these standard tags to our internal `EntityType`.

See Tree-sitter Code Navigation Systems documentation for standard tag conventions: <https://tree-sitter.github.io/tree-sitter/4-code-navigation.html> (Referenced in [Issue #660](https://github.com/tree-sitter/tree-sitter/issues/660))

## Guiding Principles

- **[IMPORTANT] Backwards Compatibility:** The external API of the `codemap.processor.tree_sitter` module (e.g., the main `TreeSitterAnalyzer.analyze` method signature and the structure of its output) **must** remain backwards compatible. Consumers of this module should not need changes. The refactoring affects internal implementation only.
- **Scalability:** The new implementation should make adding support for new languages significantly easier, primarily requiring a suitable query file rather than extensive Python code.
- **Maintainability:** Reduce language-specific code within the Python handlers, relying on standardized queries maintained alongside grammars.

## Migration Tasks

- [ ] **Research & Define Standard Tags:**
    - [ ] Review the Tree-sitter Code Navigation Systems documentation ([https://tree-sitter.github.io/tree-sitter/code-navigation-systems](https://tree-sitter.github.io/tree-sitter/code-navigation-systems)) to identify the standard set of tags relevant to our `EntityType` (e.g., definitions, references, imports, comments, documentation, etc.).
    - [ ] Create a definitive mapping from these standard Tree-sitter tags (strings like `@definition.function`, `@comment`, `@doc`) to our internal `codemap.processor.tree_sitter.base.EntityType` enum values. Handle potential variations (e.g., `@function` vs. `@definition.function`, different comment/doc tags).
- [ ] **Query File Management:**
    - [ ] Determine a strategy for locating and loading `tags.scm` (or equivalent, like `locals.scm`, `highlights.scm`) files for each supported language. Options:
        - Bundle them directly within the `codemap` package.
        - Attempt to locate them within installed `tree-sitter-<language>` Python packages or system locations.
        - Require users to configure paths (less ideal).
    - [ ] Implement robust error handling for cases where query files are missing or invalid for a language. Consider logging warnings and potentially falling back (if implemented).
- [ ] **Refactor Core Analyzer Logic:**
    - [ ] Modify `codemap.processor.tree_sitter.analyzer.TreeSitterAnalyzer` (or create a new query-focused analyzer).
    - [ ] Implement logic to load the appropriate language grammar and its corresponding query file(s) (e.g., `tags.scm`).
    - [ ] In the tree traversal logic (e.g., `_traverse_node`), execute the loaded query using `query.captures(node)` for the relevant node or subtree.
    - [ ] Prioritize using the `capture_name` (tag) from the query results and the predefined tag-to-EntityType mapping to determine the `EntityType`.
    - [ ] Remove the direct call to language-specific `get_entity_type` methods as the primary mechanism.
- [ ] **Handle Fallbacks & Existing Helpers:**
    - [ ] Decide on a fallback strategy if a query file is missing or a specific node isn't captured by any relevant tag. (Options: skip node, attempt old manual mapping if kept temporarily, log warning, assign default `UNKNOWN` type).
    - [ ] Ensure existing helper functions (like `extract_name`, `find_docstring`, `extract_imports`, `extract_calls` from `LanguageSyntaxHandler`) are still usable, likely moving them to the analyzer or a utility class, as they operate on nodes regardless of how the entity type was determined. They might need slight adjustments.
- [ ] **Testing:**
    - [ ] Update existing unit tests (`tests/processor/`) to reflect the new query-based approach.
    - [ ] Add tests specifically for the query loading, tag mapping logic, and fallback mechanisms.
    - [ ] Consider adding integration tests using sample code from multiple languages (e.g., Python, Java, Go) to verify cross-language consistency and scalability. Leverage highlighting tests format if suitable ([Tree-sitter Highlighting Tests](https://tree-sitter.github.io/tree-sitter/syntax-highlighting#unit-testing)).
- [ ] **Cleanup:**
    - [ ] Remove the old language-specific `LanguageSyntaxHandler` subclasses (e.g., `PythonSyntaxHandler`) and associated configuration classes (`PythonConfig`).
    - [ ] Remove the `get_entity_type` method from the base handler or repurpose it strictly for fallback logic if chosen.
    - [ ] Remove language-specific entity lists from `LanguageConfig` if queries provide comprehensive coverage.

## Deprecated Items (from previous TODO)

- `- check how entity recognition works in tree-sitter by default` (Superseded by query investigation)
- `- try to find a better alternative than the current impleemntation which manually maps over entities and every languages needs to be mapped separately` (This migration *is* the alternative)
- `- better support for all tree-sitter supported languages` (This migration *enables* better support)