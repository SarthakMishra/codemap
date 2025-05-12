# Generate Markdown Docs (`gen`)

Generate optimized markdown documentation and directory structures for your project:

## Command Options

```bash
codemap gen [PATH] [OPTIONS]
```

**Arguments:**

- `PATH`: Path to the codebase to analyze (file or directory, must exist; defaults to current directory)

**Options:**

- `--output`, `-o`: Output file path for the documentation (overrides config)
- `--config`, `-c`: Path to custom configuration file
- `--max-content-length`: Maximum content length for file display (set to 0 for unlimited, overrides config)
- `--lod`: Level of Detail for code analysis (signatures, structure, docs, full). Default: `docs`. Overrides config.
- `--semantic/--no-semantic`: Enable/disable semantic analysis using LSP. Default: enabled. Overrides config.
- `--tree/--no-tree`, `-t`: Include/exclude directory tree in output. Overrides config (`gen.include_tree`).
- `--entity-graph/--no-entity-graph`, `-e`: Include/exclude entity relationship graph (Mermaid) in output. Overrides config (`gen.include_entity_graph`).
- `--mermaid-entities`: Comma-separated list of entity types (e.g., 'module,class,function'). Overrides config (`gen.mermaid_entities`).
- `--mermaid-relationships`: Comma-separated list of relationship types (e.g., 'declares,imports,calls'). Overrides config (`gen.mermaid_relationships`).
- `--mermaid-legend/--no-mermaid-legend`: Show/hide the legend in the Mermaid diagram. Overrides config (`gen.mermaid_show_legend`).
- `--mermaid-unconnected/--no-mermaid-unconnected`: Remove/keep nodes with no connections in the Mermaid diagram. Overrides config (`gen.mermaid_remove_unconnected`).
- `--verbose`, `-v`: Enable verbose logging

## Examples

```bash
# Generate documentation for current directory using defaults
codemap gen
# Or using the alias:
cm gen

# Generate for a specific path with full detail and no semantic analysis
codemap gen /path/to/project --lod full --no-semantic

# Generate docs with signatures only and custom Mermaid settings
cm gen --lod signatures --mermaid-entities "class,function" --mermaid-relationships "calls"

# Generate only directory tree (implicitly disables entity graph)
codemap gen --tree --no-entity-graph

# Custom output location and content length
codemap gen -o ./docs/codebase.md --max-content-length 1500

# Use custom configuration file
codemap gen --config custom-config.yml

# Verbose mode for debugging
codemap gen -v
```

## Output Structure

The generated documentation includes:
1. Project overview and structure
2. Directory tree visualization
3. Token-optimized code summaries
4. File relationships and dependencies
5. Rich markdown formatting with syntax highlighting

## File Processing

The generator:
- Respects `.gitignore` patterns by default
- Intelligently analyzes code structure
- Optimizes content for token limits
- Generates well-structured markdown
- Handles various file types and languages 