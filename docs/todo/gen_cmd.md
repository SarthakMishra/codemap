# Generate Command Enhancement Plan

## Current Implementation Analysis

The current implementation of the generate command provides basic functionality for:
- Creating a tree-like representation of a project's directory structure
- Parsing code files to extract documentation
- Creating a single markdown file with repository documentation
- Supporting configuration options for token limits and content length
- Maintaining whitelist/blacklist filtering based on gitignore patterns

### Current Architecture:
- Uses `CodeParser` for parsing code files
- Relies on `DocumentationProcessor` for processing files
- Uses `MarkdownGenerator` for output generation
- Simple truncation based on line count rather than semantic importance
- Basic tree generation with checkboxes indicating included files

## Limitations and Areas for Improvement

1. **Token Utilization and Truncation**
   - Current truncation is based solely on line count or character count
   - No consideration of semantic importance when truncating
   - No intelligent compression strategies for token-limited contexts

2. **Metadata and Context**
   - Limited metadata about the codebase structure
   - No integration with LSP or tree-sitter for semantic analysis
   - Missing important context that helps LLMs understand the code better

3. **Pipeline Integration**
   - Not leveraging the new `processor` pipeline architecture
   - Missing integration with chunking, embedding, and analysis components
   - No semantic search capability for retrieving relevant documentation

4. **User Experience**
   - `generate` command is verbose and could be shortened to `gen`
   - Limited feedback during processing
   - No progress tracking for large repositories

## Enhancement Plan

### 1. Command Structure Updates

- **Rename Command to `gen`**
  - Update the CLI registration in `cli_app.py`
  - Maintain backward compatibility with an alias or deprecation notice
  - Update documentation and examples

```python
# Updated CLI registration
app.command(name="gen", help="Generate code documentation")(generate_command)
# Add alias for backward compatibility
app.command(name="generate", help="Generate code documentation (alias for gen)")(generate_command)
```

- **Enhance Command Arguments**
  - Add options for semantic analysis depth
  - Include embedding and vector storage options
  - Support for different output formats (markdown, JSON, HTML)

```python
def gen_command(
    path: PathArg = Path(),
    output: OutputOpt = None,
    config: ConfigOpt = None,
    map_tokens: MapTokensOpt = None,
    max_content_length: MaxContentLengthOpt = None,
    semantic_analysis: bool = True,  # Enable semantic analysis
    compression: str = "smart",      # Compression strategy
    format: str = "markdown",        # Output format
    tree: TreeFlag = False,
    is_verbose: VerboseFlag = False,
) -> None:
    """Generate documentation for the specified codebase."""
    # Implementation...
```

### 2. Integrate Processor Pipeline

- **Leverage Existing Pipeline Components**
  - Use the `ProcessingPipeline` from the processor module
  - Integrate with chunking, embedding, and storage modules
  - Utilize LSP and tree-sitter analysis for enhanced metadata

```python
def process_codebase(target_path: Path, config_data: dict) -> dict:
    """Process a codebase using the pipeline architecture."""
    # Configure the pipeline
    pipeline = ProcessingPipeline(
        repo_path=target_path,
        embedding_config=EmbeddingConfig.from_dict(config_data.get("embedding", {})),
        storage_config=StorageConfig.from_dict(config_data.get("storage", {})),
        ignored_patterns=config_data.get("ignored_patterns", None),
        enable_lsp=config_data.get("enable_lsp", True),
    )
    
    # Process files and collect results
    results = {}
    paths = list(target_path.rglob('*'))
    pipeline.batch_process(paths)
    
    # Collect processed chunks and metadata
    for path in paths:
        job = pipeline.get_job_status(path)
        if job and job.completed_at and not job.error:
            results[path] = {
                "chunks": job.chunks,
                "lsp_metadata": job.lsp_metadata,
            }
    
    return results
```

### 3. Smart Truncation and Compression

- **Semantic Importance Scoring**
  - Use LSP analysis to identify important code elements (classes, functions, etc.)
  - Score code sections based on semantic importance
  - Retain high-importance sections during truncation

```python
def score_chunk_importance(chunk: Chunk, lsp_metadata: dict) -> float:
    """Score the semantic importance of a code chunk."""
    base_score = 1.0
    
    # Higher score for chunks with symbols (classes, functions)
    if chunk.symbols:
        base_score += len(chunk.symbols) * 0.5
    
    # Higher score for chunks with imports
    if chunk.imports:
        base_score += len(chunk.imports) * 0.3
    
    # Higher score for chunks referenced by other chunks
    references = lsp_metadata.get("references", [])
    base_score += len(references) * 0.7
    
    # Higher score for recently modified code
    if chunk.git_metadata and chunk.git_metadata.last_modified:
        days_since_modified = (datetime.now() - chunk.git_metadata.last_modified).days
        base_score += max(0, 10 - days_since_modified) * 0.1
        
    return base_score
```

- **Compression Strategies**
  - Implement multiple compression strategies for token-limited contexts
  - Include options for smart, aggressive, and minimal compression
  - Allow configuration of compression preferences

```python
class CompressionStrategy:
    """Base class for code compression strategies."""
    
    def compress(self, chunks: list[Chunk], token_limit: int) -> list[Chunk]:
        """Compress chunks to fit within token limit."""
        raise NotImplementedError

class SmartCompression(CompressionStrategy):
    """Smart compression based on semantic importance."""
    
    def compress(self, chunks: list[Chunk], token_limit: int) -> list[Chunk]:
        # Sort chunks by importance
        sorted_chunks = sorted(chunks, key=lambda c: c.importance_score, reverse=True)
        
        # Keep high-importance chunks first
        result = []
        token_count = 0
        
        for chunk in sorted_chunks:
            chunk_tokens = chunk.token_count
            if token_count + chunk_tokens <= token_limit:
                result.append(chunk)
                token_count += chunk_tokens
            else:
                # Try to include a summarized version
                summarized = self._summarize_chunk(chunk)
                if token_count + summarized.token_count <= token_limit:
                    result.append(summarized)
                    token_count += summarized.token_count
        
        return result
        
    def _summarize_chunk(self, chunk: Chunk) -> Chunk:
        """Create a summarized version of a chunk."""
        # Implementation of chunk summarization
```

### 4. Enhanced Metadata Generation

- **Code Structure Metadata**
  - Generate rich metadata about code structure
  - Include dependency graphs and module relationships
  - Add file-level and project-level metrics

```python
def generate_code_metadata(parsed_results: dict) -> dict:
    """Generate rich metadata about the code structure."""
    metadata = {
        "files": {},
        "dependencies": {},
        "symbols": {},
        "project_metrics": {
            "total_files": 0,
            "total_lines": 0,
            "total_symbols": 0,
            "language_distribution": {},
        },
    }
    
    # Process each file's metadata
    for path, result in parsed_results.items():
        file_metadata = {
            "path": str(path),
            "size": path.stat().st_size,
            "language": detect_language(path),
            "symbols": [],
            "imports": [],
        }
        
        # Add symbols from chunks
        for chunk in result.get("chunks", []):
            file_metadata["symbols"].extend(chunk.symbols)
            file_metadata["imports"].extend(chunk.imports)
            
        # Add LSP metadata if available
        if "lsp_metadata" in result:
            file_metadata["lsp"] = result["lsp_metadata"]
            
        metadata["files"][str(path)] = file_metadata
        
        # Update project metrics
        metadata["project_metrics"]["total_files"] += 1
        metadata["project_metrics"]["total_lines"] += count_lines(path)
        metadata["project_metrics"]["total_symbols"] += len(file_metadata["symbols"])
        
    # Generate dependency graph
    metadata["dependencies"] = generate_dependency_graph(metadata["files"])
    
    return metadata
```

### 5. Tree-Sitter and LSP Integration

- **Semantic Analysis with Tree-Sitter**
  - Use tree-sitter to parse code structure
  - Extract detailed syntax trees for better code understanding
  - Identify important code segments based on syntax

```python
def analyze_with_tree_sitter(file_path: Path, content: str) -> dict:
    """Analyze file content using tree-sitter."""
    language = detect_language(file_path)
    parser = get_parser_for_language(language)
    
    if not parser:
        return {}
        
    tree = parser.parse(content.encode('utf-8'))
    root = tree.root_node
    
    analysis = {
        "imports": extract_imports(root, language),
        "classes": extract_classes(root, language),
        "functions": extract_functions(root, language),
        "important_nodes": identify_important_nodes(root, language),
    }
    
    return analysis
```

- **LSP Analysis Integration**
  - Leverage LSP for language-specific insights
  - Identify symbol references and definitions
  - Extract code navigation metadata

```python
def analyze_with_lsp(file_path: Path, content: str) -> dict:
    """Analyze file content using LSP."""
    language_server = get_language_server_for_file(file_path)
    
    if not language_server:
        return {}
        
    # Get document symbols
    symbols = language_server.document_symbols(file_path, content)
    
    # Get references and definitions
    references = {}
    definitions = {}
    
    for symbol in symbols:
        references[symbol.name] = language_server.find_references(file_path, symbol.position)
        definitions[symbol.name] = language_server.find_definition(file_path, symbol.position)
    
    return {
        "symbols": symbols,
        "references": references,
        "definitions": definitions,
    }
```

### 6. Implementation Strategy

1. **Phase 1: Command Structure Updates**
   - Rename command to `gen` with backward compatibility
   - Update CLI help and documentation
   - Refactor command parameters for clarity

2. **Phase 2: Processor Pipeline Integration**
   - Connect generate command to the processor pipeline
   - Implement batch processing for files
   - Add progress tracking and reporting

3. **Phase 3: Analysis Enhancements**
   - Integrate tree-sitter analysis
   - Add LSP analysis when available
   - Implement importance scoring algorithm

4. **Phase 4: Smart Truncation and Compression**
   - Implement multiple compression strategies
   - Add token counting utilities
   - Create chunk summarization functions

5. **Phase 5: Enhanced Output Generation**
   - Update MarkdownGenerator for rich metadata
   - Add support for multiple output formats
   - Implement LLM-friendly markup

### 7. Expected Outcomes

- Improved documentation quality with semantic understanding
- Better token utilization through smart truncation
- Enhanced metadata for LLM reasoning about code
- More intuitive user experience with shorter command and better feedback
- Flexible output formats for different use cases
- Intelligent resource management for large codebases
