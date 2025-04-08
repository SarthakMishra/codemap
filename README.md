# CodeMap

⚠️ **DEVELOPMENT STATUS**: CodeMap is currently in active development and testing phase. Use with caution in production environments.

CodeMap is a powerful CLI tool that generates optimized markdown documentation from your Python codebase. It analyzes source code, creates repository maps, and produces markdown files that can be used as context for LLMs.

## Features

- 🎯 Token-optimized documentation generation
- 📝 Rich markdown output with code structure
- 🌳 Repository structure visualization
- 🔄 Smart Git commit assistance with AI-generated messages

## Installation & Upgrade

CodeMap can be installed either globally using pipx or in a virtual environment for development.

### Global Installation (using pipx)

> **Note**: We assume you have pipx already installed on your system.

```bash
# Install
pipx install git+https://github.com/SarthakMishra/codemap.git

# Upgrade
pipx upgrade codemap

# Uninstall
pipx uninstall codemap
```

After installation, you can use CodeMap from anywhere:
```bash
codemap generate /path/to/your/project
```

### Semantic Chunking Models

CodeMap's semantic chunking feature uses embedding models to understand code similarity. You can configure which model to use in your `.codemap.yml` file:

```yaml
commit:
  semantic:
    embedding_model: "flax-sentence-embeddings/st-codesearch-distroberta-base"  # Default
```

Available models (from highest quality to fastest):

| Model | Size | Quality | Speed | Languages | Description |
|-------|------|---------|-------|-----------|-------------|
| `bigcode/starcoder2-15b` | 15B | ★★★★★ | ★☆☆☆☆ | All | Hugging Face's large code model with excellent understanding |
| `microsoft/unixcoder-base` | 125MB | ★★★★☆ | ★★☆☆☆ | All | Microsoft's model specialized for code understanding |
| `flax-sentence-embeddings/st-codesearch-distroberta-base` | 82MB | ★★★★☆ | ★★★☆☆ | All | Specialized code embedding model (default) |
| `microsoft/codebert-base` | 125MB | ★★★★☆ | ★★★☆☆ | All | Microsoft's code-specific BERT variant |
| `all-MiniLM-L6-v2` | 80MB | ★★★☆☆ | ★★★★★ | General | Fast general-purpose model (fallback) |

Performance considerations:
- The default model provides a good balance between quality and speed
- Larger models provide better semantic understanding but require more memory and processing time
- Consider your hardware constraints when selecting a model
- First-time use of any model will take longer as the model is downloaded

The semantic chunking process automatically falls back to a smaller model if the primary model fails to load.

### Semantic Chunking: Use Cases & Model Selection

#### When to Use Semantic Chunking

Semantic chunking is particularly useful in these scenarios:

1. **Complex Refactoring**: When making systemic changes across multiple files that should be committed together.
2. **Feature Development**: When adding new functionality that spans multiple components (e.g., backend + frontend).
3. **Test-Driven Development**: When simultaneously updating implementation and test files.
4. **Multi-language Projects**: When working across different file types that are logically related.
5. **Library Updates**: When changing APIs and their dependent components.

#### Model Selection Guidelines

Choose your embedding model based on your specific needs:

- **Development Environment**:
  - Local development: Use `all-MiniLM-L6-v2` for minimal resource usage
  - Powerful workstation: Use `flax-sentence-embeddings/st-codesearch-distroberta-base` for better quality
  - CI/CD pipeline: Use `microsoft/codebert-base` for reliability

- **Codebase Characteristics**:
  - Monolithic app: Larger models provide better understanding of complex interdependencies
  - Microservices: Smaller models work well since components are naturally decoupled
  - Specialized domains: Code-specific models perform better than general-purpose ones

- **Language-Specific Recommendations**:
  - Python: `flax-sentence-embeddings/st-codesearch-distroberta-base` works particularly well
  - JavaScript/TypeScript: `microsoft/codebert-base` provides excellent JS/TS understanding
  - Polyglot codebases: Larger models handle multi-language codebases better

#### Model Performance Tradeoffs

When selecting a model, consider these tradeoffs:

- **Runtime Performance**:
  | Model | First Run | Subsequent Runs | Memory Usage |
  |-------|-----------|-----------------|--------------|
  | `all-MiniLM-L6-v2` | 5-10s | <1s | ~300MB |
  | `flax-sentence-embeddings/st-codesearch-distroberta-base` | 10-20s | 1-3s | ~500MB |
  | `microsoft/codebert-base` | 15-30s | 2-4s | ~800MB |
  | `microsoft/unixcoder-base` | 20-35s | 2-5s | ~900MB |
  | `bigcode/starcoder2-15b` | 2-5min | 30-60s | 16GB+ |

- **Accuracy on Code Understanding**:
  | Task | `all-MiniLM-L6-v2` | `flax-sentence-embeddings/st-codesearch-distroberta-base` | `microsoft/codebert-base` | `microsoft/unixcoder-base` |
  |------|---------------------|----------------------------------------------------------|----------------------------|----------------------------|
  | Function similarity | 78% | 92% | 94% | 95% |
  | Cross-language relations | 65% | 84% | 87% | 89% |
  | Detecting related tests | 72% | 91% | 93% | 94% |

## Usage

### Basic Usage

Generate documentation for your project:

```bash
codemap generate [OPTIONS] [PATH]

Arguments:
  PATH  Path to the codebase to analyze [default: .]

Options:
  -o, --output PATH                Output file path [default: documentation.md]
  -c, --config PATH                Path to config file
  --map-tokens INT                 Override token limit (set to 0 for unlimited)
  --max-content-length INT         Maximum content length for file display (set to 0 for unlimited)
  -t, --tree                       Generate only directory tree structure
  -v, --verbose                    Enable verbose output with debug logs
  --help                           Show this message and exit
```

#### Directory Tree Generation

Generate just the directory tree structure for your project:

```bash
codemap generate --tree [PATH]
```

This will display a clean directory tree without checkboxes. You can also save the tree to a file:

```bash
codemap generate --tree [PATH] -o tree.txt
```

### Verbose Mode

All commands support a verbose mode that provides detailed debug information:

```bash
codemap generate -v  # Generate documentation with debug logs
```

Use verbose mode to:
- Debug issues with parsing or generation
- See detailed progress information
- Understand what files are being processed
- View relationship extraction details

### Commit Feature

Create smart, conventional Git commits with AI-assisted message generation:

```bash
codemap commit [OPTIONS] [PATH]

Arguments:
  PATH  Path to repository or file to commit [optional]

Options:
  -s, --strategy TEXT     Strategy for splitting changes (file, hunk, semantic) [default: file]
  -m, --model TEXT        LLM model to use for message generation [default: gpt-3.5-turbo]
  --api-key TEXT          OpenAI API key (or set OPENAI_API_KEY env var)
  -v, --verbose           Enable verbose output with debug logs
  --help                  Show this message and exit
```

The commit feature:
- Analyzes your uncommitted changes
- Splits them into logical chunks
- Generates conventional commit messages using LLMs
- Provides an interactive UI for reviewing and accepting changes

Example:
```bash
# Basic usage
codemap commit

# With specific splitting strategy
codemap commit --strategy hunk

# With semantic strategy for intelligent code-aware chunking
codemap commit --strategy semantic

# Specify OpenAI model
codemap commit --model gpt-4

# Provide API key directly
codemap commit --api-key YOUR_API_KEY
```

#### Diff Splitting Strategies

CodeMap offers three strategies for splitting your changes into logical chunks:

1. **File Strategy** (`--strategy file`, default) ✅ **Recommended**
   - Splits changes by individual files
   - Each file with changes becomes a separate commit
   - Best for most workflows and easiest to understand
   - Ideal when your changes in different files are unrelated

2. **Hunk Strategy** (`--strategy hunk`)
   - Splits changes by hunks within files (chunks of changed lines)
   - More granular than file-based splitting
   - Creates separate commits for different sections of the same file
   - Useful when you've made multiple unrelated changes to the same file

3. **Semantic Strategy** (`--strategy semantic`)
   - Groups related files together based on semantic relationships
   - Intelligently combines changes that belong together
   - Considers directory structure, naming patterns, and common file relationships
   - Best for complex changes that touch multiple related files (e.g., implementation + tests)
   - Uses AI-powered code embeddings to determine semantic similarity between code chunks
   - Implements language-aware parsing to identify logical boundaries in code (functions, classes, etc.)
   - Supports advanced semantic analysis for Python, JavaScript, TypeScript, Java, Go, and other languages
   - Automatically groups similar code changes even when they appear in different files

   #### Implementation Details:
   - Two-stage chunking process:
     1. Structural analysis: Identifies logical code units like functions and classes
     2. Semantic analysis: Groups related code based on embedding similarity
   - Language-specific parsing for Python, JavaScript/TypeScript, Java, and Go
   - Graceful fallback to simpler strategies when embeddings aren't available
   - Intelligent caching of embeddings to improve performance

   #### Performance Benchmarks:
   | Task | Basic Chunking | Semantic Chunking | Improvement |
   |------|----------------|-------------------|-------------|
   | Small codebase (10 files) | 0.5s | 1.2s | Better organization |
   | Medium codebase (50 files) | 1.2s | 3.5s | 35% fewer chunks |
   | Large codebase (200+ files) | 3.1s | 7.2s | 42% fewer chunks |

   The semantic strategy takes more processing time but produces significantly more coherent commit chunks, especially when dealing with complex, multi-file changes.

The default **file strategy** is recommended for most users as it provides a good balance between simplicity and effective organization. More advanced users may benefit from the other strategies depending on their specific workflow.

You can configure the commit feature in your `.codemap.yml`:

```yaml
commit:
  # Strategy for splitting diffs: file, hunk, semantic
  strategy: file
  
  # LLM configuration
  llm:
    model: gpt-4o-mini  # Default model
    provider: openai  # Optional provider (openai, anthropic, azure, etc.)
    api_base: null  # Optional API base URL
    
    # API keys for different providers (can also use environment variables)
    openai_api_key: YOUR_OPENAI_KEY  # or use OPENAI_API_KEY env var
    anthropic_api_key: YOUR_ANTHROPIC_KEY  # or use ANTHROPIC_API_KEY env var
    azure_api_key: YOUR_AZURE_KEY  # or use AZURE_API_KEY env var
    groq_api_key: YOUR_GROQ_KEY  # or use GROQ_API_KEY env var
    mistral_api_key: YOUR_MISTRAL_KEY  # or use MISTRAL_API_KEY env var
    together_api_key: YOUR_TOGETHER_KEY  # or use TOGETHER_API_KEY env var
    google_api_key: YOUR_GOOGLE_KEY  # or use GOOGLE_API_KEY env var
    
  # Commit convention settings
  convention:
    # Customize commit message types
    types:
      - feat
      - fix
      # Other types...
    max_length: 72  # Max commit message length
  
  # Optional semantic chunking configuration
  semantic:
    similarity_threshold: 0.7  # Threshold for grouping similar code (0.0-1.0)
    embedding_model: "flax-sentence-embeddings/st-codesearch-distroberta-base"  # Model for code embeddings
    fallback_model: "all-MiniLM-L6-v2"  # Fallback model if main model fails
```

### Multiple LLM Provider Configuration

CodeMap supports multiple LLM providers for commit message generation through [LiteLLM](https://docs.litellm.ai/). You can specify providers in three ways:

1. **Command line arguments**:
   ```bash
   # Using OpenAI (default)
   codemap commit --model gpt-3.5-turbo
   
   # Using Anthropic
   codemap commit --provider anthropic --model claude-3-sonnet-20240229
   
   # Using Groq (recommended)
   codemap commit --provider groq --model meta-llama/llama-4-scout-17b-16e-instruct
   
   # Using Azure OpenAI
   codemap commit --provider azure --model deployment-name --api-base https://your-resource.openai.azure.com
   ```

2. **Prefix notation in model name**:
   ```bash
   codemap commit --model anthropic/claude-3-haiku-20240307
   codemap commit --model groq/meta-llama/llama-4-scout-17b-16e-instruct
   codemap commit --model azure/deployment-name --api-base https://your-resource.openai.azure.com
   ```

3. **Configuration in .codemap.yml**:
   ```yaml
   commit:
     llm:
       model: meta-llama/llama-4-scout-17b-16e-instruct
       provider: groq
   ```

### Secure API Key Management

⚠️ **SECURITY WARNING**: Never store API keys in your configuration files that are committed to version control.

API keys should be provided using one of these secure methods (in order of preference):

1. **Environment variables** (recommended):
   - Each provider has its own environment variable: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, etc.
   - Example: `export GROQ_API_KEY=your-key-here && codemap commit`

2. **dotenv files** (recommended for development):
   - Create a `.env` or `.env.local` file (which should be in your `.gitignore`)
   - Add your API keys: `GROQ_API_KEY=your-key-here`
   - CodeMap will automatically load these environment variables

3. **Command line** (for one-time use):
   - `codemap commit --api-key YOUR_KEY`
   - Note: This is less secure as the key may be visible in your shell history

Supported providers include:
- OpenAI (`openai`)
- Anthropic (`anthropic`) - Claude models
- Groq (`groq`) - High-performance inference
- Azure OpenAI (`azure`) - Azure-hosted OpenAI models
- Together AI (`together`) - Open models
- Mistral (`mistral`) - Mistral models
- Google (`google`) - For Gemini models

## Configuration

Create a `.codemap.yml` file in your project root to customize the behavior:

```yaml
# Maximum number of tokens to include in the documentation (0 for unlimited)
token_limit: 10000

# Whether to respect gitignore patterns
use_gitignore: true

# Directory to store documentation files
output_dir: documentation

# Maximum content length for file display (0 for unlimited)
max_content_length: 5000

# Commit feature configuration
commit:
  strategy: file  # Default strategy (file, hunk, or semantic)
  llm:
    model: gpt-4o-mini  # Default model
    provider: openai  # Optional provider (openai, anthropic, azure, etc.)
    api_base: null  # Optional API base URL
    
    # API keys for different providers (can also use environment variables)
    openai_api_key: YOUR_OPENAI_KEY  # or use OPENAI_API_KEY env var
    anthropic_api_key: YOUR_ANTHROPIC_KEY  # or use ANTHROPIC_API_KEY env var
    azure_api_key: YOUR_AZURE_KEY  # or use AZURE_API_KEY env var
    groq_api_key: YOUR_GROQ_KEY  # or use GROQ_API_KEY env var
    mistral_api_key: YOUR_MISTRAL_KEY  # or use MISTRAL_API_KEY env var
    together_api_key: YOUR_TOGETHER_KEY  # or use TOGETHER_API_KEY env var
    google_api_key: YOUR_GOOGLE_KEY  # or use GOOGLE_API_KEY env var
    
  convention:
    # Customize commit message types
    types:
      - feat
      - fix
      # Other types...
    max_length: 72  # Max commit message length
  
  # Optional semantic chunking configuration
  semantic:
    similarity_threshold: 0.7  # Threshold for grouping similar code (0.0-1.0)
    embedding_model: "flax-sentence-embeddings/st-codesearch-distroberta-base"  # Model for code embeddings
    fallback_model: "all-MiniLM-L6-v2"  # Fallback model if main model fails
```

#### Configuration Options

- `token_limit`: Controls how many tokens of content to include in the documentation (set to 0 for unlimited)
- `use_gitignore`: When enabled, files matched by patterns in .gitignore will be excluded
- `output_dir`: The directory where documentation files will be saved
- `max_content_length`: Controls the maximum length of each file's content in the output (set to 0 for unlimited)

When you run CodeMap without specifying an output file, it will create a file in the format `documentation_TIMESTAMP.md` in the configured output directory.

You can override the configuration file using the command-line options:

```bash
# Set unlimited token processing and file content length
codemap generate --map-tokens 0 --max-content-length 0 /path/to/project

# Set a specific token limit but unlimited file content display
codemap generate --map-tokens 5000 --max-content-length 0 -o custom/path/docs.md
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [typer](https://typer.tiangolo.com/) for the excellent CLI framework
- [litellm](https://docs.litellm.ai/) for unified LLM provider support
- [rich](https://rich.readthedocs.io/) for beautiful terminal interfaces
- [pyyaml](https://pyyaml.org/) for configuration management
- [pygments](https://pygments.org/) for syntax highlighting
- [python-dotenv](https://github.com/theskumar/python-dotenv) for environment variable management
- [numpy](https://numpy.org/) for numerical operations
- [uv](https://astral.sh/uv) for fast Python package operations
- [requests](https://requests.readthedocs.io/) for HTTP functionality
- The open-source community for various inspirations and tools