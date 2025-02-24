# CodeMap

⚠️ **DEVELOPMENT STATUS**: CodeMap is currently in active development and testing phase. Use with caution in production environments.

CodeMap is a powerful CLI tool that generates optimized markdown documentation from your Python codebase using tree-sitter analysis. It analyzes source code, creates repository maps, generates ERD diagrams, and produces markdown files that can be used as context for LLMs.

## Features

- 🔍 Smart code analysis using tree-sitter
- 📊 Dependency graph generation and analysis
- 🎯 Token-optimized documentation generation
- 📝 Rich markdown output with code structure
- 🌳 Repository structure visualization
- 🔄 Automatic docstring extraction
- 🎨 Python-focused syntax parsing
- 📈 Entity Relationship Diagram (ERD) generation

## Language Support

- ✅ Python (fully supported)
- 🔄 TypeScript/JavaScript (coming soon)
- 🔜 More languages planned for future releases

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

## Usage

### Basic Usage

Generate documentation for your project:

```bash
codemap generate [OPTIONS] [PATH]

Arguments:
  PATH  Path to the codebase to analyze [default: .]

Options:
  -o, --output PATH    Output file path [default: documentation.md]
  -c, --config PATH    Path to config file
  --map-tokens INT     Override token limit
  -v, --verbose       Enable verbose output with debug logs
  --help              Show this message and exit
```

### ERD Generation

Generate an Entity Relationship Diagram (ERD) for your Python codebase:

```bash
codemap erd [OPTIONS] [PATH]

Arguments:
  PATH  Path to the codebase to analyze [default: .]

Options:
  -o, --output PATH    Output file path for the ERD (PDF format)
  -c, --config PATH    Path to config file
  -v, --verbose       Enable verbose output with debug logs
  --help              Show this message and exit
```

The ERD command will:
- Analyze your Python codebase for classes and their relationships
- Generate a visual diagram showing:
  - Classes and their attributes
  - Inheritance relationships
  - Composition/aggregation relationships
  - Many-to-many relationships
- Output a PDF file with the diagram

> **Note**: This feature requires the `graphviz` system package to be installed:
> - Ubuntu/Debian: `sudo apt-get install graphviz`
> - macOS: `brew install graphviz`
> - Windows: Download from [Graphviz Downloads](https://graphviz.org/download/)

### Verbose Mode

All commands support a verbose mode that provides detailed debug information:

```bash
codemap generate -v  # Generate documentation with debug logs
codemap erd -v      # Generate ERD with debug logs
```

Use verbose mode to:
- Debug issues with parsing or generation
- See detailed progress information
- Understand what files are being processed
- View relationship extraction details

### Configuration

Create a `.codemap.yml` file in your project root to customize the behavior:

```yaml
token_limit: 1000
include_patterns:
  - "*.py"
exclude_patterns:
  - "__pycache__"
  - "*.pyc"
  - "*.pyo"
  - "*.pyd"
  - ".git"
  - ".env"
  - "venv"
  - "build"
  - "dist"
  - "*.egg-info"
use_gitignore: true
remove_comments: false
output_format: markdown
output:
  directory: "documentation"
  filename_format: "{base}.{directory}.{timestamp}.md"
  timestamp_format: "%Y%m%d_%H%M%S"
sections:
  - "overview"
  - "dependencies"
  - "details"
analysis:
  languages:
    - python
  include_private: false
  max_depth: 5
erd:
  format: "pdf"
  rankdir: "LR"
  fontname: "Arial"
  fontsize: "10"
  node_style:
    shape: "record"
    style: "filled"
    fillcolor: "white"
    fontname: "Arial"
  edge_style:
    inheritance:
      arrowhead: "empty"
      style: "solid"
      color: "black"
    composition:
      arrowhead: "diamond"
      style: "solid"
      color: "black"
```

#### Output Configuration

The `output` section controls where and how documentation files are generated:

- `directory`: Base directory for all documentation files (created if missing)
- `filename_format`: Template for generated filenames with variables:
  - `{base}`: Base filename (default: "documentation")
  - `{directory}`: Current project directory name (omitted if empty)
  - `{timestamp}`: Current timestamp using the specified format
- `timestamp_format`: Python's strftime format for timestamps

Examples of generated filenames:
```
documentation/documentation.my-project.20240315_143022.md
documentation/documentation.20240315_143022.md  # When in root directory
```

You can override the output location using the `-o` flag:
```bash
codemap generate -o custom/path/docs.md
```

> **Note**: 
> - Missing directories are automatically created
> - Each run creates a new file with a unique timestamp
> - Use custom formats to organize documentation as needed

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/SarthakMishra/code-map.git
cd code-map

# Create a virtual environment using uv (Recommended)
uv venv

# Activate the virtual environment
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
uv sync

# Alternative: Using traditional venv and pip
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Style

We use ruff for linting and formatting:

```bash
# Using uv
uv run ruff check .
uv run ruff format .

# Using pip
ruff check .
ruff format .
```

## Future Plans

### Short-term Goals (v0.2.0)

1. Enhanced Language Support
   - [ ] Add support for Ruby
   - [ ] Add support for Rust
   - [ ] Improve TypeScript/JavaScript parsing

2. Documentation Improvements
   - [ ] Add HTML output format
   - [ ] Support custom documentation templates
   - [ ] Add PlantUML diagram generation

3. Performance Optimizations
   - [ ] Implement caching for parsed files
   - [ ] Add parallel processing for large codebases
   - [ ] Optimize memory usage for token analysis

### Long-term Goals (v1.0.0)

1. Advanced Analysis
   - [ ] Code quality metrics integration
   - [ ] Cyclomatic complexity analysis
   - [ ] Dead code detection
   - [ ] Security vulnerability scanning

2. Integration Features
   - [ ] GitHub Actions integration
   - [ ] VS Code extension
   - [ ] Pre-commit hook support
   - [ ] CI/CD pipeline templates

3. Documentation Enhancements
   - [ ] Interactive documentation viewer
   - [ ] Real-time documentation updates
   - [ ] API documentation generation
   - [ ] Automatic changelog generation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Versioning

We use semantic versioning (SemVer) for version numbers. When upgrading:
- Major version changes (x.0.0) may include breaking changes
- Minor version changes (0.x.0) add functionality in a backward-compatible manner
- Patch version changes (0.0.x) include backward-compatible bug fixes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) for providing the powerful parsing capabilities
- [typer](https://typer.tiangolo.com/) for the excellent CLI framework
- [networkx](https://networkx.org/) for graph analysis capabilities
- [uv](https://astral.sh/uv) for fast Python package operations
- The open-source community for various inspirations and tools