# CodeMap

⚠️ **DEVELOPMENT STATUS**: CodeMap is currently in active development and testing phase. Use with caution in production environments.

CodeMap is a powerful CLI tool that generates optimized markdown documentation from your codebase using tree-sitter analysis. It analyzes source code, creates repository maps, and generates markdown files that can be used as context for LLMs.

## Features

- 🔍 Smart code analysis using tree-sitter
- 📊 Dependency graph generation and analysis
- 🎯 Token-optimized documentation generation
- 📝 Rich markdown output with code structure
- 🌳 Repository structure visualization
- 🔄 Automatic docstring extraction
- 🎨 Language-aware syntax parsing

## Installation

Since CodeMap is currently in development, it can be installed globally directly from GitHub using our installation script:

```bash
sudo curl -LsSf https://raw.githubusercontent.com/SarthakMishra/code-map/main/scripts/install.sh | sudo bash
```

This script will:
1. Clone the repository to a temporary directory
2. Install CodeMap globally in your system Python environment
3. Clean up temporary files automatically
4. Make the `codemap` command available system-wide

After installation, you can use CodeMap from anywhere:

```bash
codemap generate /path/to/your/project
```

> **Note**: 
> - This installation requires sudo privileges as it installs CodeMap globally.
> - The tool is under active development, you might encounter occasional issues.
> - Please report any problems in the GitHub issues section.

## Usage

### Basic Usage

Generate documentation for your project:

```bash
codemap generate /path/to/your/project
```

The tool will analyze your codebase and create a `documentation.md` file in the current directory.

### Command Options

```bash
codemap generate [OPTIONS] [PATH]

Arguments:
  PATH  Path to the codebase to analyze [default: .]

Options:
  -o, --output PATH    Output file path [default: documentation.md]
  -c, --config PATH    Path to config file
  --map-tokens INT     Override token limit
  --help              Show this message and exit
```

### Configuration

Create a `.codemap.yml` file in your project root to customize the behavior:

```yaml
token_limit: 1000
ignore_patterns:
  - "__pycache__"
  - "*.pyc"
  - "*.pyo"
  - "*.pyd"
  - ".git"
  - ".env"
  - "venv"
use_gitignore: true
remove_comments: false
output_format: markdown
analysis:
  languages:
    - python
    - javascript
    - typescript
    - java
    - go
  include_private: false
  max_depth: 5
```

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) for providing the powerful parsing capabilities
- [typer](https://typer.tiangolo.com/) for the excellent CLI framework
- [networkx](https://networkx.org/) for graph analysis capabilities
- [uv](https://astral.sh/uv) for fast Python package operations
- The open-source community for various inspirations and tools