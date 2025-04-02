# CodeMap

⚠️ **DEVELOPMENT STATUS**: CodeMap is currently in active development and testing phase. Use with caution in production environments.

CodeMap is a powerful CLI tool that generates optimized markdown documentation from your Python codebase. It analyzes source code, creates repository maps, and produces markdown files that can be used as context for LLMs.

## Features

- 🎯 Token-optimized documentation generation
- 📝 Rich markdown output with code structure
- 🌳 Repository structure visualization

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

### Configuration

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
- [uv](https://astral.sh/uv) for fast Python package operations
- The open-source community for various inspirations and tools