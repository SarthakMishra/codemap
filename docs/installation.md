# Installation

/// warning
CodeMap currently only supports Unix-based platforms (macOS, Linux). For Windows users, we recommend using Windows Subsystem for Linux (WSL).
///

/// tip
After installation, you can use either `codemap` or the shorter alias `cm` to run the commands.
///

## Installation using pipx (Recommended)

Using `pipx` is recommended as it installs the package in an isolated environment and automatically manages the PATH.

```bash
# Ensure pipx is installed (install it if you haven't)
# python3 -m pip install --user pipx
# python3 -m pipx ensurepath

# Install codemap from PyPI
pipx install codemap
```

## Alternative: Manual Installation using pip

If you prefer not to use `pipx`, you can install using `pip` directly:

```bash
# Install with pip (user installation)
pip install --user codemap

# Make sure your PATH includes the user bin directory
# Add the following to your shell profile (e.g., ~/.bashrc, ~/.zshrc):
# export PATH="$HOME/.local/bin:$PATH"
# Or find the correct path using: python3 -m site --user-base
```

## Development Version (Latest GitHub)

If you want to try the latest development version with unreleased features:

```bash
# Using pipx
pipx install git+https://github.com/SarthakMishra/codemap.git

# Or using pip
pip install --user git+https://github.com/SarthakMishra/codemap.git
```

## Updating CodeMap

To update CodeMap to the latest version:

```bash
# If installed with pipx from PyPI
pipx upgrade codemap

# If installed with pip from PyPI
pip install --user --upgrade codemap

# If installed from GitHub
pipx upgrade codemap  # or
pip install --user --upgrade git+https://github.com/SarthakMishra/codemap.git
```

## Uninstalling

```bash
# If installed with pipx
pipx uninstall codemap

# If installed with pip
pip uninstall codemap
``` 