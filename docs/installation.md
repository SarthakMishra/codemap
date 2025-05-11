# Installation

/// warning
CodeMap currently only supports Unix-based platforms (macOS, Linux). For Windows users, we recommend using Windows Subsystem for Linux (WSL).
///

/// tip
After installation, you can use either `codemap` or the shorter alias `cm` to run the commands.
///

## Installation using uv (Recommended)

Using `uv` is recommended as it installs the package in an isolated environment and automatically manages the PATH.


```bash
# Stable version:
uv tool install codemap
```

```bash
# Development Version:
uv tool install codemap --prerelease allow
```

## Updating CodeMap

To update CodeMap to the latest version:

```bash
uv tool upgrade codemap
```

## Uninstalling

```bash
uv tool uninstall codemap
```