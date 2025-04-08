"""CLI package for CodeMap."""

# Only export what we need directly from the module
from .commit import app as commit_app

__all__ = ["commit_app"]
