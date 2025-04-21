"""Default configuration settings for the codemap tool."""

DEFAULT_CONFIG = {
    # Maximum number of tokens to include (0 for unlimited)
    "token_limit": 10000,
    # Whether to respect gitignore patterns
    "use_gitignore": True,
    # Directory to store documentation files
    "output_dir": "documentation",
    # Maximum content length per file (0 for unlimited)
    "max_content_length": 5000,
    # Commit feature configuration
    "commit": {
        # Strategy for splitting diffs: file, hunk, semantic
        "strategy": "file",
        # LLM configuration
        "llm": {
            "model": "openai/gpt-4o-mini",
            "api_base": None,
        },
        # Commit convention settings
        "convention": {
            "types": [
                "feat",  # New feature
                "fix",  # Bug fix
                "docs",  # Documentation
                "style",  # Formatting, missing semicolons, etc.
                "refactor",  # Code change that neither fixes a bug nor adds a feature
                "perf",  # Performance improvement
                "test",  # Adding or updating tests
                "build",  # Build system or external dependencies
                "ci",  # CI configuration
                "chore",  # Other changes that don't modify src or test files
            ],
            "scopes": [],
            "max_length": 72,
        },
    },
}
