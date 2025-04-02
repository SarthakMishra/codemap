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
}
