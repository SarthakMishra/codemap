"""Default configuration settings for the CodeMap tool."""

DEFAULT_CONFIG = {
    "token_limit": 1000,
    "ignore_patterns": [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".git",
        ".env",
        "venv",
    ],
    "use_gitignore": True,
    "remove_comments": False,
    "output_format": "markdown",
    "analysis": {
        "languages": ["python", "javascript", "typescript", "java", "go"],
        "include_private": False,
        "max_depth": 5,
    },
}
