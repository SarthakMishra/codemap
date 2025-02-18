"""Default configuration settings for the codemap tool."""

DEFAULT_CONFIG = {
    "token_limit": 1000,
    "include_patterns": ["*.py", "*.js", "*.ts"],
    "exclude_patterns": [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".git",
        ".env",
        ".venv",
    ],
    "use_gitignore": True,
    "remove_comments": False,
    "output_format": "markdown",
    "output": {
        "directory": "documentation",
        "filename_format": "{base}.{directory}.{timestamp}.md",
        "timestamp_format": "%Y%m%d_%H%M%S",
    },
    "sections": ["overview", "dependencies", "details"],
    "analysis": {
        "languages": ["python", "javascript", "typescript", "java", "go"],
        "include_private": False,
        "max_depth": 5,
    },
}
