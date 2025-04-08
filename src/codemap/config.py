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
            "model": "gpt-4o-mini",
            "provider": "openai",
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
        # Semantic chunking configuration
        "semantic": {
            # Threshold for semantic similarity (0.0-1.0)
            "similarity_threshold": 0.7,
            # Embedding model for code similarity
            "embedding_model": "flax-sentence-embeddings/st-codesearch-distroberta-base",
            # Fallback model if primary model fails to load
            "fallback_model": "all-MiniLM-L6-v2",
            # Language-specific settings
            "languages": {
                # Recognized code files for semantic parsing
                "extensions": [
                    "py",  # Python
                    "js",  # JavaScript
                    "ts",  # TypeScript
                    "java",  # Java
                    "kt",  # Kotlin
                    "go",  # Go
                    "c",  # C
                    "cpp",  # C++
                    "cs",  # C#
                    "rb",  # Ruby
                    "php",  # PHP
                    "swift",  # Swift
                ],
                # Cache embeddings to improve performance on subsequent runs
                "cache_embeddings": True,
                # Maximum size of embedding cache
                "max_cache_size": 1000,
            },
        },
    },
}
