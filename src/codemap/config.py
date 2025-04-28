"""Default configuration settings for the codemap tool."""

from textwrap import dedent

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
		# Whether to bypass git hooks with --no-verify when committing
		"bypass_hooks": False,
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
	# Pull request configuration
	"pr": {
		# Default branch settings
		"defaults": {
			"base_branch": None,  # Defaults to repo default if None
			"feature_prefix": "feature/",
		},
		# Git workflow strategy: github-flow, gitflow, trunk-based
		"strategy": "github-flow",
		# Branch mapping for different PR types (used by GitFlow)
		"branch_mapping": {
			"feature": {
				"base": "develop",
				"prefix": "feature/",
			},
			"release": {
				"base": "main",
				"prefix": "release/",
			},
			"hotfix": {
				"base": "main",
				"prefix": "hotfix/",
			},
			"bugfix": {
				"base": "develop",
				"prefix": "bugfix/",
			},
		},
		# Content generation settings
		"generate": {
			"title_strategy": "commits",  # Options: commits, llm, template
			"description_strategy": "commits",  # Options: commits, llm, template
			# Template for PR descriptions (used with description_strategy: "template")
			"description_template": dedent("""\
			## Changes
			{changes}

			## Testing
			{testing_instructions}

			## Screenshots
			{screenshots}
			"""),
			# Whether to use PR templates from workflow strategies
			"use_workflow_templates": True,
		},
	},
	# Server configuration - covers both daemon and API server
	"server": {
		# Network settings
		"host": "127.0.0.1",
		"port": 8765,
		# CORS settings
		"allow_cors": True,
		"cors_origins": ["*"],
		# Performance
		"max_connections": 10,
		# Daemon specific
		"log_level": "info",
		"pid_file": "~/.codemap/daemon.pid",
		"log_file": "~/.codemap/daemon.log",
		"auto_start": False,
	},
	# Processing configuration
	"processing": {
		"num_workers": 2,
		"timeout": 300,
		"chunk_size": 4096,
	},
	# Storage configuration
	"storage": {
		"data_dir": "~/.codemap/data",
		"use_cache": True,
		"cache_size_mb": 512,
	},
}
