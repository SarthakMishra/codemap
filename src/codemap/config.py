"""Default configuration settings for the codemap tool."""

from textwrap import dedent

DEFAULT_CONFIG = {
	# LLM configuration
	"llm": {
		"model": "openai/gpt-4o-mini",
		"api_base": None,
	},
	# Gen command configuration
	"gen": {
		# Maximum content length per file (0 for unlimited)
		"max_content_length": 5000,
		# Whether to respect gitignore patterns
		"use_gitignore": True,
		# Directory to store documentation files
		"output_dir": "documentation",
		# Whether to include directory tree in output
		"include_tree": True,
		# Whether to include entity relationship graph in output
		"include_entity_graph": True,
		# Enable semantic analysis using LSP
		"semantic_analysis": True,
		# Level of Detail for code analysis (signatures, structure, docs, full)
		"lod_level": "docs",
		# Mermaid entities to include (lowercase EntityType names, e.g., module, class, function)
		"mermaid_entities": ["module", "class", "function", "method", "constant", "variable", "import"],
		# Mermaid relationships to include (declares, imports, calls)
		"mermaid_relationships": ["declares", "imports", "calls"],
		# Whether to show the legend in the Mermaid diagram
		"mermaid_show_legend": True,
		# Whether to remove nodes with no connections in the Mermaid diagram
		"mermaid_remove_unconnected": False,
	},
	# Processor configuration
	"processor": {
		"enabled": True,
		"max_workers": 4,
		"ignored_patterns": [
			"**/.git/**",
			"**/__pycache__/**",
			"**/.venv/**",
			"**/node_modules/**",
			"**/*.pyc",
			"**/dist/**",
			"**/build/**",
		],
		# Default LOD level for processing
		"default_lod_level": "signatures",
	},
	# Commit feature configuration
	"commit": {
		# Strategy for splitting diffs: file, hunk, semantic
		"strategy": "file",
		# Whether to bypass git hooks with --no-verify when committing
		"bypass_hooks": False,
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
		# Commit lint rules - loaded from .commitlintrc and structured for YAML
		"lint": {
			# Header rules
			"header_max_length": {
				"level": "ERROR",
				"rule": "always",
				"value": 100,
			},
			"header_case": {
				"level": "DISABLED",
				"rule": "always",
				"value": "lower-case",
			},
			"header_full_stop": {
				"level": "ERROR",
				"rule": "never",
				"value": ".",
			},
			# Type rules
			"type_enum": {
				"level": "ERROR",
				"rule": "always",
				# Uses types from commit.convention.types
			},
			"type_case": {
				"level": "ERROR",
				"rule": "always",
				"value": "lower-case",
			},
			"type_empty": {
				"level": "ERROR",
				"rule": "never",
			},
			# Scope rules
			"scope_case": {
				"level": "ERROR",
				"rule": "always",
				"value": "lower-case",
			},
			"scope_empty": {
				"level": "DISABLED",
				"rule": "never",
			},
			"scope_enum": {
				"level": "DISABLED",
				"rule": "always",
				# Uses scopes from commit.convention.scopes
			},
			# Subject rules
			"subject_case": {
				"level": "ERROR",
				"rule": "never",
				"value": ["sentence-case", "start-case", "pascal-case", "upper-case"],
			},
			"subject_empty": {
				"level": "ERROR",
				"rule": "never",
			},
			"subject_full_stop": {
				"level": "ERROR",
				"rule": "never",
				"value": ".",
			},
			"subject_exclamation_mark": {
				"level": "DISABLED",
				"rule": "never",
			},
			# Body rules
			"body_leading_blank": {
				"level": "WARNING",
				"rule": "always",
			},
			"body_empty": {
				"level": "DISABLED",
				"rule": "never",
			},
			"body_max_line_length": {
				"level": "ERROR",
				"rule": "always",
				"value": 100,
			},
			# Footer rules
			"footer_leading_blank": {
				"level": "WARNING",
				"rule": "always",
			},
			"footer_empty": {
				"level": "DISABLED",
				"rule": "never",
			},
			"footer_max_line_length": {
				"level": "ERROR",
				"rule": "always",
				"value": 100,
			},
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
}
