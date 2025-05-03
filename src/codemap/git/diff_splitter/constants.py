"""Constants for diff splitting functionality."""

from typing import Final

# Similarity threshold
MIN_NAME_LENGTH_FOR_SIMILARITY: Final = 3

# Constants for numeric comparisons
EPSILON = 1e-10  # Small value for float comparisons

# Group size limits
MAX_FILES_PER_GROUP: Final = 10

RELATED_FILE_PATTERNS: Final = [
	# --- General Code + Test Files ---
	# Python
	(r"^(.*)\.py$", r"\1_test\.py$"),
	(r"^(.*)\.py$", r"test_\1\.py$"),
	(r"^(.*)\.py$", r"\1_spec\.py$"),
	(r"^(.*)\.py$", r"spec_\1\.py$"),
	# JavaScript / TypeScript (including JSX/TSX)
	(r"^(.*)\.(js|jsx|ts|tsx)$", r"\1\.(test|spec)\.(js|jsx|ts|tsx)$"),
	(r"^(.*)\.(js|jsx|ts|tsx)$", r"\1\.stories\.(js|jsx|ts|tsx)$"),  # Storybook
	(r"^(.*)\.(js|ts)$", r"\1\.d\.ts$"),  # JS/TS + Declaration files
	# Ruby
	(r"^(.*)\.rb$", r"\1_spec\.rb$"),
	(r"^(.*)\.rb$", r"\1_test\.rb$"),
	(r"^(.*)\.rb$", r"spec/.*_spec\.rb$"),  # Common RSpec structure
	# Java
	(r"^(.*)\.java$", r"\1Test\.java$"),
	(r"src/main/java/(.*)\.java$", r"src/test/java/\1Test\.java$"),  # Maven/Gradle structure
	# Go
	(r"^(.*)\.go$", r"\1_test\.go$"),
	# C#
	(r"^(.*)\.cs$", r"\1Tests?\.cs$"),
	# PHP
	(r"^(.*)\.php$", r"\1Test\.php$"),
	(r"^(.*)\.php$", r"\1Spec\.php$"),
	(r"src/(.*)\.php$", r"tests/\1Test\.php$"),  # Common structure
	# Rust
	(r"src/(lib|main)\.rs$", r"tests/.*\.rs$"),  # Main/Lib and integration tests
	(r"src/(.*)\.rs$", r"src/\1_test\.rs$"),  # Inline tests (less common for grouping)
	# Swift
	(r"^(.*)\.swift$", r"\1Tests?\.swift$"),
	# Kotlin
	(r"^(.*)\.kt$", r"\1Test\.kt$"),
	(r"src/main/kotlin/(.*)\.kt$", r"src/test/kotlin/\1Test\.kt$"),  # Common structure
	# --- Frontend Component Bundles ---
	# JS/TS Components + Styles (CSS, SCSS, LESS, CSS Modules)
	(r"^(.*)\.(js|jsx|ts|tsx)$", r"\1\.(css|scss|less)$"),
	(r"^(.*)\.(js|jsx|ts|tsx)$", r"\1\.module\.(css|scss|less)$"),
	(r"^(.*)\.(js|jsx|ts|tsx)$", r"\1\.styles?\.(js|ts)$"),  # Styled Components / Emotion convention
	# Vue Components + Styles
	(r"^(.*)\.vue$", r"\1\.(css|scss|less)$"),
	(r"^(.*)\.vue$", r"\1\.module\.(css|scss|less)$"),
	# Svelte Components + Styles/Scripts
	(r"^(.*)\.svelte$", r"\1\.(css|scss|less)$"),
	(r"^(.*)\.svelte$", r"\1\.(js|ts)$"),
	# Angular Components (more specific structure)
	(r"^(.*)\.component\.ts$", r"\1\.component\.html$"),
	(r"^(.*)\.component\.ts$", r"\1\.component\.(css|scss|less)$"),
	(r"^(.*)\.component\.ts$", r"\1\.component\.spec\.ts$"),  # Component + its test
	(r"^(.*)\.service\.ts$", r"\1\.service\.spec\.ts$"),  # Service + its test
	(r"^(.*)\.module\.ts$", r"\1\.routing\.module\.ts$"),  # Module + routing
	# --- Implementation / Definition / Generation ---
	# C / C++ / Objective-C
	(r"^(.*)\.h$", r"\1\.c$"),
	(r"^(.*)\.h$", r"\1\.m$"),
	(r"^(.*)\.hpp$", r"\1\.cpp$"),
	(r"^(.*)\.h$", r"\1\.cpp$"),  # Allow .h with .cpp
	(r"^(.*)\.h$", r"\1\.mm$"),
	# Protocol Buffers / gRPC
	(r"^(.*)\.proto$", r"\1\.pb\.(go|py|js|java|rb|cs|ts)$"),
	(r"^(.*)\.proto$", r"\1_pb2?\.py$"),  # Python specific proto generation
	(r"^(.*)\.proto$", r"\1_grpc\.pb\.(go|js|ts)$"),  # gRPC specific
	# Interface Definition Languages (IDL)
	(r"^(.*)\.idl$", r"\1\.(h|cpp|cs|java)$"),
	# API Specifications (OpenAPI/Swagger)
	(r"(openapi|swagger)\.(yaml|yml|json)$", r".*\.(go|py|js|java|rb|cs|ts)$"),  # Spec + generated code
	(r"^(.*)\.(yaml|yml|json)$", r"\1\.generated\.(go|py|js|java|rb|cs|ts)$"),  # Another convention
	# --- Web Development (HTML Centric) ---
	(r"^(.*)\.html$", r"\1\.(js|ts)$"),
	(r"^(.*)\.html$", r"\1\.(css|scss|less)$"),
	# --- Mobile Development ---
	# iOS (Swift)
	(r"^(.*)\.swift$", r"\1\.storyboard$"),
	(r"^(.*)\.swift$", r"\1\.xib$"),
	# Android (Kotlin/Java)
	(r"^(.*)\.(kt|java)$", r"res/layout/.*\.(xml)$"),  # Code + Layout XML (Path sensitive)
	(r"AndroidManifest\.xml$", r".*\.(kt|java)$"),  # Manifest + Code
	(r"build\.gradle(\.kts)?$", r".*\.(kt|java)$"),  # Gradle build + Code
	# --- Configuration Files ---
	# Package Managers
	(r"package\.json$", r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml)$"),
	(r"requirements\.txt$", r"(setup\.py|setup\.cfg|pyproject\.toml)$"),
	(r"pyproject\.toml$", r"(setup\.py|setup\.cfg|poetry\.lock|uv\.lock)$"),
	(r"Gemfile$", r"Gemfile\.lock$"),
	(r"Cargo\.toml$", r"Cargo\.lock$"),
	(r"composer\.json$", r"composer\.lock$"),  # PHP Composer
	(r"go\.mod$", r"go\.sum$"),  # Go Modules
	(r"pom\.xml$", r".*\.java$"),  # Maven + Java
	(r"build\.gradle(\.kts)?$", r".*\.(java|kt)$"),  # Gradle + Java/Kotlin
	# Linters / Formatters / Compilers / Type Checkers
	(
		r"package\.json$",
		r"(tsconfig\.json|\.eslintrc(\..*)?|\.prettierrc(\..*)?|\.babelrc(\..*)?|webpack\.config\.js|vite\.config\.(js|ts))$",
	),
	(r"pyproject\.toml$", r"(\.flake8|\.pylintrc|\.isort\.cfg|mypy\.ini)$"),
	# Docker
	(r"Dockerfile$", r"(\.dockerignore|docker-compose\.yml)$"),
	(r"docker-compose\.yml$", r"\.env$"),
	# CI/CD
	(r"\.github/workflows/.*\.yml$", r".*\.(sh|py|js|ts|go)$"),  # Workflow + scripts
	(r"\.gitlab-ci\.yml$", r".*\.(sh|py|js|ts|go)$"),
	(r"Jenkinsfile$", r".*\.(groovy|sh|py)$"),
	# IaC (Terraform)
	(r"^(.*)\.tf$", r"\1\.tfvars$"),
	(r"^(.*)\.tf$", r"\1\.tf$"),  # Group TF files together
	# --- Documentation ---
	(r"README\.md$", r".*$"),  # README often updated with any change
	(r"^(.*)\.md$", r"\1\.(py|js|ts|go|java|rb|rs|php|swift|kt)$"),  # Markdown doc + related code
	(r"docs/.*\.md$", r"src/.*$"),  # Documentation in docs/ related to src/
	# --- Data Science / ML ---
	(r"^(.*)\.ipynb$", r"\1\.py$"),  # Notebook + Python script
	(r"^(.*)\.py$", r"data/.*\.(csv|json|parquet)$"),  # Script + Data file (path sensitive)
	# --- General Fallbacks (Use with caution) ---
	# Files with same base name but different extensions (already covered by some specifics)
	# (r"^(.*)\..*$", r"\1\..*$"), # Potentially too broad, rely on specifics above
]
