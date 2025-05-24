"""Schemas for the CodeMap configuration."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from textwrap import dedent
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from codemap.processor.lod import LODLevel


# LLM Configuration
class LLMSchema(BaseModel):
	"""Configuration for the LLM."""

	model: str = "openai:gpt-4o-mini"
	base_url: str | None = None
	temperature: float = 0.5
	max_output_tokens: int = 1024


# Embedding Configuration
class EmbeddingChunkingSchema(BaseModel):
	"""Configuration for the embedding chunking."""

	max_hierarchy_depth: int = 2
	max_file_lines: int = 1000


class AgglomerativeClusteringSchema(BaseModel):
	"""Configuration for the agglomerative clustering."""

	metric: Literal["cosine", "euclidean", "manhattan", "l1", "l2", "precomputed"] = "precomputed"
	distance_threshold: float = 0.3
	linkage: Literal["ward", "complete", "average", "single"] = "complete"


class DBSCANSchema(BaseModel):
	"""Configuration for the DBSCAN clustering."""

	eps: float = 0.3
	min_samples: int = 2
	algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
	metric: Literal["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "precomputed"] = "precomputed"


class EmbeddingClusteringSchema(BaseModel):
	"""Configuration for the embedding clustering."""

	method: Literal["agglomerative", "dbscan"] = "agglomerative"
	agglomerative: AgglomerativeClusteringSchema = AgglomerativeClusteringSchema()
	dbscan: DBSCANSchema = DBSCANSchema()


class EmbeddingSchema(BaseModel):
	"""Configuration for the embedding."""

	model_name: str = "minishlab/potion-base-8M"
	dimension: int = 256
	dimension_metric: str = "cosine"
	max_content_length: int = 5000
	qdrant_batch_size: int = 1000
	url: str = "http://localhost:6333"
	api_key: str | None = None
	timeout: int = 120
	prefer_grpc: bool = True
	chunking: EmbeddingChunkingSchema = EmbeddingChunkingSchema()
	clustering: EmbeddingClusteringSchema = EmbeddingClusteringSchema()


class RAGSchema(BaseModel):
	"""Configuration for the RAG."""

	max_context_length: int = 10000
	max_context_results: int = 10
	similarity_threshold: float = 0.75
	system_prompt: str | None = None
	include_file_content: bool = True
	include_metadata: bool = True


class SyncSchema(BaseModel):
	"""Configuration for the sync."""

	exclude_patterns: list[str] = Field(
		default_factory=lambda: [
			r"^node_modules/",
			r"^\.venv/",
			r"^venv/",
			r"^env/",
			r"^__pycache__/",
			r"^\.mypy_cache/",
			r"^\.pytest_cache/",
			r"^\.ruff_cache/",
			r"^dist/",
			r"^build/",
			r"^\.git/",
			r"\\.pyc$",
			r"\\.pyo$",
			r"\\.so$",
			r"\\.dll$",
		]
	)


class GenSchema(BaseModel):
	"""Configuration for the gen command."""

	max_content_length: int = 5000
	use_gitignore: bool = True
	output_dir: str = "documentation"
	include_tree: bool = True
	include_entity_graph: bool = True
	semantic_analysis: bool = True
	lod_level: LODLevel = LODLevel.DOCS
	mermaid_entities: list[str] = Field(
		default_factory=lambda: ["module", "class", "function", "method", "constant", "variable", "import"]
	)
	mermaid_relationships: list[str] = Field(default_factory=lambda: ["declares", "imports", "calls"])
	mermaid_show_legend: bool = True
	mermaid_remove_unconnected: bool = False
	mermaid_styled: bool = True

	@field_validator("lod_level", mode="before")
	@classmethod
	def validate_lod_level(cls, v: LODLevel | str | int) -> LODLevel:
		"""Convert string values to LODLevel enum."""
		if isinstance(v, LODLevel):
			return v
		if isinstance(v, str):
			# Direct mapping approach to avoid enum access issues
			level_map = {
				"signatures": LODLevel.SIGNATURES,
				"structure": LODLevel.STRUCTURE,
				"docs": LODLevel.DOCS,
				"skeleton": LODLevel.SKELETON,
				"full": LODLevel.FULL,
				"1": LODLevel.SIGNATURES,
				"2": LODLevel.STRUCTURE,
				"3": LODLevel.DOCS,
				"4": LODLevel.SKELETON,
				"5": LODLevel.FULL,
			}
			# Try lowercase first, then original case
			if v.lower() in level_map:
				return level_map[v.lower()]
			if v in level_map:
				return level_map[v]
			valid_values = list(level_map.keys())
			msg = f"Invalid lod_level '{v}'. Valid values are: {valid_values}"
			raise ValueError(msg)
		if isinstance(v, int):
			# Handle numeric values
			try:
				return LODLevel(v)
			except ValueError:
				valid_numbers = [1, 2, 3, 4, 5]
				msg = f"Invalid lod_level number '{v}'. Valid numbers are: {valid_numbers}"
				raise ValueError(msg) from None
		msg = f"lod_level must be a string, number, or LODLevel enum, got {type(v)}"
		raise ValueError(msg)


class WatcherSchema(BaseModel):
	"""Configuration for the watcher."""

	enabled: bool = True
	debounce_delay: float = 1.0


class ProcessorSchema(BaseModel):
	"""Configuration for the processor."""

	enabled: bool = True
	max_workers: int = 4
	ignored_patterns: list[str] = Field(
		default_factory=lambda: [
			"**/.git/**",
			"**/__pycache__/**",
			"**/.venv/**",
			"**/node_modules/**",
			"**/*.pyc",
			"**/dist/**",
			"**/build/**",
		]
	)
	watcher: WatcherSchema = WatcherSchema()
	default_lod_level: str = "signatures"


class DiffSplitterSchema(BaseModel):
	"""Configuration for the diff splitter."""

	similarity_threshold: float = 0.6
	directory_similarity_threshold: float = 0.3
	file_move_similarity_threshold: float = 0.85
	min_chunks_for_consolidation: int = 2
	max_chunks_before_consolidation: int = 20
	max_file_size_for_llm: int = 50000
	max_log_diff_size: int = 1000
	default_code_extensions: list[str] = Field(
		default_factory=lambda: [
			"js",
			"jsx",
			"ts",
			"tsx",
			"py",
			"java",
			"c",
			"cpp",
			"h",
			"hpp",
			"cc",
			"cs",
			"go",
			"rb",
			"php",
			"rs",
			"swift",
			"scala",
			"kt",
			"sh",
			"pl",
			"pm",
		]
	)


class CommitConventionSchema(BaseModel):
	"""Configuration for the commit convention."""

	types: list[str] = Field(
		default_factory=lambda: [
			"feat",
			"fix",
			"docs",
			"style",
			"refactor",
			"perf",
			"test",
			"build",
			"ci",
			"chore",
		]
	)
	scopes: list[str] = Field(default_factory=list)
	max_length: int = 72


class LintRuleSchema(BaseModel):
	"""Configuration for the lint rule."""

	level: str  # "ERROR", "WARNING", "DISABLED"
	rule: str  # "always", "never"
	value: Any | None = None  # Can be int, str, list of str


class CommitLintSchema(BaseModel):
	"""Configuration for the commit lint."""

	header_max_length: LintRuleSchema = Field(
		default_factory=lambda: LintRuleSchema(level="ERROR", rule="always", value=100)
	)
	header_case: LintRuleSchema = Field(
		default_factory=lambda: LintRuleSchema(level="DISABLED", rule="always", value="lower-case")
	)
	header_full_stop: LintRuleSchema = Field(
		default_factory=lambda: LintRuleSchema(level="ERROR", rule="never", value=".")
	)
	type_enum: LintRuleSchema = Field(default_factory=lambda: LintRuleSchema(level="ERROR", rule="always"))
	type_case: LintRuleSchema = Field(
		default_factory=lambda: LintRuleSchema(level="ERROR", rule="always", value="lower-case")
	)
	type_empty: LintRuleSchema = Field(default_factory=lambda: LintRuleSchema(level="ERROR", rule="never"))
	scope_case: LintRuleSchema = Field(
		default_factory=lambda: LintRuleSchema(level="ERROR", rule="always", value="lower-case")
	)
	scope_empty: LintRuleSchema = Field(default_factory=lambda: LintRuleSchema(level="DISABLED", rule="never"))
	scope_enum: LintRuleSchema = Field(default_factory=lambda: LintRuleSchema(level="DISABLED", rule="always"))
	subject_case: LintRuleSchema = Field(
		default_factory=lambda: LintRuleSchema(
			level="ERROR", rule="never", value=["sentence-case", "start-case", "pascal-case", "upper-case"]
		)
	)
	subject_empty: LintRuleSchema = Field(default_factory=lambda: LintRuleSchema(level="ERROR", rule="never"))
	subject_full_stop: LintRuleSchema = Field(
		default_factory=lambda: LintRuleSchema(level="ERROR", rule="never", value=".")
	)
	subject_exclamation_mark: LintRuleSchema = Field(
		default_factory=lambda: LintRuleSchema(level="DISABLED", rule="never")
	)
	body_leading_blank: LintRuleSchema = Field(default_factory=lambda: LintRuleSchema(level="WARNING", rule="always"))
	body_empty: LintRuleSchema = Field(default_factory=lambda: LintRuleSchema(level="DISABLED", rule="never"))
	body_max_line_length: LintRuleSchema = Field(
		default_factory=lambda: LintRuleSchema(level="ERROR", rule="always", value=100)
	)
	footer_leading_blank: LintRuleSchema = Field(default_factory=lambda: LintRuleSchema(level="WARNING", rule="always"))
	footer_empty: LintRuleSchema = Field(default_factory=lambda: LintRuleSchema(level="DISABLED", rule="never"))
	footer_max_line_length: LintRuleSchema = Field(
		default_factory=lambda: LintRuleSchema(level="ERROR", rule="always", value=100)
	)


class CommitSchema(BaseModel):
	"""Configuration for the commit."""

	strategy: str = "file"
	bypass_hooks: bool = False
	use_lod_context: bool = True
	is_non_interactive: bool = False
	diff_splitter: DiffSplitterSchema = DiffSplitterSchema()
	convention: CommitConventionSchema = CommitConventionSchema()
	lint: CommitLintSchema = CommitLintSchema()


class PRDefaultsSchema(BaseModel):
	"""Configuration for the pull request defaults."""

	base_branch: str | None = None
	feature_prefix: str = "feature/"


class PRBranchMappingDetailSchema(BaseModel):
	"""Configuration for the pull request branch mapping detail."""

	base: str
	prefix: str


class PRBranchMappingSchema(BaseModel):
	"""Configuration for the pull request branch mapping."""

	feature: PRBranchMappingDetailSchema = Field(
		default_factory=lambda: PRBranchMappingDetailSchema(base="develop", prefix="feature/")
	)
	release: PRBranchMappingDetailSchema = Field(
		default_factory=lambda: PRBranchMappingDetailSchema(base="main", prefix="release/")
	)
	hotfix: PRBranchMappingDetailSchema = Field(
		default_factory=lambda: PRBranchMappingDetailSchema(base="main", prefix="hotfix/")
	)
	bugfix: PRBranchMappingDetailSchema = Field(
		default_factory=lambda: PRBranchMappingDetailSchema(base="develop", prefix="bugfix/")
	)


class PRGenerateSchema(BaseModel):
	"""Configuration for the pull request generate."""

	title_strategy: Literal["commits", "llm"] = "llm"
	description_strategy: Literal["commits", "llm"] = "llm"
	description_template: str = Field(
		default_factory=lambda: dedent(
			"""
            ## Changes
            {changes}

            ## Testing
            {testing_instructions}

            ## Screenshots
            {screenshots}
            """
		).strip()
	)
	use_workflow_templates: bool = True


class PRSchema(BaseModel):
	"""Configuration for the pull request."""

	defaults: PRDefaultsSchema = Field(default_factory=PRDefaultsSchema)
	strategy: str = "github-flow"
	branch_mapping: PRBranchMappingSchema = Field(default_factory=PRBranchMappingSchema)
	generate: PRGenerateSchema = Field(default_factory=PRGenerateSchema)


class AskSchema(BaseModel):
	"""Configuration for the ask command."""

	interactive_chat: bool = False


class GitHubConfigSchema(BaseModel):
	"""Configuration for GitHub integration (PRs, API, etc)."""

	token: str | None = None  # OAuth token for GitHub API
	repo: str | None = None  # Optional: default repo (e.g., user/repo)


class AppConfigSchema(BaseModel):
	"""Configuration for the application."""

	llm: LLMSchema = LLMSchema()
	embedding: EmbeddingSchema = EmbeddingSchema()
	rag: RAGSchema = RAGSchema()
	sync: SyncSchema = SyncSchema()
	gen: GenSchema = GenSchema()
	processor: ProcessorSchema = ProcessorSchema()
	commit: CommitSchema = CommitSchema()
	pr: PRSchema = PRSchema()
	ask: AskSchema = AskSchema()
	github: GitHubConfigSchema = GitHubConfigSchema()
	repo_root: Path | None = None

	model_config = {
		"validate_assignment": True  # Useful for ensuring type checks on assignment if loaded config is modified
	}
