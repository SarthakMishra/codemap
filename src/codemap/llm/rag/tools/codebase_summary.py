"""Codebase summary tool for PydanticAI agents to generate complete codebase documentation."""

import contextlib
import logging
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import Tool

from codemap.config import ConfigLoader
from codemap.config.config_schema import GenSchema
from codemap.gen.generator import CodeMapGenerator
from codemap.gen.utils import process_codebase
from codemap.processor.lod import LODLevel
from codemap.utils.file_utils import is_binary_file
from codemap.utils.git_utils import GitRepoContext
from codemap.utils.path_utils import filter_paths_by_gitignore

logger = logging.getLogger(__name__)

# Cache for repository paths to avoid expensive scans
_repo_paths_cache: dict[str, list[Path]] = {}

# Constants for fuzzy search scoring
MIN_FUZZY_SIMILARITY = 0.3
MAX_PATH_LENGTH_FOR_FUZZY = 100
MIN_PATH_SIMILARITY = 0.2
SOURCE_BONUS = 0.1


@lru_cache(maxsize=10)
def get_all_valid_paths_cached(repo_root_str: str) -> list[Path]:
	"""Get all valid source code paths in the repository with caching.

	Args:
	    repo_root_str: Repository root directory as string (for caching)

	Returns:
	    List of valid directory and file paths
	"""
	repo_root = Path(repo_root_str)

	# Check cache first
	if repo_root_str in _repo_paths_cache:
		logger.debug(f"Using cached paths for {repo_root_str}")
		return _repo_paths_cache[repo_root_str]

	logger.debug(f"Scanning repository paths for {repo_root_str}")
	all_paths = []

	# Get git context for ignore checking
	try:
		git_context = GitRepoContext.get_instance()
	except (OSError, ValueError, RuntimeError):
		git_context = None

	try:
		# Get all directories first (faster than files)
		directories = []
		for item in repo_root.rglob("*"):
			if not item.is_dir():
				continue

			# Check git ignore status if available
			if git_context:
				try:
					relative_path = str(item.relative_to(repo_root))
					if git_context.is_git_ignored(relative_path):
						continue
				except (ValueError, OSError):
					pass

			directories.append(item)

		# Add important source files (limit to avoid performance issues)
		important_extensions = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h", ".hpp"}
		source_files = []
		file_count = 0
		max_files = 1000  # Limit to prevent performance issues

		for ext in important_extensions:
			if file_count >= max_files:
				break
			for item in repo_root.rglob(f"*{ext}"):
				if file_count >= max_files:
					break
				if item.is_file() and not is_binary_file(item):
					# Check git ignore status if available
					if git_context:
						try:
							relative_path = str(item.relative_to(repo_root))
							if git_context.is_git_ignored(relative_path):
								continue
						except (ValueError, OSError):
							pass

					source_files.append(item)
					file_count += 1

		all_paths = directories + source_files

	except (OSError, PermissionError) as e:
		logger.warning(f"Error scanning repository paths: {e}")

	# Cache the results
	_repo_paths_cache[repo_root_str] = all_paths
	logger.debug(f"Cached {len(all_paths)} paths for {repo_root_str}")
	return all_paths


def calculate_path_similarity(search_term: str, target_path: Path, repo_root: Path) -> float:
	"""Calculate similarity score between search term and path.

	Args:
	    search_term: The search term to match against
	    target_path: The path to score
	    repo_root: Repository root for creating relative paths

	Returns:
	    Similarity score between 0.0 and 1.0 (higher is better)
	"""
	try:
		# Get relative path for scoring
		rel_path = target_path.relative_to(repo_root)
		path_str = str(rel_path)

		# Multiple scoring strategies
		scores = []

		# 1. Exact name match (highest priority)
		path_name = target_path.name
		if search_term.lower() == path_name.lower():
			return 1.0

		# 2. Exact segment match in path
		path_parts = rel_path.parts
		for part in path_parts:
			if search_term.lower() == part.lower():
				scores.append(0.95)  # Very high score for exact segment matches
				break

		# 3. Substring containment (fast check)
		if search_term.lower() in path_name.lower():
			scores.append(0.8)

		# 4. Path segment substring matches
		for part in path_parts:
			if search_term.lower() in part.lower():
				scores.append(0.7)
				break

		# 5. Fuzzy string matching (more expensive, do last)
		path_name_ratio = SequenceMatcher(None, search_term.lower(), path_name.lower()).ratio()
		if path_name_ratio > MIN_FUZZY_SIMILARITY:  # Only consider if somewhat similar
			scores.append(path_name_ratio * 0.9)  # Scale down fuzzy scores

		# 6. Full path fuzzy matching (least priority)
		if len(path_str) < MAX_PATH_LENGTH_FOR_FUZZY:  # Only for shorter paths to avoid performance issues
			path_ratio = SequenceMatcher(None, search_term.lower(), path_str.lower()).ratio()
			if path_ratio > MIN_PATH_SIMILARITY:
				scores.append(path_ratio * 0.6)  # Scale down further

		# 7. Bonus for source code directories
		if any(part in ["src", "lib", "source"] for part in path_parts) and scores:
			scores.append(max(scores) + SOURCE_BONUS)  # Small bonus

		# Return the maximum score from all strategies
		return max(scores) if scores else 0.0

	except ValueError:
		# Path is not relative to repo_root
		return 0.0


def find_matching_paths(path_name: str, repo_root: Path) -> list[Path]:
	"""Find matching paths in the codebase using optimized fuzzy search.

	Args:
	    path_name: Name or partial path to search for
	    repo_root: Repository root directory

	Returns:
	    List of matching paths found in the codebase, sorted by similarity score
	"""
	logger.info(f"Finding paths for '{path_name}' in repo_root: {repo_root}")

	# Try direct path first (exact match)
	direct_path = repo_root / path_name
	logger.info(f"Trying direct path: {direct_path} -> exists: {direct_path.exists()}")
	if direct_path.exists():
		return [direct_path]

	# Try common patterns before expensive fuzzy search
	common_patterns = [
		f"src/codemap/{path_name}",
		f"src/{path_name}",
		f"tests/{path_name}",
		f"docs/{path_name}",
	]

	exact_matches = []
	for pattern in common_patterns:
		candidate_path = repo_root / pattern
		logger.debug(f"Trying pattern '{pattern}': {candidate_path} -> exists: {candidate_path.exists()}")
		if candidate_path.exists():
			exact_matches.append(candidate_path)

	# If we found exact matches, return them
	if exact_matches:
		logger.info(f"Found {len(exact_matches)} exact matches from patterns: {exact_matches}")
		return exact_matches

	# Fuzzy search: get cached paths and score them
	logger.info(f"No exact matches found, performing optimized fuzzy search for '{path_name}'")

	all_paths = get_all_valid_paths_cached(str(repo_root))
	logger.debug(f"Found {len(all_paths)} total valid paths for fuzzy matching")

	# Early exit if no paths
	if not all_paths:
		logger.warning("No valid paths found in repository for fuzzy search")
		return []

	# Score all paths and filter by minimum threshold
	scored_paths = []
	min_score_threshold = MIN_FUZZY_SIMILARITY  # Use constant
	max_to_score = 2000  # Limit scoring to prevent performance issues

	paths_to_score = all_paths[:max_to_score] if len(all_paths) > max_to_score else all_paths

	for path in paths_to_score:
		score = calculate_path_similarity(path_name, path, repo_root)
		if score >= min_score_threshold:
			scored_paths.append((path, score))

	# Sort by score (highest first) and limit results
	scored_paths.sort(key=lambda x: x[1], reverse=True)
	max_results = 10  # Limit to top 10 matches

	matching_paths = [path for path, score in scored_paths[:max_results]]

	if matching_paths:
		logger.info(f"Found {len(matching_paths)} fuzzy matches for '{path_name}':")
		for i, (path, score) in enumerate(scored_paths[:5]):  # Log top 5
			try:
				rel_path = path.relative_to(repo_root)
				logger.info(f"  {i + 1}. {rel_path} (score: {score:.3f})")
			except ValueError:
				logger.info(f"  {i + 1}. {path} (score: {score:.3f})")
	else:
		logger.info(f"No fuzzy matches found for '{path_name}' above threshold {min_score_threshold}")

	return matching_paths


def analyze_path_structure(target_path: Path) -> tuple[str, int]:
	"""Analyze path structure and determine appropriate LOD level.

	Args:
	    target_path: Path to analyze (can be relative or absolute)

	Returns:
	    Tuple of (lod_level, max_depth)
	"""
	if target_path.is_file():
		# Single file
		return "skeleton", 0

	if not target_path.is_dir():
		# Path doesn't exist or is not accessible
		return "signatures", 0

	# Find repository root using GitRepoContext
	try:
		repo_root = GitRepoContext.get_repo_root(target_path)
		git_context = GitRepoContext.get_instance()
		logger.debug(f"Found git repository root: {repo_root}")
	except (OSError, ValueError, RuntimeError):
		# Not a git repository or error finding root, use target_path as fallback
		repo_root = target_path if target_path.is_dir() else target_path.parent
		git_context = None
		logger.debug(f"No git repository found, using fallback repo_root: {repo_root}")

	try:
		# Get all paths under target recursively using list comprehension
		all_paths = [
			item for item in target_path.rglob("*") if (item.is_file() and not is_binary_file(item)) or item.is_dir()
		]

		# Filter by gitignore patterns
		filtered_paths = filter_paths_by_gitignore(all_paths, repo_root)

		# Additional filtering using git ignore checking if available
		if git_context:
			final_paths = []
			for path in filtered_paths:
				try:
					relative_path = str(path.relative_to(repo_root))
					if not git_context.is_git_ignored(relative_path):
						final_paths.append(path)
					else:
						logger.debug(f"Skipping git-ignored path: {path}")
				except (ValueError, OSError):
					# Can't get relative path or check ignore status, include anyway
					final_paths.append(path)
			filtered_paths = final_paths

		# Separate files and directories
		files = [p for p in filtered_paths if p.is_file()]
		directories = [p for p in filtered_paths if p.is_dir()]

		if not files and not directories:
			# Empty directory or all files filtered out
			return "signatures", 0

		# Calculate maximum directory depth relative to target_path
		max_depth = 0

		for path in files + directories:
			try:
				rel_path = path.relative_to(target_path)
				path_parts = rel_path.parts

				# Calculate depth (number of directory levels)
				depth = len(path_parts) - 1 if path.is_file() else len(path_parts)
				max_depth = max(max_depth, depth)

			except ValueError:
				# Path is not relative to target_path, skip
				continue

		# Determine LOD based on depth and structure
		if max_depth == 0:
			# Only files directly in target directory, no subdirectories
			return "docs", max_depth
		if max_depth == 1:
			# 1 level deep (files in subdirectories, but no deeper nesting)
			return "structure", max_depth
		# 2+ levels deep or complex structure
		return "signatures", max_depth

	except (OSError, PermissionError) as e:
		logger.warning(f"Error analyzing path structure for {target_path}: {e}")
		return "signatures", 0


async def generate_codebase_summary(
	ctx: RunContext[ConfigLoader],
	path: str | None = None,
) -> str:
	"""Generate a comprehensive summary of the codebase or specific path.

	This tool processes the codebase at an appropriate level of detail
	based on the complexity and depth of the target path.

	Args:
	    ctx: RunContext containing ConfigLoader
	    path: Optional path to analyze (relative or absolute). If None, analyzes entire codebase.

	Returns:
	    A string containing the formatted markdown documentation.
	"""
	logger.info(f"codebase_summary tool called with path: '{path}'")
	try:
		# Get repository root first
		try:
			repo_root = GitRepoContext.get_repo_root()
		except (OSError, ValueError, RuntimeError):
			repo_root = Path.cwd()  # Fallback to current directory

		# Determine and validate target path
		if path:
			# Handle empty strings or whitespace-only paths
			path = path.strip()
			if not path:
				target_path = repo_root
				logger.info("Empty path provided, using repository root")
			else:
				# First try to find matching paths in the codebase
				matching_paths = find_matching_paths(path, repo_root)

				if not matching_paths:
					# If no matches found, provide helpful guidance instead of failing
					msg = f"Could not find a path matching '{path}' in the codebase."

					# Suggest some common paths that exist
					common_paths = ["src/codemap", "src", "tests", "docs"]
					existing_paths = [p for p in common_paths if (repo_root / p).exists()]

					if existing_paths:
						msg += f" Available directories include: {', '.join(existing_paths)}. "
						msg += (
							f"Try using a more specific path like 'src/codemap/{path}' "
							f"or browse the available directories first."
						)

					logger.info(msg)
					return msg  # Return helpful message instead of raising ModelRetry
				if len(matching_paths) == 1:
					target_path = matching_paths[0]
					logger.info(f"Found unique match for '{path}': {target_path}")
				else:
					# Multiple matches found - use the most specific one (shortest path)
					target_path = min(matching_paths, key=lambda p: len(p.parts))
					relative_matches = [str(p.relative_to(repo_root)) for p in matching_paths[:3]]
					logger.info(f"Found {len(matching_paths)} matches for '{path}', using: {target_path}")
					logger.info(f"Other matches: {relative_matches}")

				# Additional validation for accessibility
				try:
					# Test if we can access the path
					if target_path.is_dir():
						list(target_path.iterdir())
					elif target_path.is_file():
						target_path.stat()
				except (PermissionError, OSError) as e:
					msg = f"Cannot access path {target_path}: {e}"
					logger.exception(msg)
					raise ModelRetry(msg) from e
		else:
			target_path = repo_root
			logger.info("No path provided, using repository root")

		logger.info(f"Generating codebase summary for: {target_path}")

		# Get git context for ignore checking
		with contextlib.suppress(OSError, ValueError, RuntimeError):
			GitRepoContext.get_instance()

		# Note: Path validation was moved to early validation above using is_valid_analysis_path

		# Analyze path structure to determine appropriate LOD
		lod_level, depth = analyze_path_structure(target_path)

		logger.info(f"Determined LOD level: {lod_level} (depth: {depth})")

		# Configure generation based on analysis
		# Convert string to LODLevel enum
		lod_level_mapping = {
			"signatures": LODLevel.SIGNATURES,
			"structure": LODLevel.STRUCTURE,
			"docs": LODLevel.DOCS,
			"skeleton": LODLevel.SKELETON,
			"full": LODLevel.FULL,
		}
		lod_enum = lod_level_mapping.get(lod_level, LODLevel.DOCS)

		config = GenSchema(
			lod_level=lod_enum,
			include_entity_graph=False,  # No entity graph needed for summaries
			include_tree=True,  # Include file tree for better understanding
		)

		# Get configuration and create generator
		config_loader = ctx.deps
		generator = CodeMapGenerator(config)

		# Process the target path
		entities, metadata = process_codebase(target_path, config, config_loader=config_loader)

		if len(entities) == 0:
			if target_path.is_file():
				msg = f"No entities found in file: {target_path}"
			else:
				msg = (
					f"No entities found in directory: {target_path} (may be empty or contain only binary/ignored files)"
				)
			logger.warning(msg)
			raise ModelRetry(msg)

		# Generate and return the documentation
		documentation = generator.generate_documentation(entities, metadata)

		logger.info(f"Successfully generated codebase summary with {len(entities)} entities using {lod_level} LOD")
		return documentation

	except Exception as e:
		if isinstance(e, ModelRetry):
			raise
		msg = f"Failed to generate codebase summary: {e}"
		logger.exception(msg)
		raise ModelRetry(msg) from e


# Create the PydanticAI Tool instance
codebase_summary_tool = Tool(
	generate_codebase_summary,
	takes_ctx=True,
	name="codebase_summary",
	description=(
		"Generate a comprehensive summary of the complete codebase or a specific directory/file path. "
		"Automatically finds matching paths in the codebase and determines the appropriate level of detail:\n"
		"- Single file: skeleton level (basic structure)\n"
		"- Simple directory (files only): docs level (detailed documentation)\n"
		"- Medium complexity (1-2 levels): structure level (architectural overview)\n"
		"- Complex/whole codebase (2+ levels): signatures level (high-level summary)\n\n"
		"Can find paths like 'cli' even when located at 'src/codemap/cli'. "
		"Respects .gitignore patterns and skips binary files. "
		"Use this when you need to understand code architecture, file organization, or specific components. "
		"IMPORTANT: Files included in gitignore will not be included. "
		"Along with that hidden files and .git directory will not be included."
	),
	prepare=None,
)
