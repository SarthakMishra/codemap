"""Utility functions for diff splitting."""

import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from re import Pattern

import numpy as np
from pygit2.enums import FileStatus

from codemap.git.utils import ExtendedGitRepoContext, GitError

from .constants import EPSILON, MIN_NAME_LENGTH_FOR_SIMILARITY

logger = logging.getLogger(__name__)


__all__ = [
	"are_files_related",
	"calculate_semantic_similarity",
	"create_chunk_description",
	"determine_commit_type",
	"filter_valid_files",
	"get_deleted_tracked_files",
	"get_language_specific_patterns",
	"has_related_file_pattern",
	"have_similar_names",
	"is_test_environment",
	"match_test_file_patterns",
]


def get_language_specific_patterns(language: str) -> list[str]:
	"""
	Get language-specific regex patterns for code structure.

	Args:
	    language: Programming language identifier

	Returns:
	    A list of regex patterns for the language, or empty list if not supported

	"""
	# Define pattern strings (used for semantic boundary detection)
	pattern_strings = {
		"py": [
			r"^import\s+.*",  # Import statements
			r"^from\s+.*",  # From imports
			r"^class\s+\w+",  # Class definitions
			r"^def\s+\w+",  # Function definitions
			r"^if\s+__name__\s*==\s*['\"]__main__['\"]",  # Main block
		],
		"js": [
			r"^import\s+.*",  # ES6 imports
			r"^const\s+\w+\s*=\s*require",  # CommonJS imports
			r"^function\s+\w+",  # Function declarations
			r"^const\s+\w+\s*=\s*function",  # Function expressions
			r"^class\s+\w+",  # Class declarations
			r"^export\s+",  # Exports
		],
		"ts": [
			r"^import\s+.*",  # Imports
			r"^export\s+",  # Exports
			r"^interface\s+",  # Interfaces
			r"^type\s+",  # Type definitions
			r"^class\s+",  # Classes
			r"^function\s+",  # Functions
		],
		"jsx": [
			r"^import\s+.*",  # ES6 imports
			r"^const\s+\w+\s*=\s*require",  # CommonJS imports
			r"^function\s+\w+",  # Function declarations
			r"^const\s+\w+\s*=\s*function",  # Function expressions
			r"^class\s+\w+",  # Class declarations
			r"^export\s+",  # Exports
		],
		"tsx": [
			r"^import\s+.*",  # Imports
			r"^export\s+",  # Exports
			r"^interface\s+",  # Interfaces
			r"^type\s+",  # Type definitions
			r"^class\s+",  # Classes
			r"^function\s+",  # Functions
		],
		"java": [
			r"^import\s+.*",  # Import statements
			r"^public\s+class",  # Public class
			r"^private\s+class",  # Private class
			r"^(public|private|protected)(\s+static)?\s+\w+\s+\w+\(",  # Methods
		],
		"go": [
			r"^import\s+",  # Import statements
			r"^func\s+",  # Function definitions
			r"^type\s+\w+\s+struct",  # Struct definitions
		],
		"rb": [
			r"^require\s+",  # Requires
			r"^class\s+",  # Class definitions
			r"^def\s+",  # Method definitions
			r"^module\s+",  # Module definitions
		],
		"php": [
			r"^namespace\s+",  # Namespace declarations
			r"^use\s+",  # Use statements
			r"^class\s+",  # Class definitions
			r"^(public|private|protected)\s+function",  # Methods
		],
		"cs": [
			r"^using\s+",  # Using directives
			r"^namespace\s+",  # Namespace declarations
			r"^(public|private|protected|internal)\s+class",  # Classes
			r"^(public|private|protected|internal)(\s+static)?\s+\w+\s+\w+\(",  # Methods
		],
		"kt": [
			r"^import\s+.*",  # Import statements
			r"^class\s+\w+",  # Class definitions
			r"^fun\s+\w+",  # Function definitions
			r"^val\s+\w+",  # Val declarations
			r"^var\s+\w+",  # Var declarations
		],
		"scala": [
			r"^import\s+.*",  # Import statements
			r"^class\s+\w+",  # Class definitions
			r"^object\s+\w+",  # Object definitions
			r"^def\s+\w+",  # Method definitions
			r"^val\s+\w+",  # Val declarations
			r"^var\s+\w+",  # Var declarations
		],
	}

	# Return pattern strings for the language or empty list if not supported
	return pattern_strings.get(language, [])


def determine_commit_type(files: list[str]) -> str:
	"""
	Determine the appropriate commit type based on the files.

	Args:
	    files: List of file paths

	Returns:
	    Commit type string (e.g., "feat", "fix", "test", "docs", "chore")

	"""
	# Check for test files
	if any(f.startswith("tests/") or "_test." in f or "test_" in f for f in files):
		return "test"

	# Check for documentation files
	if any(f.startswith("docs/") or f.endswith(".md") for f in files):
		return "docs"

	# Check for configuration files
	if any(f.endswith((".json", ".yml", ".yaml", ".toml", ".ini", ".cfg")) for f in files):
		return "chore"

	# Default to "chore" for general updates
	return "chore"


def create_chunk_description(commit_type: str, files: list[str]) -> str:
	"""
	Create a meaningful description for a chunk.

	Args:
	    commit_type: Type of commit (e.g., "feat", "fix")
	    files: List of file paths

	Returns:
	    Description string

	"""
	if len(files) == 1:
		return f"{commit_type}: update {files[0]}"

	# Try to find a common directory using Path for better cross-platform compatibility
	try:
		common_dir = Path(os.path.commonpath(files))
		if str(common_dir) not in (".", ""):
			return f"{commit_type}: update files in {common_dir}"
	except ValueError:
		# commonpath raises ValueError if files are on different drives
		pass

	return f"{commit_type}: update {len(files)} related files"


def get_deleted_tracked_files() -> tuple[set, set]:
	"""
	Get list of deleted but tracked files from git status.

	Returns:
	    Tuple of (deleted_unstaged_files, deleted_staged_files) as sets

	"""
	deleted_unstaged_files = set()
	deleted_staged_files = set()
	try:
		# Parse git status to find deleted files
		context = ExtendedGitRepoContext.get_instance()
		status = context.repo.status()
		for filepath, flags in status.items():
			if flags & FileStatus.WT_DELETED:  # Worktree deleted (unstaged)
				deleted_unstaged_files.add(filepath)
			if flags & FileStatus.INDEX_DELETED:  # Index deleted (staged)
				deleted_staged_files.add(filepath)
		logger.debug("Found %d deleted unstaged files in git status", len(deleted_unstaged_files))
		logger.debug("Found %d deleted staged files in git status", len(deleted_staged_files))
	except GitError as e:  # Catch specific GitError from context operations
		logger.warning(
			"Failed to get git status for deleted files via context: %s. Proceeding without deleted file info.", e
		)
	except Exception:  # Catch any other unexpected error
		logger.exception("Unexpected error getting git status: %s. Proceeding without deleted file info.")

	return deleted_unstaged_files, deleted_staged_files


def get_absolute_path(file: str, repo_root: Path) -> str:
	"""
	Get the canonical absolute path string for a file.

	If 'file' is already an absolute path, it's resolved to its canonical form.
	If 'file' is a relative path, it's considered relative to 'repo_root',
	made absolute, and then resolved.

	Args:
	    file: File path string (can be relative or absolute).
	    repo_root: Path to the repository root, used as a base for relative 'file' paths.

	Returns:
	    The canonical absolute path string.
	    Returns the original 'file' string if path resolution fails.
	"""
	try:
		file_path_obj = Path(file)
		if file_path_obj.is_absolute():
			# It's already an absolute path string, resolve it to a canonical form
			# (e.g., remove '..', '.', and resolve symbolic links)
			return str(file_path_obj.resolve())

		# It's a relative path string; assume it's relative to repo_root.
		# Combine with repo_root to make it absolute, then resolve.
		absolute_path = repo_root / file_path_obj
		return str(absolute_path.resolve())
	except (ValueError, OSError, RuntimeError) as e:
		# Log the error and fallback to returning the original file string
		# if any path operation fails.
		logger.warning(
			f"Could not resolve absolute path for '{file}' with repo_root '{repo_root}'."
			f" Error: {e}. Returning original.",
			exc_info=True,
		)
		return file


def filter_valid_files(
	files: list[str], repo_root: Path, is_test_environment: bool = False
) -> tuple[list[str], list[str]]:
	"""
	Filter invalid filenames and files based on existence and Git tracking.

	Args:
	    files: List of file paths to filter
	    repo_root: Path to the repository root
	    is_test_environment: Whether running in a test environment

	Returns:
	    Tuple of (valid_files, empty_list) - The second element is always an empty list now.

	"""
	if not files:
		return [], []

	valid_files_intermediate = []
	# Keep track of files filtered due to large size if needed elsewhere,
	# but don't remove them from processing yet.

	for file in files:
		# Skip files that look like patterns or templates
		if any(char in file for char in ["*", "+", "{", "}", "\\"]) or file.startswith('"'):
			logger.warning("Skipping invalid filename in diff processing: %s", file)
			continue
		valid_files_intermediate.append(file)

	# --- File Existence and Git Tracking Checks ---
	valid_files = []  # Reset valid_files to populate after existence checks

	# Skip file existence checks in test environments
	if is_test_environment:
		logger.debug("In test environment - skipping file existence checks for %d files", len(valid_files_intermediate))
		# In test env, assume all intermediate files are valid regarding existence/tracking
		valid_files = valid_files_intermediate
	else:
		# Get deleted files
		deleted_unstaged_files, deleted_staged_files = get_deleted_tracked_files()

		# Check if files exist in the repository (tracked by git) or filesystem
		original_count = len(valid_files_intermediate)
		try:
			context = ExtendedGitRepoContext.get_instance()
			tracked_files = set(context.tracked_files.keys())

			# Keep files that either:
			# 1. Exist in filesystem
			# 2. Are tracked by git
			# 3. Are known deleted files from git status
			# 4. Are already staged deletions
			filtered_files = []
			for file in valid_files_intermediate:
				try:
					path_exists = Path(get_absolute_path(file, repo_root)).exists()
				except OSError as e:
					logger.warning("OS error checking existence for %s: %s. Skipping file.", file, e)
					continue
				except Exception:
					logger.exception("Unexpected error checking existence for %s. Skipping file.", file)
					continue

				if (
					path_exists
					or file in tracked_files
					or file in deleted_unstaged_files
					or file in deleted_staged_files
				):
					filtered_files.append(file)
				else:
					logger.warning("Skipping non-existent/untracked/not-deleted file in diff: %s", file)

			valid_files = filtered_files
			if len(valid_files) < original_count:
				logger.warning(
					"Filtered out %d files that don't exist or aren't tracked/deleted",
					original_count - len(valid_files),
				)
		except GitError as e:  # Catch GitError from context operations
			logger.warning("Failed to get tracked files from git context: %s. Filtering based on existence only.", e)
			# If we can't check git tracked files, filter by filesystem existence and git status
			filtered_files_fallback = []
			for file in valid_files_intermediate:
				try:
					path_exists = Path(file).exists()
				except OSError as e:
					logger.warning("OS error checking existence for %s: %s. Skipping file.", file, e)
					continue
				except Exception:
					logger.exception("Unexpected error checking existence for %s. Skipping file.", file)
					continue

				if path_exists or file in deleted_unstaged_files or file in deleted_staged_files:
					filtered_files_fallback.append(file)
				else:
					logger.warning("Skipping non-existent/not-deleted file in diff (git check failed): %s", file)

			valid_files = filtered_files_fallback  # Replace valid_files with the fallback list
			if len(valid_files) < original_count:
				# Adjust log message if git check failed
				logger.warning(
					"Filtered out %d files that don't exist (git check failed)",
					original_count - len(valid_files),
				)
		except Exception:  # Catch any other unexpected errors during the initial try block
			logger.exception("Unexpected error during file filtering. Proceeding with potentially incorrect list.")
			# If a catastrophic error occurs, proceed with the intermediate list
			valid_files = valid_files_intermediate

	# Return only the list of valid files. The concept of 'filtered_large_files' is removed.
	# Size checking will now happen within the splitting strategy.
	return valid_files, []  # Return empty list for the second element now.


def is_test_environment() -> bool:
	"""
	Check if the code is running in a test environment.

	Returns:
	    True if in a test environment, False otherwise

	"""
	# Check multiple environment indicators for tests
	return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules or os.environ.get("TESTING") == "1"


def calculate_semantic_similarity(emb1: list[float], emb2: list[float]) -> float:
	"""
	Calculate semantic similarity (cosine similarity) between two embedding vectors.

	Args:
	    emb1: First embedding vector
	    emb2: Second embedding vector

	Returns:
	    Similarity score between 0 and 1

	"""
	if not emb1 or not emb2:
		return 0.0

	try:
		# Convert to numpy arrays
		vec1 = np.array(emb1, dtype=np.float64)
		vec2 = np.array(emb2, dtype=np.float64)

		# Calculate cosine similarity
		dot_product = np.dot(vec1, vec2)
		norm1 = np.linalg.norm(vec1)
		norm2 = np.linalg.norm(vec2)

		if norm1 <= EPSILON or norm2 <= EPSILON:
			return 0.0

		similarity = float(dot_product / (norm1 * norm2))

		# Handle potential numeric issues
		if not np.isfinite(similarity):
			return 0.0

		return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

	except (ValueError, TypeError, ArithmeticError, OverflowError):
		logger.warning("Failed to calculate similarity")
		return 0.0


def match_test_file_patterns(file1: str, file2: str) -> bool:
	"""Check if files match common test file patterns."""
	# test_X.py and X.py patterns
	if file1.startswith("test_") and file1[5:] == file2:
		return True
	if file2.startswith("test_") and file2[5:] == file1:
		return True

	# X_test.py and X.py patterns
	if file1.endswith("_test.py") and file1[:-8] + ".py" == file2:
		return True
	return bool(file2.endswith("_test.py") and file2[:-8] + ".py" == file1)


def have_similar_names(file1: str, file2: str) -> bool:
	"""Check if files have similar base names."""
	base1 = file1.rsplit(".", 1)[0] if "." in file1 else file1
	base2 = file2.rsplit(".", 1)[0] if "." in file2 else file2

	return (base1 in base2 or base2 in base1) and min(len(base1), len(base2)) >= MIN_NAME_LENGTH_FOR_SIMILARITY


def has_related_file_pattern(file1: str, file2: str, related_file_patterns: Iterable[tuple[Pattern, Pattern]]) -> bool:
	"""
	Check if files match known related patterns.

	Args:
	    file1: First file path
	    file2: Second file path
	    related_file_patterns: Compiled regex pattern pairs to check against

	Returns:
	    True if the files match a known pattern, False otherwise

	"""
	for pattern1, pattern2 in related_file_patterns:
		if (pattern1.match(file1) and pattern2.match(file2)) or (pattern2.match(file1) and pattern1.match(file2)):
			return True
	return False


def are_files_related(file1: str, file2: str, related_file_patterns: Iterable[tuple[Pattern, Pattern]]) -> bool:
	"""
	Determine if two files are semantically related based on various criteria.

	Args:
	    file1: First file path
	    file2: Second file path
	    related_file_patterns: Compiled regex pattern pairs for pattern matching

	Returns:
	    True if the files are related, False otherwise

	"""
	# 1. Files in the same directory
	dir1 = file1.rsplit("/", 1)[0] if "/" in file1 else ""
	dir2 = file2.rsplit("/", 1)[0] if "/" in file2 else ""
	if dir1 and dir1 == dir2:
		return True

	# 2. Files in closely related directories (parent/child or same root directory)
	if dir1 and dir2:
		if dir1.startswith(dir2 + "/") or dir2.startswith(dir1 + "/"):
			return True
		# Check if they share the same top-level directory
		top_dir1 = dir1.split("/", 1)[0] if "/" in dir1 else dir1
		top_dir2 = dir2.split("/", 1)[0] if "/" in dir2 else dir2
		if top_dir1 and top_dir1 == top_dir2:
			return True

	# 3. Test files and implementation files (simple check)
	if (file1.startswith("tests/") and file2 in file1) or (file2.startswith("tests/") and file1 in file2):
		return True

	# 4. Test file patterns
	file1_name = file1.rsplit("/", 1)[-1] if "/" in file1 else file1
	file2_name = file2.rsplit("/", 1)[-1] if "/" in file2 else file2
	if match_test_file_patterns(file1_name, file2_name):
		return True

	# 5. Files with similar names
	if have_similar_names(file1_name, file2_name):
		return True

	# 6. Check for related file patterns
	return has_related_file_pattern(file1, file2, related_file_patterns)
