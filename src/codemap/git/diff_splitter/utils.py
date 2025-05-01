"""Utility functions for diff splitting."""

import logging
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from re import Pattern

import numpy as np

from codemap.git.utils import GitError, run_git_command

from .constants import EPSILON, MAX_FILE_SIZE_FOR_LLM, MIN_NAME_LENGTH_FOR_SIMILARITY

logger = logging.getLogger(__name__)


__all__ = [
	"are_files_related",
	"calculate_semantic_similarity",
	"create_chunk_description",
	"determine_commit_type",
	"extract_code_from_diff",
	"filter_valid_files",
	"get_deleted_tracked_files",
	"get_language_specific_patterns",
	"has_related_file_pattern",
	"have_similar_names",
	"is_test_environment",
	"match_test_file_patterns",
]


def extract_code_from_diff(diff_content: str) -> tuple[str, str]:
	"""
	Extract actual code content from a diff.

	Args:
	    diff_content: The raw diff content

	Returns:
	    Tuple of (old_code, new_code) extracted from the diff

	"""
	old_lines = []
	new_lines = []

	# Handle empty diff content
	if not diff_content or diff_content.isspace():
		return "", ""

	# Check if diff content is too large and truncate if necessary
	if len(diff_content) > MAX_FILE_SIZE_FOR_LLM:
		logger.warning(
			"Diff content is very large (%d bytes). Truncating to prevent API payload limits.", len(diff_content)
		)
		# Extract file name from the diff if possible
		file_match = re.search(r"diff --git a/(.*) b/(.*)", diff_content)
		file_name = file_match.group(2) if file_match else "unknown file"

		# Create a summarized message instead of full content
		return (
			f"// Large diff content for {file_name} (truncated)",
			f"// Large diff content for {file_name} (truncated)\n// Original size: {len(diff_content)} bytes",
		)

	# Split into lines and prepare to process
	lines = diff_content.split("\n")
	in_hunk = False
	in_file = False
	context_function = None
	current_file = None

	# Keep track of content size to avoid exceeding limits
	estimated_size = 0
	max_size_per_side = MAX_FILE_SIZE_FOR_LLM // 2  # Split the limit between old and new code
	size_exceeded = False

	for line in lines:
		# Skip empty lines
		if not line.strip():
			continue

		# Check for file headers
		if line.startswith("diff --git"):
			in_file = True
			in_hunk = False
			# Extract file name for context
			match = re.search(r"diff --git a/(.*) b/(.*)", line)
			if match:
				current_file = match.group(2)
			continue

		# Skip index lines, --- and +++ lines
		if line.startswith(("index ", "--- ", "+++ ", "new file mode", "deleted file mode")):
			continue

		# Check for binary file notice
		if "Binary files" in line or "GIT binary patch" in line:
			# For binary files, just add a placeholder
			old_lines.append(f"// Binary file changed: {current_file}")
			new_lines.append(f"// Binary file changed: {current_file}")
			continue

		# Check for hunk header
		if line.startswith("@@"):
			in_hunk = True
			# Try to extract function context if available
			context_match = re.search(r"@@ .+ @@ (.*)", line)
			if context_match and context_match.group(1):
				context_function = context_match.group(1).strip()
				# Add function context to both old and new lines
				if context_function:
					old_lines.append(f"// {context_function}")
					new_lines.append(f"// {context_function}")
			continue

		if not in_hunk:
			continue

		# Check if we're approaching size limits
		estimated_size += len(line)
		if estimated_size > max_size_per_side and not size_exceeded:
			size_exceeded = True
			old_lines.append(f"// Content truncated - diff too large for {current_file or 'unknown file'}")
			new_lines.append(f"// Content truncated - diff too large for {current_file or 'unknown file'}")
			logger.warning("Truncated diff content for %s due to size limits", current_file or "unknown file")
			break

		# Extract code content - handle edge cases
		if line.startswith("-"):
			old_lines.append(line[1:])
		elif line.startswith("+"):
			new_lines.append(line[1:])
		elif line.startswith(" "):
			# Context lines appear in both old and new (explicitly handle the space)
			old_lines.append(line[1:])
			new_lines.append(line[1:])
		else:
			# Handle any other lines within hunks (shouldn't normally happen, but just in case)
			old_lines.append(line)
			new_lines.append(line)

	# If we didn't find any hunks but have a file, add placeholder
	if in_file and not old_lines and not new_lines and current_file:
		old_lines.append(f"// File: {current_file}")
		new_lines.append(f"// File: {current_file}")

	# Check final sizes and truncate if needed
	old_code = "\n".join(old_lines)
	new_code = "\n".join(new_lines)

	if len(old_code) > max_size_per_side or len(new_code) > max_size_per_side:
		logger.warning("Final extracted code still exceeds size limits, truncating further")
		if len(old_code) > max_size_per_side:
			old_code = old_code[: max_size_per_side - 100] + f"\n// ... truncated ({len(old_code)} bytes total)"
		if len(new_code) > max_size_per_side:
			new_code = new_code[: max_size_per_side - 100] + f"\n// ... truncated ({len(new_code)} bytes total)"

	return old_code, new_code


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
		status_output = run_git_command(["git", "status", "--porcelain"])
		for line in status_output.splitlines():
			if line.startswith(" D"):
				# Unstaged deletion (space followed by D)
				filename = line[3:].strip()  # Skip " D " prefix and strip any whitespace
				deleted_unstaged_files.add(filename)
			elif line.startswith("D "):
				# Staged deletion (D followed by space)
				filename = line[2:].strip()  # Skip "D " prefix and strip any whitespace
				deleted_staged_files.add(filename)
		logger.debug("Found %d deleted unstaged files in git status", len(deleted_unstaged_files))
		logger.debug("Found %d deleted staged files in git status", len(deleted_staged_files))
	except GitError:
		logger.warning("Failed to get git status for deleted files")

	return deleted_unstaged_files, deleted_staged_files


def filter_valid_files(files: list[str], is_test_environment: bool = False) -> tuple[list[str], list[str]]:
	"""
	Filter invalid filenames from a list of files.

	Args:
	    files: List of file paths to filter
	    is_test_environment: Whether running in a test environment

	Returns:
	    Tuple of (valid_files, filtered_large_files) - both as lists of file paths

	"""
	if not files:
		return [], []

	valid_files = []
	filtered_large_files = []

	for file in files:
		# Skip files that look like patterns or templates
		if any(char in file for char in ["*", "+", "{", "}", "\\"]) or file.startswith('"'):
			logger.warning("Skipping invalid filename in diff processing: %s", file)
			continue

		# Skip extremely large files to prevent API payload size issues
		if not is_test_environment and Path(file).exists():
			try:
				file_size = Path(file).stat().st_size
				if file_size > MAX_FILE_SIZE_FOR_LLM:
					logger.warning(
						"Skipping very large file (%s bytes) to prevent API payload limits: %s", file_size, file
					)
					filtered_large_files.append(file)
					continue
			except OSError as e:
				logger.warning("Error checking file size for %s: %s", file, e)

		valid_files.append(file)

	# Skip file existence checks in test environments
	if is_test_environment:
		logger.debug("In test environment - skipping file existence checks for %d files", len(valid_files))
		return valid_files, filtered_large_files

	# Get deleted files
	deleted_unstaged_files, deleted_staged_files = get_deleted_tracked_files()

	# Check if files exist in the repository (tracked by git) or filesystem
	original_count = len(valid_files)
	try:
		tracked_files_output = run_git_command(["git", "ls-files"])
		tracked_files = set(tracked_files_output.splitlines())

		# Keep files that either:
		# 1. Exist in filesystem
		# 2. Are tracked by git
		# 3. Are known deleted files from git status
		# 4. Are already staged deletions
		filtered_files = []
		for file in valid_files:
			if (
				Path(file).exists()
				or file in tracked_files
				or file in deleted_unstaged_files
				or file in deleted_staged_files
			):
				filtered_files.append(file)
			else:
				logger.warning("Skipping non-existent and untracked file in diff: %s", file)

		valid_files = filtered_files
		if len(valid_files) < original_count:
			logger.warning(
				"Filtered out %d files that don't exist in the repository",
				original_count - len(valid_files),
			)
	except GitError:
		# If we can't check git tracked files, at least filter by filesystem existence and git status
		filtered_files = []
		for file in valid_files:
			if Path(file).exists() or file in deleted_unstaged_files or file in deleted_staged_files:
				filtered_files.append(file)
			else:
				logger.warning("Skipping non-existent file in diff: %s", file)

		valid_files = filtered_files
		if len(valid_files) < original_count:
			logger.warning(
				"Filtered out %d files that don't exist in the filesystem",
				original_count - len(valid_files),
			)

	return valid_files, filtered_large_files


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
