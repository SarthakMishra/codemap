"""Utility functions for diff splitting."""

import logging
import os
import re
from pathlib import Path
from re import Pattern

from codemap.git.utils import GitError, run_git_command

logger = logging.getLogger(__name__)


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

	# Skip diff header lines
	lines = diff_content.split("\n")
	in_hunk = False
	context_function = None

	for line in lines:
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

		# Extract code content
		if line.startswith("-"):
			old_lines.append(line[1:])
		elif line.startswith("+"):
			new_lines.append(line[1:])
		else:
			# Context lines appear in both old and new
			old_lines.append(line)
			new_lines.append(line)

	return "\n".join(old_lines), "\n".join(new_lines)


def get_language_specific_patterns(language: str) -> Pattern | None:
	"""
	Get language-specific regex patterns for code structure.

	Args:
	    language: Programming language identifier

	Returns:
	    Compiled regex pattern or None if language not supported

	"""
	patterns = {
		"py": r'(^class\s+\w+|^def\s+\w+|^if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:|'
		r"^import\s+|^from\s+\w+\s+import)",
		"js": r"(^function\s+\w+|^const\s+\w+\s*=\s*function|^class\s+\w+|"
		r"^\s*\w+\s*\([^)]*\)\s*{|^import\s+|^export\s+)",
		"ts": r"(^function\s+\w+|^const\s+\w+\s*=\s*function|^class\s+\w+|"
		r"^\s*\w+\s*\([^)]*\)\s*{|^import\s+|^export\s+)",
		"jsx": r"(^function\s+\w+|^const\s+\w+\s*=\s*function|^class\s+\w+|"
		r"^\s*\w+\s*\([^)]*\)\s*{|^import\s+|^export\s+)",
		"tsx": r"(^function\s+\w+|^const\s+\w+\s*=\s*function|^class\s+\w+|"
		r"^\s*\w+\s*\([^)]*\)\s*{|^import\s+|^export\s+)",
		"java": r"(^public\s+|^private\s+|^protected\s+|^class\s+\w+|"
		r"^interface\s+\w+|^enum\s+\w+|^import\s+|^package\s+)",
		"kt": r"(^public\s+|^private\s+|^protected\s+|^class\s+\w+|"
		r"^interface\s+\w+|^enum\s+\w+|^import\s+|^package\s+)",
		"scala": r"(^public\s+|^private\s+|^protected\s+|^class\s+\w+|"
		r"^interface\s+\w+|^enum\s+\w+|^import\s+|^package\s+)",
		"go": r"(^func\s+|^type\s+\w+|^import\s+|^package\s+\w+)",
	}

	if language in patterns:
		return re.compile(patterns[language], re.MULTILINE)
	return None


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

	# Try to find a common directory
	common_dir = os.path.commonpath(files)
	if common_dir and common_dir != ".":
		return f"{commit_type}: update files in {common_dir}"

	return f"{commit_type}: update {len(files)} related files"


def get_deleted_tracked_files() -> tuple[set, set]:
	"""
	Get list of deleted but tracked files from git status.

	Returns:
	    Tuple of (deleted_tracked_files, already_staged_deletions) as sets

	"""
	deleted_tracked_files = set()
	already_staged_deletions = set()
	try:
		# Parse git status to find deleted files
		status_output = run_git_command(["git", "status", "--porcelain"])
		for line in status_output.splitlines():
			if line.startswith(" D"):
				# Unstaged deletion (space followed by D)
				deleted_tracked_files.add(line[3:])
			elif line.startswith("D "):
				# Staged deletion (D followed by space)
				already_staged_deletions.add(line[3:])
		logger.debug("Found %d deleted tracked files in git status", len(deleted_tracked_files))
		logger.debug("Found %d already staged deletions in git status", len(already_staged_deletions))
	except GitError:
		logger.warning("Failed to get git status for deleted files")

	return deleted_tracked_files, already_staged_deletions


def filter_valid_files(files: list[str], is_test_environment: bool = False) -> list[str]:
	"""
	Filter invalid filenames from a list of files.

	Args:
	    files: List of file paths to filter
	    is_test_environment: Whether running in a test environment

	Returns:
	    List of valid file paths

	"""
	if not files:
		return []

	valid_files = []
	for file in files:
		# Skip files that look like patterns or templates
		if any(char in file for char in ["*", "+", "{", "}", "\\"]) or file.startswith('"'):
			logger.warning("Skipping invalid filename in diff processing: %s", file)
			continue
		valid_files.append(file)

	# Skip file existence checks in test environments
	if is_test_environment:
		return valid_files

	# Get deleted files
	deleted_tracked_files, already_staged_deletions = get_deleted_tracked_files()

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
				or file in deleted_tracked_files
				or file in already_staged_deletions
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
			if Path(file).exists() or file in deleted_tracked_files or file in already_staged_deletions:
				filtered_files.append(file)
			else:
				logger.warning("Skipping non-existent file in diff: %s", file)

		valid_files = filtered_files
		if len(valid_files) < original_count:
			logger.warning(
				"Filtered out %d files that don't exist in the filesystem",
				original_count - len(valid_files),
			)

	return valid_files
