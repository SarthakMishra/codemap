"""Utilities for interacting with Git."""

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Constant for magic number 4 in parsing ls-files output
MIN_GIT_LS_FILES_PARTS = 4
SHA1_HASH_LENGTH = 40


def _run_git_command(command: list[str], cwd: Path) -> tuple[str, str, int]:
	"""Helper function to run a Git command and capture output."""
	try:
		# Use full path to git for security if needed, but often env path is fine
		# command.insert(0, "/usr/bin/git")
		process = subprocess.run(  # noqa: S603
			command,
			cwd=cwd,
			capture_output=True,
			text=True,
			check=False,  # Don't raise CalledProcessError automatically
			encoding="utf-8",
			errors="ignore",  # Handle potential decoding errors
		)
		# Log stderr only if there's content and return code is non-zero
		if process.returncode != 0 and process.stderr:
			logger.error(
				f"Git command '{' '.join(command)}' failed with code {process.returncode}:\n{process.stderr.strip()}"
			)
		elif process.stderr:
			# Log stderr as warning even on success if there's output
			logger.warning(f"Git command '{' '.join(command)}' produced stderr:\n{process.stderr.strip()}")
		return process.stdout.strip(), process.stderr.strip(), process.returncode
	except FileNotFoundError:
		logger.exception("Git command not found. Is Git installed and in PATH?")
		raise
	except Exception:
		logger.exception(f"Failed to run Git command '{' '.join(command)}'")
		raise


def _get_exclude_patterns() -> list[str]:
	"""
	Get the list of path patterns to exclude from processing.

	Returns:
	        List of regex patterns for paths to exclude

	"""
	from codemap.config import ConfigLoader  # Ensure this local import exists

	config_loader = ConfigLoader.get_instance()

	# Get exclude patterns from config, falling back to defaults if not present
	config_patterns = config_loader.get.sync.exclude_patterns

	# Default patterns to exclude
	default_patterns = [
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
		r"\.pyc$",
		r"\.pyo$",
		r"\.so$",
		r"\.dll$",
		r"\.lib$",
		r"\.a$",
		r"\.o$",
		r"\.class$",
		r"\.jar$",
	]

	# Combine config patterns with defaults, giving precedence to config
	patterns = list(config_patterns)

	# Only add default patterns that aren't already covered
	for pattern in default_patterns:
		if pattern not in patterns:
			patterns.append(pattern)

	return patterns


def _should_exclude_path(file_path: str) -> bool:
	"""
	Check if a file path should be excluded from processing based on patterns.

	Args:
	        file_path: The file path to check

	Returns:
	        True if the path should be excluded, False otherwise

	"""
	exclude_patterns = _get_exclude_patterns()

	for pattern in exclude_patterns:
		if re.search(pattern, file_path):
			logger.debug(f"Excluding file from processing due to pattern '{pattern}': {file_path}")
			return True

	return False


def get_git_tracked_files(repo_path: Path) -> dict[str, str] | None:
	"""
	Get all tracked files in the Git repository with their blob hashes.

	Uses 'git ls-files -s' which shows staged files with mode, hash, stage, path.
	Respects .gitignore rules by using --exclude-standard flag.

	Args:
	    repo_path (Path): The path to the root of the Git repository.

	Returns:
	    dict[str, str] | None: A dictionary mapping file paths (relative to repo_path)
	                          to their Git blob hashes. Returns None on failure.

	"""
	command = ["git", "ls-files", "-s", "--full-name", "--exclude-standard"]
	stdout, _, returncode = _run_git_command(command, repo_path)

	if returncode != 0:
		logger.error(f"'git ls-files -s' failed in {repo_path}")
		return None

	tracked_files: dict[str, str] = {}
	lines = stdout.splitlines()
	for line in lines:
		if not line:
			continue
		try:
			parts = line.split()
			if len(parts) < MIN_GIT_LS_FILES_PARTS:
				logger.warning(f"Skipping malformed line in git ls-files output: {line}")
				continue

			# Extract mode, hash, stage, and path
			_mode, blob_hash, stage_str = parts[:3]
			file_path = " ".join(parts[3:])  # Handle spaces in filenames

			# Unquote path if necessary (git ls-files quotes paths with special chars)
			if file_path.startswith('"') and file_path.endswith('"'):
				file_path = file_path[1:-1].encode("latin-1").decode("unicode_escape")

			stage = int(stage_str)

			# We are interested in committed files (stage 0)
			if stage == 0 and not _should_exclude_path(file_path):
				tracked_files[file_path] = blob_hash
		except ValueError:
			logger.warning(f"Could not parse line: {line}")
		except IndexError:
			logger.warning(f"Index error parsing line: {line}")

	logger.info(f"Found {len(tracked_files)} tracked files in Git repository: {repo_path}")
	return tracked_files


def get_file_git_hash(repo_path: Path, file_path: str) -> str | None:
	"""
	Get the Git hash (blob ID) for a specific tracked file.

	Uses 'git hash-object' which computes the hash of the file content as it is
	on the filesystem currently. This matches the behavior needed for comparing
	against potentially modified files before staging.

	Args:
	    repo_path (Path): The path to the root of the Git repository.
	    file_path (str): The path to the file relative to the repository root.

	Returns:
	    str | None: The Git blob hash of the file, or None if an error occurs
	                or the file is not found/tracked.

	"""
	full_file_path = repo_path / file_path
	if not full_file_path.is_file():
		logger.warning(f"Cannot get git hash: File not found or is not a regular file: {full_file_path}")
		return None

	command = ["git", "hash-object", str(full_file_path)]
	stdout, _, returncode = _run_git_command(command, repo_path)

	if returncode != 0:
		logger.error(f"'git hash-object {file_path}' failed in {repo_path}")
		return None

	# stdout should contain just the hash
	git_hash = stdout.strip()
	if len(git_hash) == SHA1_HASH_LENGTH:  # Use constant
		return git_hash
	logger.error(f"Unexpected output from 'git hash-object {file_path}': {stdout}")
	return None
