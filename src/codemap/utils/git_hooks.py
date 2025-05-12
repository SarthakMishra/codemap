"""Utilities for managing and running git hooks directly."""

import logging
import subprocess
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

HOOK_TYPES = Literal["pre-commit", "post-commit", "commit-msg", "pre-push", "post-checkout"]


def get_git_hooks_dir(repo_root: Path | None = None) -> Path:
	"""
	Get the .git/hooks directory for the current or given repo.

	Args:
	    repo_root: Path to the repo root. Defaults to cwd.

	Returns:
	    Path to the .git/hooks directory.
	"""
	root = repo_root or Path.cwd()
	# Find .git directory
	git_dir = root / ".git"
	if not git_dir.exists():
		msg = f".git directory not found at {git_dir}"
		raise FileNotFoundError(msg)
	hooks_dir = git_dir / "hooks"
	if not hooks_dir.exists():
		msg = f".git/hooks directory not found at {hooks_dir}"
		raise FileNotFoundError(msg)
	return hooks_dir


def hook_exists(hook_type: HOOK_TYPES, repo_root: Path | None = None) -> bool:
	"""
	Check if a given hook exists and is executable.

	Args:
	    hook_type: The hook type ("pre-commit" or "pre-push").
	    repo_root: Path to the repo root. Defaults to cwd.

	Returns:
	    True if the hook exists and is executable, False otherwise.
	"""
	hooks_dir = get_git_hooks_dir(repo_root)
	hook_path = hooks_dir / hook_type
	return bool(hook_path.exists() and hook_path.is_file() and (hook_path.stat().st_mode & 0o111))


def run_hook(hook_type: HOOK_TYPES, repo_root: Path | None = None) -> int:
	"""
	Run the specified git hook directly using bash.

	Args:
	    hook_type: The hook type ("pre-commit" or "pre-push").
	    repo_root: Path to the repo root. Defaults to cwd.

	Returns:
	    The exit code of the hook process (0 if hook doesn't exist).
	"""
	hooks_dir = get_git_hooks_dir(repo_root)
	hook_path = hooks_dir / hook_type
	if not hook_path.exists():
		logger.debug(f"{hook_type} hook not found at {hook_path}")
		return 0
	if not (hook_path.stat().st_mode & 0o111):
		logger.warning(f"{hook_type} hook at {hook_path} is not executable")
		return 1
	try:
		logger.info(f"Running git hook: {hook_path}")
		result = subprocess.run(["bash", str(hook_path)], capture_output=True, text=True, check=False)  # noqa: S603, S607
		if result.stdout:
			logger.info(f"{hook_type} hook output:\n{result.stdout}")
		if result.stderr:
			logger.warning(f"{hook_type} hook error:\n{result.stderr}")
		return result.returncode
	except Exception:
		logger.exception(f"Failed to run {hook_type} hook")
		return 1


def run_all_hooks(repo_root: Path | None = None) -> dict[str, int]:
	"""
	Run all supported hooks (pre-commit, pre-push) if they exist.

	Args:
	    repo_root: Path to the repo root. Defaults to cwd.

	Returns:
	    Dict mapping hook type to exit code.
	"""
	results = {}
	hooks: list[HOOK_TYPES] = ["pre-commit", "post-commit", "commit-msg", "pre-push", "post-checkout"]
	for hook in hooks:
		if hook_exists(hook, repo_root):
			results[hook] = run_hook(hook, repo_root)
	return results
