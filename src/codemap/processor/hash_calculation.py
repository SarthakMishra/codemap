"""Module for calculating hierarchical repository checksums."""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, ClassVar

import aiofiles
import xxhash
from pygit2 import GitError

from codemap.config.config_loader import ConfigLoader

if TYPE_CHECKING:
	# This import is conditional for type hinting and will not cause a runtime error
	# if GitRepoContext is not available in the global scope when this module is loaded directly.
	# However, for actual use, GitRepoContext will need to be importable.
	from codemap.utils.git_utils import GitRepoContext

logger = logging.getLogger(__name__)


class RepoChecksumCalculator:
	"""
	Calculates a hierarchical checksum for a repository.

	Directory hashes are derived from the names and hashes of their children,
	making the checksum sensitive to content changes, additions, deletions,
	and renames.
	"""

	_instances: ClassVar[dict[Path, "RepoChecksumCalculator"]] = {}

	def __init__(
		self, repo_path: Path, git_context: "GitRepoContext | None" = None, config_loader: "ConfigLoader | None" = None
	) -> None:
		"""
		Initialize the checksum calculator.

		Prefer using get_instance() to create or retrieve instances.

		Args:
		    repo_path: Absolute path to the repository root.
		    git_context: Optional GitRepoContext, used for context like branch names
		                 (for future storage strategies) and potentially accessing
		                 configuration for checksum paths.
		    config_loader: Optional ConfigLoader, used for configuration
		"""
		if not repo_path.is_dir():
			msg = f"Repository path {repo_path} is not a valid directory."
			raise ValueError(msg)
		self.repo_path = repo_path.resolve()
		self.git_context = git_context
		self.config_loader = config_loader
		self.all_nodes_map: dict[str, dict[str, str]] | None = None  # path -> {"type": "file"|"dir", "hash": hash_val}

		if not self.config_loader:
			# Ensure we have a config loader instance to fetch default/user configs
			self.config_loader = ConfigLoader().get_instance()

		self.checksums_base_dir = self.repo_path / ".codemap_cache" / "checksums"
		self.checksums_base_dir.mkdir(parents=True, exist_ok=True)

		# Fetch exclude patterns from SyncSchema via ConfigLoader
		# This allows user overrides from .codemap.yml to be respected.
		# If no config file or specific settings, defaults from SyncSchema are used.
		app_config = self.config_loader.get
		self.exclude_patterns_str: list[str] = list(app_config.sync.exclude_patterns[:])  # Start with config patterns

		# Custom .gitignore parsing is removed. We will use pygit2.path_is_ignored later.

		# Ensure .codemap_cache (or configured equivalent) is always excluded.
		# This specific path should ideally be part of the default config in SyncSchema
		# or managed via a dedicated configuration setting if its name/location is dynamic.
		# For now, adding it directly here if not already present via a generic pattern.
		codemap_cache_pattern = r"^\.codemap_cache/"
		if codemap_cache_pattern not in self.exclude_patterns_str:
			self.exclude_patterns_str.append(codemap_cache_pattern)

		# Also explicitly exclude the checksums directory we just defined
		checksums_dir_relative = self.checksums_base_dir.relative_to(self.repo_path).as_posix()
		checksums_dir_pattern = f"^{re.escape(checksums_dir_relative)}/"
		if checksums_dir_pattern not in self.exclude_patterns_str:
			self.exclude_patterns_str.append(checksums_dir_pattern)

		self.compiled_exclude_patterns: list[Pattern[str]] = [re.compile(p) for p in self.exclude_patterns_str]
		patterns = [p.pattern for p in self.compiled_exclude_patterns]
		logger.info(f"RepoChecksumCalculator compiled CodeMap exclude patterns: {patterns}")

	@classmethod
	def get_instance(
		cls,
		repo_path: Path,
		git_context: "GitRepoContext | None" = None,
		config_loader: "ConfigLoader | None" = None,
	) -> "RepoChecksumCalculator":
		"""
		Gets a cached instance of RepoChecksumCalculator for the given repo_path..

		Args:
		    repo_path: Absolute or relative path to the repository root.
		    git_context: Optional GitRepoContext for the new instance if created.
		    config_loader: Optional ConfigLoader for the new instance if created.

		Returns:
		    An instance of RepoChecksumCalculator.
		"""
		resolved_path = repo_path.resolve()
		if resolved_path not in cls._instances:
			logger.debug(f"Creating new RepoChecksumCalculator instance for {resolved_path}")
			instance = cls(resolved_path, git_context, config_loader)
			cls._instances[resolved_path] = instance
		else:
			logger.debug(f"Reusing existing RepoChecksumCalculator instance for {resolved_path}")
			# Update context if provided, as it might have changed (e.g., branch switch)
			existing_instance = cls._instances[resolved_path]
			if git_context is not None:
				existing_instance.git_context = git_context
			if config_loader is not None:
				existing_instance.config_loader = config_loader
		return cls._instances[resolved_path]

	def _hash_string(self, data: str) -> str:
		"""Helper to hash a string using xxhash.xxh3_128_hexdigest for consistency."""
		hasher = xxhash.xxh3_128()
		hasher.update(data.encode("utf-8"))
		return hasher.hexdigest()

	async def _hash_file_content(self, file_path: Path) -> str:
		"""Calculates xxhash.xxh3_128_hexdigest for a file's content.

		Reads the file in chunks for efficiency with large files.
		"""
		hasher = xxhash.xxh3_128()
		try:
			async with aiofiles.open(file_path, "rb") as f:
				while True:
					chunk = await f.read(8192)  # 8KB chunks
					if not chunk:
						break
					hasher.update(chunk)
			return hasher.hexdigest()
		except OSError:
			logger.exception(f"Error reading file content for {file_path}")
			return self._hash_string(f"ERROR_READING_FILE:{file_path.name}")

	def _is_path_explicitly_excluded(self, path_to_check: Path) -> tuple[bool, str]:
		"""
		Checks if a path should be excluded based on configured regex patterns.

		Patterns are matched against the relative path from the repository root.

		Returns a tuple: (is_excluded, reason_for_hash_if_excluded).
		"""
		relative_path_str: str
		if self.repo_path == path_to_check:  # Root itself cannot be excluded by patterns matching children
			relative_path_str = "."
		else:
			try:
				relative_path_str = str(path_to_check.relative_to(self.repo_path).as_posix())
			except ValueError:
				logger.warning(
					f"Path {path_to_check} is not relative to repo root {self.repo_path}. Not excluding by pattern."
				)
				return False, ""

		if relative_path_str.startswith(".cache/"):
			logger.info(f"Checking exclusion for .cache path: '{relative_path_str}'")

		# 1. Check against CodeMap-specific patterns (from .codemap.yml and hardcoded)
		for pattern_idx, compiled_pattern in enumerate(self.compiled_exclude_patterns):
			if compiled_pattern.search(relative_path_str):
				original_pattern_str = self.exclude_patterns_str[pattern_idx]
				reason = f"EXCLUDED_BY_CODEMAP_CONFIG_PATTERN:{original_pattern_str}:{relative_path_str}"
				if relative_path_str == ".":
					logger.error(
						f"Repository root ('.') EXCLUDED by CodeMap config pattern: "
						f"'{original_pattern_str}' (Regex: '{compiled_pattern.pattern}')"
					)
				else:
					logger.debug(
						f"Path '{relative_path_str}' excluded by CodeMap config pattern '{original_pattern_str}'"
					)
				return True, reason

		# 2. If not excluded by CodeMap patterns, check Git's ignore status via pygit2
		if self.git_context and self.git_context.repo:
			try:
				# For the repository root ("."), trust CodeMap config patterns primarily.
				# Avoid excluding the root based on path_is_ignored(".") due to observed discrepancies
				# where `git check-ignore -v .` says not ignored, but pygit2 says it is.
				if relative_path_str == ".":
					# We already logged the result of path_is_ignored(".") earlier if it was True.
					# Here, we explicitly decide NOT to exclude the root based on that specific check.
					logger.info(
						"Skipping pygit2.path_is_ignored check for root '.' due to "
						"potential discrepancies. Only CodeMap config can exclude root."
					)

				elif self.git_context.repo.path_is_ignored(relative_path_str):
					# This is for paths OTHER than the root "."
					reason = f"EXCLUDED_BY_GITIGNORE:{relative_path_str}"
					logger.debug(f"Path '{relative_path_str}' is ignored by Git (.gitignore or similar).")
					return True, reason
			except GitError as e:  # Specifically catch GitError
				# path_is_ignored can raise GitError for various reasons.
				logger.warning(
					f"GitError checking git ignore status for '{relative_path_str}': {e}. "
					"Treating as not ignored by Git."
				)
			except TypeError as e:  # Example of another specific error if relevant
				logger.warning(
					f"TypeError checking git ignore status for '{relative_path_str}': {e}. Path type might be an issue."
				)
			# Add other specific exceptions if pygit2.path_is_ignored is known to raise them.
			# For truly unexpected errors, it might be better to let them propagate if they indicate a severe issue.
		else:
			logger.debug("GitContext not available, skipping .gitignore check for path: %s", relative_path_str)

		# If it wasn't excluded by any CodeMap pattern and (if GitContext was available) not by Git's ignore rules
		if relative_path_str.startswith(".cache/"):
			logger.info(f"Path '{relative_path_str}' (under .cache/) was NOT excluded by any method.")

		return False, ""

	async def _calculate_node_hash_recursive(
		self, current_path: Path, current_nodes_map: dict[str, dict[str, str]]
	) -> str:
		"""Recursively calculates the hash for a file or directory.

		Populates current_nodes_map with {relative_path: {"type": "file"|"dir"|"excluded"|"error_dir"|"unknown",
		"hash": hash_val}} for all processed nodes.
		Returns the hash of the current_path node.
		"""
		# Use POSIX-style paths for consistency across OS, relative to repo root.
		relative_path_str = str(current_path.relative_to(self.repo_path).as_posix())

		if relative_path_str == ".":  # Represent root as empty string for map keys if preferred, or "."
			relative_path_str = ""

		is_excluded, exclusion_hash_reason = self._is_path_explicitly_excluded(current_path)
		if is_excluded:
			node_hash = self._hash_string(exclusion_hash_reason)
			# For excluded items, we still record them as 'excluded' type for completeness if needed.
			# Or simply don't add them to the map if they shouldn't affect parent hashes.
			# Current logic: excluded items affect parent hash via their unique exclusion_hash_reason.
			current_nodes_map[relative_path_str] = {"type": "excluded", "hash": node_hash}
			return node_hash

		if current_path.is_file():
			node_hash = await self._hash_file_content(current_path)
			current_nodes_map[relative_path_str] = {"type": "file", "hash": node_hash}
			return node_hash

		if current_path.is_dir():
			children_info_for_hash: list[str] = []
			try:
				# Sort children by name for deterministic hashing.
				children_paths_sync = list(current_path.iterdir())  # Sync part
				children_paths = sorted(children_paths_sync, key=lambda p: p.name)
			except OSError:
				logger.exception(f"Error listing directory {current_path}")
				node_hash = self._hash_string(f"ERROR_LISTING_DIR:{current_path.name}")
				current_nodes_map[relative_path_str] = {"type": "error_dir", "hash": node_hash}
				return node_hash

			for child_path in children_paths:
				# The recursive call populates current_nodes_map for the child and its descendants.
				child_hash = await self._calculate_node_hash_recursive(child_path, current_nodes_map)
				# The directory's hash depends on its children's names and their hashes.
				children_info_for_hash.append(str(f"{child_path.name}:{child_hash}"))

			# Concatenate all children's "name:hash" strings.
			# An empty directory will hash an empty string.
			dir_content_representation = "".join(children_info_for_hash)
			node_hash = self._hash_string(dir_content_representation)

			current_nodes_map[relative_path_str] = {"type": "dir", "hash": node_hash}
			return node_hash
		# Handles symlinks (if is_file/is_dir is false), broken links, or other types.
		logger.warning(
			f"Path {current_path} is not a file or directory (or is a broken symlink). Assigning a fixed hash."
		)
		node_hash = self._hash_string(f"UNKNOWN_TYPE:{current_path.name}")
		# Store its hash if it needs to be part of the map.
		# Ensure relative_path_str is correctly derived for the root path itself if it's of unknown type
		map_key = relative_path_str if relative_path_str else "."  # Use "." if relative_path_str became empty (root)
		current_nodes_map[map_key] = {"type": "unknown", "hash": node_hash}
		return node_hash

	def _get_current_branch_checksum_dir(self) -> Path | None:
		if not self.git_context:
			logger.warning("GitContext not available, cannot determine current branch for checksum storage.")
			return None

		branch_name = self.git_context.get_current_branch()
		sanitized_branch_name = self.sanitize_branch_name(branch_name)

		branch_dir = self.checksums_base_dir / sanitized_branch_name
		branch_dir.mkdir(parents=True, exist_ok=True)
		return branch_dir

	def _write_checksum_data(self, root_hash: str, nodes_map: dict[str, dict[str, str]]) -> Path | None:
		branch_dir = self._get_current_branch_checksum_dir()
		if not branch_dir:
			logger.error("Could not determine branch-specific directory. Cannot write checksum data.")
			return None

		timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")
		# Optionally include part of root_hash in filename for quick identification,
		# though timestamp should be unique enough.
		# short_root_hash = root_hash[:8]
		# checksum_file_name = f"{timestamp}_{short_root_hash}.json"
		checksum_file_name = f"{timestamp}.json"
		checksum_file_path = branch_dir / checksum_file_name

		data_to_write = {"root_hash": root_hash, "nodes": nodes_map}

		try:
			with checksum_file_path.open("w", encoding="utf-8") as f:
				json.dump(data_to_write, f, indent=2)
			logger.info(f"Checksum data written to {checksum_file_path}")
			return checksum_file_path
		except OSError:
			logger.exception(f"Error writing checksum data to {checksum_file_path}")
			return None

	def _get_latest_checksum_file_for_current_branch(self) -> Path | None:
		branch_dir = self._get_current_branch_checksum_dir()
		if not branch_dir or not branch_dir.exists():
			return None

		json_files = sorted(
			[f for f in branch_dir.iterdir() if f.is_file() and f.suffix == ".json"],
			key=lambda f: f.name,  # Relies on lexicographical sort of YYYY-MM-DD_HH-MM-SS-ffffff.json
			reverse=True,
		)

		if json_files:
			return json_files[0]
		return None

	def read_latest_checksum_data_for_current_branch(self) -> tuple[str | None, dict[str, dict[str, str]] | None]:
		"""Reads the most recent checksum data file for the current git branch.

		Attempts to locate and read the latest checksum JSON file in the branch-specific
		checksum directory. The file contains repository checksum information including
		the root hash and a map of all node checksums.

		Returns:
			tuple[str | None, dict[str, dict[str, str]] | None]:
				A tuple containing:
				- The root hash string if successfully read, otherwise None
				- A dictionary mapping paths to their checksum data if successfully read, otherwise None
				Both values will be None if no checksum file exists or if reading fails.
		"""
		latest_file = self._get_latest_checksum_file_for_current_branch()
		if not latest_file:
			logger.info("No previous checksum file found for the current branch.")
			return None, None

		try:
			with latest_file.open("r", encoding="utf-8") as f:
				data = json.load(f)

			root_hash = data.get("root_hash")
			nodes_map = data.get("nodes")

			if isinstance(root_hash, str) and isinstance(nodes_map, dict):
				logger.info(f"Successfully read checksum data from {latest_file}")
				return root_hash, nodes_map
			logger.error(f"Invalid format in checksum file {latest_file}. Missing 'root_hash' or 'nodes'.")
			return None, None
		except (OSError, json.JSONDecodeError):
			logger.exception(f"Error reading or parsing checksum file {latest_file}")
			return None, None
		except Exception:  # Catch any other unexpected error
			logger.exception(f"Unexpected error reading checksum file {latest_file}")
			return None, None

	async def calculate_repo_checksum(self) -> tuple[str, dict[str, dict[str, str]]]:
		"""Calculates the checksum for the entire repository and all its constituents.

		Returns:
		    A tuple containing:
		        - str: The checksum of the repository root.
		        - dict[str, dict[str, str]]: A dictionary mapping relative paths (files and dirs)
		                          to their calculated checksums. Paths use POSIX separators.
		"""
		local_nodes_map: dict[str, dict[str, str]] = {}  # Use a local var for population
		logger.info(f"Starting checksum calculation for repository: {self.repo_path}")

		# The recursive call for the repo_path itself will calculate its hash
		# based on its children and populate local_nodes_map.
		repo_root_checksum = await self._calculate_node_hash_recursive(self.repo_path, local_nodes_map)

		self.all_nodes_map = local_nodes_map  # Store the populated map

		# Write the new checksum data
		self._write_checksum_data(repo_root_checksum, self.all_nodes_map)

		logger.info(f"Finished checksum calculation. Root checksum: {repo_root_checksum}")
		return repo_root_checksum, self.all_nodes_map  # Return the stored map

	def get_file_checksum(self, relative_path_str: str) -> str | None:
		"""
		Retrieves the pre-calculated checksum for a specific file.

		Args:
		    relative_path_str: The POSIX-style relative path of the file from the repo root.

		Returns:
		    The checksum string if the file was found in the calculated map, else None.
		"""
		if self.all_nodes_map is None:
			# Try to load from latest if map isn't populated (e.g., if only get_file_checksum is called)
			_, nodes_map = self.read_latest_checksum_data_for_current_branch()
			if nodes_map is None:  # Still none after trying to read
				logger.warning(
					"Checksum map not calculated or readable. "
					"Call calculate_repo_checksum() or ensure a "
					"valid checksum file exists."
				)
				return None
			self.all_nodes_map = nodes_map

		node_info = self.all_nodes_map.get(relative_path_str)
		if node_info and node_info.get("type") == "file":
			return node_info.get("hash")

		# If path uses OS-specific separators, try converting to POSIX
		posix_path_str = Path(relative_path_str).as_posix()
		if posix_path_str != relative_path_str:
			node_info = self.all_nodes_map.get(posix_path_str)
			if node_info and node_info.get("type") == "file":
				return node_info.get("hash")

		logger.debug(f"No file checksum found for '{relative_path_str}' in the map.")
		return None

	@staticmethod
	def sanitize_branch_name(branch_name: str) -> str:
		"""Sanitizes a branch name to be safe for directory path construction.

		Replaces typical path separators and other problematic characters.
		"""
		if not branch_name:
			return "unnamed_branch"

		# Replace common separators like / and \\ with an underscore
		sanitized = branch_name.replace("/", "_").replace("\\", "_")

		# Remove or replace any characters not suitable for directory names.
		# Whitelist approach: allow alphanumeric, underscore, hyphen, dot.
		sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "", sanitized)

		# Prevent names that are just dots or empty after sanitization
		if not sanitized or all(c == "." for c in sanitized):
			return "invalid_branch_name_after_sanitize"

		# Limit length if necessary (OS path limits)
		max_len = 50  # Arbitrary reasonable limit for a directory name component
		if len(sanitized) > max_len:
			sanitized = sanitized[:max_len]

		return sanitized
