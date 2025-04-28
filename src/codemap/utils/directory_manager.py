"""
Directory management utilities for CodeMap.

This module provides functions and classes to manage the CodeMap
directory structure across different operating systems.

"""

from __future__ import annotations

import logging
import shutil
from functools import lru_cache
from pathlib import Path

import platformdirs
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# Application name and author for platformdirs
APP_NAME = "codemap"
APP_AUTHOR = "codemap"


class DirectoryManager:
	"""Manages CodeMap directory structure and operations."""

	def __init__(self) -> None:
		"""Initialize the directory manager."""
		# Main directory locations
		self.user_data_dir = Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR))
		self.user_config_dir = Path(platformdirs.user_config_dir(APP_NAME, APP_AUTHOR))
		self.user_cache_dir = Path(platformdirs.user_cache_dir(APP_NAME, APP_AUTHOR))
		self.user_log_dir = Path(platformdirs.user_log_dir(APP_NAME, APP_AUTHOR))
		self.user_state_dir = Path(platformdirs.user_state_dir(APP_NAME, APP_AUTHOR))

		# Project-specific directory (set when working with a project)
		self.project_dir = None

	def ensure_directories(self) -> None:
		"""Ensure all required directories exist."""
		# Create main directories
		self._ensure_dir(self.user_data_dir)
		self._ensure_dir(self.user_config_dir)
		self._ensure_dir(self.user_cache_dir)
		self._ensure_dir(self.user_log_dir)
		self._ensure_dir(self.user_state_dir)

		# Create subdirectories
		self._ensure_dir(self.config_dir)
		self._ensure_dir(self.cache_dir)
		self._ensure_dir(self.db_dir)
		self._ensure_dir(self.logs_dir)
		self._ensure_dir(self.tmp_dir)

		# Create database subdirectories
		self._ensure_dir(self.vector_db_dir)
		self._ensure_dir(self.sqlite_db_dir)
		self._ensure_dir(self.kv_db_dir)

		# Create log subdirectories
		self._ensure_dir(self.daemon_logs_dir)
		self._ensure_dir(self.cli_logs_dir)
		self._ensure_dir(self.error_logs_dir)

		# Create cache subdirectories
		self._ensure_dir(self.models_cache_dir)
		self._ensure_dir(self.providers_cache_dir)

	def set_project_dir(self, project_path: str | Path) -> None:
		"""
		Set the current project directory.

		Args:
		    project_path: Path to the project

		"""
		self.project_dir = Path(project_path).resolve()

		# Create project-specific directories if needed
		project_cache_dir = self.project_dir / ".codemap_cache"
		if not project_cache_dir.exists():
			self._ensure_dir(project_cache_dir)
			# Create project subdirectories
			self._ensure_dir(project_cache_dir / "storage")
			self._ensure_dir(project_cache_dir / "embeddings")
			self._ensure_dir(project_cache_dir / "lsp")
			self._ensure_dir(project_cache_dir / "logs")

			# Create a .gitignore file in the cache directory
			self._create_gitignore(project_cache_dir)

	def clear_cache(self, cache_type: str | None = None) -> None:
		"""
		Clear the specified cache directory.

		Args:
		    cache_type: Type of cache to clear, or None for all

		"""
		if cache_type is None or cache_type == "all":
			self._clear_directory(self.cache_dir)
			self._clear_directory(self.models_cache_dir)
			self._clear_directory(self.providers_cache_dir)
		elif cache_type == "models":
			self._clear_directory(self.models_cache_dir)
		elif cache_type == "providers":
			self._clear_directory(self.providers_cache_dir)
		else:
			logger.warning("Unknown cache type: %s", cache_type)

	def get_log_file_path(self, log_type: str, name: str | None = None) -> Path:
		"""
		Get the path to a log file.

		Args:
		    log_type: Type of log ('daemon', 'cli', 'error')
		    name: Specific name for the log file (default: based on log_type)

		Returns:
		    Path to the log file

		"""
		if log_type == "daemon":
			log_dir = self.daemon_logs_dir
			name = name or "daemon.log"
		elif log_type == "cli":
			log_dir = self.cli_logs_dir
			name = name or "cli.log"
		elif log_type == "error":
			log_dir = self.error_logs_dir
			name = name or "error.log"
		else:
			log_dir = self.logs_dir
			name = name or f"{log_type}.log"

		return log_dir / name

	def get_project_cache_dir(self, create: bool = True) -> Path | None:
		"""
		Get the project-specific cache directory.

		Args:
		    create: Whether to create the directory if it doesn't exist

		Returns:
		    Path to the project cache directory, or None if no project is set

		"""
		if not self.project_dir:
			return None

		project_cache_dir = self.project_dir / ".codemap_cache"
		if create and not project_cache_dir.exists():
			self._ensure_dir(project_cache_dir)
			# Create basic structure
			self._ensure_dir(project_cache_dir / "storage")
			self._ensure_dir(project_cache_dir / "logs")
			self._create_gitignore(project_cache_dir)

		return project_cache_dir

	def _ensure_dir(self, directory: Path) -> None:
		"""
		Ensure a directory exists, creating it if necessary.

		Args:
		    directory: Directory path to ensure exists

		"""
		try:
			directory.mkdir(parents=True, exist_ok=True)
		except Exception:
			logger.exception("Failed to create directory %s", directory)
			raise

	def _clear_directory(self, directory: Path) -> None:
		"""
		Clear contents of a directory without removing the directory itself.

		Args:
		    directory: Directory to clear

		"""
		if not directory.exists() or not directory.is_dir():
			return

		try:
			for item in directory.iterdir():
				if item.is_dir():
					shutil.rmtree(item)
				else:
					item.unlink()
		except Exception:
			logger.exception("Failed to clear directory %s", directory)
			raise

	def _create_gitignore(self, directory: Path) -> None:
		"""
		Create a .gitignore file in the given directory.

		Args:
		    directory: Directory where the .gitignore file should be created

		"""
		gitignore_path = directory / ".gitignore"
		if not gitignore_path.exists():
			try:
				with gitignore_path.open("w") as f:
					f.write("# Automatically generated by CodeMap\n")
					f.write("# Cache files should not be committed\n")
					f.write("*\n")
					f.write("!*/\n")
					f.write("!.gitignore\n")
			except Exception:
				logger.exception("Failed to create .gitignore in %s", directory)

	# Directory property getters
	@property
	def config_dir(self) -> Path:
		"""Get the configuration directory."""
		return self.user_config_dir / "config"

	@property
	def cache_dir(self) -> Path:
		"""Get the cache directory."""
		return self.user_cache_dir / "cache"

	@property
	def db_dir(self) -> Path:
		"""Get the database directory."""
		return self.user_data_dir / "db"

	@property
	def logs_dir(self) -> Path:
		"""Get the logs directory."""
		return self.user_log_dir / "logs"

	@property
	def tmp_dir(self) -> Path:
		"""Get the temporary directory."""
		return self.user_cache_dir / "tmp"

	@property
	def vector_db_dir(self) -> Path:
		"""Get the vector database directory."""
		return self.db_dir / "vector"

	@property
	def sqlite_db_dir(self) -> Path:
		"""Get the SQLite database directory."""
		return self.db_dir / "sqlite"

	@property
	def kv_db_dir(self) -> Path:
		"""Get the key-value database directory."""
		return self.db_dir / "kv"

	@property
	def daemon_logs_dir(self) -> Path:
		"""Get the daemon logs directory."""
		return self.logs_dir / "daemon"

	@property
	def cli_logs_dir(self) -> Path:
		"""Get the CLI logs directory."""
		return self.logs_dir / "cli"

	@property
	def error_logs_dir(self) -> Path:
		"""Get the error logs directory."""
		return self.logs_dir / "error"

	@property
	def models_cache_dir(self) -> Path:
		"""Get the models cache directory."""
		return self.cache_dir / "models"

	@property
	def providers_cache_dir(self) -> Path:
		"""Get the providers cache directory."""
		return self.cache_dir / "providers"


@lru_cache
def get_directory_manager() -> DirectoryManager:
	"""
	Get a cached instance of the DirectoryManager.

	Returns:
	    DirectoryManager instance

	"""
	return DirectoryManager()
