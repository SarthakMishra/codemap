"""
Daemon service for CodeMap.

This module implements the core daemon functionality for running CodeMap
as a persistent background service.

"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import daemon
from lockfile.pidlockfile import PIDLockFile

from codemap.processor.pipeline import ProcessingPipeline
from codemap.utils.config_loader import ConfigLoader


class FrameType(Protocol):
	"""Frame type protocol for signal handlers."""


class CodeMapDaemon:
	"""
	Manages the CodeMap daemon process lifecycle.

	This class handles starting, stopping, and monitoring the CodeMap
	daemon process, including process forking and signal handling.

	"""

	def __init__(self, config_path: str | Path | None = None) -> None:
		"""
		Initialize the daemon with configuration.

		Args:
		    config_path: Path to configuration file. If None, default locations will be checked.

		"""
		# Convert Path to str for ConfigLoader
		config_file = str(config_path) if config_path else None
		self.config_loader = ConfigLoader(config_file=config_file)
		self.config = self.config_loader.config

		# Get daemon-specific config with defaults
		daemon_config = self.config_loader.get_daemon_config()

		# Set up daemon paths using the proper directory structure
		self.base_dir = Path.home() / ".codemap"

		# Create structured directories
		self.logs_dir = self.base_dir / "logs"
		self.run_dir = self.base_dir / "run"
		self.data_dir = Path(self.config.get("storage", {}).get("data_dir", str(self.base_dir / "data"))).expanduser()

		# Set file paths
		self.pid_file = Path(daemon_config.get("pid_file", str(self.run_dir / "daemon.pid")))
		self.log_file = Path(daemon_config.get("log_file", str(self.logs_dir / "daemon.log")))
		self.socket_file = self.run_dir / "daemon.sock"

		# Ensure directories exist
		self.base_dir.mkdir(parents=True, exist_ok=True)
		self.logs_dir.mkdir(parents=True, exist_ok=True)
		self.run_dir.mkdir(parents=True, exist_ok=True)
		self.data_dir.mkdir(parents=True, exist_ok=True)

		# Components
		self.pipeline: ProcessingPipeline | None = None
		self.api_server = None
		self.running = False
		self.stop_event = threading.Event()

		# Setup logging
		self.logger = logging.getLogger("codemap.daemon")
		self._setup_logging()

	def _setup_logging(self) -> None:
		"""Configure logging for the daemon."""
		file_handler = logging.FileHandler(self.log_file)
		formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
		file_handler.setFormatter(formatter)

		# Set up the logger
		self.logger.addHandler(file_handler)
		self.logger.setLevel(logging.INFO)

		# Create a console handler if running in foreground
		console_handler = logging.StreamHandler()
		console_handler.setFormatter(formatter)
		self.logger.addHandler(console_handler)

	def start(self) -> bool:
		"""
		Start the daemon process in the background.

		Returns:
		    bool: True if successfully started, False otherwise

		"""
		if self.is_running():
			self.logger.info("Daemon already running with PID %s", self.get_pid())
			return False

		self.logger.info("Starting CodeMap daemon...")

		try:
			context = daemon.DaemonContext(
				working_directory="/",
				umask=0o002,
				pidfile=PIDLockFile(str(self.pid_file)),
				signal_map={signal.SIGTERM: self._handle_sigterm, signal.SIGINT: self._handle_sigint},
			)

			with context:
				try:
					self._initialize_components()
					self._run_main_loop()
					return True
				except Exception:
					self.logger.exception("Error in daemon")
					self._shutdown()
					return False
		except Exception:
			self.logger.exception("Error starting daemon")
			return False

		return False

	def stop(self) -> bool:
		"""
		Stop the daemon process.

		Returns:
		    bool: True if successfully stopped, False otherwise

		"""
		if not self.is_running():
			return False

		pid = self.get_pid()
		if pid:
			try:
				os.kill(pid, signal.SIGTERM)
				# Wait for process to terminate
				for _ in range(30):  # Wait up to 30 seconds
					if not self.is_running():
						break
					time.sleep(1)
				else:
					os.kill(pid, signal.SIGKILL)

				return True
			except ProcessLookupError:
				self.pid_file.unlink(missing_ok=True)
				return True

		return False

	def status(self) -> dict[str, Any]:
		"""
		Check the status of the daemon.

		Returns:
		    Dict[str, Any]: Dictionary with daemon status information

		"""
		if self.is_running():
			pid = self.get_pid()
			return {"status": "running", "pid": pid}
		return {"status": "stopped"}

	def is_running(self) -> bool:
		"""
		Check if the daemon is currently running.

		Returns:
		    bool: True if daemon is running, False otherwise

		"""
		pid = self.get_pid()
		if not pid:
			return False

		try:
			# Send signal 0 to check if process exists
			os.kill(pid, 0)
			return True
		except (ProcessLookupError, PermissionError):
			# Process doesn't exist or we don't have permission to check
			return False

	def get_pid(self) -> int | None:
		"""
		Get the current daemon PID if available.

		Returns:
		    Optional[int]: PID of the daemon process, or None if not running

		"""
		try:
			if self.pid_file.exists():
				return int(self.pid_file.read_text().strip())
		except (OSError, ValueError):
			pass
		return None

	def _handle_sigterm(self, _signum: int, _frame: FrameType) -> None:
		"""
		Handle SIGTERM signal for graceful shutdown.

		Args:
		    _signum: Signal number (unused)
		    _frame: Current stack frame (unused)

		"""
		self.logger.info("Received SIGTERM, shutting down gracefully...")
		self._shutdown()
		sys.exit(0)

	def _handle_sigint(self, _signum: int, _frame: FrameType) -> None:
		"""
		Handle SIGINT signal for graceful shutdown.

		Args:
		    _signum: Signal number (unused)
		    _frame: Current stack frame (unused)

		"""
		self.logger.info("Received SIGINT, shutting down gracefully...")
		self._shutdown()
		sys.exit(0)

	def _initialize_components(self) -> None:
		"""Initialize daemon components."""
		try:
			# Initialize processing pipeline
			# Use the current working directory as the repository path
			repo_path = Path.cwd()
			self.pipeline = ProcessingPipeline(repo_path=repo_path)

			# Start the API server
			from codemap.daemon.api_server import APIServer

			# Get API configuration from the config loader for consistency
			api_config = self.config_loader.get_api_config()
			host = api_config.get("host", "127.0.0.1")
			port = api_config.get("port", 8765)

			# Use socket file for IPC when appropriate
			use_socket = api_config.get("use_socket", False)
			socket_path = str(self.socket_file) if use_socket else None

			# Create API server with configuration
			self.api_server = APIServer(
				host=host,
				port=port,
				socket_path=socket_path,
				pipeline=self.pipeline,
				config_loader=self.config_loader,
			)

			# Start the API server
			self.api_server.start()
			self.running = True

			self.logger.info(
				"Daemon initialized with PID file: %s, Log file: %s, Data dir: %s",
				self.pid_file,
				self.log_file,
				self.data_dir,
			)

			if socket_path:
				self.logger.info("Using socket for IPC: %s", socket_path)
			else:
				self.logger.info("API server listening on %s:%s", host, port)

		except Exception:
			self.logger.exception("Failed to initialize daemon components")
			raise

	def _run_main_loop(self) -> None:
		"""Run the main daemon loop."""
		self.logger.info("Daemon running with PID %s", os.getpid())
		self.logger.info("Waiting for tasks...")

		try:
			# Main loop - wait for stop event
			while not self.stop_event.is_set():
				time.sleep(1)
		except KeyboardInterrupt:
			self.logger.info("Received keyboard interrupt, shutting down")
		except Exception:
			self.logger.exception("Error in main loop")
			raise
		finally:
			self._shutdown()

	def _shutdown(self) -> None:
		"""Shutdown the daemon and clean up resources."""
		if not self.running:
			return

		self.logger.info("Shutting down daemon...")
		self.stop_event.set()
		self.running = False

		# Stop API server
		if self.api_server:
			try:
				self.logger.info("Stopping API server...")
				self.api_server.stop()
			except Exception:
				self.logger.exception("Error stopping API server")

		# Stop pipeline
		if self.pipeline:
			try:
				self.logger.info("Stopping processing pipeline...")
				if hasattr(self.pipeline, "stop") and callable(self.pipeline.stop):
					self.pipeline.stop()
			except Exception:
				self.logger.exception("Error stopping processing pipeline")

		self.logger.info("Daemon shutdown complete")

	def run_foreground(self) -> None:
		"""Run the daemon in the foreground (for debugging)."""
		if self.is_running():
			self.logger.info("Daemon already running with PID %s", self.get_pid())
			return

		self.logger.info("Starting CodeMap daemon in foreground...")

		try:
			self._initialize_components()
			self._run_main_loop()
		except KeyboardInterrupt:
			self.logger.info("Received keyboard interrupt, shutting down")
		except Exception:
			self.logger.exception("Error running in foreground")
		finally:
			self._shutdown()
