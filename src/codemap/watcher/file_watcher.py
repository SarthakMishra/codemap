"""File watcher module for CodeMap."""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from pathlib import Path

import anyio  # Add import
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class FileChangeHandler(FileSystemEventHandler):
	"""Handles file system events and triggers a callback."""

	def __init__(self, callback: Callable[[], Coroutine[None, None, None]], debounce_delay: float = 1.0) -> None:
		"""
		Initialize the handler.

		Args:
		    callback: An async function to call when changes are detected.
		    debounce_delay: Minimum time (seconds) between callback triggers.
		"""
		self.callback = callback
		self.debounce_delay = debounce_delay
		self._last_event_time: float = 0
		self._debounce_task: asyncio.Task | None = None

	def _schedule_callback(self) -> None:
		"""Schedule the callback execution after debounce delay, resetting timer on new events."""
		# Cancel any existing debounce task if it's running
		if self._debounce_task and not self._debounce_task.done():
			self._debounce_task.cancel()
			logger.debug("Cancelled existing debounce task due to new event.")

		# Always schedule a new debounce task
		logger.debug(f"Scheduling new debounced callback with {self.debounce_delay}s delay.")
		self._debounce_task = asyncio.create_task(self._debounced_callback())

	async def _debounced_callback(self) -> None:
		"""Wait for the debounce period and then execute the callback."""
		try:
			await asyncio.sleep(self.debounce_delay)
			logger.info("Debounce delay finished, triggering sync callback.")
			await self.callback()
			self._last_event_time = time.monotonic()  # Update time after successful execution
			logger.debug("Watcher callback executed successfully.")
		except asyncio.CancelledError:
			logger.debug("Debounce task cancelled before execution.")
			# Do not run the callback if cancelled
		except Exception:
			logger.exception("Error executing watcher callback")
		finally:
			# Clear the task reference once it's done (either completed, cancelled, or errored)
			self._debounce_task = None

	def on_any_event(self, event: FileSystemEvent) -> None:
		"""
		Catch all events and schedule the callback after debouncing.

		Args:
		    event: The file system event.
		"""
		if event.is_directory:
			return  # Ignore directory events for now, focus on file changes

		# Log the specific event detected
		event_type = event.event_type
		src_path = getattr(event, "src_path", "N/A")
		dest_path = getattr(event, "dest_path", "N/A")  # For moved events

		if event_type == "moved":
			logger.debug(f"Detected file {event_type}: {src_path} -> {dest_path}")
		else:
			logger.debug(f"Detected file {event_type}: {src_path}")

		self._schedule_callback()


class Watcher:
	"""Monitors a directory for changes and triggers a callback."""

	def __init__(
		self,
		path_to_watch: str | Path,
		on_change_callback: Callable[[], Coroutine[None, None, None]],
		debounce_delay: float = 1.0,
	) -> None:
		"""
		Initialize the watcher.

		Args:
		    path_to_watch: The directory path to monitor.
		    on_change_callback: Async function to call upon detecting changes.
		    debounce_delay: Delay in seconds to avoid rapid firing of callbacks.
		"""
		self.observer = Observer()
		self.path_to_watch = Path(path_to_watch).resolve()
		if not self.path_to_watch.is_dir():
			msg = f"Path to watch must be a directory: {self.path_to_watch}"
			raise ValueError(msg)
		self.event_handler = FileChangeHandler(on_change_callback, debounce_delay)
		self._stop_event = anyio.Event()  # Initialize the event

	async def start(self) -> None:
		"""Start monitoring the directory."""
		if not self.path_to_watch.exists():
			logger.warning(f"Watch path {self.path_to_watch} does not exist. Creating it.")
			self.path_to_watch.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

		self.observer.schedule(self.event_handler, str(self.path_to_watch), recursive=True)
		self.observer.start()
		logger.info(f"Started watching directory: {self.path_to_watch}")
		try:
			# Wait until the stop event is set
			await self._stop_event.wait()  # Replaced while loop + sleep
		except KeyboardInterrupt:
			logger.info("Watcher stopped by user (KeyboardInterrupt).")
		finally:
			# Ensure stop is called regardless of how wait() exits
			self.stop()

	def stop(self) -> None:
		"""Stop monitoring the directory."""
		if self.observer.is_alive():
			self.observer.stop()
			self.observer.join()  # Wait for observer thread to finish
			logger.info("Watchdog observer stopped.")
		# Set the event to signal the start method to exit
		self._stop_event.set()  # Set the event
		logger.info("Watcher stop event set.")
