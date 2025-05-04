"""File watcher module for CodeMap."""

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable, Coroutine
from pathlib import Path

import anyio
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class FileChangeHandler(FileSystemEventHandler):
	"""Handles file system events and triggers a callback."""

	def __init__(
		self,
		callback: Callable[[], Coroutine[None, None, None]],
		debounce_delay: float = 1.0,
		event_loop: asyncio.AbstractEventLoop | None = None,
	) -> None:
		"""
		Initialize the handler.

		Args:
		    callback: An async function to call when changes are detected.
		    debounce_delay: Minimum time (seconds) between callback triggers.
		    event_loop: The asyncio event loop to use, or None to use the current one.

		"""
		self.callback = callback
		self.debounce_delay = debounce_delay
		self._last_event_time: float = 0
		self._debounce_task: asyncio.Task | None = None
		# Set up a thread-safe way to communicate with the event loop
		self._event_queue = queue.Queue()
		self._event_processed = threading.Event()
		# Store or get the event loop
		self._event_loop = event_loop or asyncio.get_event_loop()
		# Flag to track if we're in the process of handling an event
		self._processing = False

	def _schedule_callback(self) -> None:
		"""Schedule the callback execution from a thread-safe context."""
		# If processing is already in progress, just return
		if self._processing:
			return

		# Put the event in the queue for the event loop to process
		self._event_processed.clear()
		self._event_queue.put_nowait("file_change")

		# Schedule the task in the event loop
		asyncio.run_coroutine_threadsafe(self._process_events(), self._event_loop)

	async def _process_events(self) -> None:
		"""Process events from the queue in the event loop's context."""
		if self._processing:
			return

		self._processing = True
		try:
			# Get an event from the queue
			while not self._event_queue.empty():
				_ = self._event_queue.get_nowait()

				# Cancel any existing debounce task
				if self._debounce_task and not self._debounce_task.done():
					self._debounce_task.cancel()
					logger.debug("Cancelled existing debounce task due to new event.")

				# Create a new debounce task within the event loop's context
				logger.debug(f"Scheduling new debounced callback with {self.debounce_delay}s delay.")
				self._debounce_task = self._event_loop.create_task(self._debounced_callback())
		finally:
			self._processing = False
			self._event_processed.set()

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
			# Clear the task reference once it's done
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

		# Schedule the callback in a thread-safe way
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

		# Save the current event loop to use for callbacks
		try:
			self._event_loop = asyncio.get_event_loop()
		except RuntimeError:
			# If we're not in an event loop context, create a new one
			self._event_loop = asyncio.new_event_loop()
			asyncio.set_event_loop(self._event_loop)

		self.event_handler = FileChangeHandler(on_change_callback, debounce_delay, event_loop=self._event_loop)
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
			await self._stop_event.wait()
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
		self._stop_event.set()
		logger.info("Watcher stop event set.")
