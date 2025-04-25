"""File system monitoring implementation for CodeMap."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class FileEventHandler(FileSystemEventHandler):
    """Event handler for file system events."""

    def __init__(
        self,
        on_created: Callable[[str], None] | None = None,
        on_modified: Callable[[str], None] | None = None,
        on_deleted: Callable[[str], None] | None = None,
        ignored_patterns: set[str] | None = None,
    ) -> None:
        """Initialize the event handler.

        Args:
            on_created: Callback function for file creation events
            on_modified: Callback function for file modification events
            on_deleted: Callback function for file deletion events
            ignored_patterns: Set of glob patterns to ignore

        """
        super().__init__()
        self._created_callback = on_created
        self._modified_callback = on_modified
        self._deleted_callback = on_deleted
        self.ignored_patterns = ignored_patterns or set()

    def _should_ignore(self, path: str | bytes) -> bool:
        """Check if the path should be ignored based on patterns.

        Args:
            path: File path to check

        Returns:
            bool: True if path should be ignored, False otherwise

        """
        path_str = path.decode() if isinstance(path, bytes) else str(path)
        path_obj = Path(path_str)
        return any(path_obj.match(pattern) for pattern in self.ignored_patterns)

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.

        Args:
            event: The file system event

        """
        if event.is_directory or self._should_ignore(event.src_path):
            return
        if self._created_callback:
            path_str = event.src_path.decode() if isinstance(event.src_path, bytes) else str(event.src_path)
            logger.debug("File created: %s", path_str)
            self._created_callback(path_str)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: The file system event

        """
        if event.is_directory or self._should_ignore(event.src_path):
            return
        if self._modified_callback:
            path_str = event.src_path.decode() if isinstance(event.src_path, bytes) else str(event.src_path)
            logger.debug("File modified: %s", path_str)
            self._modified_callback(path_str)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events.

        Args:
            event: The file system event

        """
        if event.is_directory or self._should_ignore(event.src_path):
            return
        if self._deleted_callback:
            path_str = event.src_path.decode() if isinstance(event.src_path, bytes) else str(event.src_path)
            logger.debug("File deleted: %s", path_str)
            self._deleted_callback(path_str)


class FileWatcher:
    """File system watcher for monitoring changes."""

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        on_created: Callable[[str], None] | None = None,
        on_modified: Callable[[str], None] | None = None,
        on_deleted: Callable[[str], None] | None = None,
        ignored_patterns: set[str] | None = None,
        recursive: bool = True,
    ) -> None:
        """Initialize the file watcher.

        Args:
            paths: Path or list of paths to watch
            on_created: Callback function for file creation events
            on_modified: Callback function for file modification events
            on_deleted: Callback function for file deletion events
            ignored_patterns: Set of glob patterns to ignore
            recursive: Whether to watch directories recursively

        """
        self.paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
        self.recursive = recursive
        self.observer = Observer()
        self.event_handler = FileEventHandler(
            on_created=on_created,
            on_modified=on_modified,
            on_deleted=on_deleted,
            ignored_patterns=ignored_patterns,
        )

        for path in self.paths:
            if not path.exists():
                msg = f"Path does not exist: {path}"
                raise ValueError(msg)
            self.observer.schedule(self.event_handler, str(path), recursive=recursive)

    def start(self) -> None:
        """Start watching for file system events."""
        logger.info("Starting file watcher for paths: %s", self.paths)
        self.observer.start()

    def stop(self) -> None:
        """Stop watching for file system events."""
        logger.info("Stopping file watcher")
        self.observer.stop()
        self.observer.join()
