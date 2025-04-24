"""Tests for the file watcher module."""

import tempfile
import time
from pathlib import Path

import pytest

from codemap.processor.watcher import FileWatcher


def test_file_watcher_initialization() -> None:
    """Test FileWatcher initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        watcher = FileWatcher(temp_dir)
        assert watcher.paths == [Path(temp_dir)]
        assert watcher.recursive is True


def test_file_watcher_invalid_path() -> None:
    """Test FileWatcher with invalid path."""
    with pytest.raises(ValueError, match="Path does not exist"):
        FileWatcher("/nonexistent/path")


def test_file_watcher_events() -> None:
    """Test FileWatcher events."""
    with tempfile.TemporaryDirectory() as temp_dir:
        created_files = []
        modified_files = []
        deleted_files = []

        def on_created(path: str) -> None:
            created_files.append(path)

        def on_modified(path: str) -> None:
            modified_files.append(path)

        def on_deleted(path: str) -> None:
            deleted_files.append(path)

        watcher = FileWatcher(
            temp_dir,
            on_created=on_created,
            on_modified=on_modified,
            on_deleted=on_deleted,
        )
        watcher.start()

        # Create a test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Hello")
        time.sleep(0.1)  # Wait for events to process

        # Modify the file
        test_file.write_text("Hello World")
        time.sleep(0.1)  # Wait for events to process

        # Delete the file
        test_file.unlink()
        time.sleep(0.1)  # Wait for events to process

        watcher.stop()

        assert len(created_files) == 1
        assert test_file.as_posix() in created_files[0]
        assert len(modified_files) > 0  # At least one modification event
        assert test_file.as_posix() in modified_files[0]
        assert len(deleted_files) == 1
        assert test_file.as_posix() in deleted_files[0]


def test_file_watcher_ignored_patterns() -> None:
    """Test FileWatcher with ignored patterns."""
    with tempfile.TemporaryDirectory() as temp_dir:
        created_files = []

        def on_created(path: str) -> None:
            created_files.append(path)

        watcher = FileWatcher(
            temp_dir,
            on_created=on_created,
            ignored_patterns={"*.ignore"},
        )
        watcher.start()

        # Create a test file that should be watched
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Hello")

        # Create a test file that should be ignored
        ignored_file = Path(temp_dir) / "test.ignore"
        ignored_file.write_text("Ignore me")

        time.sleep(0.1)  # Wait for events to process
        watcher.stop()

        assert len(created_files) == 1
        assert test_file.as_posix() in created_files[0]
        assert not any(ignored_file.as_posix() in path for path in created_files)
