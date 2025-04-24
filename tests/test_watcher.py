"""Tests for the file watcher module."""

import tempfile
import time
from pathlib import Path

import pytest

from codemap.processor.watcher import FileWatcher


@pytest.mark.unit
@pytest.mark.processor
@pytest.mark.watcher
@pytest.mark.asynchronous
class TestFileWatcher:
    """Tests for the FileWatcher class."""

    def test_initialization(self) -> None:
        """Test FileWatcher initialization."""
        # Arrange & Act
        with tempfile.TemporaryDirectory() as temp_dir:
            watcher = FileWatcher(temp_dir)

            # Assert
            assert watcher.paths == [Path(temp_dir)]
            assert watcher.recursive is True

    def test_invalid_path(self) -> None:
        """Test FileWatcher with invalid path."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Path does not exist"):
            FileWatcher("/nonexistent/path")

    def test_events(self) -> None:
        """Test FileWatcher events."""
        # Arrange
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

            # Act - Initialize and start watcher
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

            # Assert
            assert len(created_files) == 1
            assert test_file.as_posix() in created_files[0]
            assert len(modified_files) > 0  # At least one modification event
            assert test_file.as_posix() in modified_files[0]
            assert len(deleted_files) == 1
            assert test_file.as_posix() in deleted_files[0]

    def test_ignored_patterns(self) -> None:
        """Test FileWatcher with ignored patterns."""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            created_files = []

            def on_created(path: str) -> None:
                created_files.append(path)

            # Act - Initialize and start watcher with ignored pattern
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

            # Assert
            assert len(created_files) == 1
            assert test_file.as_posix() in created_files[0]
            assert not any(ignored_file.as_posix() in path for path in created_files)
