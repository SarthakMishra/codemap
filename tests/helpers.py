"""Helper functions and utilities for tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

from codemap.git.diff_splitter import DiffChunk

if TYPE_CHECKING:
	from collections.abc import Callable


def create_diff_chunk(
	files: list[str], content: str, description: str | None = None, is_llm_generated: bool = False
) -> DiffChunk:
	"""
	Create a DiffChunk for testing.

	Args:
	    files: List of files in the diff
	    content: Diff content
	    description: Optional description
	    is_llm_generated: Whether the description was LLM-generated

	Returns:
	    A DiffChunk instance

	"""
	chunk = MagicMock(spec=DiffChunk)
	chunk.files = files
	chunk.content = content
	chunk.description = description
	chunk.is_llm_generated = is_llm_generated
	return chunk


def assert_chunk_processed(chunk: DiffChunk, message: str) -> None:
	"""
	Assert that a chunk was properly processed.

	Args:
	    chunk: The processed chunk
	    message: Expected message

	"""
	assert chunk.description == message
	assert chunk.is_llm_generated


def create_file_content(path: Path, content: str) -> None:
	"""
	Create a file with specified content for testing.

	Args:
	    path: Path where to create the file
	    content: Content to write to the file

	"""
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content)


def create_python_file(path: Path, content: str | None = None) -> None:
	"""
	Create a Python file with default or specified content.

	Args:
	    path: Path where to create the file
	    content: Content to write to the file (if None, a basic Python file is created)

	"""
	if content is None:
		content = '"""Example module."""\n\ndef example_function():\n    """Example function."""\n    return True\n'
	create_file_content(path, content)


def patch_multiple(*args: str, **kwargs: str) -> Callable[[Callable], Callable]:
	"""
	Patch multiple objects at once as a decorator.

	Args:
	    *args: Patch targets
	    **kwargs: Optional patch targets with custom names

	Returns:
	    Decorator function for patching

	"""

	def decorator(func: Callable) -> Callable:
		for target in args:
			func = patch(target)(func)
		for name, target in kwargs.items():
			func = patch(target, name=name)(func)
		return func

	return decorator


def read_fixture_file(rel_path: str) -> str:
	"""
	Read a fixture file from the fixtures directory.

	Args:
	    rel_path: Relative path within fixtures directory

	Returns:
	    Content of the fixture file

	"""
	path = Path(__file__).parent / "fixtures" / rel_path
	return path.read_text(encoding="utf-8")


def create_git_commit_data(
	commit_hash: str = "abcdef1234567890",
	author: str = "Test User",
	email: str = "test@example.com",
	date: str = "2024-06-01 12:34:56 +0000",
	message: str = "Test commit message",
	diff_content: str | None = None,
) -> dict[str, Any]:
	"""
	Create standard Git commit data for tests.

	Args:
	    commit_hash: The commit hash
	    author: Author name
	    email: Author email
	    date: Commit date in Git format
	    message: Commit message
	    diff_content: Optional diff content

	Returns:
	    A dictionary with commit data

	"""
	if diff_content is None:
		diff_content = """diff --git a/file1.py b/file1.py
index 1234567..abcdefg 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@ def existing_function():
     return True

 def new_function():
-    return False
+    return True
"""

	return {
		"hash": commit_hash,
		"author": author,
		"email": email,
		"date": date,
		"message": message,
		"diff": diff_content,
		"files": ["file1.py"],
	}
