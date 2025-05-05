"""Tests for the semantic_grouping.group module."""

from codemap.git.diff_splitter import DiffChunk
from codemap.git.semantic_grouping.group import SemanticGroup


def test_semantic_group_init_empty():
	"""Test initialization of an empty SemanticGroup."""
	group = SemanticGroup()
	assert group.chunks == []
	assert group.name is None
	assert group.message is None
	assert group.approved is False


def test_semantic_group_init_with_chunks():
	"""Test initialization of a SemanticGroup with chunks."""
	chunks = [
		DiffChunk(files=["file1.py"], content="diff1"),
		DiffChunk(files=["file2.py"], content="diff2"),
	]
	group = SemanticGroup(chunks=chunks, name="Test Group")

	assert group.chunks == chunks
	assert group.name == "Test Group"
	assert group.message is None
	assert group.approved is False


def test_semantic_group_files_property():
	"""Test the files property of a SemanticGroup."""
	chunks = [
		DiffChunk(files=["file1.py"], content="diff1"),
		DiffChunk(files=["file2.py", "file3.py"], content="diff2"),
		DiffChunk(files=["file1.py", "file4.py"], content="diff3"),
	]
	group = SemanticGroup(chunks=chunks)

	# Should return a sorted list of unique files
	assert group.files == ["file1.py", "file2.py", "file3.py", "file4.py"]


def test_semantic_group_content_property():
	"""Test the content property of a SemanticGroup."""
	chunks = [
		DiffChunk(files=["file1.py"], content="diff1"),
		DiffChunk(files=["file2.py"], content="diff2"),
	]
	group = SemanticGroup(chunks=chunks)

	# Should concatenate all content with newlines
	assert group.content == "diff1\ndiff2"


def test_semantic_group_merge_with():
	"""Test merging two SemanticGroups."""
	group1 = SemanticGroup(chunks=[DiffChunk(files=["file1.py"], content="diff1")], name="Group 1")
	group2 = SemanticGroup(chunks=[DiffChunk(files=["file2.py"], content="diff2")], name="Group 2")

	merged = group1.merge_with(group2)

	# Check merged group properties
	assert len(merged.chunks) == 2
	assert merged.files == ["file1.py", "file2.py"]
	assert merged.content == "diff1\ndiff2"
	assert merged.name == "Merged: Group 1 + Group 2"

	# Original groups should be unchanged
	assert len(group1.chunks) == 1
	assert len(group2.chunks) == 1


def test_semantic_group_repr():
	"""Test the string representation of a SemanticGroup."""
	chunks = [
		DiffChunk(files=["file1.py"], content="diff1"),
		DiffChunk(files=["file2.py", "file3.py"], content="diff2"),
	]
	group = SemanticGroup(chunks=chunks)

	# Should show count of files and chunks
	assert repr(group) == "SemanticGroup(files=3, chunks=2)"
