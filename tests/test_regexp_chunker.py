"""Tests for the fallback RegExpChunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from codemap.processor.chunking.base import Chunk, EntityType
from codemap.processor.chunking.reg_exp import RegExpChunker


@pytest.fixture
def code_samples() -> dict[str, str]:
    """Provide various code samples for testing."""
    return {
        "python": '''"""Module docstring for testing.

This is a test module.
"""

def test_function():
    """Test function."""
    print("Hello")

    value = 42
    return value

# Constants and variables
CONSTANT = 42
variable = "value"
''',
        "javascript": """/**
 * Module docstring for testing.
 *
 * This is a test module.
 */

function testFunction() {
    console.log("Hello");

    const value = 42;
    return value;
}

// Constants and variables
const CONSTANT = 42;
let variable = "value";
""",
        "unknown": """This is a text file without any specific language syntax.
It has multiple lines of text.

It should still be chunked based on empty lines.

This is another block of text.
""",
    }


def test_regexp_chunker_basic() -> None:
    """Test basic functionality of the RegExpChunker."""
    chunker = RegExpChunker()

    # Test with a simple Python example
    content = "def hello():\n    print('world')"
    chunks = chunker.chunk(content, Path("test.py"))

    # Verify we get a module chunk with children
    assert len(chunks) == 1
    module = chunks[0]
    assert module.metadata.entity_type == EntityType.MODULE
    assert module.metadata.name == "test"
    assert len(module.children) > 0


def test_regexp_chunker_docstring_extraction(code_samples: dict[str, str]) -> None:
    """Test that docstrings are properly extracted."""
    chunker = RegExpChunker()

    # Test with Python code
    chunks = chunker.chunk(code_samples["python"], Path("test.py"))
    assert len(chunks) == 1
    assert chunks[0].metadata.description is not None
    assert "Module docstring for testing" in chunks[0].metadata.description

    # Test with JavaScript code
    chunks = chunker.chunk(code_samples["javascript"], Path("test.js"))
    assert len(chunks) == 1
    assert chunks[0].metadata.description is not None
    assert "Module docstring for testing" in chunks[0].metadata.description


def test_regexp_chunker_block_detection(code_samples: dict[str, str]) -> None:
    """Test that the chunker properly detects code blocks."""
    chunker = RegExpChunker()

    # Test with different language samples
    for lang, sample in code_samples.items():
        file_path = Path(f"test.{lang if lang != 'unknown' else 'txt'}")
        chunks = chunker.chunk(sample, file_path)

        # Verify module chunk
        assert len(chunks) == 1
        module = chunks[0]

        # Verify child chunks
        assert len(module.children) > 0, f"No chunks detected for {lang} sample"

        # All children should have UNKNOWN entity type since we removed language-specific detection
        for child in module.children:
            assert child.metadata.entity_type == EntityType.UNKNOWN

        # Check that there's reasonable content in the children
        total_content = sum(len(child.content) for child in module.children)
        assert total_content > 0, f"No content in children for {lang} sample"


def test_regexp_chunker_splitting() -> None:
    """Test the chunk splitting functionality."""
    chunker = RegExpChunker()

    # Create a large content string
    large_content = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100

    # Create a chunk
    chunks = chunker.chunk(large_content, Path("large.txt"))
    module = chunks[0]

    # Test splitting with a small max size
    max_size = 150
    split_chunks = chunker.split(module, max_size)

    # Verify the splitting worked
    assert len(split_chunks) > 1, "Should split into multiple chunks"

    # Check that each chunk doesn't exceed the max_size significantly
    for chunk in split_chunks:
        assert len(chunk.content) <= max_size * 1.1, f"Chunk size {len(chunk.content)} should be <= {max_size * 1.1}"

    # Check that the total content is preserved (accounting for any line breaks)
    combined_content = "".join(chunk.content for chunk in split_chunks)
    assert combined_content == module.content


def get_all_chunks(root_chunk: Chunk) -> list[Chunk]:
    """Recursively get all chunks in a hierarchy."""
    result = [root_chunk]
    for child in root_chunk.children:
        result.extend(get_all_chunks(child))
    return result
