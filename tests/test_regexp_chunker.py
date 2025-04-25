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


def get_all_chunks(root_chunk: Chunk) -> list[Chunk]:
    """Recursively get all chunks in a hierarchy."""
    result = [root_chunk]
    for child in root_chunk.children:
        result.extend(get_all_chunks(child))
    return result


@pytest.mark.unit
@pytest.mark.processor
@pytest.mark.chunking
class TestRegExpChunker:
    """Test cases for the fallback regular expression chunker."""

    def setup_method(self) -> None:
        """Set up test environment with a fresh chunker instance."""
        self.chunker = RegExpChunker()

    def test_basic_functionality(self) -> None:
        """Test basic functionality of the RegExpChunker."""
        # Arrange
        content = "def hello():\n    print('world')"

        # Act
        chunks = self.chunker.chunk(content, Path("test.py"))

        # Assert
        assert len(chunks) == 1
        module = chunks[0]
        assert module.metadata.entity_type == EntityType.MODULE
        assert module.metadata.name == "test"
        assert len(module.children) > 0

    def test_docstring_extraction(self, code_samples: dict[str, str]) -> None:
        """Test that docstrings are properly extracted from different
        languages."""
        # Arrange - using code_samples fixture

        # Act & Assert for Python
        python_chunks = self.chunker.chunk(code_samples["python"], Path("test.py"))
        assert len(python_chunks) == 1
        assert python_chunks[0].metadata.description is not None
        assert "Module docstring for testing" in python_chunks[0].metadata.description

        # Act & Assert for JavaScript
        js_chunks = self.chunker.chunk(code_samples["javascript"], Path("test.js"))
        assert len(js_chunks) == 1
        assert js_chunks[0].metadata.description is not None
        assert "Module docstring for testing" in js_chunks[0].metadata.description

    @pytest.mark.parametrize(
        ("language", "file_extension"),
        [
            ("python", "py"),
            ("javascript", "js"),
            ("unknown", "txt"),
        ],
    )
    def test_block_detection(self, code_samples: dict[str, str], language: str, file_extension: str) -> None:
        """Test that the chunker properly detects code blocks for different
        languages."""
        # Arrange
        sample = code_samples[language]
        file_path = Path(f"test.{file_extension}")

        # Act
        chunks = self.chunker.chunk(sample, file_path)

        # Assert
        assert len(chunks) == 1
        module = chunks[0]
        assert len(module.children) > 0, f"No chunks detected for {language} sample"

        # All children should have UNKNOWN entity type since we removed language-specific detection
        for child in module.children:
            assert child.metadata.entity_type == EntityType.UNKNOWN

        # Check that there's reasonable content in the children
        total_content = sum(len(child.content) for child in module.children)
        assert total_content > 0, f"No content in children for {language} sample"

    def test_chunk_splitting(self) -> None:
        """Test the chunk splitting functionality with large content."""
        # Arrange
        large_content = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
        chunks = self.chunker.chunk(large_content, Path("large.txt"))
        module = chunks[0]
        max_size = 150

        # Act
        split_chunks = self.chunker.split(module, max_size)

        # Assert
        assert len(split_chunks) > 1, "Should split into multiple chunks"

        # Check size constraints
        for chunk in split_chunks:
            assert len(chunk.content) <= max_size * 1.1, (
                f"Chunk size {len(chunk.content)} should be <= {max_size * 1.1}"
            )

        # Check content preservation
        combined_content = "".join(chunk.content for chunk in split_chunks)
        assert combined_content == module.content
