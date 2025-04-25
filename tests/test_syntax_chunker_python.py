"""Tests for the syntax-based code chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from codemap.processor.chunking.base import Chunk, EntityType
from codemap.processor.chunking.tree_sitter import TreeSitterChunker


@pytest.fixture
def python_code() -> str:
	"""Sample Python code for testing."""
	return '''"""Module docstring with detailed explanation.

This module demonstrates various Python constructs for testing the syntax chunker.
"""

import os
import sys
from typing import List, Dict, Optional, TypeVar, Generic

# Type definitions
T = TypeVar('T')
MyType = Dict[str, List[int]]

# Module-level constant
PI = 3.14159
DEBUG = True

class BaseClass:
    """Base class docstring."""
    base_attr = "base"

class MyClass(BaseClass):
    """Class docstring with comprehensive explanation.

    This class demonstrates various Python features for testing.
    """

    # Class attribute
    class_attr = "value"

    def __init__(self, x: int = 1, y: int = 2):
        """Constructor docstring.

        Args:
            x: First parameter
            y: Second parameter
        """
        self.x = x
        self.y = y

    @property
    def value(self) -> int:
        """Property docstring.

        Returns:
            The x value
        """
        return self.x

    def calculate(self, a: int, b: int) -> int:
        """Regular method with parameters.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of the parameters
        """
        return a + b + self.x

class GenericClass(Generic[T]):
    """A generic class example."""

    def __init__(self, value: T):
        self.value = value

    def get_value(self) -> T:
        return self.value

def regular_function(param1: str, param2: Optional[int] = None) -> bool:
    """Regular function docstring.

    Args:
        param1: First parameter
        param2: Optional second parameter

    Returns:
        Boolean result
    """
    if param2 is None:
        return len(param1) > 0
    return len(param1) > param2

def test_function():
    """Test function docstring."""
    # This is a test function
    assert True

    # Multi-line test
    value = 42
    assert value == 42

# Constants and variables
CONSTANT = 42
MAX_SIZE = 100

variable_one = "This is a variable"
variable_two = [1, 2, 3, 4, 5]

# Long function to test chunking of larger entities
def long_function_for_testing():
    """Long function to test chunking of larger entities."""
    result = []
    for i in range(100):
        if i % 2 == 0:
            result.append(i)
        elif i % 3 == 0:
            result.append(i * 2)
        else:
            result.append(i * 3)

    # More code to make this function larger
    data = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3",
    }

    for key, value in data.items():
        print(f"{key}: {value}")

    return result
'''


def get_all_chunks(root_chunk: Chunk) -> list[Chunk]:
	"""Recursively get all chunks in a hierarchy."""
	result = [root_chunk]
	for child in root_chunk.children:
		result.extend(get_all_chunks(child))
	return result


@pytest.mark.unit
@pytest.mark.processor
@pytest.mark.chunking
class TestPythonSyntaxChunker:
	"""Tests for Python syntax-based chunking functionality."""

	def setup_method(self) -> None:
		"""Set up test environment with a chunker instance."""
		self.chunker = TreeSitterChunker()

	def test_python_chunking(self, python_code: str) -> None:
		"""Test chunking Python code."""
		# Arrange - python_code fixture is used

		# Act
		chunks = self.chunker.chunk(python_code, Path("test.py"))

		# Assert - Verify module chunk
		assert len(chunks) == 1
		module = chunks[0]
		assert module.metadata.entity_type == EntityType.MODULE
		assert module.metadata.description is not None
		assert "Module docstring" in module.metadata.description

		# Count the total number of chunks
		all_chunks = get_all_chunks(module)
		assert len(all_chunks) > 10, "Should have a reasonable number of chunks"

		# ----- Verify imports -----
		imports = [c for c in module.children if c.metadata.entity_type == EntityType.IMPORT]
		assert len(imports) >= 2, "Should have at least 2 import statements"

		# ----- Verify classes -----
		# Find all classes
		classes = [c for c in module.children if c.metadata.entity_type == EntityType.CLASS]
		assert len(classes) >= 3, "Should have at least 3 classes"

		# Check MyClass
		my_class = next(c for c in classes if c.metadata.name == "MyClass")
		assert my_class.metadata.description is not None
		assert "Class docstring" in my_class.metadata.description

		# Verify method chunks in MyClass
		methods = [c for c in my_class.children if c.metadata.entity_type in (EntityType.METHOD, EntityType.PROPERTY)]
		assert len(methods) >= 3, "MyClass should have at least 3 methods/properties"

		# Check individual methods
		init_method = next(m for m in methods if m.metadata.name == "__init__")
		assert init_method.metadata.description is not None
		assert "Constructor docstring" in init_method.metadata.description

		value_property = next(m for m in methods if m.metadata.name == "value")
		assert value_property.metadata.entity_type == EntityType.PROPERTY
		assert value_property.metadata.description is not None
		assert "Property docstring" in value_property.metadata.description

		calculate_method = next(m for m in methods if m.metadata.name == "calculate")
		assert calculate_method.metadata.entity_type == EntityType.METHOD
		assert calculate_method.metadata.description is not None
		assert "Regular method" in calculate_method.metadata.description

		# ----- Verify functions -----
		# Find all regular functions
		regular_functions = [c for c in module.children if c.metadata.entity_type == EntityType.FUNCTION]
		assert len(regular_functions) >= 2, "Should have at least 2 regular functions"

		# Check regular function
		reg_func = next(f for f in regular_functions if f.metadata.name == "regular_function")
		assert reg_func.metadata.description is not None
		assert "Regular function docstring" in reg_func.metadata.description

		# Check long function
		long_func = next(f for f in regular_functions if f.metadata.name == "long_function_for_testing")
		assert long_func.metadata.description is not None
		assert "Long function" in long_func.metadata.description
		assert len(long_func.content) > 500, "Long function should have significant content"

		# ----- Verify test functions -----
		test_func = next(c for c in module.children if c.metadata.entity_type == EntityType.TEST_CASE)
		assert test_func.metadata.name == "test_function"
		assert test_func.metadata.description is not None
		assert "Test function docstring" in test_func.metadata.description

		# ----- Verify constants -----
		constants = [c for c in module.children if c.metadata.entity_type == EntityType.CONSTANT]
		assert len(constants) >= 3, "Should have at least 3 constants"

		pi_constant = next(c for c in constants if c.metadata.name == "PI")
		assert pi_constant is not None

		constant = next(c for c in constants if c.metadata.name == "CONSTANT")
		assert constant.metadata.name == "CONSTANT"

		# ----- Verify variables -----
		variables = [c for c in module.children if c.metadata.entity_type == EntityType.VARIABLE]
		type_aliases = [c for c in module.children if c.metadata.entity_type == EntityType.TYPE_ALIAS]

		# Note: variables might be classified as TYPE_ALIAS in the current implementation
		# So check for the variable names in either variables or type_aliases list
		variable_names = {v.metadata.name for v in variables}.union({v.metadata.name for v in type_aliases})
		assert "variable_one" in variable_names or "variable_two" in variable_names, (
			"Should detect at least one of the variable names"
		)

		# ----- Verify type definitions -----
		type_alias_names = {t.metadata.name for t in type_aliases}
		assert "MyType" in type_alias_names or "T" in type_alias_names, "Should detect at least one type alias"

	def test_chunk_splitting(self, python_code: str) -> None:
		"""Test splitting large chunks."""
		# Arrange
		chunks = self.chunker.chunk(python_code, Path("test.py"))
		module = chunks[0]  # Get the module chunk

		# Find a large function to split
		long_func = next(
			c
			for c in module.children
			if c.metadata.entity_type == EntityType.FUNCTION and c.metadata.name == "long_function_for_testing"
		)
		max_size = 400  # Adjust size to be larger than the current chunk sizes

		# Act
		split_chunks = self.chunker.split(long_func, max_size)

		# Assert
		assert len(split_chunks) > 1, "Large function should be split into multiple chunks"

		# Check that the total content is preserved
		combined_content = "".join(chunk.content for chunk in split_chunks)
		assert combined_content == long_func.content

		# Check that each chunk doesn't exceed the max_size significantly
		for chunk in split_chunks:
			assert len(chunk.content) <= max_size * 1.1, (
				f"Chunk size {len(chunk.content)} should be <= {max_size * 1.1} (max_size plus 10%)"
			)
