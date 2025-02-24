"""Tests for the ERD generator functionality."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pytest

from codemap.analyzer.tree_parser import CodeParser
from codemap.generators.erd_generator import ERDGenerator


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a copy of the sample repository for testing."""
    fixtures_path = Path(__file__).parent / "fixtures" / "sample_repo"
    repo_path = tmp_path / "sample_repo"
    shutil.copytree(fixtures_path, repo_path)
    return repo_path


@pytest.fixture
def erd_generator() -> ERDGenerator:
    """Create an ERD generator instance."""
    return ERDGenerator()


def test_generate_success(erd_generator: ERDGenerator, sample_repo: Path) -> None:
    """Test successful ERD generation."""
    # Create test files with class definitions
    (sample_repo / "models.py").write_text("""
class BaseModel:
    created_at: str
    updated_at: str | None

class User(BaseModel):
    name: str
    email: str
    orders: list[Order]

class Order(BaseModel):
    order_id: str
    user: User
    total: float
    items: list[OrderItem]

class OrderItem(BaseModel):
    order: Order
    product: Product
    quantity: int
    price: float

class Product(BaseModel):
    name: str
    price: float
    description: str | None
""")

    # Initialize parser and parse files
    parser = CodeParser()
    parsed_files = {
        sample_repo / "models.py": parser.parse_file(sample_repo / "models.py"),
    }

    # Generate ERD
    output_file = erd_generator.generate(parsed_files)
    assert output_file.exists()
    content = output_file.read_text()

    # Check that all classes are included
    assert "BaseModel" in content
    assert "User" in content
    assert "Order" in content
    assert "OrderItem" in content
    assert "Product" in content

    # Check that relationships are included
    assert "User --|> BaseModel" in content
    assert "Order --|> BaseModel" in content
    assert "OrderItem --|> BaseModel" in content
    assert "Product --|> BaseModel" in content


def test_generate_with_custom_output(erd_generator: ERDGenerator, sample_repo: Path) -> None:
    """Test ERD generation with a custom output path."""
    # Create test files
    (sample_repo / "models.py").write_text("""
class User:
    name: str
    email: str

class Profile:
    user: User
    bio: str
""")

    # Initialize parser and parse files
    parser = CodeParser()
    parsed_files = {
        sample_repo / "models.py": parser.parse_file(sample_repo / "models.py"),
    }

    # Generate ERD with custom output path
    output_file = sample_repo / "custom" / "erd.md"
    result = erd_generator.generate(parsed_files, output_file)
    assert result == output_file
    assert result.exists()

    # Check content
    content = result.read_text()
    assert "User" in content
    assert "Profile" in content
    assert "Profile --o User" in content


def test_generate_with_invalid_class_names(erd_generator: ERDGenerator, sample_repo: Path, caplog: Any) -> None:
    """Test ERD generation with invalid class names."""
    # Create a file with invalid class names
    (sample_repo / "invalid.py").write_text("""
class ValidClass:
    pass

class _InvalidClass:  # Private class
    pass

class 123InvalidClass:  # Invalid identifier
    pass
""")

    # Parse the file
    parser = CodeParser()
    parsed_files = {
        sample_repo / "invalid.py": parser.parse_file(sample_repo / "invalid.py"),
    }

    # Generate ERD
    output_file = erd_generator.generate(parsed_files)
    assert output_file.exists()
    content = output_file.read_text()

    # Check that only valid class is included
    assert "ValidClass" in content
    assert "_InvalidClass" not in content
    assert "123InvalidClass" not in content

    # Check for warning about invalid class names
    assert "Skipping invalid class name" in caplog.text
