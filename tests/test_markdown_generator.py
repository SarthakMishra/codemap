"""Tests for the markdown documentation generator."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from codemap.generators.markdown_generator import MarkdownGenerator


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a copy of the sample repository for testing."""
    fixtures_path = Path(__file__).parent / "fixtures" / "sample_repo"
    repo_path = tmp_path / "sample_repo"
    shutil.copytree(fixtures_path, repo_path)
    return repo_path


@pytest.fixture
def generator(tmp_path: Path) -> MarkdownGenerator:
    """Create a markdown generator instance."""
    return MarkdownGenerator(tmp_path, {})


def test_generate_documentation_with_files(generator: MarkdownGenerator, sample_repo: Path) -> None:
    """Test documentation generation with real files."""
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

    # Create parsed file data
    parsed_files = {
        sample_repo / "models.py": {
            "docstring": "Sample models for testing.",
            "classes": ["BaseModel", "User", "Order", "OrderItem", "Product"],
            "imports": ["datetime"],
            "references": [],
            "bases": {
                "User": ["BaseModel"],
                "Order": ["BaseModel"],
                "OrderItem": ["BaseModel"],
                "Product": ["BaseModel"],
            },
            "attributes": {
                "BaseModel": {"created_at": "datetime", "updated_at": "datetime | None"},
                "User": {"name": "str", "email": "str", "orders": "list[Order]"},
                "Order": {"order_id": "str", "user": "User", "total": "float", "items": "list[OrderItem]"},
                "OrderItem": {"order": "Order", "product": "Product", "quantity": "int", "price": "float"},
                "Product": {"name": "str", "price": "float", "description": "str | None"},
            },
        },
    }

    doc = generator.generate_documentation(parsed_files)
    assert doc
    assert "# Project Documentation" in doc
    assert "## File Structure" in doc
    assert "models.py" in doc
    assert "BaseModel" in doc
    assert "User" in doc
    assert "Order" in doc
    assert "OrderItem" in doc
    assert "Product" in doc


def test_file_sorting(generator: MarkdownGenerator, sample_repo: Path) -> None:
    """Test that files are properly sorted in the documentation."""
    # Create test files
    (sample_repo / "z.py").write_text("class Z: pass")
    (sample_repo / "a.py").write_text("class A: pass")
    (sample_repo / "m.py").write_text("class M: pass")

    # Create parsed file data
    parsed_files = {
        sample_repo / "z.py": {
            "docstring": "Z class",
            "classes": ["Z"],
            "imports": [],
            "references": [],
            "bases": {},
            "attributes": {},
        },
        sample_repo / "a.py": {
            "docstring": "A class",
            "classes": ["A"],
            "imports": [],
            "references": [],
            "bases": {},
            "attributes": {},
        },
        sample_repo / "m.py": {
            "docstring": "M class",
            "classes": ["M"],
            "imports": [],
            "references": [],
            "bases": {},
            "attributes": {},
        },
    }

    doc = generator.generate_documentation(parsed_files)
    assert doc

    # Check that files are sorted alphabetically
    z_pos = doc.find("z.py")
    a_pos = doc.find("a.py")
    m_pos = doc.find("m.py")
    assert a_pos < m_pos < z_pos


def test_markdown_escaping(tmp_path: Path) -> None:
    """Test that markdown special characters are properly escaped."""
    # Create a test file with content containing markdown special characters
    test_file = tmp_path / "special.py"
    test_file.write_text('"""A file with *special* _characters_."""\n\nclass Test:\n    pass')

    # Initialize generator
    generator = MarkdownGenerator(tmp_path, {})

    # Create parsed file data with special characters
    parsed_files = {
        test_file: {
            "docstring": "A file with *special* _characters_",
            "classes": ["Test"],
            "imports": [],
            "references": [],
            "bases": {},
            "attributes": {},
        },
    }

    # Generate documentation
    docs = generator.generate_documentation(parsed_files)
    assert docs

    # Check that markdown characters are escaped
    assert r"\*special\*" in docs
    assert r"\_characters\_" in docs
