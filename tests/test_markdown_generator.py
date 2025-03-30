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
            "classes": ["BaseModel", "User", "Order", "OrderItem", "Product"],
            "imports": ["datetime"],
            "references": [],
            "content": """
class BaseModel:
    created_at: str
    updated_at: str | None

class User(BaseModel):
    name: str
    email: str
    orders: list[Order]
""",
        },
    }

    doc = generator.generate_documentation(parsed_files)
    assert doc
    assert "# Code Map" in doc
    assert "## Overview" in doc
    assert "## File Details" in doc
    assert "models.py" in doc
    assert "BaseModel" in doc
    assert "User" in doc


def test_file_sorting(generator: MarkdownGenerator, sample_repo: Path) -> None:
    """Test that files are properly sorted in the documentation."""
    # Create test files
    (sample_repo / "z.py").write_text("class Z: pass")
    (sample_repo / "a.py").write_text("class A: pass")
    (sample_repo / "m.py").write_text("class M: pass")

    # Create parsed file data
    parsed_files = {
        sample_repo / "z.py": {
            "classes": ["Z"],
            "imports": [],
            "references": [],
            "content": "class Z: pass",
        },
        sample_repo / "a.py": {
            "classes": ["A"],
            "imports": [],
            "references": [],
            "content": "class A: pass",
        },
        sample_repo / "m.py": {
            "classes": ["M"],
            "imports": [],
            "references": [],
            "content": "class M: pass",
        },
    }

    doc = generator.generate_documentation(parsed_files)
    assert doc

    # Check that files are sorted alphabetically
    z_pos = doc.find("z.py")
    a_pos = doc.find("a.py")
    m_pos = doc.find("m.py")
    assert a_pos < m_pos < z_pos


def test_tree_generation(generator: MarkdownGenerator, sample_repo: Path) -> None:
    """Test the tree generation functionality."""
    # Create some files for the tree generation
    (sample_repo / "module1" / "file1.py").parent.mkdir(exist_ok=True, parents=True)
    (sample_repo / "module1" / "file1.py").write_text("# Test file")
    (sample_repo / "module2" / "file2.py").parent.mkdir(exist_ok=True, parents=True)
    (sample_repo / "module2" / "file2.py").write_text("# Test file 2")

    # Generate a tree
    tree = generator.generate_tree(sample_repo)
    assert tree

    # The tree should contain the directories and files
    assert "module1" in tree
    assert "module2" in tree
    assert "file1.py" in tree
    assert "file2.py" in tree
