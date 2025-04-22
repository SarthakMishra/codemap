"""Tests for the markdown documentation generator."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codemap.analyzer.tree_parser import CodeParser
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
            "language": "python",
        },
    }

    doc = generator.generate_documentation(parsed_files)
    assert doc
    assert "## Overview" in doc
    assert "## 📁 Project Structure" in doc
    assert "## 📄 Files" in doc
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
            "language": "python",
        },
        sample_repo / "a.py": {
            "classes": ["A"],
            "imports": [],
            "references": [],
            "content": "class A: pass",
            "language": "python",
        },
        sample_repo / "m.py": {
            "classes": ["M"],
            "imports": [],
            "references": [],
            "content": "class M: pass",
            "language": "python",
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


def test_different_file_extensions(generator: MarkdownGenerator, sample_repo: Path) -> None:
    """Test that documentation generation works with different file extensions."""
    # Create files with different extensions
    (sample_repo / "script.py").write_text("def main(): pass")
    (sample_repo / "style.css").write_text("body { color: #333; }")
    (sample_repo / "config.json").write_text('{"key": "value"}')
    (sample_repo / "data.yaml").write_text("key: value")
    (sample_repo / "doc.md").write_text("# Heading\n\nContent")
    (sample_repo / "script.js").write_text("function hello() { return 'world'; }")
    (sample_repo / "index.html").write_text("<html><body>Hello</body></html>")

    # Create mock parser for these files
    with patch("codemap.generators.markdown_generator.CodeParser") as mock_parser_class:
        file_filter_mock = Mock()
        file_filter_mock.should_parse.return_value = True
        file_filter_mock.gitignore_patterns = []
        file_filter_mock.matches_pattern.return_value = False

        parser = Mock()
        parser.file_filter = file_filter_mock
        mock_parser_class.return_value = parser

        # Create parsed file data with different languages
        parsed_files = {
            sample_repo / "script.py": {
                "content": "def main(): pass",
                "language": "python",
            },
            sample_repo / "style.css": {
                "content": "body { color: #333; }",
                "language": "css",
            },
            sample_repo / "config.json": {
                "content": '{"key": "value"}',
                "language": "json",
            },
            sample_repo / "data.yaml": {
                "content": "key: value",
                "language": "yaml",
            },
            sample_repo / "doc.md": {
                "content": "# Heading\n\nContent",
                "language": "markdown",
            },
            sample_repo / "script.js": {
                "content": "function hello() { return 'world'; }",
                "language": "javascript",
            },
            sample_repo / "index.html": {
                "content": "<html><body>Hello</body></html>",
                "language": "html",
            },
        }

        # Generate documentation
        doc = generator.generate_documentation(parsed_files)

        # Verify each file type is included and has correct language
        assert "script.py" in doc
        assert "Python" in doc

        assert "style.css" in doc
        assert "Css" in doc

        assert "config.json" in doc
        assert "Json" in doc

        assert "data.yaml" in doc
        assert "Yaml" in doc

        assert "doc.md" in doc
        assert "Markdown" in doc

        assert "script.js" in doc
        assert "Javascript" in doc

        assert "index.html" in doc
        assert "Html" in doc


def test_tree_with_specific_path(generator: MarkdownGenerator, sample_repo: Path) -> None:
    """Test generating a tree for a specific subdirectory."""
    # Create nested directory structure
    subdir = sample_repo / "src" / "module"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "file.py").write_text("# Test file")
    (sample_repo / "another_file.py").write_text("# Another file")

    # Generate tree for the subdir only
    with patch("codemap.generators.markdown_generator.CodeParser") as mock_parser_class:
        file_filter_mock = Mock()
        file_filter_mock.should_parse.return_value = True
        file_filter_mock.gitignore_patterns = []

        parser = Mock()
        parser.file_filter = file_filter_mock
        parser.should_parse.return_value = True
        mock_parser_class.return_value = parser

        # Generate a tree for the specific subdirectory
        tree = generator.generate_tree(subdir)

        # Should contain the file in the subdirectory
        assert "file.py" in tree
        # Should NOT contain files outside the specified path
        assert "another_file.py" not in tree


@pytest.mark.usefixtures("generator")
def test_tree_with_parsed_files_highlighting(tmp_path: Path) -> None:
    """Test tree generation with parsed_files parameter to highlight specific files."""
    # Create a clean test directory (not using sample_repo which has other files)
    test_dir = tmp_path / "test_tree"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create files
    (test_dir / "src" / "main.py").parent.mkdir(exist_ok=True, parents=True)
    (test_dir / "src" / "main.py").write_text("# Main file")
    (test_dir / "src" / "utils.py").write_text("# Utils file")
    (test_dir / "src" / "ignored.py").write_text("# Should be ignored")

    # Create a set of parsed files (only main.py and utils.py)
    parsed_files = {
        test_dir / "src" / "main.py",
        test_dir / "src" / "utils.py",
    }

    # Create a dedicated generator for this test to avoid interference
    test_generator = MarkdownGenerator(test_dir, {})

    # Generate tree with parsed_files parameter
    with patch("codemap.generators.markdown_generator.CodeParser") as mock_parser_class:
        file_filter_mock = Mock()
        file_filter_mock.should_parse.return_value = True
        file_filter_mock.gitignore_patterns = []
        file_filter_mock.matches_pattern.return_value = False

        parser = Mock(spec=CodeParser)
        parser.file_filter = file_filter_mock
        mock_parser_class.return_value = parser

        # First verify that without parsed_files, all files are included
        tree_all = test_generator.generate_tree(test_dir)
        assert "main.py" in tree_all
        assert "utils.py" in tree_all
        assert "ignored.py" in tree_all

        # Then test with parsed_files to filter
        tree_filtered = test_generator.generate_tree(test_dir, parsed_files=parsed_files)

        # In the filtered tree, only the parsed files should appear
        assert "main.py" in tree_filtered
        assert "utils.py" in tree_filtered
        assert "ignored.py" not in tree_filtered


def test_custom_directory_selection(tmp_path: Path) -> None:
    """Test that we can generate documentation for a specific subdirectory."""
    # Create a test directory structure
    root_dir = tmp_path / "project"
    root_dir.mkdir()

    # Create a subdirectory to be included in the documentation
    subdir = root_dir / "subdir"
    subdir.mkdir()

    # Create a file in the subdirectory
    test_file = subdir / "test.py"
    test_file.write_text("# Test file")

    # Create a file in the root directory (outside the target)
    root_file = root_dir / "root.py"
    root_file.write_text("# Root file")

    # Create a generator for the subdirectory only
    generator = MarkdownGenerator(subdir, {})

    # Create mock parsed files for the subdirectory only
    parsed_files = {
        test_file: {
            "content": "# Test file",
            "language": "python",
        },
    }

    # Generate documentation
    doc = generator.generate_documentation(parsed_files)

    # The documentation should include the file from the subdirectory
    assert "test.py" in doc

    # The documentation should NOT include files outside the target directory
    assert "root.py" not in doc
