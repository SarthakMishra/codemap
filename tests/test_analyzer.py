"""Tests for the code analysis functionality."""

from pathlib import Path

import pytest

from codemap.analyzer.dependency_graph import DependencyGraph
from codemap.analyzer.tree_parser import CodeParser


@pytest.fixture
def mock_repo_root(tmp_path: Path) -> Path:
    """Create a temporary repository root for testing."""
    return tmp_path


@pytest.fixture
def sample_python_file(mock_repo_root: Path) -> Path:
    """Create a sample Python file for testing."""
    file_content = '''
"""Module docstring."""
import os
from typing import List, Dict

class TestClass:
    """Test class docstring."""
    def __init__(self):
        self.value = 42

    def test_method(self) -> None:
        """Test method docstring."""
        pass

def test_function(param: str) -> bool:
    """Test function docstring."""
    return True
'''
    file_path = mock_repo_root / "test.py"
    file_path.write_text(file_content)
    return file_path


def test_code_parser_initialization() -> None:
    """Test that CodeParser initializes correctly with required attributes."""
    parser = CodeParser()
    assert parser is not None
    assert hasattr(parser, "parsers")
    assert "py" in parser.parsers
    assert "js" in parser.parsers


def test_should_parse() -> None:
    """Test file extension parsing detection."""
    parser = CodeParser()
    assert parser.should_parse(Path("test.py"))
    assert parser.should_parse(Path("test.js"))
    assert parser.should_parse(Path("test.ts"))
    assert not parser.should_parse(Path("test.txt"))
    assert not parser.should_parse(Path("test.md"))
    assert not parser.should_parse(Path(".gitignore"))


def test_python_file_parsing(sample_python_file: Path) -> None:
    """Test parsing of Python files."""
    parser = CodeParser()
    result = parser.parse_file(sample_python_file)

    assert result is not None
    assert "imports" in result
    assert "os" in result["imports"]
    assert "typing" in result["imports"]

    assert "classes" in result
    assert "TestClass" in result["classes"]

    assert "functions" in result
    assert "test_function" in result["functions"]

    assert "docstring" in result
    assert "Module docstring" in result["docstring"]


def test_dependency_graph_initialization(mock_repo_root: Path) -> None:
    """Test that DependencyGraph initializes correctly."""
    graph = DependencyGraph(mock_repo_root)
    assert graph is not None
    assert hasattr(graph, "graph")
    assert graph.repo_root == mock_repo_root


def test_dependency_graph_build(mock_repo_root: Path) -> None:
    """Test building the dependency graph."""
    # Create test files
    (mock_repo_root / "main.py").write_text("""
import utils
from models import User
""")
    (mock_repo_root / "utils.py").write_text("""
from typing import List
""")
    (mock_repo_root / "models.py").write_text("""
from utils import helper
""")

    graph = DependencyGraph(mock_repo_root)
    parser = CodeParser()

    parsed_files = {}
    for file_path in mock_repo_root.glob("*.py"):
        parsed_files[file_path] = parser.parse_file(file_path)

    graph.build_graph(parsed_files)

    # Check graph structure
    assert len(graph.graph.nodes) >= 3  # At least our 3 files
    assert graph.graph.has_edge(mock_repo_root / "main.py", mock_repo_root / "utils.py")
    assert graph.graph.has_edge(mock_repo_root / "main.py", mock_repo_root / "models.py")
    assert graph.graph.has_edge(mock_repo_root / "models.py", mock_repo_root / "utils.py")


def test_get_important_files(mock_repo_root: Path) -> None:
    """Test identification of important files."""
    graph = DependencyGraph(mock_repo_root)

    # Create mock files with different characteristics
    files = {
        mock_repo_root / "core.py": {
            "imports": ["os", "sys", "typing"],
            "classes": ["CoreClass1", "CoreClass2"],
            "functions": ["main", "helper1", "helper2"],
            "docstring": "Core functionality",
        },
        mock_repo_root / "utils.py": {
            "imports": ["typing"],
            "classes": [],
            "functions": ["utility"],
            "docstring": "Utility functions",
        },
        mock_repo_root / "empty.py": {"imports": [], "classes": [], "functions": [], "docstring": ""},
    }

    graph.build_graph(files)
    important_files = graph.get_important_files(token_limit=1000)

    # Core file should be considered more important
    assert mock_repo_root / "core.py" in important_files
    assert len(important_files) > 0


def test_parser_error_handling(mock_repo_root: Path) -> None:
    """Test error handling in the parser."""
    parser = CodeParser()
    test_file = mock_repo_root / "test.py"
    test_file.write_text("invalid python code )")

    result = parser.parse_file(test_file)

    assert result is not None
    assert "error" in result
    assert len(result["error"]) > 0


def test_circular_dependencies(mock_repo_root: Path) -> None:
    """Test handling of circular dependencies in the graph."""
    # Create files with circular dependencies
    (mock_repo_root / "a.py").write_text("import b")
    (mock_repo_root / "b.py").write_text("import c")
    (mock_repo_root / "c.py").write_text("import a")

    graph = DependencyGraph(mock_repo_root)
    parser = CodeParser()

    parsed_files = {}
    for file_path in mock_repo_root.glob("*.py"):
        parsed_files[file_path] = parser.parse_file(file_path)

    # Should not raise any errors
    graph.build_graph(parsed_files)
    important_files = graph.get_important_files(token_limit=1000)

    # All files should be included as they're all interdependent
    assert len(important_files) == 3
