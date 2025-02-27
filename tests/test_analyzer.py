"""Tests for the code analysis functionality."""

from __future__ import annotations

import shutil
from pathlib import Path

import networkx as nx
import pytest

from codemap.analyzer.dependency_graph import DependencyGraph
from codemap.analyzer.tree_parser import ERR_PARSER_INIT, CodeParser


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a copy of the sample repository for testing."""
    fixtures_path = Path(__file__).parent / "fixtures" / "sample_repo"
    repo_path = tmp_path / "sample_repo"
    shutil.copytree(fixtures_path, repo_path)
    return repo_path


def test_code_parser_initialization() -> None:
    """Test that CodeParser initializes correctly with required attributes."""
    parser = CodeParser()
    assert parser.parsers
    assert parser.config == {}
    assert parser.gitignore_patterns == []
    assert "python" in parser.parsers


def test_code_parser_initialization_with_config() -> None:
    """Test that CodeParser initializes correctly with custom config."""
    config = {"analysis": {"languages": ["python", "javascript", "typescript"]}}
    parser = CodeParser(config)
    assert "python" in parser.parsers
    assert "javascript" in parser.parsers
    assert "typescript" in parser.parsers


def test_code_parser_initialization_with_invalid_language() -> None:
    """Test that CodeParser handles invalid language gracefully."""
    config = {"analysis": {"languages": ["python", "invalid_lang"]}}
    parser = CodeParser(config)
    assert "python" in parser.parsers
    assert "invalid_lang" not in parser.parsers


def test_code_parser_initialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that CodeParser handles initialization failures correctly."""

    def mock_get_parser(lang: str) -> None:
        raise RuntimeError(f"Failed to load language {lang}")

    monkeypatch.setattr("codemap.analyzer.tree_parser.get_parser", mock_get_parser)
    with pytest.raises(RuntimeError, match=ERR_PARSER_INIT.format(lang="any")):
        CodeParser()


def test_python_file_parsing(sample_repo: Path) -> None:
    """Test parsing a Python file."""
    # Create a test file
    test_file = sample_repo / "test.py"
    test_file.write_text("""
class TestClass:
    def __init__(self):
        self.x = 1
""")

    parser = CodeParser()
    result = parser.parse_file(test_file)

    assert result["classes"] == ["TestClass"]
    assert not result["imports"]
    assert not result["bases"]


def test_supported_languages() -> None:
    """Test that all supported languages can be initialized."""
    config = {"analysis": {"languages": ["python", "javascript", "typescript", "go", "ruby", "java"]}}
    parser = CodeParser(config)

    # Check that at least Python parser is available
    assert "python" in parser.parsers

    # Log which parsers were successfully initialized
    for lang in config["analysis"]["languages"]:
        if lang in parser.parsers:
            assert parser.parsers[lang] is not None


def test_dependency_graph_build(sample_repo: Path) -> None:
    """Test building the dependency graph."""
    # Create files with dependencies
    (sample_repo / "a.py").write_text("from b import B")
    (sample_repo / "b.py").write_text("from c import C")
    (sample_repo / "c.py").write_text("from a import A")

    graph = DependencyGraph(sample_repo)
    parser = CodeParser()

    # Parse all files
    parsed_files = {}
    for file_path in sample_repo.rglob("*.py"):
        parsed_files[file_path] = parser.parse_file(file_path)

    graph.build_graph(parsed_files)
    assert isinstance(graph.graph, nx.DiGraph)
    assert len(graph.graph.nodes) > 0


def test_get_important_files(sample_repo: Path) -> None:
    """Test identification of important files."""
    graph = DependencyGraph(sample_repo)
    parser = CodeParser()

    # Parse both files
    models_file = sample_repo / "models.py"
    services_file = sample_repo / "services.py"

    parsed_files = {
        models_file: parser.parse_file(models_file),
        services_file: parser.parse_file(services_file),
    }

    graph.build_graph(parsed_files)
    important_files = graph.get_important_files(token_limit=1000)
    assert isinstance(important_files, list)
    assert len(important_files) > 0
    # models.py should be considered important as it's a dependency
    assert any(str(models_file) in str(file) for file in important_files)


def test_circular_dependencies(sample_repo: Path) -> None:
    """Test handling of circular dependencies in the graph."""
    # Create files with circular dependencies
    (sample_repo / "a.py").write_text("from b import B")
    (sample_repo / "b.py").write_text("from c import C")
    (sample_repo / "c.py").write_text("from a import A")

    graph = DependencyGraph(sample_repo)
    parser = CodeParser()

    # Parse all files
    parsed_files = {}
    for file_path in sample_repo.rglob("*.py"):
        parsed_files[file_path] = parser.parse_file(file_path)

    graph.build_graph(parsed_files)
    important_files = graph.get_important_files(token_limit=1000)
    assert isinstance(important_files, list)
    assert len(important_files) > 0


def test_parse_file_with_invalid_extension(sample_repo: Path) -> None:
    """Test parsing a file with an unsupported extension."""
    # Create a file with unsupported extension
    test_file = sample_repo / "test.txt"
    test_file.write_text("This is a test file")

    parser = CodeParser()
    result = parser.parse_file(test_file)

    # Should return empty symbols dict for unsupported file
    assert result == {
        "imports": [],
        "classes": [],
        "references": [],
        "bases": {},
        "attributes": {},
        "content": "",
    }


def test_parse_file_with_invalid_content(sample_repo: Path) -> None:
    """Test parsing a file with invalid Python content."""
    # Create a file with invalid Python syntax
    test_file = sample_repo / "invalid.py"
    test_file.write_text("this is not valid python code")

    parser = CodeParser()
    result = parser.parse_file(test_file)

    # Should still return a valid symbols dict even for invalid content
    assert isinstance(result, dict)
    assert "imports" in result
    assert "classes" in result
    assert "references" in result
    assert "bases" in result
    assert "attributes" in result
