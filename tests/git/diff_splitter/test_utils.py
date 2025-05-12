"""Tests for diff splitting utility functions."""

import logging
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pygit2.enums import FileStatus

from codemap.git.diff_splitter import utils
from tests.base import GitTestBase

logger = logging.getLogger(__name__)


@pytest.mark.unit
@pytest.mark.git
class TestGetLanguageSpecificPatterns:
	"""Tests for get_language_specific_patterns function."""

	def test_get_python_patterns(self) -> None:
		patterns = utils.get_language_specific_patterns("py")
		assert isinstance(patterns, list)
		assert len(patterns) > 0
		# Check for a known Python pattern
		assert any(p == r"^def\s+\w+" for p in patterns)

	def test_get_javascript_patterns(self) -> None:
		patterns = utils.get_language_specific_patterns("js")
		assert isinstance(patterns, list)
		assert len(patterns) > 0
		# Check for a known JS pattern
		assert any(p == r"^function\s+\w+" for p in patterns)

	def test_get_unknown_language(self) -> None:
		patterns = utils.get_language_specific_patterns("unknown_lang")
		assert patterns == []

	def test_get_empty_language(self) -> None:
		patterns = utils.get_language_specific_patterns("")
		assert patterns == []


@pytest.mark.unit
class TestDetermineCommitType:
	"""Tests for determine_commit_type function."""

	@pytest.mark.parametrize(
		("files", "expected_type"),
		[
			# Based on actual implementation logic:
			(["src/feature.py"], "chore"),  # Not specifically detected
			(["src/fix.py"], "chore"),  # Not specifically detected
			(["tests/test_feature.py"], "test"),
			(["src/module_test.py"], "test"),
			(["src/test_module.py"], "test"),
			(["docs/readme.md"], "docs"),
			(["README.md"], "docs"),
			(["style.css"], "chore"),  # Not specifically detected
			(["refactor.java"], "chore"),  # Not specifically detected
			(["build.gradle"], "chore"),  # Not specifically detected
			(["config.json"], "chore"),
			(["settings.yaml"], "chore"),
			(["options.ini"], "chore"),
			(["setup.py"], "chore"),  # Not specifically detected
			(["Dockerfile"], "chore"),  # Not specifically detected
			(["ci.yml"], "chore"),  # Falls under .yml -> chore
			(["github/workflows/main.yml"], "chore"),  # Falls under .yml -> chore
			(["src/perf.py"], "chore"),  # Not specifically detected
			(["chore.txt"], "chore"),
			(["other.config"], "chore"),  # Default case
			# Multiple files - priority: test > docs > config_chore > default_chore
			(["src/feature.py", "tests/test_feature.py"], "test"),
			(["src/fix.js", "README.md"], "docs"),
			(["docs/guide.md", "settings.toml"], "docs"),  # docs > config_chore
			(["ci.yml", "tests/test_ci.py"], "test"),  # test > config_chore
			(["refactor.go", "app.cfg"], "chore"),  # config_chore > default_chore
			(["chore.config", "style.scss"], "chore"),  # config_chore > default_chore
			(["tests/test_utils.py", "chore_script.sh"], "test"),
			([], "chore"),  # Empty list defaults to chore
		],
	)
	def test_commit_type_determination(self, files: list[str], expected_type: str) -> None:
		assert utils.determine_commit_type(files) == expected_type


@pytest.mark.unit
class TestCreateChunkDescription:
	"""Tests for create_chunk_description function."""

	def test_single_file(self) -> None:
		desc = utils.create_chunk_description("feat", ["src/component/new_feature.py"])
		assert desc == "feat: update src/component/new_feature.py"

	@patch("codemap.git.diff_splitter.utils.os.path.commonpath", return_value="src/component")
	def test_multiple_files_same_dir(self, mock_commonpath: MagicMock) -> None:
		files = ["src/component/file1.py", "src/component/file2.js"]
		desc = utils.create_chunk_description("fix", files)
		assert desc == "fix: update files in src/component"
		mock_commonpath.assert_called_once_with(files)

	@patch("codemap.git.diff_splitter.utils.os.path.commonpath", side_effect=ValueError)
	def test_multiple_files_diff_dirs_or_drives(self, mock_commonpath: MagicMock) -> None:
		files = ["src/component/file1.py", "tests/test_component.py", "docs/feature.md"]
		desc = utils.create_chunk_description("refactor", files)
		assert desc == "refactor: update 3 related files"
		mock_commonpath.assert_called_once_with(files)

	# This case now falls under the "related files" description
	def test_many_files_diff_dirs(self) -> None:
		files = [f"dir{i}/file{i}.py" for i in range(6)]
		desc = utils.create_chunk_description("chore", files)
		assert desc == "chore: update 6 related files"

	@patch("codemap.git.diff_splitter.utils.os.path.commonpath", return_value=".")
	def test_files_in_root(self, mock_commonpath: MagicMock) -> None:
		# If common path is root (.), it should use the 'related files' description
		files = ["README.md", "LICENSE"]
		desc = utils.create_chunk_description("docs", files)
		assert desc == "docs: update 2 related files"
		mock_commonpath.assert_called_once_with(files)

	@patch("codemap.git.diff_splitter.utils.os.path.commonpath", side_effect=ValueError)
	def test_mixed_root_and_subdir(self, mock_commonpath: MagicMock) -> None:
		# Common path will raise ValueError or return root, leading to 'related files'
		files = ["README.md", "src/main.py"]
		desc = utils.create_chunk_description("build", files)
		assert desc == "build: update 2 related files"
		mock_commonpath.assert_called_once_with(files)

	def test_empty_file_list(self) -> None:
		desc = utils.create_chunk_description("test", [])
		# Behavior for empty list isn't explicitly defined, assume '0 related files'
		assert desc == "test: update 0 related files"


@pytest.mark.unit
class TestFileRelationshipUtils:
	"""Tests for utility functions determining file relationships."""

	# === Tests for match_test_file_patterns ===
	@pytest.mark.parametrize(
		("file1", "file2", "expected"),
		[
			# These work because they're already base names without paths
			("module.py", "test_module.py", True),
			("test_module.py", "module.py", True),
			("test.py", "test_test.py", True),  # file named 'test.py'
			# These should be false since the function only works with base filenames
			("src/module.py", "tests/test_module.py", False),
			("tests/test_module.py", "src/module.py", False),
			("src/module.py", "src/utils.py", False),
			("test_module.py", "test_utils.py", False),
			("src/module.test.js", "src/module.js", False),
			("module.spec.ts", "module.ts", False),  # Spec pattern not implemented
			("src/module.ts", "tests/module.test.ts", False),
			("src/component.jsx", "test/component.test.jsx", False),
			("src/test/foo.py", "tests/test_foo.py", False),
		],
	)
	def test_match_test_file_patterns(self, file1: str, file2: str, expected: bool) -> None:
		assert utils.match_test_file_patterns(file1, file2) == expected

	# === Tests for have_similar_names ===
	@pytest.mark.parametrize(
		("file1", "file2", "expected"),
		[
			# Same filename different extensions
			("module.py", "module.ts", True),
			("component.jsx", "component.tsx", True),
			("utils.js", "utils.mjs", True),
			# Different filenames
			("module.py", "utils.py", False),
			# Test patterns
			("test_module.py", "module.py", True),  # Test prefix is compared
			# Path doesn't matter since only base filenames are used
			("module.c", "module.h", True),  # Header/source
			("file_a.txt", "file_b.txt", False),
			# Edge cases
			("module", "module.py", True),  # One without extension
			(".gitignore", ".dockerignore", False),  # Hidden files
			("f.py", "f.ts", False),  # Short names - below threshold
			("a", "b", False),  # Very short names (below threshold)
		],
	)
	def test_have_similar_names(self, file1: str, file2: str, expected: bool) -> None:
		assert utils.have_similar_names(file1, file2) == expected

	# === Tests for has_related_file_pattern ===
	def test_has_related_file_pattern_match(self) -> None:
		# Python doesn't allow backreferences in regex patterns directly
		simple_patterns = [(re.compile(r"file\.c$"), re.compile(r"file\.h$"))]
		assert utils.has_related_file_pattern("file.c", "file.h", simple_patterns) is True
		assert utils.has_related_file_pattern("file.h", "file.c", simple_patterns) is True  # Order invariant

	def test_has_related_file_pattern_no_match(self) -> None:
		# Since Python doesn't allow backreferences in regex patterns directly,
		# we need to test with a pattern that doesn't use backreferences
		simple_patterns = [(re.compile(r"file\.c$"), re.compile(r"file\.h$"))]
		assert utils.has_related_file_pattern("file.c", "other.h", simple_patterns) is False
		assert utils.has_related_file_pattern("file.c", "file.cpp", simple_patterns) is False

	def test_has_related_file_pattern_empty(self) -> None:
		assert utils.has_related_file_pattern("file.c", "file.h", []) is False

	# === Tests for are_files_related ===
	@pytest.mark.parametrize(
		("file1", "file2", "expected", "mock_test_return", "mock_similar_return", "mock_pattern_return", "same_dir"),
		[
			# Test pattern match - different directories
			("src/m.py", "tests/test_m.py", True, True, False, False, False),
			# Similar name match - same directory (will return early)
			("src/util.py", "src/util.ts", True, False, True, False, True),
			# Custom pattern match
			("file.c", "file.h", True, False, False, True, False),
			# No relation
			("main.py", "config.py", False, False, False, False, False),
		],
	)
	@patch("codemap.git.diff_splitter.utils.match_test_file_patterns")
	@patch("codemap.git.diff_splitter.utils.have_similar_names")
	@patch("codemap.git.diff_splitter.utils.has_related_file_pattern")
	def test_are_files_related(
		self,
		mock_has_pattern: MagicMock,
		mock_similar: MagicMock,
		mock_test: MagicMock,
		file1: str,
		file2: str,
		expected: bool,
		mock_test_return: bool,
		mock_similar_return: bool,
		mock_pattern_return: bool,
		same_dir: bool,
	) -> None:
		mock_test.return_value = mock_test_return
		mock_similar.return_value = mock_similar_return
		mock_has_pattern.return_value = mock_pattern_return

		# Extract filenames from paths as the actual function does
		file1_name = file1.rsplit("/", 1)[-1] if "/" in file1 else file1
		file2_name = file2.rsplit("/", 1)[-1] if "/" in file2 else file2

		# Use empty custom patterns for this mock-based test
		assert utils.are_files_related(file1, file2, []) == expected

		# Check mocks were called correctly with the right parameters
		# For files in the same directory, function returns early without calling any of the mocks
		if same_dir:
			mock_test.assert_not_called()
			mock_similar.assert_not_called()
			mock_has_pattern.assert_not_called()
		else:
			mock_test.assert_called_once_with(file1_name, file2_name)
			if not mock_test_return:
				mock_similar.assert_called_once_with(file1_name, file2_name)
				if not mock_similar_return:
					mock_has_pattern.assert_called_once_with(file1, file2, [])
				else:
					mock_has_pattern.assert_not_called()
			else:
				mock_similar.assert_not_called()
				mock_has_pattern.assert_not_called()


@pytest.mark.unit
class TestCalculateSemanticSimilarity:
	"""Tests for calculate_semantic_similarity function."""

	def test_identical_vectors(self) -> None:
		emb1 = [0.1, 0.2, 0.3]
		emb2 = [0.1, 0.2, 0.3]
		similarity = utils.calculate_semantic_similarity(emb1, emb2)
		assert np.isclose(similarity, 1.0)

	def test_orthogonal_vectors(self) -> None:
		emb1 = [1.0, 0.0]
		emb2 = [0.0, 1.0]
		similarity = utils.calculate_semantic_similarity(emb1, emb2)
		assert np.isclose(similarity, 0.0)

	def test_opposite_vectors(self) -> None:
		emb1 = [0.1, 0.2]
		emb2 = [-0.1, -0.2]
		similarity = utils.calculate_semantic_similarity(emb1, emb2)
		# Function clamps result to [0, 1]
		assert np.isclose(similarity, 0.0)

	def test_similar_vectors(self) -> None:
		emb1 = [0.1, 0.2, 0.7]
		emb2 = [0.11, 0.22, 0.68]
		similarity = utils.calculate_semantic_similarity(emb1, emb2)
		# Expect high similarity, close to 1.0 (but not exactly 1.0)
		assert 0.9 < similarity <= 1.0

	def test_zero_vector(self) -> None:
		emb1 = [0.0, 0.0]
		emb2 = [0.1, 0.2]
		similarity = utils.calculate_semantic_similarity(emb1, emb2)
		assert np.isclose(similarity, 0.0)
		similarity = utils.calculate_semantic_similarity(emb2, emb1)
		assert np.isclose(similarity, 0.0)
		similarity = utils.calculate_semantic_similarity(emb1, emb1)
		assert np.isclose(similarity, 0.0)

	def test_empty_vector(self) -> None:
		emb1: list[float] = []
		emb2 = [0.1, 0.2]
		similarity = utils.calculate_semantic_similarity(emb1, emb2)
		assert np.isclose(similarity, 0.0)
		similarity = utils.calculate_semantic_similarity(emb2, emb1)
		assert np.isclose(similarity, 0.0)
		similarity = utils.calculate_semantic_similarity(emb1, emb1)
		assert np.isclose(similarity, 0.0)


@pytest.mark.unit
class TestFileSystemUtils(GitTestBase):
	"""Test cases for file system utility functions."""

	@staticmethod
	def mock_path_exists_check(path_str: str) -> bool:
		"""Determine if a path exists based on its filename."""
		filename = Path(path_str).name
		if filename in ["existing.py", "also_exists.js", "untracked.md"]:
			return True
		if filename == "non_existent.txt":
			return False
		return False

	@patch("codemap.git.diff_splitter.utils.ExtendedGitRepoContext.get_instance")
	def test_get_deleted_tracked_files(self, mock_get_repo_ctx_instance: MagicMock) -> None:
		"""Test retrieving deleted tracked files using git status."""
		mock_repo_ctx = MagicMock()
		mock_get_repo_ctx_instance.return_value = mock_repo_ctx
		mock_repo = MagicMock()
		mock_repo_ctx.repo = mock_repo

		# Mock pygit2.Repository.status() output
		# This dict maps filepath to status flags
		mock_repo.status.return_value = {
			"modified.py": FileStatus.WT_MODIFIED,
			"deleted_staged.py": FileStatus.INDEX_DELETED,
			# Untracked files are typically not in repo.status() output unless explicitly requested by options.
			# The original test implied untracked.txt handling, but current get_deleted_tracked_files only checks WT_DELETED and INDEX_DELETED.
			"deleted_unstaged.py": FileStatus.WT_DELETED,
			"newly_added.js": FileStatus.INDEX_NEW,
		}

		deleted_unstaged, deleted_staged = utils.get_deleted_tracked_files()

		assert deleted_unstaged == {"deleted_unstaged.py"}
		assert deleted_staged == {"deleted_staged.py"}
		mock_get_repo_ctx_instance.assert_called_once()
		mock_repo_ctx.repo.status.assert_called_once()

	@patch("codemap.git.diff_splitter.utils.ExtendedGitRepoContext.get_instance")
	def test_get_deleted_tracked_files_no_deleted(self, mock_get_repo_ctx_instance: MagicMock) -> None:
		"""Test when no files are deleted according to git status."""
		mock_repo_ctx = MagicMock()
		mock_get_repo_ctx_instance.return_value = mock_repo_ctx
		mock_repo = MagicMock()
		mock_repo_ctx.repo = mock_repo

		mock_repo.status.return_value = {
			"modified.py": FileStatus.WT_MODIFIED,
			# "untracked.txt": ... (as above, not typically in status() like this)
		}

		deleted_unstaged, deleted_staged = utils.get_deleted_tracked_files()

		assert deleted_unstaged == set()
		assert deleted_staged == set()
		mock_get_repo_ctx_instance.assert_called_once()
		mock_repo_ctx.repo.status.assert_called_once()

	@patch("codemap.git.diff_splitter.utils.Path.exists", new=lambda _: False)
	@patch("codemap.git.diff_splitter.utils.ExtendedGitRepoContext.get_instance")
	@patch("codemap.git.diff_splitter.utils.get_deleted_tracked_files")
	@patch("codemap.git.diff_splitter.utils.is_test_environment", return_value=False)
	@patch("codemap.git.diff_splitter.utils.get_absolute_path")
	def test_filter_valid_files_normal_env(
		self,
		mock_get_absolute_path: MagicMock,
		_mock_is_test: MagicMock,
		mock_get_deleted: MagicMock,
		mock_get_repo_ctx_instance: MagicMock,
	) -> None:
		"""Test filtering files in a normal (non-test) environment."""
		files_to_check = [
			"existing.py",
			"non_existent.txt",
			"deleted_staged.log",
			"deleted_unstaged.info",
			"untracked.md",
			"also_exists.js",
			"invalid*.py",  # This will be filtered out by initial pattern check
		]
		deleted_unstaged = {"deleted_unstaged.info"}
		deleted_staged = {"deleted_staged.log"}
		# Files that ExtendedGitRepoContext.tracked_files should report
		# This is derived from the original test's `tracked_files_ls`
		tracked_files_set = {
			"existing.py",
			"deleted_staged.log",  # Still tracked even if staged for deletion
			"deleted_unstaged.info",  # Still tracked even if deleted in worktree
			"also_exists.js",
			"other_tracked.md",
		}
		repo_root = Path("/mock/repo")

		mock_get_deleted.return_value = (deleted_unstaged, deleted_staged)

		# Setup mock for ExtendedGitRepoContext instance returned by get_instance()
		mock_configured_repo_ctx = MagicMock()
		mock_get_repo_ctx_instance.return_value = mock_configured_repo_ctx
		# .tracked_files should be a dict-like object (e.g. output of repo.index)
		# For simplicity, a dict mapping path to dummy data is fine for .keys() usage.
		mock_configured_repo_ctx.tracked_files = {filename: {} for filename in tracked_files_set}

		# Mock get_absolute_path to check the path in our mock_path_exists_check
		mock_get_absolute_path.side_effect = lambda file, root: str(root / file)

		# Replace the existence check by patching Path.exists to a simple lambda
		# that always returns False, and then intercepting the exact path check in
		# utils.filter_valid_files to use our custom logic
		with patch("codemap.git.diff_splitter.utils.Path") as mock_path_cls:
			# Setup a custom Path constructor that returns a Path mock
			def path_side_effect(path_str):
				mock_path = MagicMock()
				# Make exists() return based on our mock_path_exists_check
				mock_path.exists.return_value = TestFileSystemUtils.mock_path_exists_check(path_str)
				# Ensure name is available for debug logging
				mock_path.name = Path(path_str).name
				return mock_path

			mock_path_cls.side_effect = path_side_effect

			valid_files, invalid_files_is_now_empty = utils.filter_valid_files(files_to_check, repo_root)

		# Expected valid files:
		# - existing.py (in tracked_files)
		# - deleted_staged.log (in deleted_staged AND tracked_files)
		# - deleted_unstaged.info (in deleted_unstaged AND tracked_files)
		# - also_exists.js (in tracked_files)
		# - untracked.md (not tracked, not deleted, but mock_exists returns True for it)
		assert set(valid_files) == {
			"existing.py",
			"deleted_staged.log",
			"deleted_unstaged.info",
			"also_exists.js",
			"untracked.md",
		}
		assert invalid_files_is_now_empty == []  # Second part of tuple is always empty now
		mock_get_deleted.assert_called_once()
		mock_get_repo_ctx_instance.assert_called_once()

	@patch("codemap.git.diff_splitter.utils.Path.exists")
	@patch("codemap.git.diff_splitter.utils.is_test_environment", return_value=True)
	def test_filter_valid_files_test_env(self, _mock_is_test: MagicMock, mock_path_exists: MagicMock) -> None:
		"""Test filtering files in a test environment (skips git/fs checks, but not pattern/size)."""
		mock_path_exists.return_value = False  # Assume files don't exist for simplicity
		files_to_check = ["existing.py", "non_existent.txt", "deleted.log", "invalid*.py"]
		repo_root = Path("/mock/repo")  # Mock repository root path

		# In test env, git/fs existence checks are skipped, but pattern checks still run.
		valid_files, large_files = utils.filter_valid_files(files_to_check, repo_root, is_test_environment=True)

		assert valid_files == ["existing.py", "non_existent.txt", "deleted.log"]
		assert large_files == []  # No large files simulated
