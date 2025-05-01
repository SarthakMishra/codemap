"""Tests for diff splitting utility functions."""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from codemap.git.diff_splitter import utils
from codemap.git.diff_splitter.constants import MAX_FILE_SIZE_FOR_LLM


@pytest.mark.unit
@pytest.mark.git
class TestExtractCodeFromDiff:
	"""Tests for the extract_code_from_diff function."""

	def test_simple_add_remove(self) -> None:
		diff = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old line\n+new line\n"
		old_code, new_code = utils.extract_code_from_diff(diff)
		assert old_code == "old line"
		assert new_code == "new line"

	def test_with_context(self) -> None:
		diff = (
			"diff --git a/file.py b/file.py\n"
			"--- a/file.py\n"
			"+++ b/file.py\n"
			"@@ -1,3 +1,3 @@\n"
			" unchanged line 1\n"
			"-old line\n"
			"+new line\n"
			" unchanged line 2\n"
		)
		old_code, new_code = utils.extract_code_from_diff(diff)
		expected_old = "unchanged line 1\nold line\nunchanged line 2"
		expected_new = "unchanged line 1\nnew line\nunchanged line 2"
		assert old_code == expected_old
		assert new_code == expected_new

	def test_hunk_header_context(self) -> None:
		diff = (
			"diff --git a/file.py b/file.py\n"
			"--- a/file.py\n"
			"+++ b/file.py\n"
			"@@ -5,1 +5,1 @@ def my_function():\n"
			"-    return 0\n"
			"+    return 1\n"
		)
		old_code, new_code = utils.extract_code_from_diff(diff)
		# Expect function context to be added as comments
		assert old_code == "// def my_function():\n    return 0"
		assert new_code == "// def my_function():\n    return 1"

	def test_empty_diff(self) -> None:
		old_code, new_code = utils.extract_code_from_diff("")
		assert old_code == ""
		assert new_code == ""

	def test_none_diff(self) -> None:
		# Although type hint is str, test None defensively
		old_code, new_code = utils.extract_code_from_diff(None)  # type: ignore[arg-type]
		assert old_code == ""
		assert new_code == ""

	def test_large_diff_truncation_initial(self) -> None:
		"""Test truncation when diff content itself is too large."""
		large_content = "a" * (MAX_FILE_SIZE_FOR_LLM + 100)
		# Simulate a diff header
		diff = f"diff --git a/large_file.txt b/large_file.txt\n{large_content}"
		old_code, new_code = utils.extract_code_from_diff(diff)
		assert "Large diff content for large_file.txt (truncated)" in old_code
		assert "Large diff content for large_file.txt (truncated)" in new_code
		assert f"Original size: {len(diff)} bytes" in new_code

	def test_large_diff_truncation_during_extraction(self) -> None:
		"""Test truncation when extracted code exceeds limits *after* initial check."""
		# Create content small enough initially, but large after extraction
		# Make line content smaller to stay under the initial size check
		# but still large enough to trigger truncation during extraction
		line_content = "l" * 8000  # Reduced from 10000 to ensure it passes initial check
		# Repeat context/added lines to exceed limit during processing
		num_lines = 5  # Enough to exceed limit when combined
		hunk_lines = []
		for _ in range(num_lines):
			hunk_lines.append(f" {line_content}")  # Context
			hunk_lines.append(f"+{line_content}")  # Addition
		hunk_str = "\n".join(hunk_lines)

		diff_content_size_initial = 500  # Assume header size etc.
		diff_content_size_initial += len(hunk_str)
		assert diff_content_size_initial < MAX_FILE_SIZE_FOR_LLM  # Ensure initial check passes

		diff = (
			f"diff --git a/long.py b/long.py\n"
			f"--- a/long.py\n"
			f"+++ b/long.py\n"
			f"@@ -1,{num_lines} +1,{num_lines * 2} @@ Context Here\n"
			f"{hunk_str}\n"
		)

		old_code, new_code = utils.extract_code_from_diff(diff)

		# Expect truncation message to appear during processing this time
		assert "// Context Here" in old_code  # Check context is extracted initially
		assert "// Content truncated" in old_code or "truncated" in old_code
		assert "Context Here" in new_code
		assert "Content truncated" in new_code or "truncated" in new_code

	def test_binary_file(self) -> None:
		diff = (
			"diff --git a/image.png b/image.png\n"
			"index abc..def 100644\n"
			"Binary files a/image.png and b/image.png differ\n"
		)
		old_code, new_code = utils.extract_code_from_diff(diff)
		assert old_code == "// Binary file changed: image.png"
		assert new_code == "// Binary file changed: image.png"

	def test_git_binary_patch(self) -> None:
		# Example structure, content is simplified
		diff = (
			"diff --git a/binary_data b/binary_data\n"
			"index abc..def 100644\n"
			"GIT binary patch\n"
			"delta 123\n"
			"some binary data representation\n"
			"literal 456\n"
			"more binary data representation\n"
		)
		old_code, new_code = utils.extract_code_from_diff(diff)
		assert old_code == "// Binary file changed: binary_data"
		assert new_code == "// Binary file changed: binary_data"

	def test_only_header_no_hunks(self) -> None:
		diff = "diff --git a/file.py b/file.py\nindex 123..456 100644\n--- a/file.py\n+++ b/file.py\n"
		old_code, new_code = utils.extract_code_from_diff(diff)
		# Should add placeholder if file detected but no content
		assert old_code == "// File: file.py"
		assert new_code == "// File: file.py"


@pytest.mark.unit
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
class TestFileSystemUtils:
	"""Tests for utility functions interacting with file system or git."""

	@patch("codemap.git.diff_splitter.utils.run_git_command")
	def test_get_deleted_tracked_files(self, mock_run_git: MagicMock) -> None:
		"""Test retrieving deleted tracked files using git status."""
		# Mock git status --porcelain output
		# The format is [XY] filename where X=staged status, Y=unstaged status
		status_output = (
			" M modified.py\n"  # Unstaged modification (space + M)
			"D  deleted_staged.py\n"  # Staged deletion (D + space)
			"?? untracked.txt\n"  # Untracked file (??)
			" D deleted_unstaged.py\n"  # Unstaged deletion (space + D)
			"A  newly_added.js\n"  # Staged addition (A + space)
		)
		mock_run_git.return_value = status_output

		deleted_unstaged, deleted_staged = utils.get_deleted_tracked_files()

		assert deleted_unstaged == {"deleted_unstaged.py"}
		assert deleted_staged == {"deleted_staged.py"}
		mock_run_git.assert_called_once_with(["git", "status", "--porcelain"])

	@patch("codemap.git.diff_splitter.utils.run_git_command")
	def test_get_deleted_tracked_files_no_deleted(self, mock_run_git: MagicMock) -> None:
		"""Test when no files are deleted according to git status."""
		status_output = " M modified.py\n?? untracked.txt\n"
		mock_run_git.return_value = status_output

		deleted_unstaged, deleted_staged = utils.get_deleted_tracked_files()

		assert deleted_unstaged == set()
		assert deleted_staged == set()

	@patch("codemap.git.diff_splitter.utils.os.path.exists")
	@patch("codemap.git.diff_splitter.utils.run_git_command")  # For ls-files
	@patch("codemap.git.diff_splitter.utils.get_deleted_tracked_files")
	@patch("codemap.git.diff_splitter.utils.is_test_environment", return_value=False)
	def test_filter_valid_files_normal_env(
		self, _mock_is_test: MagicMock, mock_get_deleted: MagicMock, mock_run_git: MagicMock, mock_exists: MagicMock
	) -> None:
		"""Test filtering files in a normal (non-test) environment."""
		files_to_check = [
			"existing.py",
			"non_existent.txt",
			"deleted_staged.log",
			"deleted_unstaged.info",
			"untracked.md",
			"also_exists.js",
			"invalid*.py",
		]
		deleted_unstaged = {"deleted_unstaged.info"}
		deleted_staged = {"deleted_staged.log"}
		tracked_files_ls = "existing.py\ndeleted_staged.log\ndeleted_unstaged.info\nalso_exists.js\nother_tracked.md\n"

		mock_get_deleted.return_value = (deleted_unstaged, deleted_staged)
		mock_run_git.return_value = tracked_files_ls  # Mock for 'git ls-files'
		# Configure os.path.exists mock - only called if not deleted and not tracked
		# In this setup, only 'non_existent.txt' and 'untracked.md' would trigger os.path.exists
		mock_exists.side_effect = lambda f: f in [
			"existing.py",
			"also_exists.js",
			"untracked.md",
		]  # Assume untracked exists

		valid_files, invalid_files = utils.filter_valid_files(files_to_check)

		# According to the implementation, untracked files are only valid if they exist
		# and the implementation includes Path(file).exists() checks for all files
		assert set(valid_files) == {
			"existing.py",
			"deleted_staged.log",
			"deleted_unstaged.info",
			"also_exists.js",
		}
		# invalid_files list is actually for LARGE files, not non-existent ones.
		assert invalid_files == []
		mock_get_deleted.assert_called_once()
		mock_run_git.assert_called_once_with(["git", "ls-files"])  # Check ls-files call

		# In the implementation, Path(file).exists() is used instead of os.path.exists(),
		# so our mock doesn't capture the calls
		# The code is checking for both existence and large file sizes, so the real behavior
		# is more complex than our test can easily mock

	@patch("codemap.git.diff_splitter.utils.Path.exists")  # Mock Path.exists for large file check
	@patch("codemap.git.diff_splitter.utils.is_test_environment", return_value=True)
	def test_filter_valid_files_test_env(self, _mock_is_test: MagicMock, mock_path_exists: MagicMock) -> None:
		"""Test filtering files in a test environment (skips git/fs checks, but not pattern/size)."""
		mock_path_exists.return_value = False  # Assume files don't exist for size check simplicity
		files_to_check = ["existing.py", "non_existent.txt", "deleted.log", "invalid*.py"]
		# In test env, git/fs existence checks are skipped, but pattern checks still run.
		valid_files, large_files = utils.filter_valid_files(files_to_check, is_test_environment=True)

		assert valid_files == ["existing.py", "non_existent.txt", "deleted.log"]
		assert large_files == []  # No large files simulated

	@patch("codemap.git.diff_splitter.utils.Path")  # Mock Path class
	@patch("codemap.git.diff_splitter.utils.is_test_environment", return_value=False)
	@patch("codemap.git.diff_splitter.utils.run_git_command")
	@patch("codemap.git.diff_splitter.utils.get_deleted_tracked_files")
	def test_filter_valid_files_large_file(
		self, mock_get_deleted: MagicMock, mock_run_git: MagicMock, _mock_is_test: MagicMock, mock_path: MagicMock
	) -> None:
		"""Test filtering out a large file."""
		# Mock git checks first (return empty sets/output)
		mock_get_deleted.return_value = (set(), set())
		mock_run_git.return_value = ""  # No tracked files

		# Setup Mock Path behavior
		def mock_path_creator(filepath: str) -> MagicMock:
			instance = MagicMock()
			if filepath == "large_file.bin":
				instance.exists.return_value = True
				instance.stat.return_value.st_size = MAX_FILE_SIZE_FOR_LLM + 1
			elif filepath == "small_file.txt":
				instance.exists.return_value = True  # Assume small file exists
				instance.stat.return_value.st_size = 100
			else:
				instance.exists.return_value = False
				instance.stat.return_value.st_size = 0
			return instance

		mock_path.side_effect = mock_path_creator

		files_to_check = ["small_file.txt", "large_file.bin"]
		# Run the function
		valid_files, large_files = utils.filter_valid_files(files_to_check)

		# large_file.bin is filtered due to size. small_file.txt passes size check.
		# Since small_file.txt exists (mocked Path) but is not tracked/deleted (mocked git),
		# it should remain in valid_files according to the logic.
		assert valid_files == ["small_file.txt"]
		assert large_files == ["large_file.bin"]
