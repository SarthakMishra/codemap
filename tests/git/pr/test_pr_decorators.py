"""Tests for PR generator decorators."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codemap.git.pr_generator.decorators import git_operation
from codemap.git.utils import GitError


@pytest.mark.unit
class TestGitOperationDecorator:
	"""Test cases for the @git_operation decorator."""

	def test_git_operation_success(self) -> None:
		"""Test that the decorator allows successful execution and returns value."""

		@git_operation
		def mock_op(a: int, b: int = 0) -> int:
			return a + b

		result = mock_op(1, b=2)
		assert result == 3

	def test_git_operation_reraises_git_error(self) -> None:
		"""Test that GitError raised by the function is re-raised as is."""
		error_message = "Specific git error"

		@git_operation
		def mock_op_raises_git_error() -> None:
			raise GitError(error_message)

		with pytest.raises(GitError, match=error_message):
			mock_op_raises_git_error()

	def test_git_operation_converts_other_exceptions(self) -> None:
		"""Test that other exceptions are caught and converted to GitError."""
		original_error_message = "Something went wrong"

		@git_operation
		def mock_op_raises_value_error() -> None:
			raise ValueError(original_error_message)

		expected_match = f"Git operation failed: mock_op_raises_value_error - {original_error_message}"
		with pytest.raises(GitError, match=expected_match) as exc_info:
			mock_op_raises_value_error()

		# Check that the original exception is preserved as the cause
		assert isinstance(exc_info.value.__cause__, ValueError)
		assert str(exc_info.value.__cause__) == original_error_message

	@patch("codemap.git.pr_generator.decorators.logger")
	def test_git_operation_logging(self, mock_logger: MagicMock) -> None:
		"""Test that the decorator logs start and end on success."""

		@git_operation
		def successful_op() -> str:
			return "Success"

		successful_op()

		mock_logger.debug.assert_any_call("Starting git operation: %s", "successful_op")
		mock_logger.debug.assert_any_call("Completed git operation: %s", "successful_op")

	@patch("codemap.git.pr_generator.decorators.logger")
	def test_git_operation_logging_git_error(self, mock_logger: MagicMock) -> None:
		"""Test logging when a GitError occurs."""

		@git_operation
		def raises_git_error() -> None:
			msg = "Git fail"
			raise GitError(msg)

		with pytest.raises(GitError):
			raises_git_error()

		mock_logger.debug.assert_any_call("Starting git operation: %s", "raises_git_error")
		mock_logger.debug.assert_any_call("GitError in operation: %s", "raises_git_error")
		# Ensure completion message isn't logged
		completion_call = ("Completed git operation: %s", "raises_git_error")
		assert completion_call not in [c[1] for c in mock_logger.debug.call_args_list]

	@patch("codemap.git.pr_generator.decorators.logger")
	def test_git_operation_logging_other_error(self, mock_logger: MagicMock) -> None:
		"""Test logging when another exception occurs."""
		error = ValueError("Other fail")

		@git_operation
		def raises_other_error() -> None:
			raise error

		with pytest.raises(GitError):
			raises_other_error()

		mock_logger.debug.assert_any_call("Starting git operation: %s", "raises_other_error")
		mock_logger.debug.assert_any_call("Error in git operation %s: %s", "raises_other_error", str(error))
		# Ensure completion message isn't logged
		completion_call = ("Completed git operation: %s", "raises_other_error")
		assert completion_call not in [c[1] for c in mock_logger.debug.call_args_list]
