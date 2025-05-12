"""Decorators for the PR generator module."""

import functools
import logging
from collections.abc import Callable
from typing import TypeVar, cast

from codemap.git.utils import GitError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., object])


def git_operation(func: F) -> F:
	"""
	Decorator for git operations.

	This decorator wraps functions that perform git operations, providing:
	- Logging of operation start/end
	- Standardized error handling
	- Automatic conversion of git-related exceptions to GitError

	Args:
	    func: The function to decorate

	Returns:
	    Decorated function

	"""

	@functools.wraps(func)
	def wrapper(*args: object, **kwargs: object) -> object:
		"""Wrapper function for git operations that handles logging and error conversion.

		Args:
		    *args: Positional arguments passed to the decorated function.
		    **kwargs: Keyword arguments passed to the decorated function.

		Returns:
		    The result of the decorated function if successful.

		Raises:
		    GitError: If any exception occurs during the git operation. Original GitError
		        exceptions are re-raised as-is, while other exceptions are converted to
		        GitError with a descriptive message.

		Note:
		    - Logs debug messages for operation start/end
		    - Converts non-GitError exceptions to GitError
		    - Preserves original GitError exceptions
		"""
		function_name = func.__name__
		logger.debug("Starting git operation: %s", function_name)

		try:
			result = func(*args, **kwargs)
			logger.debug("Completed git operation: %s", function_name)
			return result
		except GitError:
			# Re-raise GitError as is
			logger.debug("GitError in operation: %s", function_name)
			raise
		except Exception as e:
			# Convert other exceptions to GitError
			logger.debug("Error in git operation %s: %s", function_name, str(e))
			msg = f"Git operation failed: {function_name} - {e!s}"
			raise GitError(msg) from e

	return cast("F", wrapper)
