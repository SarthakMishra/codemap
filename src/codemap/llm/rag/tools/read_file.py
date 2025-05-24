"""Read file tool for PydanticAI agents to search and read file content."""

import logging
from pathlib import Path

import aiofiles
from pydantic_ai.tools import Tool

logger = logging.getLogger(__name__)

# Constants
MAX_FILES_TO_DISPLAY = 5


def search_files_by_name(filename: str, search_root: Path | None = None) -> list[Path]:
	"""Search for files by name in the codebase.

	Args:
	    filename: Name or partial name of the file to search for
	    search_root: Root directory to search from (defaults to current working directory)

	Returns:
	    List of matching file paths
	"""
	if search_root is None:
		search_root = Path.cwd()

	# Search for exact matches first
	matching_files = [file_path for file_path in search_root.rglob(filename) if file_path.is_file()]

	# If no exact matches, search for partial matches
	if not matching_files:
		matching_files.extend(
			[
				file_path
				for file_path in search_root.rglob("*")
				if file_path.is_file() and filename.lower() in file_path.name.lower()
			]
		)

	return matching_files


async def read_file_content(filename: str) -> str:
	"""Read file content by searching for the file name in the codebase.

	This tool searches for files matching the given filename and returns their content.
	If multiple files match, it returns content for all matches.

	Args:
	    filename: Name or partial name of the file to read

	Returns:
	    String containing the file content(s) with formatting
	"""
	try:
		# Search for matching files
		matching_files = search_files_by_name(filename)

		if not matching_files:
			return f"No files found matching '{filename}'"

		# If too many matches, limit and inform user
		if len(matching_files) > MAX_FILES_TO_DISPLAY:
			matching_files = matching_files[:MAX_FILES_TO_DISPLAY]
			result = f"Found {len(matching_files)} files matching '{filename}' (showing first 5):\n\n"
		elif len(matching_files) > 1:
			result = f"Found {len(matching_files)} files matching '{filename}':\n\n"
		else:
			result = ""

		# Read and format content for each matching file
		for i, file_path in enumerate(matching_files):
			try:
				# Read file content asynchronously
				async with aiofiles.open(file_path, encoding="utf-8") as f:
					content = await f.read()

				# Get relative path for display
				try:
					display_path = file_path.relative_to(Path.cwd())
				except ValueError:
					display_path = file_path

				# Add file header and content
				if len(matching_files) > 1:
					result += f"## File {i + 1}: {display_path}\n\n"
				else:
					result += f"## {display_path}\n\n"

				# Detect file extension for syntax highlighting
				file_ext = file_path.suffix[1:] if file_path.suffix else "text"

				result += f"```{file_ext}\n{content}\n```\n\n"

			except (OSError, UnicodeDecodeError) as e:
				result += f"## {file_path}\n\n*Error reading file: {e}*\n\n"

		return result.strip()

	except Exception as e:
		logger.exception("Error in read_file_content")
		return f"Error searching for or reading file '{filename}': {e}"


# Create the PydanticAI Tool instance
read_file_tool = Tool(
	read_file_content,
	takes_ctx=False,
	name="read_file",
	description=(
		"Search for and read file content from the codebase by filename. "
		"Provide the filename or partial filename to search for. "
		"Returns the complete file content with syntax highlighting. "
		"Can handle multiple matches if the filename is ambiguous."
	),
	prepare=None,
)
