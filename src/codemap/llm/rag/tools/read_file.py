"""Read file tool for PydanticAI agents to search and read file content."""

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiofiles
from pydantic_ai import ModelRetry
from pydantic_ai.tools import Tool

from codemap.utils.git_utils import GitRepoContext

logger = logging.getLogger(__name__)

# Constants
MAX_FILES_TO_DISPLAY = 5
MAX_CONTENT_SEARCH_FILES = 100  # Increased limit since we're using concurrent processing
MAX_WORKERS = 8  # Number of concurrent workers for file operations


async def _search_files_by_query(
	query: str, search_root: Path | None = None, search_content: bool = True
) -> list[Path]:
	"""Search for files by query in the codebase.

	This function performs a comprehensive search that includes:
	1. Exact filename matches
	2. Partial filename matches
	3. Path pattern matches
	4. Content-based keyword search (if enabled)

	Args:
	    query: Search query - can be filename, keywords, or path patterns
	    search_root: Root directory to search from (defaults to git repository root)
	    search_content: Whether to search file contents for keywords (default: True)

	Returns:
	    List of matching file paths, sorted by relevance
	"""
	if search_root is None:
		# Get repository root using GitRepoContext
		try:
			search_root = GitRepoContext.get_repo_root()
		except (OSError, ValueError, RuntimeError):
			search_root = Path.cwd()  # Fallback to current directory

	# Clean up the query
	query = query.strip()
	if query.startswith("/"):
		query = query.lstrip("/")
		query = query.removeprefix("src/")  # Remove 'src/' prefix

	matching_files = []
	query_lower = query.lower()

	# 1. Search by filename patterns
	if "/" in query:
		# Handle path patterns
		path_parts = query.split("/")
		actual_filename = path_parts[-1]

		# Search for the filename first, then filter by path
		for file_path in search_root.rglob(actual_filename):
			if file_path.is_file():
				relative_path = str(file_path.relative_to(search_root))
				if all(part in relative_path for part in path_parts[:-1]):
					matching_files.append(file_path)
	else:
		# Search for exact filename matches first
		exact_matches = [file_path for file_path in search_root.rglob(query) if file_path.is_file()]
		matching_files.extend(exact_matches)

		# Search for partial filename matches
		partial_matches = [
			file_path
			for file_path in search_root.rglob("*")
			if file_path.is_file() and query_lower in file_path.name.lower() and file_path not in exact_matches
		]
		matching_files.extend(partial_matches)

	# 2. Search by file extensions if query looks like an extension
	if query.startswith(".") or query.endswith((".py", ".js", ".ts", ".md", ".txt", ".yml", ".yaml", ".json")):
		ext_pattern = f"*{query}" if query.startswith(".") else f"*.{query}"
		ext_matches = [
			file_path
			for file_path in search_root.rglob(ext_pattern)
			if file_path.is_file() and file_path not in matching_files
		]
		matching_files.extend(ext_matches)

	# 3. Search file contents for keywords (if enabled and no filename matches)
	if search_content and len(matching_files) < MAX_FILES_TO_DISPLAY:
		content_matches = await _search_file_contents_async(query, search_root, exclude_files=set(matching_files))
		matching_files.extend(content_matches)

	# Remove duplicates while preserving order
	seen = set()
	unique_files = []
	for file_path in matching_files:
		if file_path not in seen:
			seen.add(file_path)
			unique_files.append(file_path)

	return unique_files


def _search_single_file_content(file_path: Path, pattern: re.Pattern | None, query: str) -> Path | None:
	"""Search for query in a single file's content.

	Args:
	    file_path: Path to the file to search
	    pattern: Compiled regex pattern for search (None for simple string search)
	    query: Original query string

	Returns:
	    File path if match found, None otherwise
	"""
	try:
		with file_path.open(encoding="utf-8", errors="ignore") as f:
			content = f.read()

		# Search for query in content
		if pattern:
			if pattern.search(content):
				return file_path
		elif query.lower() in content.lower():
			return file_path

		return None

	except (OSError, UnicodeDecodeError):
		# Skip files that can't be read
		return None


async def _search_file_contents_async(
	query: str, search_root: Path, exclude_files: set[Path] | None = None
) -> list[Path]:
	"""Search for query keywords in file contents using concurrent processing.

	Args:
	    query: Search query/keywords
	    search_root: Root directory to search
	    exclude_files: Set of files to exclude from search

	Returns:
	    List of files containing the query keywords
	"""
	if exclude_files is None:
		exclude_files = set()

	# Define file extensions to search
	searchable_extensions = {
		".py",
		".js",
		".ts",
		".jsx",
		".tsx",
		".md",
		".txt",
		".yml",
		".yaml",
		".json",
		".toml",
		".cfg",
		".ini",
		".sh",
		".bash",
		".zsh",
		".fish",
		".html",
		".css",
		".scss",
		".sass",
		".less",
		".vue",
		".svelte",
		".go",
		".rs",
		".java",
		".cpp",
		".c",
		".h",
		".hpp",
		".cs",
		".php",
		".rb",
		".pl",
		".r",
		".sql",
		".dockerfile",
		".makefile",
	}

	# Create regex pattern for case-insensitive search
	try:
		pattern = re.compile(re.escape(query), re.IGNORECASE)
	except re.error:
		# If query contains regex special chars that cause issues, use simple string search
		pattern = None

	# Collect files to search
	files_to_search: list[Path] = []
	for file_path in search_root.rglob("*"):
		if (
			file_path.is_file()
			and file_path not in exclude_files
			and file_path.suffix.lower() in searchable_extensions
			and len(files_to_search) < MAX_CONTENT_SEARCH_FILES
		):
			files_to_search.append(file_path)

		# Use ThreadPoolExecutor for concurrent file reading
	matching_files = []
	loop = asyncio.get_event_loop()

	with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
		# Submit all file search tasks
		tasks = [
			loop.run_in_executor(executor, _search_single_file_content, file_path, pattern, query)
			for file_path in files_to_search
		]

		# Collect results as they complete
		results = await asyncio.gather(*tasks, return_exceptions=True)

		for result in results:
			if isinstance(result, Exception):
				logger.debug(f"Error searching file: {result}")
				continue
			if result is not None and isinstance(result, Path):
				matching_files.append(result)

	return matching_files


async def _read_file_async(file_path: Path) -> str:
	"""Read file content asynchronously.

	Args:
	    file_path: Path to the file to read

	Returns:
	    File content as string

	Raises:
	    OSError: If file cannot be read
	    UnicodeDecodeError: If file encoding is invalid
	"""
	async with aiofiles.open(file_path, encoding="utf-8") as f:  # type: ignore[misc]
		return await f.read()


async def search_and_read_files(query: str, search_content: bool = True) -> str:
	"""Search for files by query and return their content.

	This tool performs a comprehensive search that includes:
	- Exact and partial filename matches
	- Path pattern matching
	- File extension matching
	- Content-based keyword search

	Args:
	    query: Search query - can be filename, keywords, path patterns, or file extensions
	    search_content: Whether to search file contents for keywords (default: True)

	Returns:
	    String containing the file content(s) with formatting

	Examples:
	    - "config.py" - finds config.py files
	    - "test_" - finds files starting with "test_"
	    - "src/utils" - finds files in src/utils directory
	    - ".py" - finds all Python files
	    - "ConfigLoader" - finds files containing "ConfigLoader" in name or content
	    - "async def" - finds files containing "async def" in content
	"""
	try:
		# Get repository root for relative path calculation
		try:
			repo_root = GitRepoContext.get_repo_root()
		except (OSError, ValueError, RuntimeError):
			repo_root = Path.cwd()  # Fallback to current directory

		# Search for matching files
		matching_files = await _search_files_by_query(query, search_content=search_content)

		if not matching_files:
			search_type = "filename and content" if search_content else "filename"
			return f"No files found matching '{query}' in {search_type}"

		# If too many matches, limit and inform user
		if len(matching_files) > MAX_FILES_TO_DISPLAY:
			matching_files = matching_files[:MAX_FILES_TO_DISPLAY]
			result = f"Found {len(matching_files)} files matching '{query}' (showing first {MAX_FILES_TO_DISPLAY}):\n\n"
		elif len(matching_files) > 1:
			result = f"Found {len(matching_files)} files matching '{query}':\n\n"
		else:
			result = ""

		# Read all files concurrently
		file_contents = await asyncio.gather(
			*[_read_file_async(file_path) for file_path in matching_files], return_exceptions=True
		)

		# Format results
		for i, (file_path, content) in enumerate(zip(matching_files, file_contents, strict=False)):
			if isinstance(content, Exception):
				msg = f"Failed to read file: {file_path}"
				logger.exception(msg)
				raise ModelRetry(msg) from content

			# Get relative path for display using repo root
			try:
				display_path = file_path.relative_to(repo_root)
			except ValueError:
				# Fallback to absolute path if file is outside repo
				display_path = file_path

			# Add file header and content
			if len(matching_files) > 1:
				result += f"## File {i + 1}: {display_path}\n\n"
			else:
				result += f"## {display_path}\n\n"

			# Detect file extension for syntax highlighting
			file_ext = file_path.suffix[1:] if file_path.suffix else "text"

			result += f"```{file_ext}\n{content}\n```\n\n"

		return result.strip()

	except Exception as e:
		msg = f"Failed to search for or read files matching '{query}': {e}"
		logger.exception(msg)
		raise ModelRetry(msg) from e


# Create the PydanticAI Tool instance
search_file_tool = Tool(
	search_and_read_files,
	takes_ctx=False,
	name="search_file",
	description=(
		"Search for and read file content from the codebase using flexible queries. "
		"Supports multiple search types: exact/partial filenames, path patterns, "
		"file extensions, and content-based keyword search. "
		"Examples: 'config.py', 'test_', 'src/utils', '.py', 'ConfigLoader', 'async def'. "
	),
	prepare=None,
)
