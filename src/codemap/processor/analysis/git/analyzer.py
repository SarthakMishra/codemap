"""Git metadata analyzer for extracting version control information for code chunks."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codemap.processor.analysis.git.models import GitMetadata

if TYPE_CHECKING:
    from codemap.processor.chunking.base import Chunk, Location

logger = logging.getLogger(__name__)


@dataclass
class GitBlameInfo:
    """Information from git blame for a specific line."""

    commit_id: str
    author: str
    author_email: str
    timestamp: datetime
    line_number: int
    line_content: str


class GitMetadataAnalyzer:
    """Analyzer for extracting Git metadata for code chunks.

    This class provides methods to retrieve Git-related information about code
    chunks, such as commit history, authorship, and changes over time.
    """

    def __init__(self, repo_path: Path | None = None) -> None:
        """Initialize the Git metadata analyzer.

        Args:
            repo_path: Path to the Git repository root. If None, it will be
                      detected from the current working directory.
        """
        self.repo_path = repo_path or self._detect_repo_root()
        self._verify_git_repo()

    def _detect_repo_root(self) -> Path:
        """Detect the Git repository root from the current directory.

        Returns:
            Path to the Git repository root

        Raises:
            ValueError: If not in a Git repository
        """
        try:
            result = subprocess.run(  # noqa: S603
                ["git", "rev-parse", "--show-toplevel"],  # noqa: S607
                capture_output=True,
                text=True,
                check=True,
            )
            return Path(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            logger.exception("Failed to detect Git repository")
            msg = "Not in a Git repository"
            raise ValueError(msg) from e

    def _verify_git_repo(self) -> None:
        """Verify that the repository path is a valid Git repository.

        Raises:
            ValueError: If the path is not a valid Git repository
        """
        git_dir = self.repo_path / ".git"
        if not git_dir.exists() or not git_dir.is_dir():
            msg = f"Not a valid Git repository: {self.repo_path}"
            raise ValueError(msg)

    def get_current_branch(self) -> str:
        """Get the name of the current Git branch.

        Returns:
            Name of the current branch
        """
        try:
            result = subprocess.run(  # noqa: S603
                ["git", "-C", str(self.repo_path), "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            logger.exception("Failed to get current branch")
            return "unknown"

    def get_current_commit(self) -> str:
        """Get the current commit hash.

        Returns:
            Current commit hash
        """
        try:
            result = subprocess.run(  # noqa: S603
                ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"],  # noqa: S607
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            logger.exception("Failed to get current commit")
            return "unknown"

    def get_blame_info(self, file_path: Path, start_line: int, end_line: int) -> list[GitBlameInfo]:
        """Get git blame information for a range of lines in a file.

        Args:
            file_path: Path to the file, relative to the repository root
            start_line: Starting line number (1-based)
            end_line: Ending line number (1-based, inclusive)

        Returns:
            List of GitBlameInfo objects, one for each line
        """
        rel_path = file_path.relative_to(self.repo_path)
        line_range = f"-L {start_line},{end_line}"

        try:
            result = subprocess.run(  # noqa: S603
                ["git", "-C", str(self.repo_path), "blame", "--line-porcelain", line_range, str(rel_path)],  # noqa: S607
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the porcelain format output
            blame_info = []
            current_info = None
            line_number = start_line

            for line in result.stdout.split("\n"):
                if line.startswith("author "):
                    current_info = GitBlameInfo(
                        commit_id="",
                        author=line[7:],
                        author_email="",
                        timestamp=datetime.now(tz=UTC),  # Placeholder, will be updated
                        line_number=line_number,
                        line_content="",
                    )
                elif line.startswith("author-mail "):
                    if current_info:
                        current_info.author_email = line[12:].strip("<>")
                elif line.startswith("author-time "):
                    if current_info:
                        timestamp = int(line[12:])
                        current_info.timestamp = datetime.fromtimestamp(timestamp, tz=UTC)
                elif line.startswith("filename "):
                    if current_info:
                        blame_info.append(current_info)
                        line_number += 1
                elif current_info and not line.startswith("\t"):
                    # This is a commit hash line
                    current_info.commit_id = line.split(" ")[0]
                elif line.startswith("\t") and current_info:
                    # This is the actual content line
                    current_info.line_content = line[1:]

            return blame_info

        except subprocess.CalledProcessError:
            logger.exception("Failed to get blame info for %s (lines %d-%d)", rel_path, start_line, end_line)
            return []

    def create_git_metadata(self, location: Location) -> GitMetadata | None:
        """Create GitMetadata for a code chunk based on its location.

        Args:
            location: Location of the code chunk

        Returns:
            GitMetadata object or None if not in a Git repository or not committed
        """
        try:
            # Get current branch and commit
            branch = self.get_current_branch()
            # We'll check if the file exists in the current commit
            self.get_current_commit()

            # Get blame info for the chunk
            blame_info = self.get_blame_info(location.file_path, location.start_line, location.end_line)

            if not blame_info:
                return None

            # Extract the most relevant commit (the oldest one)
            blame_info.sort(key=lambda x: x.timestamp)
            oldest_blame = blame_info[0]
            newest_blame = blame_info[-1]

            # Get commit message
            try:
                commit_msg_result = subprocess.run(  # noqa: S603
                    ["git", "-C", str(self.repo_path), "log", "-1", "--pretty=%B", oldest_blame.commit_id],  # noqa: S607
                    capture_output=True,
                    text=True,
                    check=True,
                )
                commit_message = commit_msg_result.stdout.strip()
            except subprocess.CalledProcessError:
                commit_message = "Unknown commit message"

            # Create the metadata
            return GitMetadata(
                is_committed=True,
                commit_id=oldest_blame.commit_id,
                commit_message=commit_message,
                timestamp=oldest_blame.timestamp,
                branch=[branch],
                last_modified_at=newest_blame.timestamp if newest_blame.timestamp != oldest_blame.timestamp else None,
            )

        except (ValueError, subprocess.SubprocessError) as e:
            logger.warning("Failed to create git metadata: %s", e)
            return None

    def enrich_chunk(self, chunk: Chunk) -> None:
        """Enrich a chunk with Git metadata.

        Args:
            chunk: The chunk to enrich
        """
        if not chunk.metadata.location:
            logger.warning("Cannot enrich chunk without location information: %s", chunk.metadata.name)
            return

        git_metadata = self.create_git_metadata(chunk.metadata.location)
        if git_metadata:
            # We can't directly modify the metadata due to it being frozen (immutable),
            # so we would need to create a new metadata object and update the chunk.
            # This would be handled in the calling code.
            logger.info("Git metadata created for chunk: %s", chunk.metadata.name)

        # Recursively enrich children
        for child in chunk.children:
            self.enrich_chunk(child)
