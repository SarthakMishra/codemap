"""Schema for the vector database."""

from pydantic import BaseModel


class GitBlameSchema(BaseModel):
	"""Metadata for a git blame."""

	commit_id: str
	date: str
	author_name: str
	start_line: int
	end_line: int


class GitMetadataSchema(BaseModel):
	"""Metadata for a git repository."""

	git_hash: str
	tracked: bool
	branch: str
	blame: list[GitBlameSchema] = []


class FileMetadataSchema(BaseModel):
	"""Metadata for a file."""

	file_path: str
	language: str
	last_modified_time: float
	file_content_hash: str


class ChunkMetadataSchema(BaseModel):
	"""Metadata for a chunk of code."""

	chunk_id: str
	content_hash: str
	start_line: int
	end_line: int
	entity_type: str
	entity_name: str
	hierarchy_path: str
	git_metadata: GitMetadataSchema
	file_metadata: FileMetadataSchema


class ChunkSchema(BaseModel):
	"""Schema for a chunk of code."""

	content: str
	metadata: ChunkMetadataSchema
