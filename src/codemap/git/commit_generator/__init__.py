"""
Commit message generation package for CodeMap.

This package provides modules for generating commit messages using LLMs.

"""

from codemap.git.diff_splitter import DiffChunk

from .generator import CommitMessageGenerator
from .prompts import DEFAULT_PROMPT_TEMPLATE, prepare_prompt
from .schemas import COMMIT_MESSAGE_SCHEMA, CommitMessageSchema, DiffChunkData
from .utils import clean_message_for_linting, lint_commit_message

# Alias for backward compatibility
MessageGenerator = CommitMessageGenerator

__all__ = [
	# Constants and schemas
	"COMMIT_MESSAGE_SCHEMA",
	"DEFAULT_PROMPT_TEMPLATE",
	# Classes
	"CommitMessageGenerator",
	"CommitMessageSchema",
	"DiffChunk",
	"DiffChunkData",
	"MessageGenerator",  # Add the alias to __all__
	# Functions
	"clean_message_for_linting",
	"lint_commit_message",
	"prepare_prompt",
]
