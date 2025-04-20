"""Git commit feature package for CodeMap."""

from .command import CommitCommand, setup_message_generator
from .diff_splitter import DiffChunk, DiffSplitter, SplitStrategy
from .interactive import ChunkAction, ChunkResult, CommitUI, process_all_chunks
from .message_generator import LLMError, MessageGenerator

__all__ = [
    "ChunkAction",
    "ChunkResult",
    "CommitCommand",
    "CommitUI",
    "DiffChunk",
    "DiffSplitter",
    "LLMError",
    "MessageGenerator",
    "SplitStrategy",
    "process_all_chunks",
    "setup_message_generator",
]
