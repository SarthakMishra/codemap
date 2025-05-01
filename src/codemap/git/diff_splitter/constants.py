"""Constants for diff splitting functionality."""

from typing import Final

# Chunk consolidation thresholds
MIN_CHUNKS_FOR_CONSOLIDATION: Final = 2
MAX_CHUNKS_BEFORE_CONSOLIDATION: Final = 20

# Similarity thresholds
MIN_NAME_LENGTH_FOR_SIMILARITY: Final = 3
DEFAULT_SIMILARITY_THRESHOLD: Final = 0.4
DIRECTORY_SIMILARITY_THRESHOLD: Final = 0.3

# Maximum size of diff content to log in debug mode
MAX_LOG_DIFF_SIZE = 1000

# Maximum size of file content to include in LLM prompts (bytes)
# This helps prevent 413 Payload Too Large errors
MAX_FILE_SIZE_FOR_LLM: Final = 100000  # 100KB

# Model configuration
MODEL_NAME = "sarthak1/Qodo-Embed-M-1-1.5B-M2V-Distilled"

# Default code extensions
DEFAULT_CODE_EXTENSIONS: Final = {
	"js",
	"jsx",
	"ts",
	"tsx",
	"py",
	"java",
	"c",
	"cpp",
	"h",
	"hpp",
	"cc",
	"cs",
	"go",
	"rb",
	"php",
	"rs",
	"swift",
	"scala",
	"kt",
	"sh",
	"pl",
	"pm",
}

# Constants for numeric comparisons
EPSILON = 1e-10  # Small value for float comparisons

# Group size limits
MAX_FILES_PER_GROUP: Final = 10
