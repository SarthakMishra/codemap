"""Constants for diff splitting functionality."""

# Chunk consolidation thresholds
MIN_CHUNKS_FOR_CONSOLIDATION = 1
MAX_CHUNKS_BEFORE_CONSOLIDATION = 10

# Similarity thresholds
MIN_NAME_LENGTH_FOR_SIMILARITY = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DIRECTORY_SIMILARITY_THRESHOLD = 0.5

# Model configuration
MODEL_NAME = "Qodo/Qodo-Embed-1-1.5B"

# Default code extensions
DEFAULT_CODE_EXTENSIONS = {
	"py",
	"js",
	"ts",
	"java",
	"kt",
	"go",
	"c",
	"cpp",
	"cs",
	"rb",
	"php",
	"swift",
	"jsx",
	"tsx",
}
