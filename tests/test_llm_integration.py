"""Test script for direct OpenRouter API integration."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.codemap.git.message_generator import DiffChunkData, MessageGenerator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load environment variables from .env.local
env_file = Path(".env.local")
if env_file.exists():
    load_dotenv(env_file)
    logger.info("Loaded environment from %s", env_file)
else:
    logger.warning("Warning: %s not found", env_file)


# Test OpenRouter integration
def test_openrouter_integration() -> None:
    """Test direct OpenRouter integration."""
    logger.info("Starting OpenRouter integration test")

    # Get the current directory
    repo_root = Path.cwd()
    logger.info("Using repo root: %s", repo_root)

    # Create a message generator with OpenRouter
    generator = MessageGenerator(
        repo_root=repo_root,
        model="qwen/qwen2.5-coder-7b-instruct",  # From .codemap.yml
        provider="openrouter",  # From .codemap.yml
    )

    logger.info("Created MessageGenerator with model=%s, provider=%s", generator.model, generator.provider)

    # Print environment variable for debugging
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    logger.info("OPENROUTER_API_KEY present: %s", openrouter_key is not None)
    if openrouter_key:
        # Show a few characters for verification
        logger.info("OPENROUTER_API_KEY starts with: %s...", openrouter_key[:10])

    # Create a simple test diff chunk
    chunk = DiffChunkData(
        files=[".env.example"],
        content="""diff --git a/.env.example b/.env.example
index 105c41b..fdcb59a 100644
--- a/.env.example
+++ b/.env.example
@@ -10,3 +10,4 @@
 # MISTRAL_API_KEY=...
 # TOGETHER_API_KEY=...
 # GOOGLE_API_KEY=...
+# OPENROUTER_API_KEY=..
""",
    )

    logger.info("Attempting to generate message with chunk")

    # Try to generate a message
    try:
        message, is_llm = generator.generate_message(chunk)
        logger.info("Message generation result: is_llm=%s, message=%s", is_llm, message)
    except Exception:
        logger.exception("Error generating message")

    logger.info("Test completed")


if __name__ == "__main__":
    test_openrouter_integration()
