"""Test script for direct OpenRouter API integration."""

import logging
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.codemap.git.message_generator import DiffChunkData
from tests.base import LLMTestBase
from tests.helpers import create_diff_chunk

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.slow
class TestLLMIntegration(LLMTestBase):
    """Tests for LLM integration functionality."""

    def setup_method(self) -> None:
        """Set up the test environment."""
        # Load environment variables from .env.local
        env_file = Path(".env.local")
        if env_file.exists():
            load_dotenv(env_file)
            logger.info("Loaded environment from %s", env_file)
        else:
            logger.warning("Warning: %s not found", env_file)

        # Get the current directory
        self.repo_root = Path.cwd()
        logger.info("Using repo root: %s", self.repo_root)

    @pytest.mark.skipif("OPENROUTER_API_KEY" not in os.environ, reason="OPENROUTER_API_KEY not set in environment")
    def test_openrouter_integration(self) -> None:
        """Test direct OpenRouter integration."""
        logger.info("Starting OpenRouter integration test")

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

        # Create actual generator instead of using the mock
        from src.codemap.git.message_generator import MessageGenerator

        generator = MessageGenerator(
            repo_root=self.repo_root,
            model="qwen/qwen2.5-coder-7b-instruct",
            provider="openrouter",
        )

        # Try to generate a message
        try:
            message, is_llm = generator.generate_message(chunk)
            logger.info("Message generation result: is_llm=%s, message=%s", is_llm, message)
            assert is_llm, "Message should be LLM-generated"
            assert message, "Message should not be empty"
        except Exception:
            logger.exception("Error generating message")
            pytest.fail("Message generation failed")

        logger.info("Test completed")

    def test_llm_mock(self) -> None:
        """Test that the LLM mock works correctly."""
        # This uses the mock from LLMTestBase
        chunk = create_diff_chunk(
            files=[".env.example"],
            content="""diff --git a/.env.example b/.env.example
+# OPENROUTER_API_KEY=..
""",
        )

        # Set a custom response
        self.mock_llm_response(response="docs: add OpenRouter API key environment variable", success=True)

        # Test the mock
        message, is_llm = self.message_generator.generate_message(chunk)
        assert message == "docs: add OpenRouter API key environment variable"
        assert is_llm is True
