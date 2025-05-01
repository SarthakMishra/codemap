"""Test script for direct OpenRouter API integration."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.codemap.git.commit_generator import DiffChunkData, MessageGenerator
from src.codemap.git.commit_generator.prompts import DEFAULT_PROMPT_TEMPLATE
from src.codemap.llm import create_client
from src.codemap.utils.config_loader import ConfigLoader
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
		self._load_environment_variables()

		# Get the current directory
		self.repo_root = Path.cwd()
		logger.info("Using repo root: %s", self.repo_root)

	def _load_environment_variables(self) -> None:
		"""Load environment variables from .env.local file."""
		env_file = Path(".env.local")
		if env_file.exists():
			load_dotenv(env_file)
			logger.info("Loaded environment from %s", env_file)
		else:
			logger.warning("Warning: %s not found", env_file)

	def _create_test_diff_chunk(self) -> DiffChunkData:
		"""Create a test diff chunk for OpenRouter API testing."""
		return DiffChunkData(
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

	def _create_message_generator(
		self, model: str = "qwen/qwen2.5-coder-7b-instruct", provider: str = "openrouter"
	) -> MessageGenerator:
		"""Create a real message generator for testing."""
		# Create LLM client
		llm_client = create_client(
			repo_path=self.repo_root, model=model, api_key=os.environ.get(f"{provider.upper()}_API_KEY")
		)

		# Get prompt template and create config loader
		config_loader = ConfigLoader(repo_root=self.repo_root)

		return MessageGenerator(
			repo_root=self.repo_root,
			llm_client=llm_client,
			prompt_template=DEFAULT_PROMPT_TEMPLATE,
			config_loader=config_loader,
		)

	def _generate_and_verify_message(self, generator: MessageGenerator, chunk: DiffChunkData) -> tuple[str, bool]:
		"""Generate a message and verify the output."""
		logger.info("Attempting to generate message with chunk")

		message, is_llm = generator.generate_message(chunk)

		logger.info("Message generation result: is_llm=%s, message=%s", is_llm, message)
		assert is_llm, "Message should be LLM-generated"
		assert message, "Message should not be empty"

		return message, is_llm

	@pytest.mark.skipif("OPENROUTER_API_KEY" not in os.environ, reason="OPENROUTER_API_KEY not set in environment")
	def test_openrouter_integration(self) -> None:
		"""Test direct OpenRouter integration."""
		logger.info("Starting OpenRouter integration test")

		# Create a simple test diff chunk
		chunk = self._create_test_diff_chunk()

		# Create actual generator instead of using the mock
		generator = self._create_message_generator()

		# Try to generate a message
		try:
			self._generate_and_verify_message(generator, chunk)
		except Exception as e:
			logger.exception("Error generating message")
			pytest.fail(f"Message generation failed: {e!s}")

		logger.info("Test completed successfully")

	def test_llm_mock(self) -> None:
		"""Test that the LLM mock works correctly."""
		# Create a simple diff chunk
		chunk = create_diff_chunk(
			files=[".env.example"],
			content="""diff --git a/.env.example b/.env.example
+# OPENROUTER_API_KEY=..
""",
		)

		# Define expected response
		expected_response = "docs: add OpenRouter API key environment variable"

		# Set up the mock response
		self.mock_llm_response(response=expected_response, success=True)

		# Generate message using the mock
		message, is_llm = self.message_generator.generate_message(chunk)

		# Verify results
		assert message == expected_response, f"Expected '{expected_response}' but got '{message}'"
		assert is_llm is True, "Expected is_llm to be True"
