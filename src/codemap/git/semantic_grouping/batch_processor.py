"""
Batch processing for semantic groups commit message generation.

This module provides functionality to generate commit messages for
multiple semantic groups in batch using LiteLLM's batch_completion.

"""

import logging
from typing import TYPE_CHECKING

from codemap.git.diff_splitter import DiffChunk
from codemap.git.interactive import CommitUI
from codemap.git.semantic_grouping.context_processor import process_chunks_with_lod
from codemap.utils.config_loader import ConfigLoader

if TYPE_CHECKING:
	from codemap.git.semantic_grouping import SemanticGroup

logger = logging.getLogger(__name__)


def batch_generate_messages(
	groups: list["SemanticGroup"],
	prompt_template: str,
	config_loader: ConfigLoader,
	model: str | None = None,
) -> list["SemanticGroup"]:
	"""
	Generate commit messages for multiple semantic groups in batch.

	Args:
	    groups: List of SemanticGroup objects
	    prompt_template: Template to use for prompt generation
	    config_loader: ConfigLoader instance
	    model: Optional model name override

	Returns:
	    List of SemanticGroup objects with messages added

	Raises:
	    LLMError: If batch processing fails

	"""
	if not groups:
		return []

	# Get config values
	llm_config = config_loader.get("llm", {})
	max_tokens = llm_config.get("max_context_tokens", 4000)
	use_lod_context = llm_config.get("use_lod_context", True)
	model_name = model or llm_config.get("model", "openai/gpt-4o-mini")

	# Prepare temporary chunks and prompts for each group
	temp_chunks = []
	messages_list = []

	# Add this at the top of the function, right after getting the config values
	ui = CommitUI()

	for group in groups:
		try:
			# Create a temporary DiffChunk with optimized content if needed
			if use_lod_context and len(group.chunks) > 1:
				logger.debug("Processing semantic group with %d chunks using LOD context", len(group.chunks))
				try:
					optimized_content = process_chunks_with_lod(group.chunks, max_tokens)
					if optimized_content:
						temp_chunk = DiffChunk(files=group.files, content=optimized_content)
					else:
						temp_chunk = DiffChunk(files=group.files, content=group.content)
				except Exception:
					logger.exception("Error in LOD context processing, falling back to original content")
					temp_chunk = DiffChunk(files=group.files, content=group.content)
			else:
				temp_chunk = DiffChunk(files=group.files, content=group.content)

			# Store the temp chunk for reference
			temp_chunks.append(temp_chunk)

			# Prepare the prompt for this group
			# We need to import and use methods from CommitMessageGenerator
			from codemap.git.commit_generator.generator import MinimalMessageGenerator

			# Creating a minimal generator just to use its _prepare_prompt method
			temp_generator = MinimalMessageGenerator(
				prompt_template=prompt_template,
				config_loader=config_loader,
			)

			prompt = temp_generator.prepare_prompt(temp_chunk)

			# Format as messages for batch_completion
			messages = [{"role": "user", "content": prompt}]
			messages_list.append(messages)

		except Exception:
			logger.exception(f"Error preparing prompt for group {group.files}")
			# Add empty messages for this group to maintain index alignment
			messages_list.append([{"role": "user", "content": "Skip this group due to error"}])

	# Use the LLM module's batch generation
	try:
		from codemap.git.commit_generator.schemas import COMMIT_MESSAGE_SCHEMA
		from codemap.llm.utils import batch_generate_completions

		# Execute batch completion using the LLM module
		responses = batch_generate_completions(
			messages_list=messages_list,
			model=model_name,
			config_loader=config_loader,
			response_format={"type": "json_object", "schema": COMMIT_MESSAGE_SCHEMA},
			temperature=llm_config.get("temperature", 0.7),
			max_tokens=llm_config.get("max_tokens", 1024),
		)

		# Process responses and update groups
		for i, (response, group) in enumerate(zip(responses, groups, strict=False)):
			try:
				# Extract content from response
				if response and hasattr(response, "choices") and response.choices:
					content = response.choices[0].message.content

					# If it's JSON, extract the message
					if content.startswith("{") and content.endswith("}"):
						import json

						try:
							parsed = json.loads(content)
							if "message" in parsed:
								content = parsed["message"]
						except json.JSONDecodeError:
							pass  # Keep original content if JSON parsing fails

					# Set the message on the group
					group.message = content
				else:
					logger.warning(f"Empty or invalid response for group {i}")
					group.message = f"update: changes to {len(group.files)} files"
			except Exception:
				logger.exception(f"Error processing response for group {i}")
				group.message = f"update: changes to {len(group.files)} files"

	except Exception:
		logger.exception("Batch completion failed")
		# Just use the already initialized UI
		ui.show_warning("LLM call failed. Using fallback commit messages.")

		# Provide fallback messages for all groups
		for group in groups:
			if not group.message:  # Don't override if already set
				fallback_msg = f"update: changes to {len(group.files)} files"
				group.message = fallback_msg
				# Log which groups received fallback messages
				logger.warning(f"Using fallback message for files: {group.files}")

	return groups
