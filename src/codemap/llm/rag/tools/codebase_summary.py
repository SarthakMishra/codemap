"""Codebase summary tool for PydanticAI agents to generate complete codebase documentation."""

import logging
from pathlib import Path

from pydantic_ai import ModelRetry
from pydantic_ai.tools import Tool

from codemap.config import ConfigLoader
from codemap.config.config_schema import GenSchema
from codemap.gen.generator import CodeMapGenerator
from codemap.gen.utils import process_codebase

logger = logging.getLogger(__name__)


async def generate_codebase_summary() -> str:
	"""Generate a comprehensive summary of the entire codebase.

	This tool processes the complete codebase at the structure level
	and returns a formatted documentation string with file tree included.

	Returns:
	    A string containing the formatted markdown documentation of the entire codebase.
	"""
	try:
		# Use current working directory as the target
		target_path = Path.cwd()

		logger.info(f"Generating codebase summary for: {target_path}")

		# Configure for structure-level documentation with tree
		config = GenSchema(
			lod_level="signatures",
			include_entity_graph=False,  # No entity graph needed
			include_tree=True,  # Include file tree
		)

		# Get configuration and create generator
		config_loader = ConfigLoader.get_instance()
		generator = CodeMapGenerator(config)

		# Process the entire codebase
		entities, metadata = process_codebase(target_path, config, config_loader=config_loader)

		if len(entities) == 0:
			msg = f"No entities found for {target_path}"
			logger.exception(msg)
			raise ModelRetry(msg)

		# Generate and return the documentation
		documentation = generator.generate_documentation(entities, metadata)

		logger.info(f"Successfully generated codebase summary with {len(entities)} entities")
		return documentation

	except Exception as e:
		msg = f"Failed to generate codebase summary: {e}"
		logger.exception(msg)
		raise ModelRetry(msg) from e


# Create the PydanticAI Tool instance
codebase_summary_tool = Tool(
	generate_codebase_summary,
	takes_ctx=False,
	name="codebase_summary",
	description=(
		"Generate a comprehensive summary of the entire codebase. "
		"This provides an overview of all files, modules, classes, and functions "
		"with their relationships and includes a file tree structure. "
		"Use this when you need a complete understanding of the codebase architecture."
	),
	prepare=None,
)
