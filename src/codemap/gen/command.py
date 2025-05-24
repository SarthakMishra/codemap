"""Command implementation for code documentation generation."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from codemap.config import ConfigLoader
from codemap.config.config_schema import GenSchema
from codemap.utils.cli_utils import console, progress_indicator

from .utils import process_codebase

if TYPE_CHECKING:
	from pathlib import Path

logger = logging.getLogger(__name__)


def show_error(message: str) -> None:
	"""
	Display an error message using the console.

	Args:
		message: The error message to display
	"""
	console.print(f"[red]Error:[/red] {message}")


class GenCommand:
	"""Main implementation of the gen command."""

	def __init__(self, config: GenSchema, config_loader: ConfigLoader | None = None) -> None:
		"""
		Initialize the gen command.

		Args:
		    config: Generation configuration
		    config_loader: Optional ConfigLoader instance to use

		"""
		self.config = config
		self.config_loader = config_loader or ConfigLoader()
		logger.debug("GenCommand initialized with ConfigLoader")

	def execute(self, target_path: Path, output_path: Path) -> bool:
		"""
		Execute the gen command.

		Args:
		    target_path: Path to the target codebase
		    output_path: Path to write the output

		Returns:
		    True if successful, False otherwise

		"""
		from .generator import CodeMapGenerator
		from .utils import write_documentation

		start_time = time.time()

		try:
			# Create generator
			generator = CodeMapGenerator(self.config)

			# Process codebase with progress tracking
			with progress_indicator("Processing codebase..."):
				# Process the codebase
				entities, metadata = process_codebase(target_path, self.config, config_loader=self.config_loader)

			# Generate documentation
			console.print("[green]Processing complete. Generating documentation...")
			content = generator.generate_documentation(entities, metadata)

			# Write documentation to output file
			write_documentation(output_path, content)

			# Show completion message with timing
			elapsed = time.time() - start_time
			console.print(f"[green]Generation completed in {elapsed:.2f} seconds.")

			return True

		except Exception as e:
			logger.exception("Error during gen command execution")
			show_error(f"Generation failed: {e}")
			return False
