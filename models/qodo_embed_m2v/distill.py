#!/usr/bin/env python
"""
Script to distill Qodo-Embed-1-1.5B using Model2Vec.

This script performs the following operations:
1. Downloads the Qodo-Embed-1-1.5B model
2. Distills it using Model2Vec to create a smaller, faster static model
3. Saves the distilled model for further use
"""

import argparse
import logging
import shutil
import time
from pathlib import Path

from model2vec.distill import distill

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
	"""Run the distillation process for Qodo-Embed-1-1.5B."""
	parser = argparse.ArgumentParser(description="Distill Qodo-Embed-1-1.5B using Model2Vec")
	parser.add_argument(
		"--model_name", default="Qodo/Qodo-Embed-1-1.5B", help="Model name or path for the source model"
	)
	parser.add_argument("--output_dir", default="models/qodo_embed_m2v", help="Directory to save the distilled model")
	parser.add_argument(
		"--pca_dims", type=int, default=384, help="Dimensions for PCA reduction (smaller = faster but less accurate)"
	)
	parser.add_argument("--save_to_hub", action="store_true", help="Whether to push the model to HuggingFace Hub")
	parser.add_argument("--hub_model_id", default=None, help="Model ID for HuggingFace Hub (if saving to hub)")
	parser.add_argument("--skip_readme", action="store_true", default=True, help="Skip generating the README file")

	args = parser.parse_args()

	# Create output directory if it doesn't exist
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	logger.info(f"Starting distillation of {args.model_name}")
	logger.info(f"Distilled model will be saved to {output_dir}")
	logger.info(f"Using PCA dimensions: {args.pca_dims}")
	logger.info(f"Skipping README generation: {args.skip_readme}")

	# Record start time for benchmarking
	start_time = time.time()

	# Run the distillation
	try:
		logger.info("Starting Model2Vec distillation...")
		m2v_model = distill(
			model_name=args.model_name,
			pca_dims=args.pca_dims,
		)

		distill_time = time.time() - start_time
		logger.info(f"Distillation completed in {distill_time:.2f} seconds")

		# Save the distilled model
		m2v_model.save_pretrained(args.output_dir)
		logger.info(f"Model saved to {args.output_dir}")

		# Remove README.md if it was created and we want to skip it
		if args.skip_readme and (output_dir / "README.md").exists():
			(output_dir / "README.md").unlink()
			logger.info("Removed auto-generated README.md")

		# Get model size information
		model_size_mb = sum(
			f.stat().st_size for f in output_dir.glob("**/*") if f.is_file() and f.name != "README.md"
		) / (1024 * 1024)
		logger.info(f"Distilled model size: {model_size_mb:.2f} MB")

		# Push to hub if requested
		if args.save_to_hub:
			if args.hub_model_id:
				logger.info(f"Pushing model to HuggingFace Hub as {args.hub_model_id}")

				# Create a temporary README for Hub upload if needed
				readme_path = output_dir / "README.md"
				had_readme = readme_path.exists()

				if args.skip_readme and had_readme:
					# Backup the README
					shutil.move(readme_path, output_dir / "README.md.bak")

				# Push to Hub
				m2v_model.push_to_hub(args.hub_model_id)

				# Restore state
				if args.skip_readme:
					if had_readme:
						# Restore the backup
						shutil.move(output_dir / "README.md.bak", readme_path)
					elif (output_dir / "README.md").exists():
						# Remove README created during push_to_hub
						(output_dir / "README.md").unlink()
			else:
				logger.error("--hub_model_id must be specified when using --save_to_hub")

		logger.info("Distillation process completed successfully!")

	except Exception:
		logger.exception("Error during distillation")
		raise


if __name__ == "__main__":
	main()
