#!/usr/bin/env python
"""
Script to evaluate the performance of the distilled Qodo-Embed model.

This script performs the following:
1. Loads both the original Qodo-Embed-1-1.5B model and the distilled version
2. Compares them on:
   - Embedding similarity
   - Inference speed
   - Memory usage
3. Outputs a comprehensive evaluation report
"""

import argparse
import gc
import logging
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# For transformer models
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Sample texts for evaluation
SAMPLE_TEXTS = [
	"def process_data_stream(source_iterator):",
	"implement binary search tree",
	"how to handle memory efficient data streaming",
	"""class LazyLoader:
        def __init__(self, source):
            self.generator = iter(source)
            self._cache = []""",
	"""def dfs_traversal(root):
        if not root:
            return []
        visited = []
        stack = [root]
        while stack:
            node = stack.pop()
            visited.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return visited""",
]


def load_models(original_model_name: str, distilled_model_path: str) -> tuple[Any, StaticModel]:
	"""Load both the original and distilled models."""
	logger.info(f"Loading original model: {original_model_name}")

	try:
		# Try to load as a sentence transformer first
		original_model = SentenceTransformer(original_model_name)
		model_type = "sentence_transformer"
	except Exception:
		# If that fails, try loading as a Hugging Face transformer
		AutoTokenizer.from_pretrained(original_model_name)
		original_model = AutoModel.from_pretrained(original_model_name)
		model_type = "huggingface"

	logger.info(f"Loading distilled model from: {distilled_model_path}")
	distilled_model = StaticModel.from_pretrained(distilled_model_path)

	return (original_model, model_type), distilled_model


def measure_memory_usage(model: Any) -> float:
	"""Measure memory usage of a model in MB."""
	gc.collect()
	torch.cuda.empty_cache() if torch.cuda.is_available() else None

	process = psutil.Process(os.getpid())
	memory_before = process.memory_info().rss / (1024 * 1024)  # MB

	# Force model to allocate memory if it hasn't already
	if isinstance(model, StaticModel) or hasattr(model, "encode"):
		_ = model.encode(["Test"])
	else:
		# For HF models, we need to handle differently
		pass

	gc.collect()
	torch.cuda.empty_cache() if torch.cuda.is_available() else None

	process = psutil.Process(os.getpid())
	memory_after = process.memory_info().rss / (1024 * 1024)  # MB

	return memory_after - memory_before


def compute_embeddings(
	original_model: Any, original_model_type: str, distilled_model: StaticModel, texts: list[str]
) -> tuple[np.ndarray, np.ndarray]:
	"""Compute embeddings using both models."""
	# Original model embeddings
	if original_model_type == "sentence_transformer":
		original_embeddings = original_model.encode(texts)
	else:
		# For HF models, we need more custom code
		# Simple mean pooling function for HF models
		def mean_pooling(model_output, attention_mask):
			token_embeddings = model_output[0]
			input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
			return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
				input_mask_expanded.sum(1), min=1e-9
			)

		tokenizer = AutoTokenizer.from_pretrained(original_model.config._name_or_path)
		encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

		with torch.no_grad():
			model_output = original_model(**encoded_input)
			original_embeddings = mean_pooling(model_output, encoded_input["attention_mask"]).numpy()

	# Distilled model embeddings
	distilled_embeddings = distilled_model.encode(texts)

	return original_embeddings, distilled_embeddings


def measure_inference_speed(model: Any, model_type: str, texts: list[str], n_runs: int = 5) -> float:
	"""Measure inference speed in texts/second."""
	# Warmup
	if model_type in {"sentence_transformer", "static_model"}:
		_ = model.encode(texts[:1])
	else:
		# Warmup for HF models
		tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
		encoded_input = tokenizer(texts[:1], padding=True, truncation=True, return_tensors="pt")
		with torch.no_grad():
			_ = model(**encoded_input)

	# Measure speed
	start_time = time.time()

	if model_type in {"sentence_transformer", "static_model"}:
		for _ in range(n_runs):
			_ = model.encode(texts)
	else:
		# For HF models
		tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
		for _ in range(n_runs):
			encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
			with torch.no_grad():
				_ = model(**encoded_input)

	total_time = time.time() - start_time
	return (len(texts) * n_runs) / total_time


def compute_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
	"""Compute cosine similarity between embeddings, handling different dimensions.

	For embeddings with different dimensions, we compute similarity by comparing
	how they rank the same texts (semantically equivalent).
	"""
	# Ensure embeddings1 and embeddings2 are 2D arrays with shapes (n_samples, n_features)
	if embeddings1.ndim == 1:
		embeddings1 = embeddings1.reshape(1, -1)
	if embeddings2.ndim == 1:
		embeddings2 = embeddings2.reshape(1, -1)

	# Check and transpose if needed to ensure samples are in rows
	if embeddings2.shape[0] != len(SAMPLE_TEXTS) and embeddings2.shape[1] == len(SAMPLE_TEXTS):
		embeddings2 = embeddings2.T

	logger.info(f"Embeddings shapes: original={embeddings1.shape}, distilled={embeddings2.shape}")

	# If dimensions differ, we compute similarity matrix based on how each model ranks text pairs
	# This is a form of semantic similarity evaluation rather than direct vector comparison
	similarity_matrix = np.zeros((len(SAMPLE_TEXTS), len(SAMPLE_TEXTS)))

	# Compute similarity matrices within each embedding space
	sim1 = cosine_similarity(embeddings1)
	sim2 = cosine_similarity(embeddings2)

	# The similarity between samples i and j is the correlation between how they rank other samples
	for i in range(len(SAMPLE_TEXTS)):
		for j in range(len(SAMPLE_TEXTS)):
			# For diagonal elements (same sample), use a direct measure of how similar
			# the two models rank that sample against all others
			if i == j:
				# Pearson correlation between the rankings (excluding self-comparison)
				rankings1 = np.delete(sim1[i], i)
				rankings2 = np.delete(sim2[i], i)
				# Higher correlation means the models agree on the semantic similarity
				similarity_matrix[i, j] = np.corrcoef(rankings1, rankings2)[0, 1]
			else:
				# For off-diagonal elements, compare how similarly both models relate samples i and j
				similarity_matrix[i, j] = 1 - abs(sim1[i, j] - sim2[i, j])

	return similarity_matrix


def format_size(size_bytes: float) -> str:
	"""Format size in bytes to human-readable format."""
	for unit in ["B", "KB", "MB", "GB"]:
		if size_bytes < 1024.0:
			return f"{size_bytes:.2f} {unit}"
		size_bytes /= 1024.0
	return f"{size_bytes:.2f} TB"


def plot_comparison(results: dict[str, Any], output_dir: str) -> None:
	"""Generate comparison plots and save them."""
	output_path = Path(output_dir)
	output_path.mkdir(exist_ok=True, parents=True)

	# Speed comparison
	plt.figure(figsize=(10, 6))
	models = ["Original", "Distilled"]
	speeds = [results["original_speed"], results["distilled_speed"]]
	plt.bar(models, speeds, color=["#1f77b4", "#ff7f0e"])
	plt.ylabel("Texts per second")
	plt.title("Inference Speed Comparison")
	plt.savefig(output_path / "speed_comparison.png", dpi=300, bbox_inches="tight")

	# Memory comparison
	plt.figure(figsize=(10, 6))
	memories = [results["original_memory"], results["distilled_memory"]]
	plt.bar(models, memories, color=["#1f77b4", "#ff7f0e"])
	plt.ylabel("Memory Usage (MB)")
	plt.title("Memory Usage Comparison")
	plt.savefig(output_path / "memory_comparison.png", dpi=300, bbox_inches="tight")

	# Size comparison
	plt.figure(figsize=(10, 6))
	sizes = [results["original_size"], results["distilled_size"]]
	plt.bar(models, sizes, color=["#1f77b4", "#ff7f0e"])
	plt.ylabel("Model Size (MB)")
	plt.title("Model Size Comparison")
	plt.savefig(output_path / "size_comparison.png", dpi=300, bbox_inches="tight")

	# Similarity matrix heatmap
	plt.figure(figsize=(8, 6))
	plt.imshow(results["similarity_matrix"], cmap="viridis", interpolation="nearest")
	plt.colorbar(label="Cosine Similarity")
	plt.title("Embedding Similarity Between Original and Distilled Models")
	plt.xticks([])
	plt.yticks(range(len(SAMPLE_TEXTS)), [t[:20] + "..." if len(t) > 20 else t for t in SAMPLE_TEXTS])
	plt.savefig(output_path / "similarity_matrix.png", dpi=300, bbox_inches="tight")


def evaluate_models(original_model_name: str, distilled_model_path: str, output_dir: str):
	"""Evaluate the original and distilled models."""
	# Load models
	(original_model, original_model_type), distilled_model = load_models(original_model_name, distilled_model_path)

	# Measure model sizes
	original_model_size = sum(p.numel() * 4 for p in original_model.parameters()) / (
		1024 * 1024
	)  # MB (assuming float32)
	distilled_model_size = sum(f.stat().st_size for f in Path(distilled_model_path).glob("**/*") if f.is_file()) / (
		1024 * 1024
	)  # MB

	# Measure memory usage
	original_memory = measure_memory_usage(original_model)
	distilled_memory = measure_memory_usage(distilled_model)

	# Compute embeddings
	original_embeddings, distilled_embeddings = compute_embeddings(
		original_model, original_model_type, distilled_model, SAMPLE_TEXTS
	)

	# Compute similarity between embeddings
	similarity_matrix = compute_cosine_similarity(original_embeddings, distilled_embeddings)
	similarity_diagonal = np.diag(similarity_matrix)
	avg_similarity = np.mean(similarity_diagonal)

	# Measure inference speed
	original_speed = measure_inference_speed(original_model, original_model_type, SAMPLE_TEXTS, n_runs=5)
	distilled_speed = measure_inference_speed(distilled_model, "static_model", SAMPLE_TEXTS, n_runs=5)

	# Collect results
	results = {
		"original_size": original_model_size,
		"distilled_size": distilled_model_size,
		"original_memory": original_memory,
		"distilled_memory": distilled_memory,
		"similarity_matrix": similarity_matrix,
		"avg_similarity": avg_similarity,
		"original_speed": original_speed,
		"distilled_speed": distilled_speed,
		"speed_improvement": distilled_speed / original_speed if original_speed > 0 else float("inf"),
		"size_reduction": original_model_size / distilled_model_size if distilled_model_size > 0 else float("inf"),
		"memory_reduction": original_memory / distilled_memory if distilled_memory > 0 else float("inf"),
	}

	# Generate plots
	plot_comparison(results, output_dir)

	# Print results
	logger.info("\n" + "=" * 50)
	logger.info("Model Evaluation Results")
	logger.info("=" * 50)
	logger.info(f"Original Model Size: {results['original_size']:.2f} MB")
	logger.info(f"Distilled Model Size: {results['distilled_size']:.2f} MB")
	logger.info(f"Size Reduction Factor: {results['size_reduction']:.2f}x")
	logger.info("\n")
	logger.info(f"Original Model Memory: {results['original_memory']:.2f} MB")
	logger.info(f"Distilled Model Memory: {results['distilled_memory']:.2f} MB")
	logger.info(f"Memory Reduction Factor: {results['memory_reduction']:.2f}x")
	logger.info("\n")
	logger.info(f"Original Model Speed: {results['original_speed']:.2f} texts/second")
	logger.info(f"Distilled Model Speed: {results['distilled_speed']:.2f} texts/second")
	logger.info(f"Speed Improvement Factor: {results['speed_improvement']:.2f}x")
	logger.info("\n")
	logger.info(f"Average Embedding Similarity: {results['avg_similarity']:.4f}")
	logger.info("=" * 50)

	return results


def main() -> None:
	"""Run the evaluation process."""
	parser = argparse.ArgumentParser(description="Evaluate the distilled model against the original")
	parser.add_argument("--original_model", default="Qodo/Qodo-Embed-1-1.5B", help="Original model name or path")
	parser.add_argument("--distilled_model", default="models/qodo_embed_m2v", help="Path to the distilled model")
	parser.add_argument(
		"--output_dir", default="models/qodo_embed_m2v/evaluation", help="Directory to save evaluation results"
	)

	args = parser.parse_args()

	# Create output directory
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Run evaluation
	try:
		evaluate_models(args.original_model, args.distilled_model, args.output_dir)
		logger.info(f"Evaluation completed. Results saved to {args.output_dir}")
	except Exception:
		logger.exception("Error during evaluation")
		raise


if __name__ == "__main__":
	main()
