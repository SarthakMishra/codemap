"""
Module for resolving file integrity constraints in semantic groups.

This module provides functionality for ensuring that changes to the same file are kept
in the same commit, even when semantic clustering might separate them. This ensures that
file integrity is maintained during the commit process.

Key components:
- FileIntegrityResolver: Main class that analyzes file overlaps between semantic groups
  and decides whether to merge groups or reassign chunks to maintain file integrity

The resolution process involves:
1. Detecting violations (files that appear in multiple semantic groups)
2. Calculating semantic similarity between groups with overlapping files
3. Deciding whether to merge groups (if sufficiently similar) or reassign chunks
4. Iteratively resolving violations until all files are in exactly one group

"""

import logging
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from codemap.git.diff_splitter import DiffChunk

if TYPE_CHECKING:
	from codemap.config import ConfigLoader
	from codemap.git.semantic_grouping import SemanticGroup

logger = logging.getLogger(__name__)

# Type variable for chunks
T = TypeVar("T", bound=DiffChunk)


class FileIntegrityResolver:
	"""
	Resolves file integrity constraints for semantic groups.

	File integrity refers to the requirement that all changes to a specific file should
	be included in the same commit, even if they are semantically different. This prevents
	fragmented changes to the same file across multiple commits, which can lead to broken builds
	or inconsistent states.

	The resolver works by:
	1. Identifying files that appear in multiple semantic groups
	2. Calculating the semantic similarity between these overlapping groups
	3. Either merging similar groups or reassigning chunks from less relevant groups
	   to the most appropriate group

	This process ensures that each file is modified in exactly one commit, while still
	maintaining semantic coherence within commits when possible.

	"""

	def __init__(
		self,
		similarity_threshold: float = 0.6,
		config_loader: "ConfigLoader | None" = None,
	) -> None:
		"""
		Initialize the resolver.

		Args:
		    similarity_threshold: Threshold for group similarity to trigger merging (0.0-1.0).
		    config_loader: Optional ConfigLoader instance.
		"""
		if config_loader:
			self.config_loader = config_loader
		else:
			from codemap.config import ConfigLoader

			self.config_loader = ConfigLoader()

		self.similarity_threshold = similarity_threshold

		# Import here to avoid making sklearn a hard dependency
		try:
			from sklearn.metrics.pairwise import cosine_similarity

			self.cosine_similarity = cosine_similarity
		except ImportError as e:
			logger.exception("Failed to import scikit-learn. Please install it with: uv add scikit-learn")
			msg = "scikit-learn is required for file integrity resolution"
			raise ImportError(msg) from e

	def calculate_group_similarity(
		self, group1: "SemanticGroup", group2: "SemanticGroup", chunk_embeddings: dict[DiffChunk, np.ndarray]
	) -> float:
		"""
		Calculate similarity between two groups based on their chunks' embeddings.

		This method computes the average pairwise cosine similarity between all combinations
		of chunks from the two groups. The similarity is based on the semantic embeddings
		of the chunks' content.

		Process:
		1. Extract embeddings for all chunks in both groups
		2. Compute pairwise cosine similarities between each pair of chunks
		3. Return the average similarity score

		Args:
		    group1: First semantic group to compare
		    group2: Second semantic group to compare
		    chunk_embeddings: Dict mapping chunks to their embeddings

		Returns:
		    float: Similarity score between 0 and 1, where:
		        - 0 indicates completely unrelated changes
		        - 1 indicates identical or extremely similar changes
		        - Values around 0.6-0.8 typically indicate related functionality

		"""
		# Get embeddings for chunks in each group
		embeddings1 = [chunk_embeddings[chunk] for chunk in group1.chunks if chunk in chunk_embeddings]
		embeddings2 = [chunk_embeddings[chunk] for chunk in group2.chunks if chunk in chunk_embeddings]

		if not embeddings1 or not embeddings2:
			return 0.0

		# Calculate pairwise similarities
		similarities = []
		for emb1 in embeddings1:
			for emb2 in embeddings2:
				sim = self.cosine_similarity([emb1], [emb2])[0][0]
				similarities.append(sim)

		# Return average similarity
		return sum(similarities) / len(similarities) if similarities else 0.0

	def resolve_violations(
		self, groups: list["SemanticGroup"], chunk_embeddings: dict[DiffChunk, np.ndarray]
	) -> list["SemanticGroup"]:
		"""
		Resolve file integrity violations by merging or reassigning chunks.

		A violation occurs when the same file appears in multiple semantic groups.
		This needs to be resolved because a file should be modified in only one commit.

		Args:
		    groups: List of SemanticGroup objects to resolve
		    chunk_embeddings: Dict mapping chunks to their embeddings

		Returns:
		    List of SemanticGroup objects with all violations resolved

		"""
		# Keep iterating until no violations remain
		while True:
			# Build file to groups mapping
			file_to_groups: dict[str, list[int]] = {}
			for i, group in enumerate(groups):
				for file in group.files:
					if file not in file_to_groups:
						file_to_groups[file] = []
					file_to_groups[file].append(i)

			# Find violations (files in multiple groups)
			violations = {file: indices for file, indices in file_to_groups.items() if len(indices) > 1}

			if not violations:
				break  # No violations, we're done

			# Process the first violation
			file = next(iter(violations))
			group_indices = violations[file]

			# Try to find groups to merge based on similarity
			max_similarity = 0
			groups_to_merge = None

			# Calculate similarities between all pairs of groups containing this file
			for i in range(len(group_indices)):
				for j in range(i + 1, len(group_indices)):
					idx1, idx2 = group_indices[i], group_indices[j]
					similarity = self.calculate_group_similarity(groups[idx1], groups[idx2], chunk_embeddings)

					if similarity > max_similarity:
						max_similarity = similarity
						groups_to_merge = (idx1, idx2)

			# Decide whether to merge or reassign based on similarity threshold
			if max_similarity >= self.similarity_threshold and groups_to_merge:
				# STRATEGY 1: Merge groups if they're similar enough
				idx1, idx2 = groups_to_merge
				merged_group = groups[idx1].merge_with(groups[idx2])

				# Replace the first group with the merged one and remove the second
				groups[idx1] = merged_group
				groups.pop(idx2)
			else:
				# STRATEGY 2: Reassign chunks to the primary group for this file
				# Find the primary group (group with most chunks containing this file)
				file_chunks_count = []
				for idx in group_indices:
					count = sum(1 for chunk in groups[idx].chunks if file in chunk.files)
					file_chunks_count.append((idx, count))

				# Sort by count descending
				file_chunks_count.sort(key=lambda x: x[1], reverse=True)
				primary_idx = file_chunks_count[0][0]

				# Move chunks containing this file to the primary group
				for idx in group_indices:
					if idx != primary_idx:
						# Find chunks containing this file
						chunks_to_move = [chunk for chunk in groups[idx].chunks if file in chunk.files]

						# Move chunks to primary group
						groups[primary_idx].chunks.extend(chunks_to_move)

						# Remove moved chunks from original group
						groups[idx].chunks = [chunk for chunk in groups[idx].chunks if file not in chunk.files]

				# Remove empty groups
				groups = [group for group in groups if group.chunks]

		return groups
