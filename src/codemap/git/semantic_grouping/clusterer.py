"""
Module for clustering diff chunks based on their embeddings.

This module provides functionality to group related code changes together based on their
semantic similarity, using vector embeddings and clustering algorithms. The clustering
process helps identify related changes that should be committed together.

Key components:
- DiffClusterer: Main class that implements clustering algorithms for diff chunks
- ClusteringParams: Type definition for parameters used by clustering algorithms

The module supports multiple clustering methods:
1. Agglomerative (hierarchical) clustering: Builds a hierarchy of clusters based on distances
   between embeddings, using a distance threshold to determine final cluster boundaries
2. DBSCAN: Density-based clustering that groups points in high-density regions,
   treating low-density points as noise/outliers

"""

import logging
from typing import TYPE_CHECKING, TypedDict, TypeVar, cast

import numpy as np

from codemap.git.diff_splitter import DiffChunk

if TYPE_CHECKING:
	from codemap.config import ConfigLoader

logger = logging.getLogger(__name__)


# Define a type for clustering parameters
class ClusteringParams(TypedDict, total=False):
	"""
	Type definition for clustering algorithm parameters.

	These parameters configure the behavior of the clustering algorithms:

	For agglomerative clustering:
	- n_clusters: Optional limit on number of clusters (None means no limit)
	- distance_threshold: Maximum distance for clusters to be merged (lower = more clusters)
	- metric: Distance metric to use (e.g., "precomputed" for precomputed distance matrix)
	- linkage: Strategy for calculating distances between clusters ("average", "single", etc.)

	For DBSCAN:
	- eps: Maximum distance between points in the same neighborhood
	- min_samples: Minimum points required to form a dense region
	- metric: Distance metric to use

	"""

	n_clusters: int | None
	distance_threshold: float | None
	metric: str
	linkage: str
	eps: float
	min_samples: int


T = TypeVar("T")


class DiffClusterer:
	"""
	Clusters diff chunks based on their semantic embeddings.

	This class provides methods to group related code changes by their semantic similarity,
	using vector embeddings and standard clustering algorithms from scikit-learn.

	Clustering helps identify code changes that are related to each other and should be
	grouped in the same commit, even if they appear in different files.

	The class supports multiple clustering algorithms:
	1. Agglomerative clustering: Hierarchical clustering that's good for finding natural
	   groupings without needing to specify the exact number of clusters
	2. DBSCAN: Density-based clustering that can identify outliers and works well with
	   irregularly shaped clusters

	"""

	def __init__(self, config_loader: "ConfigLoader", **kwargs: object) -> None:
		"""
		Initialize the clusterer.

		Args:
		    config_loader: ConfigLoader to use for configuration (follows DI pattern)
		    **kwargs: Additional parameters for the clustering algorithm:
		        - For agglomerative: distance_threshold, linkage, etc.
		        - For DBSCAN: eps, min_samples, etc.

		Raises:
		    ImportError: If scikit-learn is not installed

		"""
		self.config = config_loader.get.embedding.clustering
		self.method = self.config.method
		self.kwargs = kwargs

		# Import here to avoid making sklearn a hard dependency
		try:
			from sklearn.cluster import DBSCAN, AgglomerativeClustering
			from sklearn.metrics.pairwise import cosine_similarity

			self.AgglomerativeClustering = AgglomerativeClustering
			self.DBSCAN = DBSCAN
			self.cosine_similarity = cosine_similarity
		except ImportError as e:
			logger.exception("Failed to import scikit-learn. Please install it with: uv add scikit-learn")
			msg = "scikit-learn is required for clustering"
			raise ImportError(msg) from e

	def cluster(self, chunk_embeddings: list[tuple[DiffChunk, np.ndarray]]) -> list[list[DiffChunk]]:
		"""
		Cluster chunks based on their embeddings.

		              Process:
		              1. Extracts chunks and embeddings from input tuples
		              2. Computes a similarity matrix using cosine similarity
		              3. Converts similarity to distance matrix (1 - similarity)
		              4. Applies clustering algorithm based on the chosen method
		              5. Organizes chunks into clusters based on labels
		              6. Handles special cases like noise points in DBSCAN

		Args:
		    chunk_embeddings: List of (chunk, embedding) tuples where each embedding
		        is a numpy array representing the semantic vector of a code chunk

		Returns:
		    List of lists, where each inner list contains chunks in the same cluster.
		    With DBSCAN, noise points (label -1) are returned as individual single-item clusters.

		Examples:
		    >>> embedder = DiffEmbedder()
		    >>> chunk_embeddings = embedder.embed_chunks(diff_chunks)
		    >>> clusterer = DiffClusterer(method="agglomerative", distance_threshold=0.5)
		    >>> clusters = clusterer.cluster(chunk_embeddings)
		    >>> for i, cluster in enumerate(clusters):
		    ...     print(f"Cluster {i} has {len(cluster)} chunks")

		"""
		if not chunk_embeddings:
			return []

		# Extract chunks and embeddings
		chunks = [ce[0] for ce in chunk_embeddings]
		embeddings = np.array([ce[1] for ce in chunk_embeddings])

		# Compute similarity matrix (1 - cosine distance)
		similarity_matrix = self.cosine_similarity(embeddings)

		# Convert to distance matrix (1 - similarity)
		distance_matrix = 1 - similarity_matrix

		# Apply clustering
		if self.method == "agglomerative":
			# Default parameters if not provided
			params = {
				"n_clusters": None,
				"distance_threshold": self.config.agglomerative.distance_threshold,
				"metric": self.config.agglomerative.metric,
				"linkage": self.config.agglomerative.linkage,
			}
			params.update(cast("dict[str, float | str | None]", self.kwargs))

			clustering = self.AgglomerativeClustering(**params)
			labels = clustering.fit_predict(distance_matrix)

		elif self.method == "dbscan":
			# Default parameters if not provided
			params = {
				"eps": self.config.dbscan.eps,
				"min_samples": self.config.dbscan.min_samples,
				"metric": self.config.dbscan.metric,
			}
			params.update(cast("dict[str, float | int | str]", self.kwargs))

			clustering = self.DBSCAN(**params)
			labels = clustering.fit_predict(distance_matrix)

		else:
			msg = f"Unsupported clustering method: {self.method}"
			raise ValueError(msg)

		# Group chunks by cluster label
		clusters: dict[int, list[DiffChunk]] = {}
		labels_list: list[int] = labels.tolist()  # Convert numpy array to list for type safety
		for i, label in enumerate(labels_list):
			# Convert numpy integer to Python int
			label_key = int(label)
			if label_key not in clusters:
				clusters[label_key] = []
			clusters[label_key].append(chunks[i])

		# Convert to list of lists and handle noise points (-1 label in DBSCAN)
		result: list[list[DiffChunk]] = []
		for label, cluster_chunks in sorted(clusters.items()):
			if label != -1:  # Regular cluster
				result.append(cluster_chunks)
			else:  # Noise points - each forms its own cluster
				result.extend([[chunk] for chunk in cluster_chunks])

		return result
