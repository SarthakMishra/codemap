"""
Client for interacting with the CodeMap daemon.

This module provides a client interface for communicating with the
CodeMap daemon via its HTTP API.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from pathlib import Path

# Optional requests dependency
REQUESTS_AVAILABLE = False
try:
	# We use a conditional import to handle optional dependency
	import requests  # type: ignore[import] # noqa: F401

	REQUESTS_AVAILABLE = True
except ImportError:
	pass

# HTTP Status codes as constants
HTTP_NOT_FOUND = 404


class DaemonClient:
	"""
	Client for interacting with the CodeMap daemon.

	This class provides methods for communicating with the CodeMap daemon
	via its HTTP API.

	"""

	def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
		"""
		Initialize the daemon client.

		Args:
		    host: The hostname or IP address of the daemon (default: 127.0.0.1)
		    port: The port number of the daemon (default: 8765)

		"""
		self.host = host
		self.port = port
		self.base_url = f"http://{host}:{port}/api"
		self.logger = logging.getLogger("codemap.daemon.client")

		if not REQUESTS_AVAILABLE:
			self.logger.warning("The 'requests' package is not installed. API calls will fail.")

	def _check_requests_available(self) -> None:
		"""Check if requests is available and raise an exception if not."""
		if not REQUESTS_AVAILABLE:
			msg = "The 'requests' package is required for API calls"
			raise RuntimeError(msg)

	def check_status(self) -> dict[str, Any]:
		"""
		Check the status of the daemon.

		Returns:
		    Dict[str, Any]: Status information from the daemon

		Raises:
		    RuntimeError: If the daemon is not running or cannot be contacted

		"""
		self._check_requests_available()

		# Import is checked above
		import requests
		from requests import RequestException

		try:
			response = requests.get(f"{self.base_url}/status", timeout=5)
			response.raise_for_status()
			return response.json()
		except RequestException as e:
			self.logger.exception("Failed to contact daemon")
			msg = f"Failed to contact daemon: {e}"
			raise RuntimeError(msg) from e

	def get_job_status(self, job_id: str) -> dict[str, Any]:
		"""
		Get the status of a specific job.

		Args:
		    job_id: ID of the job to check

		Returns:
		    Dict[str, Any]: Job status information

		Raises:
		    RuntimeError: If the daemon is not running or the job is not found

		"""
		self._check_requests_available()

		import requests
		from requests import RequestException

		try:
			response = requests.get(f"{self.base_url}/jobs/{job_id}", timeout=5)
			response.raise_for_status()
			return response.json()
		except RequestException as e:
			if hasattr(e, "response") and e.response is not None and e.response.status_code == HTTP_NOT_FOUND:
				msg = f"Job {job_id} not found"
				raise RuntimeError(msg) from e

			self.logger.exception("Failed to get job status")
			msg = f"Failed to get job status: {e}"
			raise RuntimeError(msg) from e

	def list_jobs(self, status_filter: str | None = None) -> list[dict[str, Any]]:
		"""
		List all jobs, optionally filtered by status.

		Args:
		    status_filter: Filter jobs by status (active, completed, failed)

		Returns:
		    List[Dict[str, Any]]: List of jobs matching the filter

		Raises:
		    RuntimeError: If the daemon is not running or cannot be contacted

		"""
		self._check_requests_available()

		import requests
		from requests import RequestException

		try:
			url = f"{self.base_url}/jobs"
			params = {}
			if status_filter:
				params["status"] = status_filter

			response = requests.get(url, params=params, timeout=5)
			response.raise_for_status()
			return response.json().get("jobs", [])
		except RequestException as e:
			self.logger.exception("Failed to list jobs")
			msg = f"Failed to list jobs: {e}"
			raise RuntimeError(msg) from e

	def process_file(self, file_path: str | Path) -> str:
		"""
		Request the daemon to process a file.

		Args:
		    file_path: Path to the file to process

		Returns:
		    str: Job ID for tracking the processing task

		Raises:
		    RuntimeError: If the daemon is not running or cannot process the file

		"""
		self._check_requests_available()

		import requests
		from requests import RequestException

		try:
			# Convert Path to string if needed
			file_path_str = str(file_path)

			response = requests.post(f"{self.base_url}/process", json={"file_path": file_path_str}, timeout=5)
			response.raise_for_status()
			return response.json().get("job_id")
		except RequestException as e:
			self.logger.exception("Failed to process file")
			msg = f"Failed to process file: {e}"
			raise RuntimeError(msg) from e

	def analyze_repository(self, repo_path: str | Path) -> str:
		"""
		Request the daemon to analyze a repository.

		Args:
		    repo_path: Path to the repository to analyze

		Returns:
		    str: Job ID for tracking the analysis task

		Raises:
		    RuntimeError: If the daemon is not running or cannot analyze the repository

		"""
		self._check_requests_available()

		import requests
		from requests import RequestException

		try:
			# Convert Path to string if needed
			repo_path_str = str(repo_path)

			response = requests.post(f"{self.base_url}/analyze", json={"repo_path": repo_path_str}, timeout=5)
			response.raise_for_status()
			return response.json().get("job_id")
		except RequestException as e:
			self.logger.exception("Failed to analyze repository")
			msg = f"Failed to analyze repository: {e}"
			raise RuntimeError(msg) from e
