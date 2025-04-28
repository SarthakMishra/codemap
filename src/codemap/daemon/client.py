"""
Client for interacting with the CodeMap daemon.

This module provides a client interface for communicating with the
CodeMap daemon via its HTTP API.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar

from codemap.utils.config_loader import ConfigLoader

T = TypeVar("T")

# Optional requests dependency
REQUESTS_AVAILABLE = False
try:
	# We use a conditional import to handle optional dependency
	import requests  # type: ignore[import] # noqa: F401

	REQUESTS_AVAILABLE = True
except ImportError:
	pass


# Optional requests-unixsocket dependency
REQUESTS_UNIX_AVAILABLE = False
try:
	import requests_unixsocket  # type: ignore[import] # noqa: F401

	REQUESTS_UNIX_AVAILABLE = True
except ImportError:
	pass


# HTTP Status codes as constants
HTTP_NOT_FOUND = 404

# Error messages
ERR_NO_JOB_ID = "No job ID returned from server"


class DaemonClient:
	"""
	Client for interacting with the CodeMap daemon.

	This class provides methods for communicating with the CodeMap daemon
	via its HTTP API.

	"""

	def __init__(
		self,
		host: str | None = None,
		port: int | None = None,
		socket_path: str | None = None,
		config_file: str | Path | None = None,
	) -> None:
		"""
		Initialize the daemon client.

		Args:
		    host: The hostname or IP address of the daemon (default from config or 127.0.0.1)
		    port: The port number of the daemon (default from config or 8765)
		    socket_path: Path to Unix socket for communication (default from config)
		    config_file: Path to configuration file (optional)

		"""
		# Load configuration if available
		self.config_loader = ConfigLoader(str(config_file) if config_file else None)

		# Get API config with connection details
		api_config = self.config_loader.get_api_config()

		# Determine connection method and details
		self.use_socket = socket_path is not None or api_config.get("use_socket", False)

		# Socket path takes precedence if provided
		if self.use_socket:
			self.socket_path = socket_path or api_config.get("socket_file")
			if not self.socket_path:
				msg = "Socket path not provided and not found in configuration"
				raise ValueError(msg)

			# Expand path if needed
			self.socket_path = Path(self.socket_path).expanduser()

			# Set the base URL for Unix socket
			# Convert Path object to string before using string replace method
			socket_path_str = str(self.socket_path)
			self.base_url = f"http+unix://{socket_path_str.replace('/', '%2F')}/api"
			self.host = None
			self.port = None
		else:
			# Use HTTP connection
			self.host = host or api_config.get("host", "127.0.0.1")
			self.port = port or api_config.get("port", 8765)
			self.base_url = f"http://{self.host}:{self.port}/api"
			self.socket_path = None

		self.logger = logging.getLogger("codemap.daemon.client")

		if not REQUESTS_AVAILABLE:
			self.logger.warning("The 'requests' package is not installed. API calls will fail.")

		self.logger.debug("Initialized daemon client: %s", self.base_url)

	def _check_requests_available(self) -> None:
		"""Check if requests is available and raise an exception if not."""
		if not REQUESTS_AVAILABLE:
			msg = "The 'requests' package is required for API calls"
			raise RuntimeError(msg)

		# For Unix socket connections, check if requests-unixsocket is available
		if self.use_socket and not REQUESTS_UNIX_AVAILABLE:
			msg = "The 'requests-unixsocket' package is required for Unix socket connections"
			self.logger.error(msg)
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

		try:
			# Handle Unix socket connection or regular HTTP
			if self.use_socket:
				import requests_unixsocket

				with requests_unixsocket.Session() as session:
					response = session.get(f"{self.base_url}/status", timeout=5)
			else:
				import requests

				response = requests.get(f"{self.base_url}/status", timeout=5)

			response.raise_for_status()
			return response.json()
		except Exception as e:
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
		try:
			return self._make_request("get", f"jobs/{job_id}")
		except RuntimeError as e:
			if "not found" in str(e).lower():
				msg = f"Job {job_id} not found"
				raise RuntimeError(msg) from e
			raise

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
		params = {}
		if status_filter:
			params["status"] = status_filter

		response = self._make_request("get", "jobs", params=params)
		return response.get("jobs", [])

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
		# Convert Path to string if needed
		file_path_str = str(file_path)

		response = self._make_request("post", "process", json={"file_path": file_path_str})
		job_id = response.get("job_id")
		if not job_id:
			raise RuntimeError(ERR_NO_JOB_ID)
		return str(job_id)

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
		# Convert Path to string if needed
		repo_path_str = str(repo_path)

		response = self._make_request("post", "analyze", json={"repo_path": repo_path_str})
		job_id = response.get("job_id")
		if not job_id:
			raise RuntimeError(ERR_NO_JOB_ID)
		return str(job_id)

	def _make_request(
		self, method: str, endpoint: str, **kwargs: dict[str, object] | list[object] | str | float | bool | None
	) -> dict[str, Any]:
		"""
		Make a request to the daemon API.

		Args:
		    method: HTTP method (get, post, etc.)
		    endpoint: API endpoint
		    **kwargs: Additional arguments for the request

		Returns:
		    Response data as JSON

		Raises:
		    RuntimeError: If the request fails

		"""
		self._check_requests_available()

		try:
			# Handle Unix socket connection or regular HTTP
			if self.use_socket:
				import requests_unixsocket

				with requests_unixsocket.Session() as session:
					response = getattr(session, method.lower())(
						f"{self.base_url}/{endpoint}", **kwargs, timeout=kwargs.get("timeout", 5)
					)
			else:
				import requests

				response = getattr(requests, method.lower())(
					f"{self.base_url}/{endpoint}", **kwargs, timeout=kwargs.get("timeout", 5)
				)

			response.raise_for_status()
			return response.json()
		except Exception as e:
			# Handle HTTP exceptions specifically
			if hasattr(e, "__dict__") and "response" in e.__dict__:
				response_obj = e.__dict__["response"]
				if hasattr(response_obj, "status_code") and response_obj.status_code == HTTP_NOT_FOUND:
					msg = f"Endpoint /{endpoint} not found"
					raise RuntimeError(msg) from e

			self.logger.exception("Failed to make %s request to /%s", method, endpoint)
			msg = f"Request failed: {e}"
			raise RuntimeError(msg) from e
