"""
HTTP API server for the CodeMap daemon.

This module provides an HTTP API for interacting with the CodeMap
daemon. The API allows for executing tasks, checking status, and
managing jobs.

"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Annotated, Any, NoReturn

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from codemap.processor.pipeline import ProcessingPipeline
from codemap.utils.config_loader import ConfigLoader

# Security schemes
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


# Custom exceptions
class DaemonError(Exception):
	"""Base exception for daemon-related errors."""


class FeatureNotImplementedError(DaemonError):
	"""Exception raised when a requested feature is not implemented."""


class InvalidRequestError(DaemonError):
	"""Exception raised when a request is invalid."""


class ProcessingError(DaemonError):
	"""Exception raised when an error occurs during processing."""


class AuthenticationError(DaemonError):
	"""Exception raised when authentication fails."""


# API Request/Response models
class ErrorResponse(BaseModel):
	"""Model for standardized error responses."""

	error: str
	detail: str | None = None
	status_code: int = Field(default=500, description="HTTP status code")


class ProcessFileRequest(BaseModel):
	"""Model for file processing requests."""

	file_path: str


class AnalyzeRepoRequest(BaseModel):
	"""Model for repository analysis requests."""

	repo_path: str


class JobResponse(BaseModel):
	"""Model for job responses."""

	job_id: str


class StatusResponse(BaseModel):
	"""Model for status responses."""

	status: str
	jobs: dict[str, int]


class JobDetailResponse(BaseModel):
	"""Model for detailed job information."""

	id: str
	status: str
	created_at: float
	completed_at: float | None = None
	error: str | None = None


class APIKeyManager:
	"""Manages API keys for authentication."""

	def __init__(self, config: dict[str, Any]) -> None:
		"""
		Initialize the API key manager.

		Args:
		        config: Configuration dictionary

		"""
		self.logger = logging.getLogger("codemap.daemon.api.auth")
		self.api_keys: dict[str, dict[str, Any]] = {}
		self.key_file: Path | None = None

		# Load API keys from config
		self._load_api_keys(config)

	def _load_api_keys(self, _config: dict[str, Any]) -> None:
		"""
		Load API keys from configuration.

		Args:
		        _config: Configuration dictionary

		"""
		# Use the helper method from ConfigLoader for consistent access
		config_loader = ConfigLoader()
		api_config = config_loader.get_api_config()
		auth_config = api_config.get("auth", {})

		# Check if authentication is enabled
		self.auth_enabled = auth_config.get("enabled", False)
		if not self.auth_enabled:
			self.logger.info("API authentication is disabled")
			return

		# Load API keys
		keys = auth_config.get("api_keys", [])
		for key_info in keys:
			if isinstance(key_info, dict) and "key" in key_info and "name" in key_info:
				key = key_info["key"]
				self.api_keys[key] = {
					"name": key_info["name"],
					"created_at": key_info.get("created_at", time.time()),
					"scopes": key_info.get("scopes", ["*"]),
				}

		# Set the key file path for persistent storage
		key_file = auth_config.get("key_file")
		if key_file:
			self.key_file = Path(key_file).expanduser().resolve()
			self.logger.info("API key file set to: %s", self.key_file)

		# If no keys are configured but auth is enabled, generate a default key
		if not self.api_keys and self.auth_enabled:
			self.logger.warning("No API keys configured but authentication is enabled. Generating a default key.")
			self.generate_key("default", ["*"])

	def generate_key(self, name: str, scopes: list[str] | None = None) -> str:
		"""
		Generate a new API key.

		Args:
		        name: A name for the key (for reference)
		        scopes: Optional list of scopes for this key

		Returns:
		        str: The generated API key

		"""
		if scopes is None:
			scopes = ["*"]  # Default to all scopes if none provided

		# Generate a unique key
		key = secrets.token_urlsafe(32)

		# Store the key details
		self.api_keys[key] = {
			"name": name,
			"scopes": scopes,
			"created_at": time.time(),
		}

		# Save keys to the persistent storage
		self._save_keys()

		return key

	def validate_key(self, api_key: str) -> bool:
		"""
		Validate an API key.

		Args:
		        api_key: The API key to validate

		Returns:
		        bool: True if the key is valid, False otherwise

		"""
		if not self.auth_enabled:
			return True

		return api_key in self.api_keys

	def _save_keys(self) -> None:
		"""Save API keys to the persistent storage file if available."""
		if not self.key_file:
			return

		try:
			# Prepare keys for saving (with all necessary info)
			keys_to_save = []
			for key, info in self.api_keys.items():
				keys_to_save.append(
					{"key": key, "name": info["name"], "created_at": info["created_at"], "scopes": info["scopes"]}
				)

			# Write to file
			with Path(self.key_file).open("w") as f:
				json.dump({"api_keys": keys_to_save}, f, indent=2)

			self.logger.debug("Saved %d API keys to %s", len(keys_to_save), self.key_file)
		except Exception:
			self.logger.exception("Failed to save API keys to %s", self.key_file)

	def get_key_info(self, api_key: str) -> dict[str, Any] | None:
		"""
		Get information about an API key.

		Args:
		        api_key: The API key to get information for

		Returns:
		        Optional[Dict[str, Any]]: Key information or None if key is invalid

		"""
		return self.api_keys.get(api_key)


class APIServer:
	"""
	HTTP API server for the CodeMap daemon.

	This class implements a FastAPI-based server that exposes the CodeMap
	daemon's functionality via a REST API.

	"""

	def __init__(self, host: str, port: int, pipeline: ProcessingPipeline, config: dict[str, Any]) -> None:
		"""
		Initialize the API server.

		Args:
		    host: Hostname or IP address to bind to
		    port: Port number to listen on
		    pipeline: The ProcessingPipeline instance to expose
		    config: Configuration dictionary

		"""
		self.host = host
		self.port = port
		self.pipeline = pipeline
		self.config = config
		self.logger = logging.getLogger("codemap.daemon.api")

		# Set up API key manager
		self.key_manager = APIKeyManager(config)

		# Set up FastAPI
		self.app = self._create_app()
		self.server: uvicorn.Server | None = None
		self.server_thread: threading.Thread | None = None

	def get_api_key(self, api_key_header: str = Security(API_KEY_HEADER)) -> str:
		"""
		Dependency for routes to validate API key from header.

		Args:
		    api_key_header: API key from request header

		Returns:
		    str: Validated API key

		Raises:
		    HTTPException: If authentication failed or API key missing

		"""
		# If auth is disabled, return a dummy key
		if not self.key_manager.auth_enabled:
			return "disabled"

		# If auth is enabled, require a valid key
		if not api_key_header:
			raise HTTPException(
				status_code=status.HTTP_401_UNAUTHORIZED,
				detail="Missing API key",
				headers={"WWW-Authenticate": "ApiKey"},
			)

		if not self.key_manager.validate_key(api_key_header):
			raise HTTPException(
				status_code=status.HTTP_403_FORBIDDEN,
				detail="Invalid API key",
				headers={"WWW-Authenticate": "ApiKey"},
			)

		return api_key_header

	def _create_app(self) -> FastAPI:
		"""
		Create and configure the FastAPI application.

		Returns:
		    FastAPI: Configured FastAPI application

		"""
		app = FastAPI(
			title="CodeMap API",
			description="API for interacting with the CodeMap daemon",
			version="0.1.0",
		)

		# Configure CORS
		cors_config = self.config.get("server", {}).get("cors", {})
		if cors_config.get("allow_cors", True):
			origins = cors_config.get("origins", ["*"])
			app.add_middleware(
				CORSMiddleware,
				allow_origins=origins,
				allow_credentials=True,
				allow_methods=["*"],
				allow_headers=["*"],
			)

		# Set up processing pipeline for routes
		pipeline = self.pipeline

		# Add custom exception handlers
		@app.exception_handler(FeatureNotImplementedError)
		async def feature_not_implemented_handler(_request: Request, exc: FeatureNotImplementedError) -> JSONResponse:
			"""Handle FeatureNotImplementedError exceptions."""
			return JSONResponse(
				status_code=status.HTTP_501_NOT_IMPLEMENTED,
				content={"error": "Feature not implemented", "detail": str(exc)},
			)

		@app.exception_handler(InvalidRequestError)
		async def invalid_request_handler(_request: Request, exc: InvalidRequestError) -> JSONResponse:
			"""Handle InvalidRequestError exceptions."""
			return JSONResponse(
				status_code=status.HTTP_400_BAD_REQUEST,
				content={"error": "Invalid request", "detail": str(exc)},
			)

		@app.exception_handler(ProcessingError)
		async def processing_error_handler(_request: Request, exc: ProcessingError) -> JSONResponse:
			"""Handle ProcessingError exceptions."""
			return JSONResponse(
				status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
				content={"error": "Processing error", "detail": str(exc)},
			)

		@app.exception_handler(AuthenticationError)
		async def auth_error_handler(_request: Request, exc: AuthenticationError) -> JSONResponse:
			"""Handle AuthenticationError exceptions."""
			return JSONResponse(
				status_code=status.HTTP_401_UNAUTHORIZED,
				content={"error": "Authentication error", "detail": str(exc)},
				headers={"WWW-Authenticate": "ApiKey"},
			)

		@app.exception_handler(Exception)
		async def general_exception_handler(_request: Request, _exc: Exception) -> JSONResponse:
			"""Handle unexpected exceptions."""
			self.logger.exception("Unexpected error")
			return JSONResponse(
				status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
				content={"error": "Internal server error", "detail": "An unexpected error occurred"},
			)

		def feature_not_implemented(detail: str) -> NoReturn:
			"""
			Raise FeatureNotImplementedError with the given detail.

			Args:
			        detail: Error detail message

			Raises:
			        FeatureNotImplementedError: Always

			"""
			raise FeatureNotImplementedError(detail)

		def check_pipeline_method(method_name: str, required: bool = True) -> bool:
			"""
			Check if a method exists on the pipeline.

			Args:
			        method_name: Name of the method to check
			        required: Whether the method is required

			Returns:
			        bool: True if the method exists, False otherwise

			"""
			has_method = hasattr(pipeline, method_name) and callable(getattr(pipeline, method_name))
			if not has_method and required:
				feature_not_implemented(f"Pipeline does not implement {method_name}")
			return has_method

		# Health check endpoint (no auth required)
		@app.get("/health")
		async def health_check() -> dict:
			"""Get health status of the daemon."""
			return {"status": "ok"}

		# API endpoints
		@app.get("/api/status", response_model=StatusResponse)
		async def get_status(_api_key: Annotated[str, Depends(self.get_api_key)]) -> dict:
			"""Get the current daemon status."""
			active_jobs = pipeline.get_active_jobs() if check_pipeline_method("get_active_jobs", required=False) else []
			completed_jobs = (
				pipeline.get_completed_jobs() if check_pipeline_method("get_completed_jobs", required=False) else []
			)
			failed_jobs = pipeline.get_failed_jobs() if check_pipeline_method("get_failed_jobs", required=False) else []

			return {
				"status": "running",
				"jobs": {
					"active": len(active_jobs),
					"completed": len(completed_jobs),
					"failed": len(failed_jobs),
				},
			}

		@app.get("/api/jobs/{job_id}", response_model=JobDetailResponse)
		async def get_job(job_id: str, _api_key: Annotated[str, Depends(self.get_api_key)]) -> dict:
			"""
			Get details of a specific job.

			Args:
			        job_id: Job ID to retrieve
			        _api_key: API key for authentication

			"""
			if not check_pipeline_method("get_job"):
				feature_not_implemented("Job tracking not implemented")

			job = pipeline.get_job(job_id)
			if not job:
				raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

			return job

		@app.get("/api/jobs")
		async def list_jobs(
			_api_key: Annotated[str, Depends(self.get_api_key)], status_filter: str | None = None
		) -> dict:
			"""
			List all jobs, optionally filtered by status.

			Args:
			        _api_key: API key for authentication
			        status_filter: Filter jobs by status

			"""
			# Check that all required job tracking methods are available
			if not (
				check_pipeline_method("get_active_jobs", required=False)
				and check_pipeline_method("get_completed_jobs", required=False)
				and check_pipeline_method("get_failed_jobs", required=False)
			):
				feature_not_implemented("Job tracking not implemented")

			jobs = []

			if status_filter == "active" or not status_filter:
				jobs.extend(pipeline.get_active_jobs())

			if status_filter == "completed" or not status_filter:
				jobs.extend(pipeline.get_completed_jobs())

			if status_filter == "failed" or not status_filter:
				jobs.extend(pipeline.get_failed_jobs())

			return {"jobs": jobs}

		@app.post("/api/process", response_model=JobResponse)
		async def process_file(
			request: ProcessFileRequest, _api_key: Annotated[str, Depends(self.get_api_key)]
		) -> dict:
			"""
			Process a file.

			Args:
			        request: File processing request
			        _api_key: API key for authentication

			"""
			file_path = request.file_path

			# Validate file path
			path = Path(file_path).expanduser().resolve()

			def _check_path() -> tuple[bool, str | None]:
				"""Check if the path exists and is a file."""
				if not path.exists():
					return False, f"File not found: {file_path}"
				if not path.is_file():
					return False, f"Not a file: {file_path}"
				return True, None

			def _validate_path() -> None:
				"""Validate the path and raise an exception if invalid."""
				is_valid, error_msg = _check_path()
				if not is_valid:
					raise InvalidRequestError(error_msg)

			try:
				_validate_path()
			except Exception as e:
				if not isinstance(e, InvalidRequestError):
					self.logger.exception("Error validating file path: %s", file_path)
					msg = f"Invalid file path: {e}"
					raise InvalidRequestError(msg) from e
				raise

			# Process the file
			try:
				if not check_pipeline_method("process_file", required=False):
					feature_not_implemented("File processing not implemented")

				job_id = pipeline.process_file(str(path))
				return {"job_id": job_id}
			except Exception as e:
				self.logger.exception("Error processing file: %s", file_path)
				msg = f"Error processing file: {e}"
				raise ProcessingError(msg) from e

		@app.post("/api/analyze", response_model=JobResponse)
		async def analyze_repository(
			request: AnalyzeRepoRequest, _api_key: Annotated[str, Depends(self.get_api_key)]
		) -> dict:
			"""
			Analyze a repository.

			Args:
			        request: Repository analysis request
			        _api_key: API key for authentication

			"""
			repo_path = request.repo_path

			# Validate repository path
			path = Path(repo_path).expanduser().resolve()

			def _check_path() -> tuple[bool, str | None]:
				"""Check if the path exists and is a directory."""
				if not path.exists():
					return False, f"Repository not found: {repo_path}"
				if not path.is_dir():
					return False, f"Not a directory: {repo_path}"
				return True, None

			def _validate_path() -> None:
				"""Validate the path and raise an exception if invalid."""
				is_valid, error_msg = _check_path()
				if not is_valid:
					raise InvalidRequestError(error_msg)

			try:
				_validate_path()
			except Exception as e:
				if not isinstance(e, InvalidRequestError):
					self.logger.exception("Error validating repository path: %s", repo_path)
					msg = f"Invalid repository path: {e}"
					raise InvalidRequestError(msg) from e
				raise

			try:
				# Check if the process_repository method exists in the pipeline
				if not check_pipeline_method("process_repository", required=True):
					feature_not_implemented("Repository analysis not implemented")

				job_id = pipeline.process_repository(path)
				return {"job_id": job_id}
			except Exception as e:
				self.logger.exception("Error processing repository: %s", repo_path)
				msg = f"Error processing repository: {e}"
				raise ProcessingError(msg) from e

		# API key management endpoints
		@app.get("/api/keys")
		async def list_api_keys(_api_key: Annotated[str, Depends(self.get_api_key)]) -> dict:
			"""
			List all API keys.

			Args:
			        _api_key: API key for authentication

			"""
			if not self.key_manager.auth_enabled:
				return {"api_keys": []}

			# Return a list of keys (without exposing the actual key values)
			keys = []
			for key, info in self.key_manager.api_keys.items():
				# Create a masked version of the key for display
				masked_key = key[:4] + "..." + key[-4:]

				keys.append(
					{
						"id": key,  # Using the full key as ID
						"name": info["name"],
						"masked_key": masked_key,
						"created_at": info["created_at"],
						"scopes": info["scopes"],
					}
				)

			return {"api_keys": keys}

		@app.post("/api/keys")
		async def create_api_key(
			_api_key: Annotated[str, Depends(self.get_api_key)], name: str, scopes: list[str] | None = None
		) -> dict:
			"""
			Create a new API key.

			Args:
			        _api_key: API key for authentication
			        name: Name for the new key
			        scopes: List of scopes for the key

			"""
			if not self.key_manager.auth_enabled:
				return {"error": "API key authentication is disabled"}

			new_key = self.key_manager.generate_key(name, scopes)
			info = self.key_manager.get_key_info(new_key)
			if not info:
				msg = "Failed to create API key"
				raise ProcessingError(msg)

			return {
				"key": new_key,
				"name": info["name"],
				"created_at": info["created_at"],
				"scopes": info["scopes"],
			}

		self.logger.debug("Created FastAPI application with %d routes", len(app.routes))
		return app

	def start(self) -> None:
		"""
		Start the API server in a background thread.

		Raises:
		        RuntimeError: If the server is already started

		"""
		if self.server_thread and self.server_thread.is_alive():
			msg = "API server is already running"
			raise RuntimeError(msg)

		self.logger.info("Starting API server on %s:%d", self.host, self.port)

		config = uvicorn.Config(
			app=self.app,
			host=self.host,
			port=self.port,
			log_level="info",
		)
		self.server = uvicorn.Server(config)

		# Start the server in a background thread
		self.server_thread = threading.Thread(target=self.server.run, daemon=True)
		self.server_thread.start()

		self.logger.info("API server started, waiting for requests")

	def stop(self) -> None:
		"""
		Stop the API server.

		Raises:
		        RuntimeError: If the server is not running

		"""
		if not self.server_thread or not self.server_thread.is_alive():
			msg = "API server is not running"
			raise RuntimeError(msg)

		self.logger.info("Stopping API server")

		if self.server:
			self.server.should_exit = True
			self.server_thread.join(timeout=1.0)

		if self.server_thread.is_alive():
			self.logger.warning("API server thread did not exit gracefully, forcing shutdown")
		else:
			self.logger.info("API server stopped")
