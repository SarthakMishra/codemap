"""Utilities for working with Docker containers directly via Python."""

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound

if TYPE_CHECKING:
	from docker.models.containers import Container

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60  # seconds
HTTP_OK = 200  # HTTP status code for OK

# Container configuration constants
QDRANT_IMAGE = "qdrant/qdrant:latest"
QDRANT_CONTAINER_NAME = "codemap-qdrant"
QDRANT_HOST_PORT = 6333
QDRANT_HTTP_PORT = 6333
QDRANT_GRPC_PORT = 6334
QDRANT_STORAGE_PATH = ".codemap_cache/qdrant"

POSTGRES_IMAGE = "postgres:latest"
POSTGRES_CONTAINER_NAME = "codemap-postgres"
POSTGRES_HOST_PORT = 5432
POSTGRES_ENV = {"POSTGRES_PASSWORD": "postgres", "POSTGRES_USER": "postgres", "POSTGRES_DB": "codemap"}
POSTGRES_STORAGE_PATH = ".codemap_cache/postgres_data"


async def is_docker_running() -> bool:
	"""Check if the Docker daemon is running."""

	def _check_docker_sync() -> bool:
		client = None
		try:
			client = docker.from_env()
			client.ping()
			return True
		except DockerException:
			return False
		finally:
			if client is not None:
				client.close()
		return False

	return await asyncio.to_thread(_check_docker_sync)


async def is_container_running(container_name: str) -> bool:
	"""
	Check if a specific Docker container is running.

	Args:
	    container_name: Name of the container to check

	Returns:
	    True if the container is running, False otherwise

	"""

	def _check_container_sync(name: str) -> bool:
		client = None
		try:
			client = docker.from_env()
			try:
				container = cast("Container", client.containers.get(name))
				return container.status == "running"
			except NotFound:
				return False
		except DockerException:
			logger.exception("Docker error while checking container status for %s", name)
			return False
		finally:
			if client is not None:
				client.close()
		return False

	return await asyncio.to_thread(_check_container_sync, container_name)


async def pull_image_if_needed(image_name: str) -> bool:
	"""
	Pull a Docker image if it's not already available locally.

	Args:
	    image_name: Name of the image to pull

	Returns:
	    True if successful, False otherwise

	"""

	def _pull_image_sync(name: str) -> bool:
		client = None
		try:
			client = docker.from_env()
			try:
				client.images.get(name)
				logger.info(f"Image {name} already exists locally")
				return True
			except ImageNotFound:
				logger.info(f"Pulling image {name}...")
				try:
					client.images.pull(name)
					logger.info(f"Successfully pulled image {name}")
					return True
				except APIError:
					logger.exception(f"Failed to pull image {name}")
					return False
		except DockerException:  # Catch potential errors from images.get()
			logger.exception(f"Docker error while checking image {name}")
			return False
		finally:
			if client is not None:
				client.close()
		return False

	return await asyncio.to_thread(_pull_image_sync, image_name)


async def ensure_volume_path_exists(path: str) -> None:
	"""
	Ensure the host path for a volume exists.

	Args:
	    path: Path to ensure exists

	"""

	def _ensure_path_sync(p: str) -> None:
		Path(p).mkdir(parents=True, exist_ok=True)

	await asyncio.to_thread(_ensure_path_sync, path)


async def start_qdrant_container() -> bool:
	"""
	Start the Qdrant container.

	Returns:
	    True if successful, False otherwise

	"""

	def _start_qdrant_sync() -> bool:
		client = None
		try:
			client = docker.from_env()

			# Ensure image is available (This function is already async, called separately)
			# if not await pull_image_if_needed(QDRANT_IMAGE):
			# 	return False

			# Ensure storage directory exists (This function is already async, called separately)
			# await ensure_volume_path_exists(QDRANT_STORAGE_PATH)

			# Check if container already exists
			try:
				container = cast("Container", client.containers.get(QDRANT_CONTAINER_NAME))
				if container.status == "running":
					logger.info(f"Container {QDRANT_CONTAINER_NAME} is already running")
					return True

				# If container exists but is not running, start it
				logger.info(f"Starting existing container {QDRANT_CONTAINER_NAME}")
				container.start()
				logger.info(f"Started container {QDRANT_CONTAINER_NAME}")
				return True

			except NotFound:
				# Container doesn't exist, create and start it
				abs_storage_path = str(Path(QDRANT_STORAGE_PATH).absolute())

				logger.info(f"Creating and starting container {QDRANT_CONTAINER_NAME}")

				# Define volume binding in Docker SDK format
				volumes: list[str] = [f"{abs_storage_path}:/qdrant/storage:rw"]

				# Define port mapping
				ports: dict[str, int | list[int] | tuple[str, int] | None] = {
					f"{QDRANT_HTTP_PORT}/tcp": QDRANT_HOST_PORT,
					f"{QDRANT_GRPC_PORT}/tcp": QDRANT_GRPC_PORT,
				}

				restart_policy = {"Name": "always"}

				client.containers.run(
					image=QDRANT_IMAGE,
					name=QDRANT_CONTAINER_NAME,
					ports=ports,
					volumes=volumes,
					detach=True,
					restart_policy=restart_policy,  # type: ignore[arg-type]
				)
				logger.info(f"Created and started container {QDRANT_CONTAINER_NAME}")
				return True

		except DockerException:
			logger.exception("Docker error while starting Qdrant container")
			return False
		finally:
			if client is not None:
				client.close()
		return False

	# Ensure image is available
	if not await pull_image_if_needed(QDRANT_IMAGE):
		return False

	# Ensure storage directory exists
	await ensure_volume_path_exists(QDRANT_STORAGE_PATH)

	return await asyncio.to_thread(_start_qdrant_sync)


async def start_postgres_container() -> bool:
	"""
	Start the PostgreSQL container.

	Returns:
	    True if successful, False otherwise

	"""

	def _start_postgres_sync() -> bool:
		client = None
		try:
			client = docker.from_env()

			# Ensure image is available (This function is already async, called separately)
			# if not await pull_image_if_needed(POSTGRES_IMAGE):
			# 	return False

			# Ensure storage directory exists (This function is already async, called separately)
			# await ensure_volume_path_exists(POSTGRES_STORAGE_PATH)

			# Check if container already exists
			try:
				container = cast("Container", client.containers.get(POSTGRES_CONTAINER_NAME))
				if container.status == "running":
					logger.info(f"Container {POSTGRES_CONTAINER_NAME} is already running")
					return True

				# If container exists but is not running, start it
				logger.info(f"Starting existing container {POSTGRES_CONTAINER_NAME}")
				container.start()
				logger.info(f"Started container {POSTGRES_CONTAINER_NAME}")
				return True

			except NotFound:
				# Container doesn't exist, create and start it
				abs_storage_path = str(Path(POSTGRES_STORAGE_PATH).absolute())

				logger.info(f"Creating and starting container {POSTGRES_CONTAINER_NAME}")

				# Define volume binding in Docker SDK format
				volumes: list[str] = [f"{abs_storage_path}:/var/lib/postgresql/data:rw"]

				# Define port mapping
				ports: dict[str, int | list[int] | tuple[str, int] | None] = {"5432/tcp": POSTGRES_HOST_PORT}

				restart_policy = {"Name": "always"}

				client.containers.run(
					image=POSTGRES_IMAGE,
					name=POSTGRES_CONTAINER_NAME,
					ports=ports,
					volumes=volumes,
					environment=POSTGRES_ENV,
					detach=True,
					restart_policy=restart_policy,  # type: ignore[arg-type]
				)
				logger.info(f"Created and started container {POSTGRES_CONTAINER_NAME}")
				return True

		except DockerException:
			logger.exception("Docker error while starting PostgreSQL container")
			return False
		finally:
			if client is not None:
				client.close()
		return False

	# Ensure image is available
	if not await pull_image_if_needed(POSTGRES_IMAGE):
		return False

	# Ensure storage directory exists
	await ensure_volume_path_exists(POSTGRES_STORAGE_PATH)

	return await asyncio.to_thread(_start_postgres_sync)


async def check_qdrant_health(url: str = f"http://localhost:{QDRANT_HOST_PORT}") -> bool:
	"""
	Check if Qdrant service is healthy and ready to accept connections.

	Args:
	    url: Base URL of the Qdrant service

	Returns:
	    True if Qdrant is healthy, False otherwise

	"""
	from httpx import AsyncClient, RequestError

	health_url = f"{url}/healthz"
	start_time = time.time()

	async with AsyncClient() as client:
		try:
			async with asyncio.timeout(DEFAULT_TIMEOUT):
				while True:
					try:
						response = await client.get(health_url)
						if response.status_code == HTTP_OK:
							logger.info("Qdrant service is healthy (responded 200 OK)")
							return True
					except RequestError:
						pass

					if time.time() - start_time >= DEFAULT_TIMEOUT:
						break

					# Wait before trying again
					await asyncio.sleep(1)
		except TimeoutError:
			pass

	logger.error(f"Qdrant service did not become healthy within {DEFAULT_TIMEOUT} seconds")
	return False


async def ensure_qdrant_running(
	wait_for_health: bool = True, qdrant_url: str = f"http://localhost:{QDRANT_HOST_PORT}"
) -> tuple[bool, str]:
	"""
	Ensure the Qdrant container is running, starting it if needed.

	Args:
	    wait_for_health: Whether to wait for Qdrant to be healthy
	    qdrant_url: URL of the Qdrant service

	Returns:
	    Tuple of (success, message)

	"""
	if not await is_docker_running():
		return False, "Docker daemon is not running"

	# Check if Qdrant service is already running
	qdrant_running = False

	from httpx import AsyncClient, HTTPError, RequestError

	try:
		# Try a direct HTTP request first to see if Qdrant is up
		async with AsyncClient(timeout=3.0) as client:
			try:
				response = await client.get(f"{qdrant_url}/health")
				if response.status_code == HTTP_OK:
					logger.info("Qdrant is already available via HTTP")
					qdrant_running = True
			except RequestError:
				# HTTP request failed, now check if it's running in Docker
				qdrant_running = await is_container_running(QDRANT_CONTAINER_NAME)
	except (HTTPError, ConnectionError, OSError) as e:
		logger.warning(f"Error checking Qdrant service: {e}")

	# Start services if needed
	if not qdrant_running:
		logger.info("Qdrant service is not running, starting container...")
		started = await start_qdrant_container()
		if not started:
			return False, "Failed to start Qdrant container"

		if wait_for_health:
			# Wait for Qdrant to be healthy
			logger.info(f"Waiting for Qdrant service to be healthy (timeout: {DEFAULT_TIMEOUT}s)...")
			healthy = await check_qdrant_health(qdrant_url)
			if not healthy:
				return False, "Qdrant service failed to become healthy within the timeout period"

	return True, "Qdrant container is running"


async def ensure_postgres_running() -> tuple[bool, str]:
	"""
	Ensure the PostgreSQL container is running, starting it if needed.

	Returns:
	    Tuple of (success, message)

	"""
	if not await is_docker_running():
		return False, "Docker daemon is not running"

	# Check if PostgreSQL container is already running
	postgres_running = await is_container_running(POSTGRES_CONTAINER_NAME)

	# Start container if needed
	if not postgres_running:
		logger.info("PostgreSQL service is not running, starting container...")
		started = await start_postgres_container()
		if not started:
			return False, "Failed to start PostgreSQL container"

	return True, "PostgreSQL container is running"


async def stop_container(container_name: str) -> bool:
	"""
	Stop a Docker container.

	Args:
	    container_name: Name of the container to stop

	Returns:
	    True if successful, False otherwise

	"""

	def _stop_sync(name: str) -> bool:
		client = None
		try:
			client = docker.from_env()
			try:
				container = cast("Container", client.containers.get(name))
				if container.status == "running":
					logger.info(f"Stopping container {name}")
					container.stop(timeout=10)  # Wait up to 10 seconds for clean shutdown
					logger.info(f"Stopped container {name}")
				return True
			except NotFound:
				logger.info(f"Container {name} does not exist")
				return True
		except DockerException:  # Catch DockerException from client.containers.get or container.stop
			logger.exception(f"Docker error while stopping container {name}")
			return False
		finally:
			if client is not None:
				client.close()
		return False

	return await asyncio.to_thread(_stop_sync, container_name)


async def stop_all_codemap_containers() -> tuple[bool, str]:
	"""
	Stop all CodeMap containers.

	Returns:
	    Tuple of (success, message)

	"""
	containers = [QDRANT_CONTAINER_NAME, POSTGRES_CONTAINER_NAME]
	success = True

	for container_name in containers:
		if not await stop_container(container_name):
			success = False

	if success:
		return True, "All CodeMap containers stopped successfully"
	return False, "Failed to stop some CodeMap containers"
