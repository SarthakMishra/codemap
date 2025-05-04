"""Tests for the docker_utils.py module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from docker.errors import DockerException, ImageNotFound, NotFound

from codemap.utils.docker_utils import (
	QDRANT_CONTAINER_NAME,
	QDRANT_IMAGE,
	QDRANT_STORAGE_PATH,
	check_qdrant_health,
	ensure_volume_path_exists,
	is_container_running,
	is_docker_running,
	pull_image_if_needed,
	start_qdrant_container,
	stop_all_codemap_containers,
	stop_container,
)


@pytest.mark.unit
class TestDockerUtils:
	"""Tests for docker_utils.py functions."""

	@pytest.mark.asyncio
	async def test_is_docker_running_success(self) -> None:
		"""Test the is_docker_running function when Docker is running."""
		with patch("docker.from_env") as mock_docker:
			# Setup mock
			mock_client = MagicMock()
			mock_docker.return_value = mock_client
			mock_client.ping.return_value = True

			# Call the function
			result = await is_docker_running()

			# Assertions
			assert result is True
			mock_client.ping.assert_called_once()
			mock_client.close.assert_called_once()

	@pytest.mark.asyncio
	async def test_is_docker_running_failure(self) -> None:
		"""Test the is_docker_running function when Docker is not running."""
		with patch("docker.from_env") as mock_docker:
			# Setup mock to raise DockerException
			mock_docker.side_effect = DockerException("Docker not running")

			# Call the function
			result = await is_docker_running()

			# Assertions
			assert result is False

	@pytest.mark.asyncio
	async def test_is_container_running_container_exists_and_running(self) -> None:
		"""Test is_container_running when container exists and is running."""
		with patch("docker.from_env") as mock_docker:
			# Setup mock
			mock_client = MagicMock()
			mock_docker.return_value = mock_client

			mock_container = MagicMock()
			mock_container.status = "running"
			mock_client.containers.get.return_value = mock_container

			# Call the function
			result = await is_container_running("test-container")

			# Assertions
			assert result is True
			mock_client.containers.get.assert_called_once_with("test-container")
			mock_client.close.assert_called_once()

	@pytest.mark.asyncio
	async def test_is_container_running_container_exists_not_running(self) -> None:
		"""Test is_container_running when container exists but not running."""
		with patch("docker.from_env") as mock_docker:
			# Setup mock
			mock_client = MagicMock()
			mock_docker.return_value = mock_client

			mock_container = MagicMock()
			mock_container.status = "exited"
			mock_client.containers.get.return_value = mock_container

			# Call the function
			result = await is_container_running("test-container")

			# Assertions
			assert result is False
			mock_client.containers.get.assert_called_once_with("test-container")
			mock_client.close.assert_called_once()

	@pytest.mark.asyncio
	async def test_is_container_running_container_not_found(self) -> None:
		"""Test is_container_running when container does not exist."""
		with patch("docker.from_env") as mock_docker:
			# Setup mock
			mock_client = MagicMock()
			mock_docker.return_value = mock_client
			mock_client.containers.get.side_effect = NotFound("Container not found")

			# Call the function
			result = await is_container_running("test-container")

			# Assertions
			assert result is False
			mock_client.containers.get.assert_called_once_with("test-container")
			mock_client.close.assert_called_once()

	@pytest.mark.asyncio
	async def test_is_container_running_docker_exception(self) -> None:
		"""Test is_container_running when a Docker exception occurs."""
		with patch("docker.from_env") as mock_docker:
			# Setup mock
			mock_client = MagicMock()
			mock_docker.return_value = mock_client
			mock_client.containers.get.side_effect = DockerException("Docker error")

			# Call the function
			result = await is_container_running("test-container")

			# Assertions
			assert result is False
			mock_client.close.assert_called_once()

	@pytest.mark.asyncio
	async def test_pull_image_if_needed_image_exists(self) -> None:
		"""Test pull_image_if_needed when image already exists."""
		with patch("docker.from_env") as mock_docker:
			# Setup mock
			mock_client = MagicMock()
			mock_docker.return_value = mock_client

			# Mock that image exists
			mock_client.images.get.return_value = MagicMock()

			# Call the function
			result = await pull_image_if_needed("test-image")

			# Assertions
			assert result is True
			mock_client.images.get.assert_called_once_with("test-image")
			mock_client.images.pull.assert_not_called()
			mock_client.close.assert_called_once()

	@pytest.mark.asyncio
	async def test_pull_image_if_needed_image_not_exists_pull_success(self) -> None:
		"""Test pull_image_if_needed when image doesn't exist and pull succeeds."""
		with patch("docker.from_env") as mock_docker:
			# Setup mock
			mock_client = MagicMock()
			mock_docker.return_value = mock_client

			# Mock image not found then pull success
			mock_client.images.get.side_effect = ImageNotFound("Image not found")
			mock_client.images.pull.return_value = MagicMock()

			# Call the function
			result = await pull_image_if_needed("test-image")

			# Assertions
			assert result is True
			mock_client.images.get.assert_called_once_with("test-image")
			mock_client.images.pull.assert_called_once_with("test-image")
			mock_client.close.assert_called_once()

	@pytest.mark.asyncio
	async def test_ensure_volume_path_exists(self) -> None:
		"""Test ensure_volume_path_exists creates directory if needed."""
		with patch("pathlib.Path.mkdir") as mock_mkdir:
			# Call the function
			await ensure_volume_path_exists("/test/path")

			# Assertions
			mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

	@pytest.mark.asyncio
	async def test_start_qdrant_container_already_running(self) -> None:
		"""Test start_qdrant_container when container already running."""
		with (
			patch("docker.from_env") as mock_docker,
			patch("codemap.utils.docker_utils.pull_image_if_needed") as mock_pull,
			patch("codemap.utils.docker_utils.ensure_volume_path_exists") as mock_ensure_path,
		):
			# Setup mocks
			mock_pull.return_value = True
			mock_client = MagicMock()
			mock_docker.return_value = mock_client

			mock_container = MagicMock()
			mock_container.status = "running"
			mock_client.containers.get.return_value = mock_container

			# Call the function
			result = await start_qdrant_container()

			# Assertions
			assert result is True
			mock_pull.assert_called_once_with(QDRANT_IMAGE)
			mock_ensure_path.assert_called_once_with(QDRANT_STORAGE_PATH)
			mock_client.containers.get.assert_called_once_with(QDRANT_CONTAINER_NAME)
			mock_container.start.assert_not_called()
			mock_client.close.assert_called_once()

	@pytest.mark.asyncio
	async def test_check_qdrant_health_success(self) -> None:
		"""Test check_qdrant_health when Qdrant is healthy."""
		with patch("httpx.AsyncClient.get") as mock_get:
			# Setup mock response
			mock_response = AsyncMock()
			mock_response.status_code = 200
			mock_get.return_value = mock_response

			# Call the function
			result = await check_qdrant_health("http://test-url")

			# Assertions
			assert result is True
			mock_get.assert_called_once()

	@pytest.mark.asyncio
	async def test_stop_container_success(self) -> None:
		"""Test stop_container when container exists and stops successfully."""
		with patch("docker.from_env") as mock_docker:
			# Setup mock
			mock_client = MagicMock()
			mock_docker.return_value = mock_client

			mock_container = MagicMock()
			mock_container.status = "running"
			mock_client.containers.get.return_value = mock_container

			# Call the function
			result = await stop_container("test-container")

			# Assertions
			assert result is True
			mock_client.containers.get.assert_called_once_with("test-container")
			mock_container.stop.assert_called_once()
			mock_client.close.assert_called_once()

	@pytest.mark.asyncio
	async def test_stop_container_not_found(self) -> None:
		"""Test stop_container when container doesn't exist."""
		with patch("docker.from_env") as mock_docker:
			# Setup mock
			mock_client = MagicMock()
			mock_docker.return_value = mock_client
			mock_client.containers.get.side_effect = NotFound("Container not found")

			# Call the function
			result = await stop_container("test-container")

			# Assertions
			assert result is True  # Should return True since there's nothing to stop
			mock_client.containers.get.assert_called_once_with("test-container")
			mock_client.close.assert_called_once()

	@pytest.mark.asyncio
	async def test_stop_all_codemap_containers(self) -> None:
		"""Test stop_all_codemap_containers successfully stops all containers."""
		with (
			patch("codemap.utils.docker_utils.stop_container") as mock_stop,
		):
			# Setup mock
			mock_stop.return_value = True

			# Call the function
			result, message = await stop_all_codemap_containers()

			# Assertions
			assert result is True
			assert "stopped successfully" in message
			# Should attempt to stop both Qdrant and Postgres containers
			assert mock_stop.call_count == 2
