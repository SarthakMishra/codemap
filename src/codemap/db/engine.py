"""Handles SQLModel engine creation and session management for CodeMap."""

import asyncio  # Added for running async ensure_postgres_running
import logging
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

# Removed StaticPool import as it's SQLite specific
# Import necessary docker utilities and constants
from codemap.utils.docker_utils import (
	POSTGRES_ENV,
	POSTGRES_HOST_PORT,
	ensure_postgres_running,
)

# Import models to ensure they are registered with SQLModel metadata

logger = logging.getLogger(__name__)

_engine_cache: dict[str, Engine] = {}  # Cache key will be the db url string
_engine_lock = asyncio.Lock()  # Lock for async engine creation/check


# Updated function signature: removed db_path, no longer needed for Postgres
async def get_engine(echo: bool = False) -> Engine:
	"""
	Gets or creates the SQLAlchemy engine for the PostgreSQL database.

	Ensures the PostgreSQL Docker container is running before creating the engine.

	Args:
	        echo (bool): Whether to echo SQL statements to the log.

	Returns:
	        Engine: The SQLAlchemy Engine instance.

	Raises:
	        RuntimeError: If the PostgreSQL container cannot be started or connection fails.

	"""
	# Construct the PostgreSQL connection URL
	# Uses values imported from docker_utils
	db_user = POSTGRES_ENV.get("POSTGRES_USER", "postgres")
	db_password = POSTGRES_ENV.get("POSTGRES_PASSWORD", "postgres")
	db_name = POSTGRES_ENV.get("POSTGRES_DB", "codemap")
	db_host = "localhost"  # Assuming Docker runs on localhost
	db_port = POSTGRES_HOST_PORT

	database_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

	async with _engine_lock:
		if database_url in _engine_cache:
			return _engine_cache[database_url]

		# Ensure PostgreSQL container is running
		logger.info("Checking and ensuring PostgreSQL container is running...")
		success, message = await ensure_postgres_running()
		if not success:
			logger.error(f"Failed to ensure PostgreSQL is running: {message}")
			# Consider a more specific exception type
			msg = f"PostgreSQL service setup failed: {message}"
			raise RuntimeError(msg)
		logger.info("PostgreSQL container check successful.")

		# Create the engine (using synchronous create_engine for now)
		# Removed connect_args and poolclass specific to SQLite
		try:
			engine = create_engine(database_url, echo=echo)
			# Test connection - optional but good practice
			with engine.connect():
				logger.info("Successfully connected to PostgreSQL.")

			_engine_cache[database_url] = engine
			logger.info(f"SQLModel engine created and cached for PostgreSQL at: {db_host}:{db_port}/{db_name}")
			return engine
		except Exception:
			logger.exception(f"Failed to create PostgreSQL engine or connect to {database_url}")
			raise  # Re-raise the original exception


def create_db_and_tables(engine_instance: Engine) -> None:
	"""Creates the database and all tables defined in SQLModel models."""
	logger.info("Ensuring database tables exist...")
	try:
		SQLModel.metadata.create_all(engine_instance)
		logger.info("Database tables ensured.")
	except Exception:
		logger.exception("Error creating database tables")
		raise


@contextmanager
def get_session(engine_instance: Engine) -> Generator[Session, None, None]:
	"""Provides a context-managed SQLModel session from a given engine."""
	session = Session(engine_instance)
	try:
		yield session
	except Exception:
		logger.exception("Session rollback due to exception")
		session.rollback()
		raise
	finally:
		session.close()
