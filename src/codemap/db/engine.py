"""Handles SQLModel engine creation and session management for CodeMap."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy.engine import Engine  # Import Engine for type hint
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

# Import models to ensure they are registered with SQLModel metadata

logger = logging.getLogger(__name__)

_engine_cache: dict[Path, Engine] = {}


def get_engine(db_path: str | Path, echo: bool = False) -> Engine:
	"""
	Gets or creates the SQLAlchemy engine for the given database path.

	Args:
	    db_path (str | Path): The path to the SQLite database file.
	    echo (bool): Whether to echo SQL statements to the log.

	Returns:
	    Engine: The SQLAlchemy Engine instance.

	"""
	db_path = Path(db_path).resolve()  # Use resolved absolute path as key
	if db_path in _engine_cache:
		return _engine_cache[db_path]

	# Ensure the directory exists
	try:
		db_path.parent.mkdir(parents=True, exist_ok=True)
	except OSError:
		logger.exception(f"Error creating database directory {db_path.parent}")
		raise

	sqlite_url = f"sqlite:///{db_path}"
	# Necessary for SQLite usage in multi-threaded apps (like web servers or task queues)
	connect_args = {"check_same_thread": False}

	# Create the engine
	engine = create_engine(
		sqlite_url,
		echo=echo,
		connect_args=connect_args,
		poolclass=StaticPool,  # Recommended for SQLite
	)
	_engine_cache[db_path] = engine
	logger.info(f"SQLModel engine created and cached for database at: {db_path}")
	return engine


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
