"""
Daemon module for CodeMap.

This package contains modules for running CodeMap as a background service:
- service: Core daemon functionality for process lifecycle management
- api_server: HTTP API server for remote interaction with the daemon
- client: Client library for interacting with the daemon

"""

# Forward declarations for type hints
from typing import TYPE_CHECKING, TypeVar

# Import types for type checking only
if TYPE_CHECKING:
	from .api_server import APIServer
	from .client import DaemonClient
	from .service import CodeMapDaemon

# Define type variable for return types
T = TypeVar("T", bound=object)

# These imports will be resolved when the modules are implemented
__all__ = ["APIServer", "CodeMapDaemon", "DaemonClient"]


# Import symbols only when accessed to avoid circular imports
def __getattr__(name: str) -> object:
	if name == "CodeMapDaemon":
		from .service import CodeMapDaemon

		return CodeMapDaemon
	if name == "APIServer":
		from .api_server import APIServer

		return APIServer
	if name == "DaemonClient":
		from .client import DaemonClient

		return DaemonClient

	msg = f"module {__name__!r} has no attribute {name!r}"
	raise AttributeError(msg)
