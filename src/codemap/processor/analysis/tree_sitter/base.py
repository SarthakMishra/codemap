"""
Base classes and interfaces for tree-sitter analysis.

This module defines the core data structures and interfaces for tree-sitter analysis.
It provides:
- Entity type definitions for tree-sitter nodes
- Metadata structures for tree-sitter nodes.
- Base tree-sitter analysis interface

"""

from __future__ import annotations

import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class EntityType(Enum):
	"""Types of code entities that can be extracted."""

	# File-level entities
	MODULE = auto()
	NAMESPACE = auto()
	PACKAGE = auto()

	# Type definitions
	CLASS = auto()
	INTERFACE = auto()
	PROTOCOL = auto()  # Similar to interface but for structural typing
	STRUCT = auto()
	ENUM = auto()
	TYPE_ALIAS = auto()

	# Functions and methods
	FUNCTION = auto()
	METHOD = auto()
	PROPERTY = auto()  # For getter/setter methods
	TEST_CASE = auto()
	TEST_SUITE = auto()

	# Variables and constants
	VARIABLE = auto()
	CONSTANT = auto()
	CLASS_FIELD = auto()  # For class-level variables/fields

	# Code organization
	IMPORT = auto()
	DECORATOR = auto()

	# Documentation
	COMMENT = auto()
	DOCSTRING = auto()

	# Special cases
	UNKNOWN = auto()


# Export the EntityType class
__all__ = ["EntityType"]
