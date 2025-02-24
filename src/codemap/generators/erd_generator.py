"""Entity Relationship Diagram generator for the codebase."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# Error messages
ERR_NO_CLASSES: Final = "No classes found in the codebase"


class ERDGenerator:
    """Generates Entity Relationship Diagrams from parsed code."""

    def __init__(self) -> None:
        """Initialize the ERD generator."""
        self._entities: set[str] = set()
        self._relationships: set[tuple[str, str, str, str]] = set()

    def _extract_entities(self, parsed_files: dict[Path, dict[str, Any]]) -> None:
        """Extract entities from parsed files.

        Args:
            parsed_files: Dictionary mapping file paths to their parsed symbols.
        """
        for symbols in parsed_files.values():
            if not symbols:
                continue
            for class_name in symbols.get("classes", []):
                if self._is_valid_class_name(class_name):
                    self._entities.add(class_name)
                else:
                    logger.warning("Invalid class name found: %s", class_name)

    def _process_inheritance_relationships(self, symbols: dict[str, Any]) -> None:
        """Process inheritance relationships from a symbols dictionary.

        Args:
            symbols: Dictionary containing parsed symbols.
        """
        if not symbols:
            return

        for class_name, bases in symbols.get("bases", {}).items():
            if not self._is_valid_class_name(class_name):
                continue
            for base in bases:
                if base in self._entities:
                    self._relationships.add((base, class_name, "inherits", ""))

    def _process_composition_relationships(self, symbols: dict[str, Any]) -> None:
        """Process composition relationships from a symbols dictionary.

        Args:
            symbols: Dictionary containing parsed symbols.
        """
        if not symbols:
            return

        for class_name, attributes in symbols.get("attributes", {}).items():
            if not self._is_valid_class_name(class_name):
                continue
            for attr_name, attr_type in attributes.items():
                if attr_type in self._entities:
                    desc = f"has {attr_name}"
                    self._relationships.add((class_name, attr_type, "contains", desc))
                    logger.debug("Added composition: %s -> %s", class_name, attr_type)

    def _extract_relationships(self, parsed_files: dict[Path, dict[str, Any]]) -> None:
        """Extract relationships from parsed files.

        Args:
            parsed_files: Dictionary mapping file paths to their parsed symbols.
        """
        for symbols in parsed_files.values():
            self._process_inheritance_relationships(symbols)
            self._process_composition_relationships(symbols)

    def _generate_markdown(self) -> str:
        """Generate markdown representation of the ERD.

        Returns:
            Markdown string containing the ERD.

        This method has been split into smaller methods to reduce complexity.
        """
        header = self._generate_header()
        entities = self._generate_entities_section()
        relationships = self._generate_relationships_section()
        return f"{header}\n\n{entities}\n\n{relationships}"

    def _generate_header(self) -> str:
        """Generate the header section of the markdown document.

        Returns:
            Header section as a string.
        """
        return "# Entity Relationship Diagram\n\n```mermaid\nerDiagram"

    def _generate_entities_section(self) -> str:
        """Generate the entities section of the markdown document.

        Returns:
            Entities section as a string.
        """
        lines = [f"  {entity}" for entity in sorted(self._entities)]
        return "\n".join(lines)

    def _generate_relationships_section(self) -> str:
        """Generate the relationships section of the markdown document.

        Returns:
            Relationships section as a string.
        """
        inheritance_rels = []
        composition_rels = []

        for source, target, rel_type, _ in sorted(self._relationships):
            if rel_type == "inherits":
                inheritance_rels.append(f"  {source} --|> {target}")
            else:  # contains
                composition_rels.append(f"  {source} ||--o{target}")

        return "\n".join(inheritance_rels + composition_rels + ["\n```"])

    def _is_valid_class_name(self, name: str) -> bool:
        """Check if a class name is valid.

        Args:
            name: The class name to check.

        Returns:
            True if the name is valid, False otherwise.
        """
        return bool(name and name.isidentifier() and not name.startswith("_"))

    def generate(self, parsed_files: dict[Path, dict[str, Any]], output_path: Path | None = None) -> Path:
        """Generate an ERD from parsed files.

        Args:
            parsed_files: Dictionary mapping file paths to their parsed symbols.
            output_path: Optional path to write the ERD to.

        Returns:
            Path where the ERD was written.

        Raises:
            ValueError: If no classes are found in the codebase.
        """
        # Extract entities and relationships
        self._extract_entities(parsed_files)
        if not self._entities:
            logger.warning("No entities found in the codebase")
            raise ValueError(ERR_NO_CLASSES)

        self._extract_relationships(parsed_files)

        # Generate markdown content
        content = self._generate_markdown()

        # Write to file
        output_path = output_path or Path("erd.md")
        try:
            output_path.write_text(content)
            logger.info("Successfully generated ERD at %s", output_path)
            return output_path
        except Exception:
            logger.exception("Failed to write ERD to file")
            raise
