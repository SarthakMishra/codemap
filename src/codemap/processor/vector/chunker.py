"""Handles chunking of code files using tree-sitter or regex fallback."""

import logging
import re
from pathlib import Path
from typing import Any

from tree_sitter import Node, Parser  # Import Node and Parser for typing

# Add LLM imports
from codemap.llm import LLMClient, LLMError, create_client

# Import the analyzer and related components
from codemap.processor.tree_sitter.analyzer import TreeSitterAnalyzer, get_language_by_extension
from codemap.processor.tree_sitter.base import EntityType
from codemap.processor.tree_sitter.languages import LanguageSyntaxHandler  # Import base class for typing

from . import config

logger = logging.getLogger(__name__)

Chunk = dict[str, Any]  # Structure matching schema fields (excluding embedding)

# Instantiate the analyzer globally (or manage it elsewhere if needed)
# Loading parsers can take time, so doing it once is efficient.
analyzer = TreeSitterAnalyzer()


def chunk_file(file_path: Path, file_content: str, git_hash: str, config_data: dict[str, Any]) -> list[Chunk] | None:
	"""
	Chunks a file based on its type (tree-sitter or fallback).

	Args:
	    file_path (Path): The path to the file (relative to repo root).
	    file_content (str): The content of the file.
	    git_hash (str): The git blob hash of the file content.
	    config_data (Dict[str, Any]): The loaded application configuration.

	Returns:
	    List[Chunk] | None: A list of chunk dictionaries, or None on error.

	"""
	# Determine language using the analyzer's utility function
	language = get_language_by_extension(file_path)

	# Check if a parser and handler exist for this language
	parser_instance = analyzer.get_parser(language) if language else None
	handler_instance = analyzer.get_syntax_handler(language) if language else None

	if language and parser_instance and handler_instance:
		logger.debug(f"Using tree-sitter chunking for: {file_path} (language: {language})")
		try:
			# Pass parser, handler, and config_data to the tree-sitter chunking function
			return _chunk_with_treesitter(
				file_path, file_content, git_hash, parser_instance, handler_instance, config_data
			)
		except Exception:
			logger.exception(f"Error during tree-sitter chunking for {file_path}. Falling back to regex.")
			# Fallback to regex if tree-sitter fails unexpectedly
			try:
				return _chunk_with_regex(file_path, file_content, git_hash)
			except Exception:
				logger.exception(f"Error during regex fallback chunking (after tree-sitter failure) for {file_path}")
				return None
	else:
		if language:
			logger.debug(
				f"No tree-sitter parser/handler for language '{language}'. Using regex fallback for: {file_path}"
			)
		else:
			logger.debug(f"Could not determine language. Using regex fallback for: {file_path}")

		try:
			return _chunk_with_regex(file_path, file_content, git_hash)
		except Exception:
			logger.exception(f"Error during regex fallback chunking for {file_path}")
			return None


# Helper function to get node text safely
def _get_node_text(node: Node, content_bytes: bytes) -> str:
	try:
		return content_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
	except IndexError:
		return ""


def _chunk_with_treesitter(
	file_path: Path,
	file_content: str,
	git_hash: str,
	parser: Parser,  # Use imported Parser type hint
	handler: LanguageSyntaxHandler,
	config_data: dict[str, Any],  # Add config_data parameter
) -> list[Chunk]:
	"""Performs chunking using tree-sitter based on entities."""
	chunks: list[Chunk] = []
	content_bytes = file_content.encode("utf-8")
	tree = parser.parse(content_bytes)
	root_node = tree.root_node

	if not root_node:
		logger.warning(f"Tree-sitter parsing returned no root node for {file_path}")
		return []  # Return empty list on parsing failure

	# --- Module Level Chunk (Collect top-level info) ---
	module_chunk_texts = []

	# Get file-level docstring first
	file_docstring, doc_node = handler.find_docstring(root_node, content_bytes)
	module_doc_end_line = doc_node.end_point[0] + 1 if doc_node else 0
	if file_docstring:
		module_chunk_texts.append(file_docstring.strip())

	# Collect top-level imports, vars, consts, func/class signatures
	for node in root_node.children:
		# Skip the docstring node itself if we already processed it
		if doc_node and node.id == doc_node.id:
			continue

		entity_type = handler.get_entity_type(node, root_node, content_bytes)

		if entity_type == EntityType.IMPORT:
			import_text = _get_node_text(node, content_bytes).strip()
			if import_text:
				module_chunk_texts.append(import_text)
		elif entity_type in (EntityType.VARIABLE, EntityType.CONSTANT):
			var_text = _get_node_text(node, content_bytes).strip()
			if var_text:
				module_chunk_texts.append(var_text)
		elif entity_type in (EntityType.FUNCTION, EntityType.CLASS):
			# Use getattr for safety, provide a fallback lambda returning first line
			signature = getattr(
				handler,
				"extract_signature",
				lambda n, cb: _get_node_text(n, cb).splitlines()[0] if _get_node_text(n, cb) else "",
			)(node, content_bytes)
			if signature:
				module_chunk_texts.append(signature.strip())
				# Add docstring immediately after signature if found
				doc, _ = handler.find_docstring(node, content_bytes)
				if doc:
					indented_doc = "> " + "\n> ".join(doc.strip().splitlines())
					module_chunk_texts.append(indented_doc)
		# Add other top-level elements if needed

	# Create the module chunk if we collected any text
	if module_chunk_texts:
		module_chunk_combined_text = "\n".join(module_chunk_texts)
		# Use file end line or docstring end line for module chunk extent
		module_end_line = max(module_doc_end_line, 1)
		chunks.append(
			{
				config.FIELD_FILE_PATH: str(file_path),
				config.FIELD_ENTITY_NAME: file_path.name + " (module overview)",
				config.FIELD_CHUNK_TYPE: config.CHUNK_TYPE_MODULE,
				config.FIELD_CHUNK_TEXT: module_chunk_combined_text[: config.MAX_CHUNK_TEXT_LENGTH],
				config.FIELD_GIT_HASH: git_hash,
				config.FIELD_START_LINE: 1,
				config.FIELD_END_LINE: module_end_line,  # Approximate end line for module info
			}
		)

	# --- Iterate through nodes for Class/Function/Method Chunks ---
	nodes_to_visit = list(root_node.children)
	visited_node_ids = {root_node.id}  # Avoid processing nodes multiple times

	while nodes_to_visit:
		node = nodes_to_visit.pop(0)

		if node.id in visited_node_ids or handler.should_skip_node(node):
			continue
		visited_node_ids.add(node.id)

		entity_type = handler.get_entity_type(node, node.parent, content_bytes)
		name = handler.extract_name(node, content_bytes) or "<anonymous>"
		docstring, _ = handler.find_docstring(node, content_bytes)
		start_line = node.start_point[0] + 1
		end_line = node.end_point[0] + 1

		chunk_text = ""
		chunk_type = ""
		entity_name_for_chunk = name
		process_children = False  # Flag to control adding children to the visit queue

		# --- Class Chunk ---
		if entity_type == EntityType.CLASS:
			chunk_type = config.CHUNK_TYPE_CLASS
			entity_name_for_chunk = f"Class: {name}"
			# Use getattr for safety, provide a fallback lambda returning first line
			signature = getattr(
				handler,
				"extract_signature",
				lambda _, cb: _get_node_text(_, cb).splitlines()[0] if _get_node_text(_, cb) else "",
			)(node, content_bytes)

			# Start chunk text with signature and docstring
			class_chunk_parts = [signature]
			if docstring:
				class_chunk_parts.append(docstring.strip())

			# Add class variables and method signatures
			class_body_node = handler.get_body_node(node)
			if class_body_node:
				member_signatures = []
				for child in class_body_node.children:
					child_entity_type = handler.get_entity_type(child, class_body_node, content_bytes)
					# Extract signatures for methods and fields/properties
					if child_entity_type == EntityType.METHOD:
						method_sig = getattr(
							handler,
							"extract_signature",
							lambda _, cb: _get_node_text(_, cb).splitlines()[0] if _get_node_text(_, cb) else "",
						)(child, content_bytes)
						if method_sig:
							member_signatures.append(method_sig.strip())
					elif child_entity_type in (
						EntityType.CLASS_FIELD,
						EntityType.PROPERTY,
						EntityType.VARIABLE,
						EntityType.CONSTANT,
					):
						# Extract the full definition line for fields/properties
						field_text = _get_node_text(child, content_bytes).strip()
						if field_text:
							member_signatures.append(field_text)

				if member_signatures:
					class_chunk_parts.append("\n--- Members ---")  # Add a separator
					class_chunk_parts.extend(member_signatures)

			chunk_text = "\n\n".join(filter(None, class_chunk_parts))
			process_children = True  # Keep processing children (methods) individually

		# --- Function/Method Chunk ---
		elif entity_type in (EntityType.FUNCTION, EntityType.METHOD):
			chunk_type = config.CHUNK_TYPE_METHOD if entity_type == EntityType.METHOD else config.CHUNK_TYPE_FUNCTION
			prefix = "Method" if entity_type == EntityType.METHOD else "Function"

			# Find parent class name if it's a method
			parent_class_name = None
			if entity_type == EntityType.METHOD:
				# Use getattr for safety, provide a fallback lambda returning None
				parent_class_node = getattr(handler, "get_enclosing_node_of_type", lambda _, __: None)(
					node, EntityType.CLASS
				)
				if parent_class_node:
					parent_class_name = handler.extract_name(parent_class_node, content_bytes)

			if parent_class_name:
				entity_name_for_chunk = f"{prefix}: {parent_class_name}.{name}"
			else:
				entity_name_for_chunk = f"{prefix}: {name}"

			# Use getattr for safety, provide a fallback lambda returning first line
			signature = getattr(
				handler,
				"extract_signature",
				lambda _, __: _get_node_text(_, __).splitlines()[0] if _get_node_text(_, __) else "",
			)(node, content_bytes)

			# Get body text (placeholder/fallback)
			body_node = getattr(handler, "get_body_node", lambda _: _)(
				node
			)  # Simple fallback: assume node is body if method missing
			body_text = _get_node_text(body_node, content_bytes).strip() if body_node else ""

			# --- Summarization Logic ---
			summarized_body = None
			# Get vector config safely
			processor_config = config_data.get("processor", {})
			vector_config = processor_config.get("vector", {})
			summarize_bodies = vector_config.get("summarize_bodies", False)
			min_length = vector_config.get("min_body_length_for_summary", 1000)

			if summarize_bodies and len(body_text) >= min_length:
				try:
					# Pass necessary llm config parts if create_client expects overrides
					llm_config_overrides = config_data.get("llm", {})
					llm_client: LLMClient = create_client(
						repo_path=None,  # Assuming repo path isn't strictly needed here
						model=llm_config_overrides.get("model"),
						api_key=llm_config_overrides.get("api_key"),
						api_base=llm_config_overrides.get("api_base"),
					)
					summary_prompt = (
						f"Provide a concise one-paragraph summary of the purpose "
						f"and main logic of the following code snippet:\n\n"
						f"```\n{body_text}\n```\n\nSummary:"
					)
					# Use reasonable defaults for summarization parameters
					summarized_body = llm_client.generate_text(
						prompt=summary_prompt,
						max_tokens=150,  # type: ignore[arg-type] # Limit summary length
						temperature=0.2,  # type: ignore[arg-type] # Lower temp for more factual summary
					)
					logger.debug(f"Successfully summarized body for {entity_name_for_chunk} in {file_path}")
				except LLMError as e:
					logger.warning(
						f"LLM summarization failed for {entity_name_for_chunk} in {file_path}: {e}. Using full body."
					)
				except Exception:  # Catch any other unexpected errors
					log_message = (
						f"Unexpected error during summarization for "
						f"{entity_name_for_chunk} in {file_path}. Using full body."
					)
					logger.exception(log_message)

			# Use summarized body if available, otherwise use original (potentially truncated) body
			final_body_text = summarized_body if summarized_body else body_text
			# --- End Summarization Logic ---

			# Combine parts, ensuring separation and stripping whitespace
			parts = [signature]
			if docstring:
				parts.append(docstring.strip())
			if final_body_text:  # Use the potentially summarized body
				# Add a clear separator for the body/summary
				parts.append("---\\nImplementation Summary:" if summarized_body else "---\\nImplementation:")
				parts.append(final_body_text)  # Already stripped or summarized
			chunk_text = "\\n\\n".join(filter(None, parts))  # Join non-empty parts with double newline

			process_children = False  # Don't typically chunk things inside a function/method body

		# --- Create Chunk if relevant type identified ---
		if chunk_type:
			final_chunk_text = chunk_text[: config.MAX_CHUNK_TEXT_LENGTH]
			if final_chunk_text:  # Only add chunk if there is text content
				chunks.append(
					{
						config.FIELD_FILE_PATH: str(file_path),
						config.FIELD_ENTITY_NAME: entity_name_for_chunk,
						config.FIELD_CHUNK_TYPE: chunk_type,
						config.FIELD_CHUNK_TEXT: final_chunk_text,
						config.FIELD_GIT_HASH: git_hash,
						config.FIELD_START_LINE: start_line,
						config.FIELD_END_LINE: end_line,
					}
				)

		# --- Add children to visit stack conditionally ---
		if process_children and node.children:
			# Add children in reverse order to maintain approximate top-down processing with pop(0)
			nodes_to_visit.extend(reversed(node.children))

	logger.info(f"Created {len(chunks)} chunks using tree-sitter for {file_path}.")
	return chunks


# Simple regex for splitting by double newlines (paragraph-like separation)
# More sophisticated regex could be used, but start simple for fallback.
# This regex finds blocks of text separated by one or more blank lines.
# It captures the non-blank content.
REGEX_FALLBACK_SPLITTER = re.compile(r"(?:\n\s*){2,}")  # Split on 2+ newlines (maybe blank lines in between)


def _chunk_with_regex(file_path: Path, file_content: str, git_hash: str) -> list[Chunk]:
	"""Performs fallback chunking using regex (splitting by double newlines)."""
	chunks = []
	lines = file_content.splitlines(keepends=True)  # Keep ends to calculate lines accurately

	# Split content by double (or more) newlines
	text_blocks = REGEX_FALLBACK_SPLITTER.split(file_content)

	start_offset = 0
	for block in text_blocks:
		trimmed_block = block.strip()
		if not trimmed_block:
			# Adjust offset for the split delimiter length we skipped
			# Find the length of the delimiter that followed this empty block
			match = REGEX_FALLBACK_SPLITTER.search(file_content, start_offset)
			if match:
				start_offset = match.end()
			continue  # Skip empty blocks resulting from split

		# Find the actual start and end lines of this block in the original content
		# This is approximate and might need refinement
		try:
			block_start_offset = file_content.index(block, start_offset)
			block_end_offset = block_start_offset + len(block)

			# Count preceding newlines to find start line more accurately
			start_line = file_content.count("\n", 0, block_start_offset) + 1
			# Count total newlines in the block itself
			num_newlines_in_block = block.count("\n")
			end_line = start_line + num_newlines_in_block
			# Adjust if block doesn't end with newline but original content might have more lines
			if not block.endswith("\n") and start_line + num_newlines_in_block <= len(lines):
				pass  # End line is correct
			elif block.endswith("\n"):
				end_line = start_line + num_newlines_in_block - 1  # inclusive end line

			end_line = max(start_line, end_line)  # Ensure end_line >= start_line

			# Truncate chunk_text if needed
			chunk_text_final = trimmed_block[: config.MAX_CHUNK_TEXT_LENGTH]

			chunk_data = {
				config.FIELD_FILE_PATH: str(file_path),
				config.FIELD_ENTITY_NAME: file_path.name,  # Use filename as entity for fallback
				config.FIELD_CHUNK_TYPE: config.CHUNK_TYPE_FALLBACK,
				config.FIELD_CHUNK_TEXT: chunk_text_final,
				config.FIELD_GIT_HASH: git_hash,
				config.FIELD_START_LINE: start_line,
				config.FIELD_END_LINE: end_line,
			}
			chunks.append(chunk_data)

			# Update start offset for next search
			start_offset = block_end_offset

		except ValueError:
			logger.warning(f"Could not reliably find block offset for fallback chunk in {file_path}. Skipping block.")
			# Attempt to find the next split point to recover
			match = REGEX_FALLBACK_SPLITTER.search(file_content, start_offset)
			if match:
				start_offset = match.end()
			else:
				break  # Cannot proceed

	logger.info(f"Created {len(chunks)} chunks using regex fallback for {file_path}.")
	return chunks
