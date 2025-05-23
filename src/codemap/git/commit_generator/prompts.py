# ruff: noqa: E501
"""Prompt templates for commit message generation."""

from __future__ import annotations

from typing import Any

from codemap.config import ConfigLoader

from .schemas import CommitMessageSchema

COMMIT_SYSTEM_PROMPT = """
**Conventional Commit 1.0.0 Specification:**

1.  **Type:** REQUIRED. Must be lowercase.
    *   `feat`: New feature (MINOR SemVer).
    *   `fix`: Bug fix (PATCH SemVer).
    *   Other types (`build`, `chore`, `ci`, `docs`, `style`, `refactor`, `perf`, `test`, etc.) are allowed.
2.  **Scope:** OPTIONAL. Lowercase noun(s) in parentheses describing the code section (e.g., `(parser)`).
    *   Keep short (1-2 words).
3.  **Description:** REQUIRED. Concise, imperative, present tense summary of *what* changed and *why* based on the diff.
    *   Must follow the colon and space.
    *   Must be >= 10 characters.
    *   Must NOT end with a period.
4.  **Body:** OPTIONAL. Explain *why* and *how*. Start one blank line after the description.
	*	Use the body only if extra context is needed to understand the changes.
	*	Do not use the body to add unrelated information.
	*	Do not use the body to explain *what* was changed.
	*	Try to keep the body concise and to the point.
5.  **Footer(s):** OPTIONAL. Format `Token: value` or `Token # value`.
    *   Start one blank line after the body.
    *   Use `-` for spaces in tokens (e.g., `Reviewed-by`).
6.  **BREAKING CHANGE:** Indicate with `!` before the colon in the header (e.g., `feat(api)!: ...`)
    *   OR with a `BREAKING CHANGE: <description>` footer (MUST be uppercase).
    *   Correlates with MAJOR SemVer.
    *   If `!` is used, the description explains the break.
7.  **Special Case - Binary Files:**
    *   For binary file changes, use `chore` type with a scope indicating the file type (e.g., `(assets)`, `(images)`, `(builds)`)
    *   Be specific about what changed (e.g., "update image assets", "add new icon files", "replace binary database")
    *   If the diff content is empty or shows binary file changes, focus on the filenames to determine the purpose

---

You are an AI assistant specialized in writing git commit messages.
You are tasked with generating Conventional Commit messages from Git diffs.
Follow the user's requirements carefully and to the letter.
Your response must be a valid JSON object matching the provided schema.
"""

# Default prompt template for commit message generation
DEFAULT_PROMPT_TEMPLATE = """
**File Summary:**
{files_summary}

**Git diff:**
{diff}

**Commit Message Format:**
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Output Requirements:**

- Type must be one of: {convention.types}
- Length of the entire header line (`<type>[scope]: <description>`) must be less than {convention.max_length} characters
- Strictly omit footers: `Related Issue #`, `Closes #`, `REVIEWED-BY`, `TRACKING #`, `APPROVED`.
- Following JSON Schema must be followed for Output:
{schema}
- Return your answer as json.

---
Please analyze the `Git diff` and `File Summary` carefully and generate an appropriate commit message for all the changes made in all the files.
Write commits like an experienced developer. Use simple language and avoid technical jargon.
"""

# Context for move operations
MOVE_CONTEXT = """
---
This diff group contains file moves. Here is the list of files that are relocated:
{files}

These files are moved from {source_dir} to {target_dir}.
"""


def get_lint_prompt_template() -> str:
	"""
	Get the prompt template for lint feedback.

	Returns:
	    The prompt template with lint feedback placeholders

	"""
	return """
You are a helpful assistant that fixes conventional commit messages that have linting errors.

1. The conventional commit format is:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```
2. Types include: {convention.types}
3. Scope must be short (1-2 words), concise, and represent the specific component affected
4. The description should be a concise, imperative present tense summary of the code changes,
   focusing on *what* was changed and *why*.
5. The optional body should focus on the *why* and *how* of the changes.

IMPORTANT: The provided commit message has the following issues:
{lint_feedback}

Original commit message:
{original_message}

Brief file context (without full diff):
{files_summary}

Please fix these issues and ensure the generated message adheres to the commit convention.

IMPORTANT:
- Strictly follow the format <type>[optional scope]: <description>
- Do not include any other text, explanation, or surrounding characters
- Do not include any `Related Issue #`, `Closes #`, `REVIEWED-BY`, `TRACKING #`, `APPROVED` footers.
- Respond with a valid JSON object following this schema:

{schema}

Return your answer as json.
"""


def file_info_to_human_summary(file_info: dict[str, Any]) -> str:
	"""
	Convert file_info dict to a human-readable summary (used in both initial and regeneration prompts).

	Args:
	    file_info: Dictionary with information about files

	Returns:
	    Human-readable summary string
	"""
	files_summary = []
	for file_path, info in file_info.items():
		extension = info.get("extension", "")
		directory = info.get("directory", "")
		module = info.get("module", "")
		summary = f"- {file_path} ({extension} file in {directory})"
		if module:
			summary += f", part of {module} module"
		files_summary.append(summary)
	return "\n".join(files_summary) if files_summary else "No file information available"


def prepare_prompt(
	template: str,
	diff_content: str,
	file_info: dict[str, Any],
	config_loader: ConfigLoader,
	extra_context: dict[str, Any] | None = None,
) -> str:
	"""
	Prepare the prompt for the LLM.

	Args:
	    template: Prompt template to use
	    diff_content: Diff content to include
	    file_info: Information about files in the diff
	    config_loader: ConfigLoader instance to use for configuration
	    extra_context: Optional additional context values for the template

	Returns:
	    Formatted prompt

	"""
	context = {
		"diff": diff_content,
		# Use human-readable summary for files
		"files_summary": file_info_to_human_summary(file_info),
		"convention": config_loader.get.commit.convention,
		"schema": CommitMessageSchema,
	}

	# Add any extra context values
	if extra_context:
		context.update(extra_context)

	try:
		return template.format(**context)
	except KeyError as e:
		msg = f"Prompt template formatting error. Missing key: {e}"
		raise ValueError(msg) from e


def prepare_lint_prompt(
	template: str,
	file_info: dict[str, Any],
	config_loader: ConfigLoader,
	lint_messages: list[str],
	original_message: str | None = None,
) -> str:
	"""
	Prepare a prompt with lint feedback for regeneration.

	Args:
	    template: Prompt template to use
	    file_info: Information about files in the diff
	    config_loader: ConfigLoader instance to use for configuration
	    lint_messages: List of linting error messages
	    original_message: The original failed commit message

	Returns:
	    Enhanced prompt with linting feedback

	"""
	# Create specific feedback for linting issues
	lint_feedback = "\n".join([f"- {msg}" for msg in lint_messages])

	# Use the shared summary function
	files_summary_text = file_info_to_human_summary(file_info)

	# If original_message wasn't provided, use a placeholder
	message_to_fix = original_message or "No original message provided"

	# Create an enhanced context with linting feedback
	context = {
		"convention": config_loader.get.commit.convention,
		"schema": CommitMessageSchema,
		"lint_feedback": lint_feedback,
		"original_message": message_to_fix,
		"files_summary": files_summary_text,
	}

	try:
		return template.format(**context)
	except KeyError as e:
		msg = f"Lint prompt template formatting error. Missing key: {e}"
		raise ValueError(msg) from e
