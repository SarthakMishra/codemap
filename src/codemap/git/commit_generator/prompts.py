"""Prompt templates for commit message generation."""

from __future__ import annotations

from typing import Any

from .schemas import COMMIT_MESSAGE_SCHEMA

# Default prompt template for commit message generation
DEFAULT_PROMPT_TEMPLATE = """
You are an AI assistant generating Conventional Commit 1.0.0 messages from Git diffs.

**Format:**
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Instructions & Rules:**

1.  **Type:** REQUIRED. Must be lowercase and one of: {convention[types]}.
    *   `feat`: New feature (MINOR SemVer).
    *   `fix`: Bug fix (PATCH SemVer).
    *   Other types (`build`, `chore`, `ci`, `docs`, `style`, `refactor`, `perf`, `test`, etc.) are allowed.
2.  **Scope:** OPTIONAL. Lowercase noun(s) in parentheses describing the code section (e.g., `(parser)`).
    *   Keep short (1-2 words).
3.  **Description:** REQUIRED. Concise, imperative, present tense summary of *what* changed and *why* based on the diff.
    *   Must follow the colon and space.
    *   Must be >= 10 characters.
    *   Must NOT end with a period.
    *   The entire header line (`<type>[scope]: <description>`) must be <= {convention[max_length]} characters.
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

**Input:**

*   File notes: {files}
*   Git diff: {diff}

**Output Requirements:**

*   Respond with ONLY the raw commit message string.
*   NO extra text, explanations, or markdown formatting (like ```).
*   STRICTLY OMIT footers: `Related Issue #`, `Closes #`, `REVIEWED-BY`, `TRACKING #`, `APPROVED`.

**(IMPORTANT) Following JSON Schema must be followed for Output:**
{schema}

---
Analyze the following diff and generate the commit message:

{diff}
"""


def get_lint_prompt_template() -> str:
	"""
	Get the prompt template for lint feedback.

	Returns:
	    The prompt template with lint feedback placeholders

	"""
	return """
You are a helpful assistant that generates conventional commit messages based on code changes.
Given a Git diff, please generate a concise and descriptive commit message following these conventions:

1. Use the format:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```
2. Types include: {convention[types]}
3. Scope must be short (1-2 words), concise, and represent the specific component affected
4. The description should be a concise, imperative present tense summary of the *specific code changes*
   in the diff chunk (e.g., "add feature", "fix bug", "update documentation").
   Focus on *what* was changed and *why*.
5. The optional body should be a multi-paragraph summary of the changes, focusing on the *why* and *how* of the changes.
6. The optional footer(s) should be a list of one or more footers, each with a token and a value.
7. Your response must ONLY contain the commit message string, formatted as:
  ```
  <type>[optional scope]: <description>

  [optional body]

  [optional footer(s)]
  ```
   with absolutely no other text, explanation, or surrounding characters (like quotes or markdown).

IMPORTANT: The previous commit message had the following issues:
{lint_feedback}

Please fix these issues and ensure the generated message adheres to the commit convention.

---
Here are some notes about the files changed:
{files}
---
Analyze the following diff and respond with ONLY the commit message string:

{diff}

---
IMPORTANT:
- Strictly follow the format <type>[optional scope]: <description>
- Do not include any other text, explanation, or surrounding characters (like quotes or markdown).
- Strictly do not include any `Related Issue #`, `Closes #`, `REVIEWED-BY`, `TRACKING #`, `APPROVED` footers.
- Strictly follow the JSON schema provided while generating output in JSON format:

{schema}
"""


def prepare_prompt(
	template: str,
	diff_content: str,
	file_info: dict[str, Any],
	convention: dict[str, Any],
	extra_context: dict[str, Any] | None = None,
) -> str:
	"""
	Prepare the prompt for the LLM.

	Args:
	    template: Prompt template to use
	    diff_content: Diff content to include
	    file_info: Information about files in the diff
	    convention: Commit convention settings
	    extra_context: Optional additional context values for the template

	Returns:
	    Formatted prompt

	"""
	context = {
		"diff": diff_content,
		"files": file_info,
		"convention": convention,
		"schema": COMMIT_MESSAGE_SCHEMA,
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
	diff_content: str,
	file_info: dict[str, Any],
	convention: dict[str, Any],
	lint_messages: list[str],
) -> str:
	"""
	Prepare a prompt with lint feedback for regeneration.

	Args:
	    template: Prompt template to use
	    diff_content: Diff content to include
	    file_info: Information about files in the diff
	    convention: Commit convention settings
	    lint_messages: List of linting error messages

	Returns:
	    Enhanced prompt with linting feedback

	"""
	# Create specific feedback for linting issues
	lint_feedback = "\n".join([f"- {msg}" for msg in lint_messages])

	# Extract conventional commits guidelines from the template
	# Instead of trying to extract from DEFAULT_PROMPT_TEMPLATE, just use the formatted rules directly
	conventional_commits_spec = """
1. **Type:** Must be lowercase and one of the allowed types.
2. **Scope:** Optional, lowercase noun describing the section.
3. **Description:** Imperative, present tense summary.
4. **Body:** Optional explanation of why and how.
5. **Breaking Change:** Indicated with ! or BREAKING CHANGE footer.
"""

	# Create an enhanced context with linting feedback
	context = {
		"diff": diff_content,
		"files": file_info,
		"convention": convention,
		"schema": COMMIT_MESSAGE_SCHEMA,
		"lint_feedback": lint_feedback,
		"conventional_commits_spec": conventional_commits_spec,
	}

	try:
		return template.format(**context)
	except KeyError as e:
		msg = f"Lint prompt template formatting error. Missing key: {e}"
		raise ValueError(msg) from e
