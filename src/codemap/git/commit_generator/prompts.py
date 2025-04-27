"""Prompt templates for commit message generation."""

from __future__ import annotations

from typing import Any

from .schemas import COMMIT_MESSAGE_SCHEMA

# Default prompt template for commit message generation
DEFAULT_PROMPT_TEMPLATE = """
# Conventional Commits 1.0.0

The commit message should be structured as follows:

---

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```
---

<br />
The commit contains the following structural elements, to communicate intent to the
consumers of your library:

1. **fix:** a commit of the _type_ `fix` patches a bug in your codebase
  (this correlates with [`PATCH`] in Semantic Versioning).
2. **feat:** a commit of the _type_ `feat` introduces a new feature to the codebase
  (this correlates with [`MINOR`] in Semantic Versioning).
3. **BREAKING CHANGE:** a commit that has a footer `BREAKING CHANGE:`, or appends a `!` after the type/scope,
  introduces a breaking API change (correlating with [`MAJOR`] in Semantic Versioning).
A BREAKING CHANGE can be part of commits of any _type_.
4. _types_ other than `fix:` and `feat:` are allowed, for example @commitlint/config-conventional
  (based on the Angular convention) recommends `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`,
  `perf:`, `test:`, and others.
5. _footers_ other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to
  [git trailer format](https://git-scm.com/docs/git-interpret-trailers).

Additional types are not mandated by the Conventional Commits specification, and have no implicit effect
in Semantic Versioning (unless they include a BREAKING CHANGE).

A scope may be provided to a commit's type, to provide additional contextual information and is contained within
parenthesis, e.g., `feat(parser): add ability to parse arrays`.

## Examples

### Commit message with description and breaking change footer
```
feat: allow provided config object to extend other configs

BREAKING CHANGE: `extends` key in config file is now used for extending other config files
```

### Commit message with `!` to draw attention to breaking change
```
feat!: send an email to the customer when a product is shipped
```

### Commit message with scope and `!` to draw attention to breaking change
```
feat(api)!: send an email to the customer when a product is shipped
```

### Commit message with both `!` and BREAKING CHANGE footer
```
chore!: drop support for Node 6

BREAKING CHANGE: use JavaScript features not available in Node 6.
```

### Commit message with no body
```
docs: correct spelling of CHANGELOG
```

### Commit message with scope
```
feat(lang): add Polish language
```

### Commit message with multi-paragraph body and multiple footers
```
fix: prevent racing of requests

Introduce a request id and a reference to latest request. Dismiss
incoming responses other than from latest request.

Remove timeouts which were used to mitigate the racing issue but are
obsolete now.

Reviewed-by: Z
Refs: #123
```

## Specification

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY",
and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

1. Commits MUST be prefixed with a type, which consists of a noun, `feat`, `fix`, etc., followed
by the OPTIONAL scope, OPTIONAL `!`, and REQUIRED terminal colon and space.
2. The type `feat` MUST be used when a commit adds a new feature to your application or library.
3. The type `fix` MUST be used when a commit represents a bug fix for your application.
4. A scope MAY be provided after a type. A scope MUST consist of a noun describing a
section of the codebase surrounded by parenthesis, e.g., `fix(parser):`
5. A description MUST immediately follow the colon and space after the type/scope prefix.
6. The description is a short summary of the code changes, e.g., _fix: array parsing issue when multiple spaces were
contained in string_.
7. A longer commit body MAY be provided after the short description, providing additional contextual information about
the code changes. The body MUST begin one blank line after the description.
8. A commit body is free-form and MAY consist of any number of newline separated paragraphs.
9. One or more footers MAY be provided one blank line after the body. Each footer MUST consist of
 a word token, followed by either a `:<space>` or `<space>#` separator, followed by a string value.
10. A footer's token MUST use `-` in place of whitespace characters, e.g., `Acked-by`.
An exception is made for `BREAKING CHANGE`, which MAY also be used as a token.
11. A footer's value MAY contain spaces and newlines, and parsing MUST terminate when the next valid footer
  token/separator pair is observed.
12. Breaking changes MUST be indicated in the type/scope prefix of a commit, or as an entry in the
  footer.
13. If included as a footer, a breaking change MUST consist of the uppercase text BREAKING CHANGE, followed by a colon,
space, and description, e.g., _BREAKING CHANGE: environment variables now take precedence over config files_.
14. If included in the type/scope prefix, breaking changes MUST be indicated by a
  `!` immediately before the `:`. If `!` is used, `BREAKING CHANGE:` MAY be omitted from the footer section,
  and the commit description SHALL be used to describe the breaking change.
15. Types other than `feat` and `fix` MAY be used in your commit messages, e.g., _docs: update ref docs._
16. The units of information that make up Conventional Commits MUST NOT be treated as case sensitive by implementors,
with the exception of BREAKING CHANGE which MUST be uppercase.
17. BREAKING-CHANGE MUST be synonymous with BREAKING CHANGE, when used as a token in a footer.
---

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

## Commit Linting Rules
Your generated commit message will be validated against the following rules:
1. Type must be one of the allowed types: {convention[types]}
2. Type must be lowercase
3. Subject must not end with a period
4. Subject must be at least 10 characters long
5. Header line (first line) should be no longer than {convention[max_length]} characters
6. If a scope is provided, it must be in lowercase
7. Header must have a space after the colon
8. Description must start with an imperative verb (e.g., "add", not "adds" or "added")

---
Here are some notes about the files changed:
{files}
---
Analyze the following diff and respond with ONLY the commit message string:

{diff}

---
IMPORTANT:
- Strictly follow the format and instructions above.
- Do not include any other text, explanation, or surrounding characters (like quotes or markdown).
- Strictly do not include any `Related Issue #`, `Closes #`, `REVIEWED-BY`, `TRACKING #`, `APPROVED` footers.
- Strictly follow the JSON schema provided while generating output in JSON format:

{schema}
"""


def get_lint_prompt_template() -> str:
	"""
	Get the prompt template for lint feedback.

	Returns:
	    The prompt template with lint feedback placeholders

	"""
	return """
{conventional_commits_spec}

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
) -> str:
	"""
	Prepare the prompt for the LLM.

	Args:
	    template: Prompt template to use
	    diff_content: Diff content to include
	    file_info: Information about files in the diff
	    convention: Commit convention settings

	Returns:
	    Formatted prompt

	"""
	context = {
		"diff": diff_content,
		"files": file_info,
		"convention": convention,
		"schema": COMMIT_MESSAGE_SCHEMA,
	}

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

	# Get the conventional commits spec from the DEFAULT_PROMPT_TEMPLATE
	conventional_commits_spec = DEFAULT_PROMPT_TEMPLATE.split("# Conventional Commits 1.0.0")[1].split(
		"---\n\nYou are a helpful assistant"
	)[0]

	# Create an enhanced context with linting feedback
	context = {
		"diff": diff_content,
		"files": file_info,
		"convention": convention,
		"schema": COMMIT_MESSAGE_SCHEMA,
		"lint_feedback": lint_feedback,
		"conventional_commits_spec": "# Conventional Commits 1.0.0" + conventional_commits_spec,
	}

	try:
		return template.format(**context)
	except KeyError as e:
		msg = f"Lint prompt template formatting error. Missing key: {e}"
		raise ValueError(msg) from e
