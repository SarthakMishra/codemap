"""Prompt templates for PR generation."""

from __future__ import annotations

PR_TITLE_PROMPT = """Based on the following commits, generate a clear, concise PR title that captures the
essence of the changes.
Follow these guidelines:
- Focus on the most important change
- If there are multiple related changes, summarize them
- Keep it under 80 characters
- Start with a capital letter
- Don't use a period at the end
- Use present tense (e.g., "Add feature" not "Added feature")
- Be descriptive and specific (e.g., "Fix memory leak in data processing" not just "Fix bug")
- Include the type of change if clear (Feature, Fix, Refactor, etc.)

Commits:
{commit_list}

PR Title:
---

IMPORTANT:
- Do not include any other text in your response except the PR title.
- Do not wrap the PR title in quotes.
- Do not add any explanations or other text to your response.
"""


PR_DESCRIPTION_PROMPT = """
Based on the following commits, generate a comprehensive PR description following this template:

## What type of PR is this? (check all applicable)

- [ ] Refactor
- [ ] Feature
- [ ] Bug Fix
- [ ] Optimization
- [ ] Documentation Update

## Description
[Fill this section with a detailed description of the changes]

## Related Tickets & Documents
- Related Issue #
- Closes #

## Added/updated tests?
- [ ] Yes
- [ ] No, and this is why: [explanation]
- [ ] I need help with writing tests

Consider the following guidelines:
- Check the appropriate PR type boxes based on the commit messages
- Provide a clear, detailed description of the changes
- Include any relevant issue numbers that this PR relates to or closes
- Indicate if tests were added, and if not, explain why
- Use bullet points for clarity

Commits:
{commit_list}

PR Description:
---

IMPORTANT:
- Do not include any other text in your response except the PR description.
- Do not wrap the PR description in quotes.
- Do not add any explanations or other text to your response.
"""


def format_commits_for_prompt(commits: list[str]) -> str:
	"""
	Format commit messages as a bulleted list.

	Args:
	    commits: List of commit messages

	Returns:
	    Formatted commit list as a string

	"""
	return "\n".join([f"- {commit}" for commit in commits])
