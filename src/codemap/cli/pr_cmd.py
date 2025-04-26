"""Command for generating and managing pull requests."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated

import questionary
import typer
from rich.panel import Panel

from codemap.git import DiffSplitter, SplitStrategy
from codemap.git.command import CommitCommand
from codemap.utils import validate_repo_path
from codemap.utils.cli_utils import console, loading_spinner, setup_logging
from codemap.utils.config_loader import ConfigLoader
from codemap.utils.git_utils import (
	GitDiff,
	GitError,
	get_repo_root,
	get_staged_diff,
	get_unstaged_diff,
	get_untracked_files,
	run_git_command,
)
from codemap.utils.llm_utils import create_universal_generator, generate_message
from codemap.utils.pr_strategies import create_strategy
from codemap.utils.pr_utils import (
	PullRequest,
	branch_exists,
	checkout_branch,
	create_branch,
	create_pull_request,
	generate_pr_content_from_template,
	generate_pr_description_from_commits,
	generate_pr_description_with_llm,
	generate_pr_title_from_commits,
	generate_pr_title_with_llm,
	generate_release_pr_content,
	get_branch_relation,
	get_commit_messages,
	get_current_branch,
	get_default_branch,
	get_existing_pr,
	push_branch,
	suggest_branch_name,
	update_pull_request,
)

logger = logging.getLogger(__name__)


class PRAction(str, Enum):
	"""Actions for PR command."""

	CREATE = "create"
	UPDATE = "update"


@dataclass
class PROptions:
	"""Options for the PR command."""

	repo_path: Path | None
	branch_name: str | None = field(default=None)
	base_branch: str | None = field(default=None)
	title: str | None = field(default=None)
	description: str | None = field(default=None)
	commit_first: bool = field(default=True)
	force_push: bool = field(default=False)
	pr_number: int | None = field(default=None)
	interactive: bool = field(default=True)
	model: str | None = field(default=None)
	api_base: str | None = field(default=None)
	api_key: str | None = field(default=None)


def _exit_with_error(message: str, exit_code: int = 1, exception: Exception | None = None) -> None:
	"""
	Exit with an error message.

	Args:
	    message: Error message to display
	    exit_code: Exit code to use
	    exception: Exception that caused the error

	"""
	console.print(message)
	if exception is None:
		raise typer.Exit(exit_code)
	raise typer.Exit(exit_code) from exception


def _validate_branch_name(branch_name: str) -> bool:
	"""
	Validate a branch name.

	Args:
	    branch_name: Branch name to validate

	Returns:
	    True if valid, False otherwise

	"""
	# Check if branch name is valid
	if not branch_name or not re.match(r"^[a-zA-Z0-9_.-]+$", branch_name):
		console.print("[red]Invalid branch name. Use only letters, numbers, underscores, dots, and hyphens.[/red]")
		return False
	return True


def _handle_branch_creation(options: PROptions) -> str | None:
	"""
	Handle branch creation or selection.

	Args:
	    options: PR options

	Returns:
	    Branch name or None if cancelled

	"""
	# Load PR configuration
	config_loader = ConfigLoader(repo_root=options.repo_path)
	workflow_strategy_name = config_loader.get_workflow_strategy()

	# Create workflow strategy
	workflow = create_strategy(workflow_strategy_name)

	current_branch = get_current_branch()
	default_branch = get_default_branch()

	# If we're already on a feature branch (not the default branch), ask if we want to use it
	if current_branch != default_branch and options.interactive:
		branch_type = workflow.detect_branch_type(current_branch) or "feature"
		use_current = questionary.confirm(
			f"You are on branch '{current_branch}' ({branch_type}). Do you want to use this branch for the PR?",
			default=True,
		).ask()
		if use_current:
			return current_branch

	# If branch name is provided, use it
	if options.branch_name:
		branch_name = options.branch_name
		if not _validate_branch_name(branch_name):
			return None
	elif options.interactive:
		# Get all branches with metadata
		branches_with_metadata = workflow.get_all_branches_with_metadata()

		# Group branches by type
		grouped_branches = workflow.get_branches_by_type()

		# Get appropriate branch types from workflow strategy
		branch_types = workflow.get_branch_types()

		# Ask for branch type (feature, release, hotfix, etc.)
		branch_type = questionary.select(
			"Select branch type:",
			choices=branch_types,
			default=branch_types[0] if branch_types else "feature",
			qmark="🔀",
		).ask()

		# Get existing branches of this type to show as options
		branches_of_type = grouped_branches.get(branch_type, [])

		# Get uncommitted changes to suggest a branch name if creating new branch
		try:
			# Get all changes
			staged = get_staged_diff()
			unstaged = get_unstaged_diff()
			untracked_files = get_untracked_files()

			# Combine into a single diff
			all_files = list(set(staged.files + unstaged.files + untracked_files))
			combined_content = staged.content + unstaged.content

			diff = GitDiff(
				files=all_files,
				content=combined_content,
				is_staged=False,  # Mixed staged/unstaged
			)
		except GitError:
			# Return an empty diff in case of error
			diff = GitDiff(files=[], content="", is_staged=False)

		# Generate a suggested branch name based on the changes
		suggested_name = ""
		if diff.files:
			# Use the diff splitter to get semantic chunks
			splitter = DiffSplitter(options.repo_path)
			chunks = splitter.split_diff(diff, str(SplitStrategy.SEMANTIC))

			if chunks:
				# Ensure repo_path is not None before passing it, as required by create_universal_generator
				if options.repo_path is None:
					# Handle the case where repo_path is None, maybe log an error or skip suggestion
					logger.error(
						"Repository path is required for generating branch name suggestion but was not provided."
					)
					suggested_name = ""  # Fallback to empty suggestion
				else:
					# Set up message generator for the first chunk
					generator = create_universal_generator(
						repo_path=options.repo_path,  # Now known to be Path
						model=options.model,
						api_key=options.api_key,
						api_base=options.api_base,
					)

					# Generate a commit message for the first chunk
					try:
						message, _ = generate_message(chunks[0], generator)
						# Extract the first line as the commit message
						first_line = message.split("\n")[0] if "\n" in message else message
						suggested_name = suggest_branch_name(first_line, workflow_strategy_name)
					except (ValueError, RuntimeError, ConnectionError) as e:
						# Fallback to a simple branch name
						logger.warning("Error generating branch name: %s", e)
						suggested_name = suggest_branch_name(f"update-{chunks[0].files[0]}", workflow_strategy_name)

		# Create formatted branch options for selection
		branch_options = []

		# Add option to create a new branch
		branch_options.append(
			{"name": f"Create new branch: {suggested_name}" if suggested_name else "Create new branch", "value": "new"}
		)

		# Add existing branches of this type with metadata
		for branch in branches_of_type:
			meta = branches_with_metadata.get(branch, {})
			last_commit = meta.get("last_commit_date", "unknown date")
			commit_count = meta.get("commit_count", "0")
			location = []
			if meta.get("is_local", False):
				location.append("local")
			if meta.get("is_remote", False):
				location.append("remote")
			location_str = ", ".join(location)

			branch_options.append(
				{"name": f"{branch} ({last_commit}, {commit_count} commits, {location_str})", "value": branch}
			)

		# Ask user to select a branch or create a new one
		if branch_options:
			branch_selection = questionary.select(
				"Select or create a branch:", choices=branch_options, qmark="🌿"
			).ask()

			if branch_selection == "new":
				# Ask for a new branch name with suggested name as default
				branch_name = questionary.text(
					"Enter branch name:",
					default=suggested_name,
				).ask()

				if not branch_name or not _validate_branch_name(branch_name):
					return None

				# Create a new branch
				try:
					create_branch(branch_name)
					console.print(f"[green]Created and switched to new branch: {branch_name}[/green]")
				except GitError as e:
					console.print(f"[red]Error creating branch: {e}[/red]")
					return None
			else:
				# Use selected existing branch
				branch_name = branch_selection

				# Check if local branch exists
				if branch_name in workflow.get_local_branches():
					try:
						checkout_branch(branch_name)
						console.print(f"[green]Switched to existing branch: {branch_name}[/green]")
					except GitError as e:
						console.print(f"[red]Error checking out branch: {e}[/red]")
						return None
				else:
					# Branch exists remotely but not locally
					try:
						run_git_command(["git", "checkout", "-b", branch_name, f"origin/{branch_name}"])
						console.print(f"[green]Checked out remote branch: {branch_name}[/green]")
					except GitError as e:
						console.print(f"[red]Error checking out remote branch: {e}[/red]")
						return None
		else:
			# No branches found, create a new one
			branch_name = questionary.text(
				"Enter branch name:",
				default=suggested_name,
			).ask()

			if not branch_name or not _validate_branch_name(branch_name):
				return None

			# Create a new branch
			try:
				create_branch(branch_name)
				console.print(f"[green]Created and switched to new branch: {branch_name}[/green]")
			except GitError as e:
				console.print(f"[red]Error creating branch: {e}[/red]")
				return None
	else:
		console.print("[red]No branch name provided and interactive mode is disabled.[/red]")
		return None

	return branch_name


def _handle_commits(options: PROptions) -> bool:
	"""
	Handle committing changes.

	Args:
	    options: PR options

	Returns:
	    True if successful, False otherwise

	"""
	if not options.commit_first:
		return True

	# Check if there are uncommitted changes
	try:
		# Get all changes
		staged = get_staged_diff()
		unstaged = get_unstaged_diff()
		untracked_files = get_untracked_files()

		# Combine into a single diff
		all_files = list(set(staged.files + unstaged.files + untracked_files))
		combined_content = staged.content + unstaged.content

		diff = GitDiff(
			files=all_files,
			content=combined_content,
			is_staged=False,  # Mixed staged/unstaged
		)
	except GitError:
		# Return an empty diff in case of error
		diff = GitDiff(files=[], content="", is_staged=False)

	if not diff.files:
		console.print("[yellow]No uncommitted changes to commit.[/yellow]")
		return True

	# Ask if user wants to commit changes
	if options.interactive:
		commit_changes = questionary.confirm(
			f"Found {len(diff.files)} uncommitted files. Do you want to commit them now?",
			default=True,
		).ask()
		if not commit_changes:
			return True

	# Use the commit command to commit changes
	try:
		# Set up the splitter
		splitter = DiffSplitter(options.repo_path)
		chunks = splitter.split_diff(diff, str(SplitStrategy.SEMANTIC))
		if not chunks:
			console.print("[yellow]No changes to commit after filtering.[/yellow]")
			return True

		# Set up message generator - we don't need to store it since
		# CommitCommand will handle message generation internally
		# Ensure repo_path is not None before passing it
		if options.repo_path is None:
			# Handle None repo_path by using current directory
			logger.warning("Repository path not provided, using current directory")
			repo_path = Path.cwd()
			try:
				# Try to get repo root to validate it's a git repo
				repo_path = get_repo_root(repo_path)
			except GitError as e:
				console.print(f"[red]Error: Not a valid git repository: {e}[/red]")
				return False
		else:
			repo_path = options.repo_path

		create_universal_generator(
			repo_path=repo_path,  # Now guaranteed to be a valid Path
			model=options.model,
			api_key=options.api_key,
			api_base=options.api_base,
		)

		# Make sure to stage all files before analyzing
		try:
			# Use git add . to stage all files for analysis
			with loading_spinner("Staging files for analysis..."):
				run_git_command(["git", "add", "."])
		except GitError as e:
			logger.warning("Failed to stage all changes: %s", e)
			# Continue with the process even if staging fails

		# Process all chunks using the CommitCommand
		command = CommitCommand(path=options.repo_path, model=options.model or "gpt-4o-mini")

		# Explicitly initialize the sentence transformers model with proper loading spinners
		with loading_spinner("Checking semantic analysis capabilities..."):
			model_available = command.splitter._check_sentence_transformers_availability()  # noqa: SLF001

		if not model_available:
			console.print(
				"[yellow]Semantic analysis will be limited. To enable full capabilities, install: "
				"pip install sentence-transformers numpy[/yellow]"
			)

		if command.splitter._sentence_transformers_available:  # noqa: SLF001
			with loading_spinner(
				"Loading embedding model for semantic analysis (first use may download model files)..."
			):
				model_loaded = command.splitter._check_model_availability()  # noqa: SLF001

			if not model_loaded:
				console.print(
					"[yellow]Semantic analysis will use simplified approach due to model loading issues.[/yellow]"
				)
			else:
				console.print("[green]Semantic analysis model loaded successfully.[/green]")

		result = command.process_all_chunks(
			chunks,
			interactive=options.interactive,
		)
	except (OSError, ValueError, RuntimeError, ConnectionError) as e:
		console.print(f"[red]Error committing changes: {e}[/red]")
		return False
	else:
		return result


def _handle_push(options: PROptions, branch_name: str | None) -> bool:
	"""
	Handle pushing changes to remote.

	Args:
	    options: PR options
	    branch_name: Branch name to push

	Returns:
	    True if successful, False otherwise

	"""
	# Ensure branch_name is not None
	if branch_name is None:
		console.print("[red]Branch name cannot be None.[/red]")
		return False

	# Ask if user wants to push changes
	if options.interactive:
		push_changes = questionary.confirm(
			f"Push branch '{branch_name}' to remote?",
			default=True,
		).ask()
		if not push_changes:
			console.print("[yellow]Not pushing branch to remote.[/yellow]")
			return True

	# Push branch
	try:
		push_branch(branch_name, force=options.force_push)
		console.print(f"[green]Pushed branch '{branch_name}' to remote.[/green]")
	except GitError as e:
		console.print(f"[red]Error pushing branch: {e}[/red]")
		return False
	else:
		return True


def _generate_title(
	options: PROptions, title_strategy: str, commits: list[str], branch_name: str, branch_type: str
) -> str:
	"""
	Generate PR title based on the chosen strategy.

	Args:
	    options: PR options
	    title_strategy: Strategy to use for title generation
	    commits: List of commit messages
	    branch_name: Branch name
	    branch_type: Branch type

	Returns:
	    Generated PR title

	"""
	if options.title:
		return options.title

	# Generate based on strategy
	if not commits:
		# For empty PRs, generate title based on branch name
		if branch_type == "release":
			# For release branches, suggest version number
			return f"Release {branch_name.replace('release/', '')}"
		# For other branches, use branch name
		clean_name = branch_name.replace(f"{branch_type}/", "").replace("-", " ").replace("_", " ")
		return f"{branch_type.capitalize()}: {clean_name.capitalize()}"
	if title_strategy == "llm" and options.repo_path:
		# Use LLM to generate title
		return generate_pr_title_with_llm(
			commits,
			model=options.model,
			api_key=options.api_key,
			api_base=options.api_base,
		)
	# Use commit messages to generate title
	return generate_pr_title_from_commits(commits)


def _generate_description(
	options: PROptions,
	description_strategy: str,
	commits: list[str],
	branch_name: str,
	branch_type: str,
	workflow_strategy_name: str,
	base_branch: str,
	content_config: dict,
) -> str:
	"""
	Generate PR description based on the chosen strategy.

	Args:
	    options: PR options
	    description_strategy: Strategy to use for description generation
	    commits: List of commit messages
	    branch_name: Branch name
	    branch_type: Branch type
	    workflow_strategy_name: Workflow strategy name
	    base_branch: Base branch for PR
	    content_config: Content generation configuration

	Returns:
	    Generated PR description

	"""
	if options.description:
		# Check if the description is a file path or a string
		desc_path = Path(options.description)
		if desc_path.exists() and desc_path.is_file():
			with desc_path.open("r", encoding="utf-8") as f:
				return f.read()
		else:
			return options.description

	if not commits:
		# For empty PRs, check if it's a release PR
		if branch_type == "release" and workflow_strategy_name == "gitflow":
			# For release PRs in GitFlow, generate release notes
			content = generate_release_pr_content(base_branch, branch_name)
			return content["description"]
		# For other empty PRs, generate a simple description based on branch name
		return f"Changes in {branch_name}"
	if description_strategy == "llm" and options.repo_path:
		# Use LLM to generate description
		return generate_pr_description_with_llm(
			commits,
			model=options.model,
			api_key=options.api_key,
			api_base=options.api_base,
		)
	if description_strategy == "template" and not content_config.get("use_workflow_templates", True):
		# Use template from config
		template = content_config.get("description_template", "")
		if template:
			# Generate a basic description from commits
			commit_description = "\n".join([f"- {commit}" for commit in commits])
			return template.format(
				changes=commit_description,
				testing_instructions="Please test these changes thoroughly.",
				screenshots="",
			)

	# Fallback to commit-based description
	return generate_pr_description_from_commits(commits)


def _handle_pr_creation(options: PROptions, branch_name: str | None) -> PullRequest | None:
	"""
	Handle PR creation.

	Args:
	    options: PR options
	    branch_name: Branch name to create PR from

	Returns:
	    Created PR or None if cancelled

	"""
	if branch_name is None:
		return None

	# Load PR configuration
	config_loader = ConfigLoader(repo_root=options.repo_path)
	workflow_strategy_name = config_loader.get_workflow_strategy()
	config_loader.get_pr_config()
	content_config = config_loader.get_content_generation_config()

	# Create workflow strategy
	workflow = create_strategy(workflow_strategy_name)

	# Detect branch type
	branch_type = workflow.detect_branch_type(branch_name) or "feature"

	# Determine base branch
	if options.base_branch:
		base_branch = options.base_branch
	else:
		# Get base branch from branch mapping
		branch_mapping = config_loader.get_branch_mapping(branch_type)
		base_branch = branch_mapping.get("base") or get_default_branch()

	if options.interactive:
		# Display a rich branch selector with relationships between branches
		all_branches = set(workflow.get_local_branches() + workflow.get_remote_branches())

		# Create formatted choices with branch metadata
		branch_choices = []
		for b in all_branches:
			if b == branch_name:  # Skip current branch
				continue

			meta = workflow.get_branch_metadata(b)
			relation, commit_count = get_branch_relation(b, branch_name)
			relation_str = "ancestor" if relation else "unrelated"

			branch_choices.append(
				{
					"name": f"{b} ({meta.get('last_commit_date', 'unknown')}, {relation_str}, {commit_count} commits)",
					"value": b,
				}
			)

		# Sort choices by relation
		branch_choices.sort(key=lambda x: "0" if "ancestor" in x["name"] else "1" + x["name"])

		# Add default branch at the top if it exists
		default = get_default_branch()
		branch_choices = [c for c in branch_choices if c["value"] != default]

		# Only add default branch to choices if it actually exists in the repository
		default_exists = branch_exists(default, include_remote=True)
		if default_exists:
			branch_choices.insert(0, {"name": f"{default} (default branch)", "value": default})
		elif len(branch_choices) > 0:
			# If default doesn't exist but we have other branches, use the first one as fallback
			default = branch_choices[0]["value"]
			logger.warning("Default branch '%s' doesn't exist. Using '%s' as fallback.", get_default_branch(), default)

		# Ask for base branch
		selected_base = questionary.select(
			"Select base branch:",
			choices=branch_choices,
			default=base_branch if base_branch in [c["value"] for c in branch_choices] else default,
			qmark="🔀",
		).ask()

		base_branch = selected_base

	# Check if PR already exists
	existing_pr = get_existing_pr(branch_name)
	if existing_pr:
		if options.interactive:
			update_existing = questionary.confirm(
				f"PR #{existing_pr.number} already exists for branch '{branch_name}'. Do you want to update it?",
				default=True,
			).ask()
			if update_existing:
				return _handle_pr_update(options, existing_pr)
			return existing_pr
		console.print(f"[yellow]PR #{existing_pr.number} already exists for branch '{branch_name}'.[/yellow]")
		return existing_pr

	# Get commit messages for content generation
	commits = get_commit_messages(base_branch, branch_name)

	# Generate PR title and description based on options and strategies
	if options.title and options.description:
		# User provided both title and description
		title = options.title

		# Check if the description is a file path or a string
		desc_path = Path(options.description)
		if desc_path.exists() and desc_path.is_file():
			with desc_path.open("r", encoding="utf-8") as f:
				description = f.read()
		else:
			description = options.description
	else:
		# Title and/or description need to be generated
		title_strategy = content_config.get("title_strategy", "commits")
		description_strategy = content_config.get("description_strategy", "commits")

		# Determine PR content generation method
		use_workflow_templates = content_config.get("use_workflow_templates", True)
		if use_workflow_templates and (description_strategy == "template" or title_strategy == "template"):
			# Get first commit message or generic description for template
			commit_desc = commits[0] if commits else f"Update {branch_name}"
			# Remove conventional commit prefix if present (e.g., feat(scope): message)
			if commits:
				commit_desc = re.sub(r"^[a-z]+(\([^)]+\))?:\s*", "", commit_desc)

			# Use PR templates based on workflow strategy
			pr_content = generate_pr_content_from_template(
				branch_name=branch_name,
				description=commit_desc,
				strategy_name=workflow_strategy_name,
			)

			# Only use template-generated title if title_strategy is template
			if title_strategy == "template":
				title = options.title or pr_content["title"]
			else:
				# Generate title with other strategies
				title = _generate_title(options, title_strategy, commits, branch_name, branch_type)

			# Only use template-generated description if description_strategy is template
			if description_strategy == "template":
				description = options.description or pr_content["description"]
			else:
				# Generate description with other strategies
				description = _generate_description(
					options,
					description_strategy,
					commits,
					branch_name,
					branch_type,
					workflow_strategy_name,
					base_branch,
					content_config,
				)
		else:
			# Generate title with fallbacks
			title = _generate_title(options, title_strategy, commits, branch_name, branch_type)

			# Generate description with fallbacks
			description = _generate_description(
				options,
				description_strategy,
				commits,
				branch_name,
				branch_type,
				workflow_strategy_name,
				base_branch,
				content_config,
			)

	# Show preview and allow editing in interactive mode
	if options.interactive:
		# Show preview
		console.print("\n[bold]Pull Request Preview:[/bold]")
		console.print(Panel(f"[bold]Title:[/bold] {title}"))
		console.print(Panel(f"[bold]Description:[/bold]\n\n{description}"))

		# Allow editing
		edit_pr = questionary.confirm("Do you want to edit the PR content?", default=False).ask()
		if edit_pr:
			# Edit title
			new_title = questionary.text("Edit PR title:", default=title).ask()
			if new_title:
				title = new_title

			# Edit description
			with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as tmp:
				tmp.write(description)
				tmp_path = tmp.name

			try:
				editor = os.environ.get("EDITOR", "nano")
				# Using subprocess.run with user input (editor) is considered safe in this context
				# as it's using a system environment variable with a fallback to a known safe value
				subprocess.run([editor, tmp_path], check=True)  # noqa: S603

				# Replace open() with Path.open()
				description = Path(tmp_path).read_text(encoding="utf-8")
			except subprocess.SubprocessError:
				console.print("[yellow]Failed to open editor. Using original description.[/yellow]")
			finally:
				with contextlib.suppress(OSError):
					# Replace os.unlink with Path.unlink()
					Path(tmp_path).unlink(missing_ok=True)

	# Ask for confirmation in interactive mode
	if options.interactive:
		create_pr = questionary.confirm(
			f"Create PR from '{branch_name}' to '{base_branch}'?",
			default=True,
		).ask()
		if not create_pr:
			return None

	# Create PR
	try:
		with loading_spinner(f"Creating PR from '{branch_name}' to '{base_branch}'"):
			pr = create_pull_request(base_branch, branch_name, title, description)
		console.print(f"\n[green]Created PR #{pr.number}: {pr.title}[/green]")
		console.print(f"[bold blue]URL:[/bold blue] {pr.url}")
		return pr
	except GitError as e:
		console.print(f"[red]Error creating PR: {e}[/red]")
		return None


def _handle_pr_update(options: PROptions, pr: PullRequest | None) -> PullRequest | None:
	"""
	Handle PR update.

	Args:
	    options: PR options
	    pr: Existing PR

	Returns:
	    Updated PR or None if cancelled

	"""
	# Ensure PR is not None
	if pr is None:
		console.print("[red]PR cannot be None.[/red]")
		return None

	# Get base branch
	base_branch = options.base_branch or get_default_branch()

	# Get commit messages
	try:
		commits = get_commit_messages(base_branch, pr.branch)
	except GitError as e:
		console.print(f"[red]Error getting commit messages: {e}[/red]")
		commits = []

	# Generate PR title and description with AI if possible
	try:
		# Display a spinner while generating PR content
		with loading_spinner("Generating PR content with AI..."):
			# Try AI-generated title and description first
			title = options.title or pr.title
			if not title:
				title = generate_pr_title_with_llm(
					commits, model=options.model, api_key=options.api_key, api_base=options.api_base
				)

			description = options.description or pr.description
			if not description:
				description = generate_pr_description_with_llm(
					commits, model=options.model, api_key=options.api_key, api_base=options.api_base
				)
	except (ValueError, RuntimeError, ConnectionError) as e:
		console.print(f"[yellow]AI generation failed: {e}[/yellow]")
		console.print("[yellow]Falling back to rule-based PR generation...[/yellow]")
		# Fallback to rule-based generation
		title = options.title or pr.title or generate_pr_title_from_commits(commits)
		description = options.description or pr.description or generate_pr_description_from_commits(commits)

	# In interactive mode, allow editing title and description
	if options.interactive:
		title = questionary.text("PR title:", default=title).ask()
		if not title:
			console.print("[red]PR title cannot be empty.[/red]")
			return None

		# Show description preview
		console.print("\nPR description preview:")
		console.print(Panel(description, title="Description"))

		edit_description = questionary.confirm("Edit description?", default=False).ask()
		if edit_description:
			# Use a temporary file for editing
			with tempfile.NamedTemporaryFile(mode="w+", suffix=".md", delete=False) as temp:
				temp.write(description)
				temp_path = temp.name

			try:
				# Try to use the user's preferred editor
				editor = os.environ.get("EDITOR", "nano")
				subprocess.run([editor, temp_path], check=True)  # noqa: S603

				with Path(temp_path).open() as temp:
					description = temp.read()
			except OSError as e:
				console.print(f"[red]Error editing description: {e}[/red]")
			finally:
				Path(temp_path).unlink()

	# Update PR
	try:
		updated_pr = update_pull_request(pr.number, title, description)
		console.print(f"[green]Updated PR #{updated_pr.number}: {updated_pr.url}[/green]")
	except GitError as e:
		console.print(f"[red]Error updating PR: {e}[/red]")
		return None
	else:
		return updated_pr


def _load_llm_config(repo_path: Path | None) -> dict:
	"""
	Load LLM configuration from ConfigLoader.

	Args:
	    repo_path: Path to the repository

	Returns:
	    Dictionary with LLM configuration values

	"""
	# Create a config loader instance
	config_loader = ConfigLoader(repo_root=repo_path)

	# Get the LLM configuration
	return config_loader.get_llm_config()


def validate_workflow_strategy(value: str | None) -> str | None:
	"""Validate workflow strategy."""
	if value is not None and value not in ["github-flow", "gitflow", "trunk-based"]:
		msg = "Workflow must be one of: github-flow, gitflow, trunk-based"
		raise typer.BadParameter(msg)
	return value


def pr_command(
	path: Annotated[
		Path,
		typer.Argument(
			exists=True,
			help="Path to the codebase to analyze",
			show_default=True,
		),
	] = Path(),
	action: Annotated[PRAction, typer.Argument(help="Action to perform: create or update")] = PRAction.CREATE,
	branch_name: Annotated[str | None, typer.Option("--branch", "-b", help="Target branch name")] = None,
	branch_type: Annotated[
		str | None, typer.Option("--type", "-t", help="Branch type (feature, release, hotfix, bugfix)")
	] = None,
	base_branch: Annotated[
		str | None,
		typer.Option("--base", help="Base branch for the PR (defaults to repo default)"),
	] = None,
	title: Annotated[str | None, typer.Option("--title", help="Pull request title")] = None,
	description: Annotated[
		str | None,
		typer.Option("--desc", "-d", help="Pull request description (file path or text)"),
	] = None,
	no_commit: Annotated[
		bool,
		typer.Option("--no-commit", help="Skip the commit process before creating PR"),
	] = False,
	force_push: Annotated[bool, typer.Option("--force-push", "-f", help="Force push the branch")] = False,
	pr_number: Annotated[
		int | None,
		typer.Option("--pr", help="PR number to update (required for update action)"),
	] = None,
	workflow: Annotated[
		str | None,
		typer.Option(
			"--workflow",
			"-w",
			help="Git workflow strategy (github-flow, gitflow, trunk-based)",
			callback=validate_workflow_strategy,
		),
	] = None,
	non_interactive: Annotated[bool, typer.Option("--non-interactive", help="Run in non-interactive mode")] = False,
	model: Annotated[
		str | None,
		typer.Option("--model", "-m", help="LLM model for content generation"),
	] = None,
	api_base: Annotated[str | None, typer.Option("--api-base", help="API base URL for LLM")] = None,
	api_key: Annotated[str | None, typer.Option("--api-key", help="API key for LLM")] = None,
	is_verbose: Annotated[
		bool,
		typer.Option(
			"--verbose",
			"-v",
			help="Enable verbose logging",
		),
	] = False,
) -> None:
	"""Create or update a pull request."""
	# Configure logging
	setup_logging(is_verbose=is_verbose)

	# Helper function to raise typer.Exit with proper context
	def exit_command(code: int = 1) -> None:
		raise typer.Exit(code) from None

	try:
		# Get absolute path to repo
		repo_path = validate_repo_path(path)

		# Get PR configuration from config loader
		config_loader = ConfigLoader(repo_root=repo_path)
		config_loader.get_pr_config()

		# Set workflow strategy from command line or config - use ternary operator
		workflow_strategy = workflow if workflow else config_loader.get_workflow_strategy()

		# Create workflow strategy instance
		strategy = create_strategy(workflow_strategy)

		# Set up PR options
		options = PROptions(
			repo_path=repo_path,
			branch_name=branch_name,
			base_branch=base_branch,
			title=title,
			description=description,
			commit_first=not no_commit,
			force_push=force_push,
			pr_number=pr_number,
			interactive=not non_interactive,
			model=model,
			api_base=api_base,
			api_key=api_key,
		)

		# Load LLM config from file if not provided via CLI
		if not options.model or not options.api_key or not options.api_base:
			llm_config = _load_llm_config(repo_path)
			options.model = options.model or llm_config.get("model")
			options.api_key = options.api_key or llm_config.get("api_key")
			options.api_base = options.api_base or llm_config.get("api_base")

		# Perform requested action
		if action == PRAction.CREATE:
			# Configure branch type if provided
			if branch_type:
				# Validate branch type against workflow strategy
				valid_types = strategy.get_branch_types()
				if branch_type not in valid_types:
					console.print(f"[red]Invalid branch type for {workflow_strategy}: {branch_type}[/red]")
					console.print(f"[red]Valid types: {', '.join(valid_types)}[/red]")
					exit_command(1)

				# If branch name is provided, ensure it has the right prefix
				if options.branch_name:
					prefix = strategy.get_branch_prefix(branch_type)
					if prefix and not options.branch_name.startswith(prefix):
						options.branch_name = f"{prefix}{options.branch_name}"

			# Handle commits if needed
			if options.commit_first:
				commit_success = _handle_commits(options)
				if not commit_success:
					return

			# Handle branch creation/selection
			branch_name = _handle_branch_creation(options)
			if not branch_name:
				return

			# Handle push
			push_success = _handle_push(options, branch_name)
			if not push_success:
				return

			# Handle PR creation
			pr = _handle_pr_creation(options, branch_name)
			if not pr:
				return
		else:  # update
			# If PR number is not provided, get existing PR for current branch
			if not options.pr_number:
				current_branch = get_current_branch()
				existing_pr = get_existing_pr(current_branch)
				if existing_pr:
					options.pr_number = existing_pr.number
				else:
					console.print("[red]No PR found for current branch. Please specify a PR number.[/red]")
					exit_command(1)

			# Handle PR update
			pr = _handle_pr_update(options, None)
			if not pr:
				return
	except (GitError, ValueError) as e:
		_exit_with_error(f"Error: {e}", exception=e)
	except typer.Exit:
		raise
	except KeyboardInterrupt:
		console.print("\n[yellow]Operation cancelled by user.[/yellow]")
		exit_command(130)
	except Exception as e:
		logger.exception("Unexpected error in PR command")
		_exit_with_error(f"Unexpected error: {e}", exception=e)
