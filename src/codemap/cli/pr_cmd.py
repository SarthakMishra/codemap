"""Command for generating and managing pull requests."""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, cast

import questionary
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from codemap.cli.commit_cmd import create_universal_generator
from codemap.git.commit_generator.command import CommitCommand
from codemap.git.commit_generator.generator import CommitMessageGenerator
from codemap.git.diff_splitter.schemas import DiffChunk
from codemap.git.diff_splitter.splitter import DiffSplitter
from codemap.git.pr_generator.generator import PRGenerator
from codemap.git.pr_generator.schemas import PullRequest
from codemap.git.pr_generator.strategies import branch_exists, create_strategy
from codemap.git.pr_generator.utils import (
	checkout_branch,
	create_branch,
	generate_pr_description_from_commits,
	generate_pr_description_with_llm,
	generate_pr_title_from_commits,
	generate_pr_title_with_llm,
	get_commit_messages,
	get_current_branch,
	get_default_branch,
	get_existing_pr,
	push_branch,
)
from codemap.git.utils import (
	GitDiff,
	GitError,
	get_repo_root,
	get_staged_diff,
	get_unstaged_diff,
	get_untracked_files,
	run_git_command,
	validate_repo_path,
)
from codemap.llm.utils import create_client
from codemap.utils.cli_utils import (
	exit_with_error,
	progress_indicator,
	setup_logging,
	show_error,
	show_warning,
)
from codemap.utils.config_loader import ConfigLoader

# Constants
MAX_PREVIEW_LINES = 10  # Maximum number of lines to show in description preview (unused, keeping full description)
MAX_DESCRIPTION_LENGTH = 100  # Maximum length for prefilling text input


def generate_message(
	chunk: DiffChunk, generator: CommitMessageGenerator, use_simple_mode: bool = False
) -> tuple[str, bool]:
	"""
	Generate a commit message for a diff chunk.

	This is a placeholder and should be properly imported from the
	appropriate module.

	"""
	if hasattr(generator, "generate_message_with_linting") and not use_simple_mode:
		message, used_llm, _ = generator.generate_message_with_linting(chunk)
	else:
		message, used_llm = generator.generate_message(chunk)
	return message, used_llm


def generate_release_pr_content(base_branch: str, branch_name: str) -> dict[str, str]:
	"""
	Generate PR content for a release.

	This is a placeholder and should be properly imported from the appropriate module.

	Args:
	        base_branch: The branch to merge into (e.g. main)
	        branch_name: The release branch name (e.g. release/1.0.0)

	Returns:
	        Dictionary with title and description

	"""
	# Extract version from branch name
	version = branch_name.replace("release/", "")
	title = f"Release {version}"
	# Include base branch information in the description
	description = f"# Release {version}\n\nThis pull request merges release {version} into {base_branch}."
	return {"title": title, "description": description}


logger = logging.getLogger(__name__)
console = Console()


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
	exit_with_error(message, exit_code, exception)


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
		show_error("Invalid branch name. Use only letters, numbers, underscores, dots, and hyphens.")
		return False
	return True


def _handle_branch_creation(options: PROptions) -> str | None:
	"""
	Handle branch creation or selection.

	Args:
	    options: PR options

	Returns:
	    Branch name if successful, None otherwise

	"""
	# Load PR configuration
	config_loader = ConfigLoader(repo_root=options.repo_path)
	workflow_strategy_name = config_loader.get_workflow_strategy()

	# Create workflow strategy
	workflow = create_strategy(workflow_strategy_name)

	# If branch name is already provided, validate and use it
	if options.branch_name:
		if not _validate_branch_name(options.branch_name):
			return None
		# Ensure branch exists
		if not branch_exists(options.branch_name):
			# Create branch if it doesn't exist
			try:
				create_branch(options.branch_name)
				console.print(f"[green]Created and switched to branch: {options.branch_name}[/green]")
			except GitError as e:
				show_error(f"Error creating branch: {e}")
				return None
		else:
			# Branch exists, make sure we're on it
			try:
				checkout_branch(options.branch_name)
				console.print(f"[green]Switched to branch: {options.branch_name}[/green]")
			except GitError as e:
				show_error(f"Error checking out branch: {e}")
				return None
		return options.branch_name

	# If interactive mode, let user select or create branch
	if options.interactive:
		# Get current branch
		current_branch = get_current_branch()

		# Ask if user wants to use current branch
		use_current = questionary.confirm(
			f"Use current branch '{current_branch}' for PR?",
			default=True,
		).ask()

		if use_current:
			return current_branch

		# Get default branch from repository
		default_branch = get_default_branch()
		if not default_branch:
			default_branch = "main"  # Fallback

		# Suggest a branch name based on PR type
		suggested_name = "feature/new-feature"  # Default suggestion

		# Get all branches with metadata
		branches_with_metadata = workflow.get_all_branches_with_metadata()
		branch_options = [{"name": "[Create new branch]", "value": "new"}]

		for branch, meta in branches_with_metadata.items():
			# Skip default branch
			if branch == default_branch:
				continue

			# Get last commit and commit count
			last_commit = meta.get("last_commit", "unknown")
			commit_count = meta.get("commit_count", 0)

			# Build location string (local, remote)
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
		show_warning("No uncommitted changes to commit.")
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
		# Initialize empty chunks
		chunks = []

		# Set up the splitter
		if options.repo_path is not None:
			splitter = DiffSplitter(repo_root=options.repo_path)
			chunks, filtered_large_files = splitter.split_diff(diff)
			if filtered_large_files:
				console.print(f"[yellow]Skipped {len(filtered_large_files)} large files due to size limits.[/yellow]")
			if not chunks:
				show_warning("No changes to commit after filtering.")
				return True
		else:
			# Handle None repo_path by using current directory
			logger.warning("Repository path not provided, using current directory")
			repo_path = Path.cwd()
			try:
				# Try to get repo root to validate it's a git repo
				repo_path = get_repo_root(repo_path)
			except GitError as e:
				show_error(f"Error: Not a valid git repository: {e}")
				return False

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
				show_error(f"Error: Not a valid git repository: {e}")
				return False
		else:
			repo_path = options.repo_path

		# Create the universal generator
		# Note: We don't actually use this directly, but it's called to ensure CommitCommand
		# has access to the initialized generator
		create_universal_generator(
			repo_path=repo_path,  # Now guaranteed to be a valid Path
			model=options.model,
			api_key=options.api_key,
			api_base=options.api_base,
		)

		# Make sure to stage all files before analyzing
		try:
			# Use git add . to stage all files for analysis
			with progress_indicator("Staging files for analysis", style="spinner"):
				run_git_command(["git", "add", "."])
		except GitError as e:
			logger.warning("Failed to stage all changes: %s", e)
			# Continue with the process even if staging fails

		# Process all chunks using the CommitCommand
		command = CommitCommand(path=options.repo_path, model=options.model or "gpt-4o-mini")

		# Explicitly initialize the sentence transformers model with proper progress indication
		with progress_indicator("Checking semantic analysis capabilities", style="spinner"):
			model_available = command.splitter._check_sentence_transformers_availability()  # noqa: SLF001

		if not model_available:
			show_warning(
				"Semantic analysis will be limited. To enable full capabilities, install: "
				"pip install sentence-transformers numpy"
			)

		if command.splitter._sentence_transformers_available:  # noqa: SLF001
			with progress_indicator(
				"Loading embedding model for semantic analysis (first use may download model files)", style="spinner"
			):
				model_loaded = command.splitter._check_model_availability()  # noqa: SLF001

			if not model_loaded:
				show_warning("Semantic analysis will use simplified approach due to model loading issues.")
			else:
				console.print("[green]Semantic analysis model loaded successfully.[/green]")

		result = command.process_all_chunks(
			chunks,
			interactive=options.interactive,
		)
	except (OSError, ValueError, RuntimeError, ConnectionError) as e:
		show_error(f"Error committing changes: {e}")
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
		show_error("Branch name cannot be None.")
		return False

	# Ask if user wants to push changes
	if options.interactive:
		push_changes = questionary.confirm(
			f"Push branch '{branch_name}' to remote?",
			default=True,
		).ask()
		if not push_changes:
			show_warning("Not pushing branch to remote.")
			return True

	# Push branch
	try:
		push_branch(branch_name, force=options.force_push)
		console.print(f"[green]Pushed branch '{branch_name}' to remote.[/green]")
	except GitError as e:
		show_error(f"Error pushing branch: {e}")
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
		client = create_client(model=options.model, api_key=options.api_key, api_base=options.api_base)
		return generate_pr_title_with_llm(commits, llm_client=client)
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
		client = create_client(model=options.model, api_key=options.api_key, api_base=options.api_base)
		return generate_pr_description_with_llm(commits, llm_client=client)
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
	    Created PR if successful, None otherwise

	"""
	# Ensure branch_name is not None
	if branch_name is None:
		show_error("Branch name cannot be None.")
		return None

	# Ask if user wants to create PR
	if options.interactive:
		create_pr = questionary.confirm(
			f"Create PR from branch '{branch_name}'?",
			default=True,
		).ask()
		if not create_pr:
			show_warning("Not creating PR.")
			return None

	try:
		# Load PR configuration
		with progress_indicator("Loading PR configuration", style="spinner"):
			config_loader = ConfigLoader(repo_root=options.repo_path)
			pr_config = config_loader.config.get("pr", {})
			workflow_strategy_name = config_loader.get_workflow_strategy()

			# Determine base branch
			base_branch = options.base_branch
			if base_branch is None:
				with contextlib.suppress(GitError):
					base_branch = get_default_branch()
				if base_branch is None:
					try:
						# Create workflow strategy
						workflow = create_strategy(workflow_strategy_name)
						branch_type = ""
						if "/" in branch_name:
							branch_type = branch_name.split("/")[0]
						base_branch = workflow.get_default_base(branch_type) or "main"
					except Exception:
						logger.exception("Failed to determine base branch from workflow strategy")
						base_branch = "main"

		# Check for existing PR first
		existing_pr = get_existing_pr(branch_name)
		if existing_pr:
			show_warning(f"PR #{existing_pr.number} already exists for this branch.")
			return existing_pr

		# Get recent commits
		try:
			with progress_indicator("Fetching recent commits", style="spinner"):
				commits = get_commit_messages(base_branch, branch_name)
		except GitError as e:
			show_error(f"Error fetching commits: {e}")
			commits = []

		# Determine branch type
		branch_type = ""
		if "/" in branch_name:
			branch_type = branch_name.split("/")[0]

		# Get PR content configuration
		content_config = pr_config.get("content", {})

		# Get title and description strategies
		title_strategy = content_config.get("title_strategy", "commits")
		description_strategy = content_config.get("description_strategy", "commits")

		# Generate title
		with progress_indicator("Generating PR title", style="spinner"):
			# If title is already provided, use it
			if options.title:
				title = options.title
			else:
				# Otherwise generate it based on the strategy
				title = _generate_title(options, title_strategy, commits, branch_name, branch_type)

			# Make sure we have a title
			if not title:
				title = f"Update {branch_name}"

		# Generate description
		with progress_indicator("Generating PR description", style="spinner"):
			# If description is already provided, use it
			if options.description:
				# Check if it's a file path or a string
				description_path = Path(options.description)
				if description_path.exists() and description_path.is_file():
					try:
						with Path.open(description_path, encoding="utf-8") as f:
							description = f.read()
					except OSError as e:
						show_error(f"Error reading description file: {e}")
						description = options.description
				else:
					description = options.description
			else:
				# Generate description based on strategy
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

			# Make sure we have a description
			if not description:
				description = f"Update from branch {branch_name}"

		# Create the PR
		with progress_indicator("Creating PR", style="spinner"):
			# Initialize PR generator
			llm_client = create_client(
				model=options.model,
				api_key=options.api_key,
				api_base=options.api_base,
			)

			repo_path = cast("Path", options.repo_path)
			generator = PRGenerator(
				repo_path=repo_path,
				llm_client=llm_client,
			)

			# Create the PR
			pr = generator.create_pr(base_branch, branch_name, title, description)

		# Show PR information
		console.print(f"\n[green]Created PR #{pr.number}: {pr.title}[/green]")
		console.print(f"[green]URL: {pr.url}[/green]")

		# Show description (full)
		console.print("\n[bold blue]Description:[/bold blue]")
		# Description panel (full description)
		desc_panel = Panel(
			Markdown(description), title="[bold]PR Description[/bold]", border_style="green", padding=(1, 2)
		)
		console.print(desc_panel)

		return pr
	except GitError as e:
		show_error(f"Error creating PR: {e}")
		return None


def _handle_pr_update(options: PROptions, pr: PullRequest | None) -> PullRequest | None:
	"""
	Handle PR update process.

	Args:
	    options: PR command options
	    pr: Existing PR to update

	Returns:
	    Updated PullRequest object if successful, None otherwise

	"""
	if not options.repo_path:
		_exit_with_error("Repository path is required.")
		return None

	if not pr:
		# If PR number is provided but PR object is not, try to get the PR
		if options.pr_number:
			try:
				pr_number = options.pr_number
				current_branch = get_current_branch()

				# Create LLM client for PR generation
				llm_client = create_client(
					model=options.model,
					api_key=options.api_key,
					api_base=options.api_base,
				)

				# Create PR generator
				repo_path = cast("Path", options.repo_path)
				pr_generator = PRGenerator(
					repo_path=repo_path,
					llm_client=llm_client,
				)

				# Get the PR information directly from GitHub
				logger.info("Attempting to update PR #%s", pr_number)

				# For now, create a minimal PR object with just the needed fields
				pr = PullRequest(
					number=pr_number,
					url=f"https://github.com/unknown/unknown/pull/{pr_number}",
					title="",  # Will be updated soon
					description="",  # Will be updated soon
					branch=current_branch,
				)
			except Exception as e:
				logger.exception("Error retrieving PR information")
				_exit_with_error(f"Failed to retrieve PR information: {e}")
				return None
		else:
			_exit_with_error("No PR provided for update.")
			return None

	try:
		# Create LLM client for PR generation
		llm_client = create_client(
			model=options.model,
			api_key=options.api_key,
			api_base=options.api_base,
		)

		# Create PR generator
		repo_path = cast("Path", options.repo_path)
		pr_generator = PRGenerator(
			repo_path=repo_path,
			llm_client=llm_client,
		)

		# Get base branch
		base_branch = get_default_branch()

		# Get data for PR
		config_loader = ConfigLoader(repo_root=options.repo_path)
		content_config = config_loader.get_content_generation_config()
		title_strategy = content_config.get("title_strategy", "conventional")
		description_strategy = content_config.get("description_strategy", "conventional")

		# Detect branch type using strategy
		workflow_strategy_name = config_loader.get_workflow_strategy()
		workflow = create_strategy(workflow_strategy_name)
		branch_type = workflow.detect_branch_type(pr.branch) or "feature"

		# Get commits for title/description generation
		commits = get_commit_messages(base_branch, pr.branch)

		# Generate title
		title = options.title
		if title is None:
			# Use existing PR title if available
			title = pr.title if pr.title else ""
			if options.interactive:
				update_title = questionary.confirm(f"Update title? (Current: {title})", default=False).ask()
				if update_title:
					title = _generate_title(options, title_strategy, commits, pr.branch, branch_type)

		# Generate description
		description = options.description
		if description is None:
			# Use existing PR description if available
			current_desc = pr.description if pr.description else ""
			if options.interactive:
				show_limit = 100
				description_preview = (
					f"{current_desc[:show_limit]}..." if len(current_desc) > show_limit else current_desc
				)
				update_desc = questionary.confirm(
					f"Update description? (Current: {description_preview})", default=False
				).ask()
				if update_desc:
					description = _generate_description(
						options,
						description_strategy,
						commits,
						pr.branch,
						branch_type,
						workflow_strategy_name,
						base_branch,
						content_config,
					)
				else:
					description = current_desc
			else:
				description = current_desc

		# Update PR
		with progress_indicator(f"Updating PR #{pr.number}", style="spinner"):
			updated_pr = pr_generator.update_pr(cast("int", pr.number), title, description)

		console.print(f"[green]Updated PR #{updated_pr.number}: {updated_pr.url}[/green]")

		# Display the updated title and description in panels
		title_panel = Panel(
			Text(title, style="green"), title="[bold]PR Title[/bold]", border_style="green", padding=(1, 2)
		)
		console.print(title_panel)

		# Description panel (full description)
		desc_panel = Panel(
			Markdown(description), title="[bold]PR Description[/bold]", border_style="green", padding=(1, 2)
		)
		console.print(desc_panel)

		return updated_pr
	except GitError as e:
		console.print(f"[red]Error in PR update: {e}[/red]")
		return None


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
	valid_strategies = ["github-flow", "gitflow", "trunk-based"]
	if value is None or value in valid_strategies:
		return value
	console.print(f"[red]Invalid workflow strategy: {value}. Must be one of: {', '.join(valid_strategies)}[/red]")
	msg = f"Must be one of: {', '.join(valid_strategies)}"
	raise typer.BadParameter(msg)


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

			# Handle branch creation/selection first
			branch_name = _handle_branch_creation(options)
			if not branch_name:
				return

			# Handle commits if needed (after branch is created/selected)
			if options.commit_first:
				commit_success = _handle_commits(options)
				if not commit_success:
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
