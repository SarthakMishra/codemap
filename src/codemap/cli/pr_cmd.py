"""CLI command for generating pull requests using the refactored lazy-loading pattern."""

import logging
from enum import Enum
from pathlib import Path
from typing import Annotated, cast

import asyncer
import typer

from codemap.utils.cli_utils import progress_indicator


class PRAction(str, Enum):
	"""Actions for PR command."""

	CREATE = "create"
	UPDATE = "update"


def validate_workflow_strategy(value: str | None) -> str | None:
	"""Validate workflow strategy - lightweight callback for typer."""
	# Avoid heavy imports like Console here
	valid_strategies = ["github-flow", "gitflow", "trunk-based"]
	if value is None or value in valid_strategies:
		return value
	msg = f"Invalid workflow strategy: {value}. Must be one of: {', '.join(valid_strategies)}"
	raise typer.BadParameter(msg)


# --- Command Argument Annotations (Keep these lightweight) ---

ActionArg = Annotated[PRAction, typer.Argument(help="Action to perform: create or update")]
BranchNameOpt = Annotated[str | None, typer.Option("--branch", "-b", help="Target branch name")]
BranchTypeOpt = Annotated[
	str | None, typer.Option("--type", "-t", help="Branch type (feature, release, hotfix, bugfix)")
]
BaseBranchOpt = Annotated[
	str | None,
	typer.Option("--base", help="Base branch for the PR (defaults to repo default)"),
]
TitleOpt = Annotated[str | None, typer.Option("--title", help="Pull request title")]
DescriptionOpt = Annotated[
	str | None,
	typer.Option("--desc", "-d", help="Pull request description (file path or text)"),
]
NoCommitOpt = Annotated[
	bool,
	typer.Option("--no-commit", help="Skip the commit process before creating PR"),
]
ForcePushOpt = Annotated[bool, typer.Option("--force-push", "-f", help="Force push the branch")]
PRNumberOpt = Annotated[
	int | None,
	typer.Option("--pr", help="PR number to update (required for update action)"),
]
WorkflowOpt = Annotated[
	str | None,
	typer.Option(
		"--workflow",
		"-w",
		help="Git workflow strategy (github-flow, gitflow, trunk-based)",
		callback=validate_workflow_strategy,
	),
]
NonInteractiveOpt = Annotated[bool, typer.Option("--non-interactive", help="Run in non-interactive mode")]
ModelOpt = Annotated[
	str | None,
	typer.Option("--model", "-m", help="LLM model for content generation"),
]
ApiBaseOpt = Annotated[str | None, typer.Option("--api-base", help="API base URL for LLM")]
ApiKeyOpt = Annotated[str | None, typer.Option("--api-key", help="API key for LLM")]
BypassHooksFlag = Annotated[
	bool, typer.Option("--bypass-hooks", "--no-verify", help="Bypass git hooks with --no-verify")
]


# --- Registration Function ---


def register_command(app: typer.Typer) -> None:
	"""Register the pr command with the Typer app."""

	@app.command("pr")
	@asyncer.runnify
	async def pr_command_entrypoint(
		action: ActionArg = PRAction.CREATE,
		branch_name: BranchNameOpt = None,
		branch_type: BranchTypeOpt = None,
		base_branch: BaseBranchOpt = None,
		title: TitleOpt = None,
		description: DescriptionOpt = None,
		no_commit: NoCommitOpt = False,
		force_push: ForcePushOpt = False,
		pr_number: PRNumberOpt = None,
		workflow: WorkflowOpt = None,
		non_interactive: NonInteractiveOpt = False,
		bypass_hooks: BypassHooksFlag = False,
	) -> None:
		"""Create or update a GitHub/GitLab pull request with generated content."""
		await _pr_command_impl(
			action=action,
			branch_name=branch_name,
			branch_type=branch_type,
			base_branch=base_branch,
			title=title,
			description=description,
			no_commit=no_commit,
			force_push=force_push,
			pr_number=pr_number,
			workflow=workflow,
			non_interactive=non_interactive,
			bypass_hooks=bypass_hooks,
		)


# --- Command Implementation (Heavy imports inside) ---


async def _pr_command_impl(
	action: PRAction,
	branch_name: str | None,
	branch_type: str | None,
	base_branch: str | None,
	title: str | None,
	description: str | None,
	no_commit: bool,
	force_push: bool,
	pr_number: int | None,
	workflow: str | None,
	non_interactive: bool,
	bypass_hooks: bool,
) -> None:
	"""Actual implementation of the pr command with heavy imports."""
	# --- Heavy Imports ---

	with progress_indicator("Setting up environment..."):
		import contextlib
		from dataclasses import dataclass, field

		import questionary
		from rich.console import Console
		from rich.markdown import Markdown
		from rich.panel import Panel
		from rich.rule import Rule
		from rich.text import Text

		from codemap.config import ConfigLoader
		from codemap.config.config_schema import PRGenerateSchema
		from codemap.git.commit_generator.command import SemanticCommitCommand
		from codemap.git.pr_generator.command import PRWorkflowCommand
		from codemap.git.pr_generator.pr_git_utils import PRGitUtils
		from codemap.git.pr_generator.strategies import (
			WorkflowStrategy,
			branch_exists,
			create_strategy,
			get_default_branch,
		)
		from codemap.git.pr_generator.utils import (
			PRCreationError,
			generate_pr_description_from_commits,
			generate_pr_description_with_llm,
			generate_pr_title_from_commits,
			generate_pr_title_with_llm,
			get_all_open_prs,
			get_existing_pr,
		)
		from codemap.git.pr_generator.utils import validate_branch_name as validate_branch_name_util
		from codemap.git.utils import (
			ExtendedGitRepoContext,
			GitError,
		)
		from codemap.llm import LLMError
		from codemap.llm.client import LLMClient
		from codemap.utils.cli_utils import (
			exit_with_error,
			handle_keyboard_interrupt,
		)

	# --- Setup ---
	logger = logging.getLogger(__name__)
	console = Console()
	interactive = not non_interactive

	def _exit_command(code: int = 1) -> None:
		"""Helper function to exit the command using typer.Exit.

		Args:
			code: Exit code to return (default: 1).

		Raises:
			typer.Exit: Always raises this exception to exit the command.
		"""
		raise typer.Exit(code) from None

	# --- Dataclass for Options (similar to pr_cmd_old) ---
	@dataclass
	class PROptions:
		"""Internal options bundle."""

		branch_name: str | None = field(default=None)
		branch_type: str | None = field(default=None)
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
		workflow_strategy_name: str = field(default="github-flow")  # Default strategy
		bypass_hooks: bool = field(default=False)

	# --- Helper Functions (Adapted from pr_cmd_old) ---

	def _resolve_description(desc_input: str | None) -> str | None:
		"""Resolves description from input (text or file path)."""
		if not desc_input:
			return None
		try:
			desc_path = Path(desc_input)
			if desc_path.exists() and desc_path.is_file():
				with desc_path.open("r", encoding="utf-8") as f:
					return f.read()
		except OSError:  # Handle cases where input is not a valid path
			pass
		return desc_input  # Assume it's plain text

	def _handle_branch_creation(options: PROptions, workflow: WorkflowStrategy) -> str | None:
		"""Handle branch creation or selection."""
		pgu = PRGitUtils.get_instance()
		current_branch_name = pgu.get_current_branch()
		logger.info(f"Current branch: {current_branch_name}")

		# 1. Use provided branch name if valid
		if options.branch_name:
			if not validate_branch_name_util(options.branch_name):
				return None
			# Ensure branch exists or create it
			if not branch_exists(options.branch_name, pgu_instance=pgu):
				try:
					pgu.create_branch(options.branch_name)
					console.print(f"[green]Created and switched to new branch: {options.branch_name}[/green]")
					return options.branch_name
				except GitError:
					logger.exception("Error creating branch")
					return None
			else:
				# Branch exists, switch to it if not already there
				if pgu.get_current_branch() != options.branch_name:
					try:
						pgu.checkout_branch(options.branch_name)
						console.print(f"[green]Switched to existing branch: {options.branch_name}[/green]")
					except GitError:
						logger.exception("Error switching to branch")
						return None
				return options.branch_name

		# 2. Interactive mode
		if options.interactive:
			use_current = questionary.confirm(
				f"Use current branch '{current_branch_name}' for PR?",
				default=True,
			).ask()
			if use_current:
				return current_branch_name

			# Suggest a name based on type (if provided) or default
			suggested_name = "feature/new-feature"
			if options.branch_type:
				prefix = workflow.get_branch_prefix(options.branch_type)
				suggested_name = f"{prefix}new-{options.branch_type}" if prefix else f"new-{options.branch_type}"

			# Get existing branches for selection
			default_repo_branch = get_default_branch(pgu_instance=pgu) or "main"
			branch_options = [{"name": "[Create new branch]", "value": "_new_"}]
			try:
				# Use workflow strategy to get branches with metadata
				branches_with_metadata = workflow.get_all_branches_with_metadata()
				for branch, meta in branches_with_metadata.items():
					if branch == default_repo_branch:
						continue
					last_commit = meta.get("last_commit", "unknown")
					commit_count = meta.get("commit_count", 0)
					location = []
					if meta.get("is_local"):
						location.append("local")
					if meta.get("is_remote"):
						location.append("remote")
					location_str = ", ".join(location) or "unknown"
					branch_options.append(
						{
							"name": f"{branch} ({last_commit}, {commit_count} commits, {location_str})",
							"value": branch,
						}
					)
			except GitError as e:
				logger.warning(f"Could not retrieve detailed branch list: {e}")
				# Fallback to simple local branches
				try:
					local_branches = workflow.get_local_branches()
					# Use extend with a generator expression
					branch_options.extend(
						{"name": branch, "value": branch} for branch in local_branches if branch != default_repo_branch
					)
				except GitError:
					logger.exception("Failed to list any branches.")
					return None

			chosen_branch = questionary.select(
				"Select or create a branch:",
				choices=branch_options,
				qmark="\U0001f33f",  # Herb emoji
			).ask()

			if chosen_branch == "_new_":
				new_branch_name = questionary.text(
					"Enter new branch name:",
					default=suggested_name,
					validate=lambda name: True if validate_branch_name_util(name) else "Invalid branch name format.",
				).ask()
				if not new_branch_name:
					console.print("[yellow]Branch creation cancelled.[/yellow]")
					return None
				try:
					pgu.create_branch(new_branch_name)
					console.print(f"[green]Created and switched to new branch: {new_branch_name}[/green]")
					return new_branch_name
				except GitError:
					logger.exception("Error creating branch")
					return None
			elif chosen_branch:
				# Existing branch selected
				try:
					if pgu.get_current_branch() != chosen_branch:
						if chosen_branch in workflow.get_local_branches():
							pgu.checkout_branch(chosen_branch)
							console.print(f"[green]Switched to existing local branch: {chosen_branch}[/green]")
						else:
							# Create local branch from remote and checkout
							pgu.create_branch(chosen_branch, from_reference=f"origin/{chosen_branch}")
							console.print(
								f"[green]Created and switched to local branch '{chosen_branch}' from remote.[/green]"
							)
					return chosen_branch
				except GitError:
					logger.exception(f"Error switching to branch '{chosen_branch}'")
					return None
			else:
				# User cancelled selection
				console.print("[yellow]Branch selection cancelled.[/yellow]")
				return None

		# 3. Non-interactive mode
		else:
			logger.exception("No branch name provided and non-interactive mode enabled.")
			return None

	async def _handle_commits(options: PROptions) -> bool:
		"""Handle committing changes using CommitCommand."""
		if not options.commit_first:
			logger.info("Skipping commit step as requested.")
			return True

		try:
			# Check for changes (staged, unstaged, untracked)
			git_context = ExtendedGitRepoContext().get_instance()
			staged = git_context.get_staged_diff()
			unstaged = git_context.get_unstaged_diff()
			untracked = git_context.get_untracked_files()
			if not staged.files and not unstaged.files and not untracked:
				logger.warning("No changes detected to commit.")
				return True

			num_files = len(set(staged.files + unstaged.files + untracked))
			commit_now = True
			if options.interactive:
				commit_now = questionary.confirm(
					f"Found {num_files} files with changes. Commit them now?", default=True
				).ask()

			if not commit_now:
				logger.info("User chose not to commit changes.")
				return True

			# Use CommitCommand for the commit process
			logger.info("Starting commit process...")
			commit_command = SemanticCommitCommand(
				bypass_hooks=options.bypass_hooks,
			)
			# The run method handles staging, splitting, generation, and committing
			# Properly await the async run method
			success = await commit_command.run(interactive=options.interactive)

			if not success:
				# CommitCommand.run should raise exceptions or show errors,
				# but we can check the return value just in case.
				logger.exception("Commit process failed.")
				return False

			logger.info("Commit process completed.")
			return True

		except GitError:
			logger.exception("Git error during commit preparation")
			return False
		except Exception:
			logger.exception("Unexpected error during commit handling")
			logger.exception("Failed to handle commits")
			return False

	def _handle_push(options: PROptions, branch_name: str) -> bool:
		"""Handle pushing the branch to remote."""
		push_now = True
		if options.interactive:
			push_now = questionary.confirm(f"Push branch '{branch_name}' to remote?", default=True).ask()

		if not push_now:
			logger.warning(f"Skipping push for branch '{branch_name}'.")
			return True

		pgu = PRGitUtils.get_instance()
		try:
			with progress_indicator(f"Pushing branch '{branch_name}'...", style="spinner"):
				pgu.push_branch(branch_name, force=options.force_push, ignore_hooks=options.bypass_hooks)
			console.print(f"[green]Successfully pushed branch '{branch_name}' to remote.[/green]")
			return True
		except GitError:
			logger.exception(f"Error pushing branch '{branch_name}'")
			return False

	def _interactive_pr_review(
		initial_title: str,
		initial_description: str,
		options: PROptions,
		commits: list[str],
		branch_name: str,
		branch_type: str,
		base_branch: str,
		content_config: PRGenerateSchema,
		workflow_strategy_name: str,
	) -> tuple[str | None, str | None]:
		"""Interactive loop for reviewing and editing PR title and description."""
		title = initial_title
		description = initial_description

		while True:
			console.print(Rule("PR Preview"))
			console.print(Panel(Text(title), title="[bold]Title[/bold]", border_style="blue"))
			# Use Markdown for description preview
			console.print(Panel(Markdown(description), title="[bold]Description[/bold]", border_style="blue"))

			action = questionary.select(
				"Review the generated PR content:",
				choices=[
					"Confirm and Proceed",
					"Edit Title",
					"Edit Description",
					"Regenerate",
					"Cancel",
				],
				default="Confirm and Proceed",
				qmark="\U0001f4dd",  # Memo emoji
			).ask()

			# Handle None case explicitly
			if action is None:
				console.print("[yellow]PR operation cancelled.[/yellow]")
				return None, None
			if action == "Confirm and Proceed":
				return title, description
			if action == "Cancel":
				console.print("[yellow]PR operation cancelled.[/yellow]")
				return None, None
			if action == "Edit Title":
				new_title = questionary.text("Enter new title:", default=title).ask()
				if new_title is not None:
					title = new_title
			elif action == "Edit Description":
				new_description = questionary.text(
					"Edit description:",
					default=description,
					multiline=True,
				).ask()
				if new_description is not None:
					description = new_description
			elif action == "Regenerate":
				console.print("[cyan]Regenerating title and description...[/cyan]")
				title_strategy = content_config.title_strategy
				description_strategy = content_config.description_strategy
				title = _generate_title(options, title_strategy, commits, branch_name, branch_type)
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
			# Continue loop for edit/regenerate actions
		# Should not be reached
		return None, None

	# --- Generation Helpers (Adapted from pr_cmd_old, now use options dataclass) ---
	def _generate_title(
		options: PROptions, title_strategy: str, commits: list[str], branch_name: str, branch_type: str
	) -> str:
		"""Generate PR title based on the chosen strategy."""
		if options.title:
			return options.title

		if not commits:  # Handle empty commits case
			if branch_type == "release":
				version = branch_name.replace("release/", "")
				return f"Release {version}"
			clean_name = branch_name.replace(f"{branch_type}/", "").replace("-", " ").replace("_", " ")
			return f"{branch_type.capitalize()}: {clean_name.capitalize()}"

		if title_strategy == "llm":
			try:
				client = LLMClient(
					config_loader=config_loader,
				)
				return generate_pr_title_with_llm(commits=commits, llm_client=client)
			except (LLMError, ConnectionError, TimeoutError, ValueError, RuntimeError):
				logger.exception("LLM title generation failed. Falling back to commit-based title.")
				# Fall through to commit-based
		# Default to commit-based
		return generate_pr_title_from_commits(commits)

	def _generate_description(
		options: PROptions,
		description_strategy: str,
		commits: list[str],
		branch_name: str,
		branch_type: str,
		workflow_strategy_name: str,
		base_branch: str,
		content_config: PRGenerateSchema,
	) -> str:
		"""Generate PR description based on the chosen strategy."""
		# Handle explicit description (file or text)
		resolved_desc = _resolve_description(options.description)
		if resolved_desc is not None:
			return resolved_desc

		if not commits:  # Handle empty commits case
			if branch_type == "release" and workflow_strategy_name == "gitflow":
				# Use the method from PRWorkflowCommand if available, otherwise simple description
				try:
					# Access via the instance created later
					# This logic might be better placed within the main flow where pr_workflow exists
					# For now, keep simple fallback
					version = branch_name.replace("release/", "")
					return f"# Release {version}\n\nThis pull request merges release {version} into {base_branch}."
				except AttributeError:
					logger.warning("PRWorkflowCommand instance not available for release content generation.")
					return f"Changes related to release branch {branch_name}"
			else:
				return f"Changes related to branch {branch_name}"

		if description_strategy == "llm":
			try:
				client = LLMClient(
					config_loader=config_loader,
				)
				return generate_pr_description_with_llm(commits, llm_client=client)
			except (LLMError, ConnectionError, TimeoutError, ValueError, RuntimeError):
				logger.exception("LLM description generation failed. Falling back to commit-based description.")
				# Fall through to commit-based

		if description_strategy == "template":
			# Logic for template-based generation (from pr_cmd_old)
			template = content_config.description_template
			if template:
				commit_summary = "\n".join([f"- {commit}" for commit in commits])
				try:
					return template.format(
						changes=commit_summary,
						testing_instructions="Please test these changes thoroughly.",  # Example placeholder
						screenshots="",  # Example placeholder
					)
				except KeyError:
					logger.exception("Description template missing key. Falling back to commit-based.")
					# Fall through

		# Default to commit-based
		return generate_pr_description_from_commits(commits)

	# --- Main Logic ---
	try:
		# 2. Load Configuration
		config_loader = ConfigLoader.get_instance()
		pr_config = config_loader.get.pr
		llm_config = config_loader.get.llm
		content_config = pr_config.generate

		# Determine workflow strategy (CLI > Config > Default)
		workflow_strategy_name = workflow or pr_config.strategy
		try:
			strategy = create_strategy(workflow_strategy_name)
		except ValueError as e:
			exit_with_error(f"Invalid workflow strategy: {e}")
			return  # Should not be reached due to exit_with_error

		# 3. Consolidate Options
		opts = PROptions(
			branch_name=branch_name,
			branch_type=branch_type,
			base_branch=base_branch,
			title=title,
			description=description,
			commit_first=not no_commit,
			force_push=force_push,
			pr_number=pr_number,
			interactive=interactive,
			model=llm_config.model,
			workflow_strategy_name=workflow_strategy_name,
			bypass_hooks=bypass_hooks,
		)
		logger.debug(f"Resolved PR options: {opts}")

		# 4. Instantiate Workflow Command
		# Create LLM client separately first
		llm_client = LLMClient(
			config_loader=config_loader,
		)
		pr_workflow = PRWorkflowCommand(llm_client=llm_client, config_loader=config_loader)

		# 5. Execute Action
		if action == PRAction.CREATE:
			console.print(Rule("Starting PR Creation Process", style="bold blue"))

			# 5a. Handle Branch
			final_branch_name = _handle_branch_creation(opts, strategy)
			if not final_branch_name:
				_exit_command(1)  # Exit if branch handling failed

			# Cast to str to satisfy the type checker after the None check
			final_branch_name = cast("str", final_branch_name)

			opts.branch_name = final_branch_name  # Update options with final name

			# 5b. Handle Commits (Optional)
			if opts.commit_first and not await _handle_commits(opts):
				_exit_command(1)  # Exit if commit handling failed

			# 5c. Handle Push
			if not _handle_push(opts, final_branch_name):
				_exit_command(1)  # Exit if push failed

			# 5d. Determine Base Branch (before generating content)
			final_base_branch = opts.base_branch
			if not final_base_branch:
				pgu_temp_instance = PRGitUtils.get_instance()
				with contextlib.suppress(GitError):
					final_base_branch = get_default_branch(pgu_instance=pgu_temp_instance)
				if not final_base_branch:
					detected_branch_type = strategy.detect_branch_type(final_branch_name) or ""
					final_base_branch = strategy.get_default_base(detected_branch_type)
				if not final_base_branch:
					final_base_branch = "main"  # Final fallback
					logger.warning(f"Could not determine base branch, falling back to '{final_base_branch}'")
			opts.base_branch = final_base_branch  # Store resolved base branch

			# Interactive base branch selection (if needed)
			if opts.interactive and not base_branch:  # Only if not specified via CLI
				logger.info("Interactively verifying base branch.")
				try:
					remote_branches = strategy.get_remote_branches()
					# Filter out the current branch itself from choices
					choices = sorted([b for b in remote_branches if b != final_branch_name])
					default_choice = final_base_branch if final_base_branch in choices else None
					if choices:
						selected_base = questionary.select(
							"Select the base branch for the PR:",
							choices=choices,
							default=default_choice,
							qmark="\U0001f3af",  # Direct hit emoji
						).ask()
						if selected_base:
							opts.base_branch = selected_base
						else:
							console.print(
								f"[yellow]Selection cancelled. Using base branch: '{opts.base_branch}'[/yellow]"
							)
					else:
						logger.warning(f"No other remote branches found to select as base. Using '{opts.base_branch}'.")
				except GitError:
					logger.exception("Could not list remote branches for selection")

			# Ensure opts.base_branch is not None before proceeding
			if not opts.base_branch:
				exit_with_error("Base branch could not be determined or selected.")
				return

			# Check for existing PR AFTER push and base branch confirmation
			if final_branch_name:  # Ensure branch name is not None
				existing_pr = get_existing_pr(final_branch_name)
				if existing_pr:
					logger.warning(
						f"PR #{existing_pr.number} already exists for branch "
						f"'{final_branch_name}'. Update it instead? "
						f"({existing_pr.url})"
					)
					# Optionally switch to update flow or exit
					_exit_command(0)  # Exit gracefully if PR exists

			# 5e. Generate Title & Description
			console.print(Rule("Generating PR Content", style="bold blue"))
			if not opts.base_branch:  # Should not happen due to earlier checks
				exit_with_error("Base branch is unexpectedly None before generating content.")
				return

			pgu = PRGitUtils.get_instance()
			commits = pgu.get_commit_messages(opts.base_branch, final_branch_name)
			detected_branch_type = strategy.detect_branch_type(final_branch_name) or "feature"  # Default type
			title_strategy = content_config.title_strategy
			description_strategy = content_config.description_strategy

			initial_title = _generate_title(opts, title_strategy, commits, final_branch_name, detected_branch_type)
			initial_description = _generate_description(
				opts,
				description_strategy,
				commits,
				final_branch_name,
				detected_branch_type,
				opts.workflow_strategy_name,
				opts.base_branch,
				content_config,
			)

			# 5f. Interactive Review (if applicable)
			final_title, final_description = initial_title, initial_description
			if opts.interactive:
				final_title, final_description = _interactive_pr_review(
					initial_title,
					initial_description,
					opts,
					commits,
					final_branch_name,
					detected_branch_type,
					opts.base_branch,
					content_config,
					opts.workflow_strategy_name,
				)
				if final_title is None or final_description is None:
					_exit_command(0)  # User cancelled

			# 5g. Create PR using PRWorkflowCommand
			console.print(Rule("Creating Pull Request", style="bold blue"))
			try:
				with progress_indicator("Creating PR on GitHub/GitLab...", style="spinner"):
					pr = pr_workflow.create_pr_workflow(
						base_branch=opts.base_branch,
						head_branch=final_branch_name,
						title=final_title,
						description=final_description,
					)
				console.print(f"[green]Successfully created PR #{pr.number}: {pr.url}[/green]")
				# Ensure title and description are not None before passing to Panel
				final_title_str = final_title or "(Title not generated)"
				final_desc_str = final_description or "(Description not generated)"
				console.print(Panel(Text(final_title_str), title="[bold]Final Title[/bold]", border_style="green"))
				console.print(
					Panel(Markdown(final_desc_str), title="[bold]Final Description[/bold]", border_style="green")
				)
			except (PRCreationError, GitError) as e:
				# Handle specific errors like unrelated histories if needed
				error_message = str(e).lower()
				if "unrelated histories" in error_message or "no history in common" in error_message:
					suggestion = (
						f"\n[bold yellow]Suggestion:[/bold yellow]\n"
						f"The branch '{final_branch_name}' might not share history with base '{opts.base_branch}'.\n"
						f"Consider rebasing manually:\n"
						f"  `git checkout {opts.base_branch}`\n"
						f"  `git pull origin {opts.base_branch}`\n"
						f"  `git checkout {final_branch_name}`\n"
						f"  `git rebase {opts.base_branch}`\n"
						f"  `git push --force-with-lease origin {final_branch_name}`\n"
						f"Then run the command again."
					)
					exit_with_error(f"PR Creation Failed: {e}{suggestion}", exception=e)
				else:
					exit_with_error(f"PR Creation Failed: {e}", exception=e)

		else:  # action == PRAction.UPDATE
			console.print(Rule("Starting PR Update Process", style="bold blue"))

			# 5a. Determine PR number and current branch
			pr_num_to_update = opts.pr_number
			pgu = PRGitUtils.get_instance()
			current_branch = pgu.get_current_branch()
			if not pr_num_to_update:
				# Check existing PR for current branch only if branch name is known
				if current_branch:
					existing_pr = get_existing_pr(current_branch)
					if existing_pr:
						pr_num_to_update = existing_pr.number
						console.print(
							f"[cyan]Found existing PR #{pr_num_to_update} for current branch '{current_branch}'.[/cyan]"
						)
					else:
						# Interactive selection from all open PRs
						open_prs = get_all_open_prs()
						if not open_prs:
							exit_with_error("No open PRs found for this repository.")
							return
						import questionary

						choices = [
							questionary.Choice(title=f"#{pr.number}: {pr.title} [{pr.branch}]", value=pr.number)
							for pr in open_prs
						]
						pr_num_to_update = questionary.select(
							"Select a PR to update:", choices=choices, qmark="🔢"
						).ask()
						if not pr_num_to_update:
							exit_with_error("No PR selected for update.")
							return
				else:
					exit_with_error("Could not determine current branch to find existing PR.")
					return

			# Ensure current_branch is not None before proceeding
			if not current_branch:
				exit_with_error("Could not determine the current branch for the update operation.")
				return

			# Ensure pr_num_to_update is not None before proceeding
			if pr_num_to_update is None:
				# This should be caught by the earlier logic, but double-check
				exit_with_error("Internal error: PR number for update is None.")
				return

			# 5b. Handle Commits (Optional, might be needed before push/update)
			if opts.commit_first and not await _handle_commits(opts):
				_exit_command(1)

			# 5c. Handle Push (Optional but likely needed before update)
			if not _handle_push(opts, current_branch):
				_exit_command(1)

			# 5d. Update PR using PRWorkflowCommand
			console.print(Rule(f"Updating PR #{pr_num_to_update}", style="bold blue"))
			try:
				# Need to resolve description path/text before passing
				resolved_description = _resolve_description(opts.description)

				# We need base_branch and head_branch if we intend to regenerate content
				update_base_branch = opts.base_branch
				update_head_branch = current_branch
				# If regeneration is needed, ensure both branches are set
				if opts.title is None or resolved_description is None:
					if not update_base_branch:
						update_base_branch = get_default_branch(pgu_instance=pgu)
					if not update_head_branch:
						update_head_branch = current_branch

				# --- Interactive Review Step (mirrors PR creation) ---
				# Get commits for the PR branch (for context in review)
				commits = pgu.get_commit_messages(update_base_branch or "main", update_head_branch)
				detected_branch_type = strategy.detect_branch_type(update_head_branch) or "feature"
				review_title = opts.title
				review_description = resolved_description
				# If title/desc are None, regenerate as in creation
				if review_title is None:
					review_title = _generate_title(
						opts, content_config.title_strategy, commits, update_head_branch, detected_branch_type
					)
				if review_description is None:
					review_description = _generate_description(
						opts,
						content_config.description_strategy,
						commits,
						update_head_branch,
						detected_branch_type,
						opts.workflow_strategy_name,
						update_base_branch or "main",
						content_config,
					)
				if opts.interactive:
					review_title, review_description = _interactive_pr_review(
						review_title,
						review_description,
						opts,
						commits,
						update_head_branch,
						detected_branch_type,
						update_base_branch or "main",
						content_config,
						opts.workflow_strategy_name,
					)
					if review_title is None or review_description is None:
						_exit_command(0)  # User cancelled

				with progress_indicator(f"Updating PR #{pr_num_to_update} on GitHub/GitLab...", style="spinner"):
					updated_pr = pr_workflow.update_pr_workflow(
						pr_number=pr_num_to_update,
						title=review_title,  # Use reviewed/edited title
						description=review_description,  # Use reviewed/edited description
						base_branch=update_base_branch if opts.title is None or resolved_description is None else None,
						head_branch=update_head_branch if opts.title is None or resolved_description is None else None,
					)
				console.print(f"[green]Successfully updated PR #{updated_pr.number}: {updated_pr.url}[/green]")
				# Ensure title and description are not None before passing to Panel
				updated_title_str = updated_pr.title or "(Title not available)"
				updated_desc_str = updated_pr.description or "(Description not available)"
				console.print(Panel(Text(updated_title_str), title="[bold]Final Title[/bold]", border_style="green"))
				console.print(
					Panel(Markdown(updated_desc_str), title="[bold]Final Description[/bold]", border_style="green")
				)

			except (PRCreationError, GitError) as e:
				exit_with_error(f"PR Update Failed: {e}", exception=e)

	except typer.Exit:
		raise  # Let typer handle its own exit exceptions
	except KeyboardInterrupt:
		handle_keyboard_interrupt()  # Use standardized handler
	except (GitError, ValueError, PRCreationError) as e:
		# Catch known errors from git utils or workflow
		exit_with_error(f"Error: {e}", exception=e)
	except Exception as e:
		# Catch unexpected errors
		logger.exception("An unexpected error occurred in the PR command")
		exit_with_error(f"An unexpected error occurred: {e}", exception=e)
