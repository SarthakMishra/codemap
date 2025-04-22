"""Command for generating and managing pull requests."""

from __future__ import annotations

import json
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
from codemap.utils.pr_utils import (
    PullRequest,
    branch_exists,
    checkout_branch,
    create_branch,
    create_pull_request,
    generate_pr_description_from_commits,
    generate_pr_description_with_llm,
    generate_pr_title_from_commits,
    generate_pr_title_with_llm,
    get_commit_messages,
    get_current_branch,
    get_default_branch,
    get_existing_pr,
    push_branch,
    suggest_branch_name,
    update_pull_request,
)

from .cli_types import PathArg, VerboseFlag

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
    """Exit with an error message.

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
    """Validate a branch name.

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
    """Handle branch creation or selection.

    Args:
        options: PR options

    Returns:
        Branch name or None if cancelled
    """
    current_branch = get_current_branch()
    default_branch = get_default_branch()

    # If we're already on a feature branch (not the default branch), ask if we want to use it
    if current_branch != default_branch and options.interactive:
        use_current = questionary.confirm(
            f"You are on branch '{current_branch}'. Do you want to use this branch for the PR?",
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
        # Get uncommitted changes to suggest a branch name
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
                        suggested_name = suggest_branch_name([first_line])
                    except (ValueError, RuntimeError, ConnectionError) as e:
                        # Fallback to a simple branch name
                        logger.warning("Error generating branch name: %s", e)
                        suggested_name = suggest_branch_name([f"update-{chunks[0].files[0]}"])

        # Ask for branch name
        branch_name = questionary.text(
            "Enter branch name:",
            default=suggested_name,
        ).ask()

        if not branch_name or not _validate_branch_name(branch_name):
            return None
    else:
        console.print("[red]No branch name provided and interactive mode is disabled.[/red]")
        return None

    # Check if branch exists
    if branch_exists(branch_name):
        if options.interactive:
            use_existing = questionary.confirm(
                f"Branch '{branch_name}' already exists. Do you want to use it?",
                default=True,
            ).ask()
            if not use_existing:
                return None

        # Checkout existing branch
        try:
            checkout_branch(branch_name)
            console.print(f"[green]Switched to existing branch: {branch_name}[/green]")
        except GitError as e:
            console.print(f"[red]Error checking out branch: {e}[/red]")
            return None
    else:
        # Create new branch
        try:
            create_branch(branch_name)
            console.print(f"[green]Created and switched to new branch: {branch_name}[/green]")
        except GitError as e:
            console.print(f"[red]Error creating branch: {e}[/red]")
            return None

    return branch_name


def _handle_commits(options: PROptions) -> bool:
    """Handle committing changes.

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
    """Handle pushing changes to remote.

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


def _handle_pr_creation(options: PROptions, branch_name: str | None) -> PullRequest | None:
    """Handle PR creation.

    Args:
        options: PR options
        branch_name: Branch name

    Returns:
        Created PR or None if cancelled
    """
    # Ensure branch_name is not None
    if branch_name is None:
        console.print("[red]Branch name cannot be None.[/red]")
        return None

    # Set base branch
    base_branch = options.base_branch or get_default_branch()

    # Check if PR already exists
    existing_pr = get_existing_pr(branch_name)
    if existing_pr:
        if options.interactive:
            use_existing = questionary.confirm(
                f"PR #{existing_pr.number} already exists for branch '{branch_name}'. Do you want to update it?",
                default=True,
            ).ask()
            if use_existing:
                return _handle_pr_update(options, existing_pr)
            return None
        console.print(f"[yellow]PR #{existing_pr.number} already exists for branch '{branch_name}'.[/yellow]")
        return existing_pr

    # Get commit messages
    try:
        commits = get_commit_messages(base_branch, branch_name)
    except GitError as e:
        console.print(f"[red]Error getting commit messages: {e}[/red]")
        commits = []

    # Generate PR title and description with AI if possible
    try:
        # Display a spinner while generating PR content
        with loading_spinner("Generating PR content with AI..."):
            # Try AI-generated title and description first
            title = options.title
            if not title:
                title = generate_pr_title_with_llm(
                    commits, model=options.model, api_key=options.api_key, api_base=options.api_base
                )

            description = options.description
            if not description:
                description = generate_pr_description_with_llm(
                    commits, model=options.model, api_key=options.api_key, api_base=options.api_base
                )
    except (ValueError, RuntimeError, ConnectionError) as e:
        console.print(f"[yellow]AI generation failed: {e}[/yellow]")
        console.print("[yellow]Falling back to rule-based PR generation...[/yellow]")
        # Fallback to rule-based generation
        title = options.title or generate_pr_title_from_commits(commits)
        description = options.description or generate_pr_description_from_commits(commits)

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

    # Create PR
    try:
        pr = create_pull_request(base_branch, branch_name, title, description)
        console.print(f"[green]Created PR #{pr.number}: {pr.url}[/green]")
    except GitError as e:
        console.print(f"[red]Error creating PR: {e}[/red]")
        return None
    else:
        return pr


def _handle_pr_update(options: PROptions, pr: PullRequest | None) -> PullRequest | None:
    """Handle PR update.

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
    """Load LLM configuration from ConfigLoader.

    Args:
        repo_path: Path to the repository

    Returns:
        Dictionary with LLM configuration values
    """
    from codemap.utils.config_loader import ConfigLoader

    # Create a config loader instance
    config_loader = ConfigLoader(repo_root=repo_path)

    # Get the LLM configuration
    return config_loader.get_llm_config()


def pr_command(
    path: PathArg = Path(),
    action: Annotated[PRAction, typer.Argument(help="Action to perform: create or update")] = PRAction.CREATE,
    branch_name: Annotated[str | None, typer.Option("--branch", "-b", help="Target branch name")] = None,
    base_branch: Annotated[
        str | None,
        typer.Option("--base", help="Base branch for the PR (defaults to repo default)"),
    ] = None,
    title: Annotated[str | None, typer.Option("--title", "-t", help="Pull request title")] = None,
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
    non_interactive: Annotated[bool, typer.Option("--non-interactive", help="Run in non-interactive mode")] = False,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="LLM model for content generation"),
    ] = None,
    api_base: Annotated[str | None, typer.Option("--api-base", help="API base URL for LLM")] = None,
    api_key: Annotated[str | None, typer.Option("--api-key", help="API key for LLM")] = None,
    is_verbose: VerboseFlag = False,
) -> None:
    """Generate and manage pull requests.

    Creates or updates pull requests with AI-generated content.
    Handles branch creation, commits, and pushing changes.
    """
    setup_logging(is_verbose=is_verbose)
    logger.info("Starting PR command with action: %s", action)

    try:
        repo_path = validate_repo_path(path)
        if not repo_path:
            _exit_with_error("[red]Error:[/red] Not a valid Git repository")

        # Load LLM config from ConfigLoader
        llm_config = _load_llm_config(repo_path)

        # Command line options take precedence over config file
        if not model:
            model = llm_config["model"]
            # No need to extract provider here as it's not used

        if not api_key:
            api_key = llm_config["api_key"]

        api_base = api_base or llm_config["api_base"]

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
            api_key=api_key,
            api_base=api_base,
        )

        if action == PRAction.CREATE:
            # Handle branch creation or selection
            branch = _handle_branch_creation(options)
            if not branch:
                _exit_with_error("[red]Branch creation/selection cancelled.[/red]")

            # Handle commits if needed
            if options.commit_first and not _handle_commits(options):
                _exit_with_error("[red]Commit process failed or was cancelled.[/red]")

            # Handle push
            if not _handle_push(options, branch):
                _exit_with_error("[red]Push failed or was cancelled.[/red]")

            # Handle PR creation
            pr = _handle_pr_creation(options, branch)
            if not pr:
                _exit_with_error("[red]PR creation failed or was cancelled.[/red]")

        elif action == PRAction.UPDATE:
            current_branch = get_current_branch()

            # Find PR if number not provided
            pr = None
            if pr_number:
                # Try to get PR details
                try:
                    # Use gh CLI to get PR details
                    cmd = ["gh", "pr", "view", str(pr_number), "--json", "number,title,body,headRefName,url"]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
                    if result.returncode == 0:
                        pr_data = json.loads(result.stdout)
                        pr = PullRequest(
                            branch=pr_data.get("headRefName", current_branch),
                            title=pr_data.get("title", ""),
                            description=pr_data.get("body", ""),
                            url=pr_data.get("url", ""),
                            number=pr_data.get("number"),
                        )

                        # Check if we need to switch branches
                        if pr.branch != current_branch:
                            if options.interactive:
                                switch = questionary.confirm(
                                    f"PR #{pr.number} is for branch '{pr.branch}', but you're on '{current_branch}'. "
                                    "Switch branches?",
                                    default=True,
                                ).ask()
                                if switch:
                                    try:
                                        checkout_branch(pr.branch)
                                        console.print(f"[green]Switched to branch: {pr.branch}[/green]")
                                        current_branch = pr.branch
                                    except GitError as e:
                                        error_msg = f"[red]Error switching branches: {e}[/red]"
                                        _exit_with_error(error_msg, exception=e)
                            else:
                                _exit_with_error(
                                    f"[red]PR #{pr.number} is for branch '{pr.branch}', but you're on "
                                    f"'{current_branch}'.[/red]",
                                )
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as e:
                    error_msg = f"[red]Error getting PR details: {e}[/red]"
                    _exit_with_error(error_msg, exception=e)
            else:
                # Try to find PR for current branch
                pr = get_existing_pr(current_branch)
                if not pr:
                    _exit_with_error(f"[red]No PR found for branch '{current_branch}'.[/red]")

            # Handle commits if needed
            if options.commit_first and not _handle_commits(options):
                _exit_with_error("[red]Commit process failed or was cancelled.[/red]")

            # Handle push
            if not _handle_push(options, current_branch):
                _exit_with_error("[red]Push failed or was cancelled.[/red]")

            # Handle PR update
            updated_pr = _handle_pr_update(options, pr)
            if not updated_pr:
                _exit_with_error("[red]PR update failed or was cancelled.[/red]")

    except GitError as e:
        error_msg = f"[red]Git error: {e}[/red]"
        _exit_with_error(error_msg, exception=e)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Error in PR command")
        _exit_with_error("", exception=e)
