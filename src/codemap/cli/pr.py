"""Command for generating and managing pull requests."""

from __future__ import annotations

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
from rich.console import Console
from rich.panel import Panel

from codemap.git import GitWrapper
from codemap.git.commit.diff_splitter import DiffSplitter, SplitStrategy
from codemap.git.commit.interactive import process_all_chunks
from codemap.git.utils.git_utils import GitError
from codemap.git.utils.pr_utils import (
    PullRequest,
    branch_exists,
    checkout_branch,
    create_branch,
    create_pull_request,
    get_commit_messages,
    get_current_branch,
    get_default_branch,
    get_existing_pr,
    push_branch,
    suggest_branch_name,
    update_pull_request,
)
from codemap.utils import loading_spinner, validate_repo_path
from codemap.utils.llm_utils import create_universal_generator, generate_message

app = typer.Typer(help="Generate and manage pull requests")
console = Console()
logger = logging.getLogger(__name__)


class PRAction(str, Enum):
    """Actions for PR command."""

    CREATE = "create"
    UPDATE = "update"


@dataclass
class PROptions:
    """Options for the PR command."""

    repo_path: Path
    branch_name: str | None = field(default=None)
    base_branch: str | None = field(default=None)
    title: str | None = field(default=None)
    description: str | None = field(default=None)
    commit_first: bool = field(default=True)
    force_push: bool = field(default=False)
    pr_number: int | None = field(default=None)
    interactive: bool = field(default=True)
    model: str | None = field(default=None)
    provider: str | None = field(default=None)
    api_base: str | None = field(default=None)
    api_key: str | None = field(default=None)


@app.callback()
def callback() -> None:
    """Generate and manage pull requests."""


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
        git = GitWrapper(options.repo_path)
        diff = git.get_uncommitted_changes()

        # Generate a suggested branch name based on the changes
        suggested_name = ""
        if diff.files:
            # Use the diff splitter to get semantic chunks
            splitter = DiffSplitter(options.repo_path)
            chunks = splitter.split_diff(diff, str(SplitStrategy.SEMANTIC))

            if chunks:
                # Set up message generator for the first chunk
                generator = create_universal_generator(
                    repo_path=options.repo_path,
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
                except (ValueError, RuntimeError, ConnectionError):
                    # Fallback to a simple branch name
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
    git = GitWrapper(options.repo_path)
    diff = git.get_uncommitted_changes()
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

        # Set up message generator
        generator = create_universal_generator(
            repo_path=options.repo_path,
            model=options.model,
            api_key=options.api_key,
            api_base=options.api_base,
        )

        # Process all chunks
        result = process_all_chunks(options.repo_path, chunks, generator, git, interactive=options.interactive)
    except (OSError, ValueError, RuntimeError, ConnectionError) as e:
        console.print(f"[red]Error committing changes: {e}[/red]")
        return False
    else:
        return result == 0


def _handle_push(options: PROptions, branch_name: str) -> bool:
    """Handle pushing changes to remote.

    Args:
        options: PR options
        branch_name: Branch name to push

    Returns:
        True if successful, False otherwise
    """
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


def _handle_pr_creation(options: PROptions, branch_name: str) -> PullRequest | None:
    """Handle PR creation.

    Args:
        options: PR options
        branch_name: Branch name

    Returns:
        Created PR or None if cancelled
    """
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
            from codemap.git.utils.pr_utils import generate_pr_description_with_llm, generate_pr_title_with_llm

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
        from codemap.git.utils.pr_utils import generate_pr_description_from_commits, generate_pr_title_from_commits

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


def _handle_pr_update(options: PROptions, pr: PullRequest) -> PullRequest | None:
    """Handle PR update.

    Args:
        options: PR options
        pr: Existing PR

    Returns:
        Updated PR or None if cancelled
    """
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
            from codemap.git.utils.pr_utils import generate_pr_description_with_llm, generate_pr_title_with_llm

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
        from codemap.git.utils.pr_utils import generate_pr_description_from_commits, generate_pr_title_from_commits

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


def _load_llm_config(repo_path: Path) -> dict:
    """Load LLM configuration from .codemap.yml file.

    Args:
        repo_path: Path to the repository

    Returns:
        Dictionary with LLM configuration values
    """
    config = {
        "model": "gpt-4o-mini",  # Default fallback model
        "api_base": None,
        "api_key": None,
    }

    config_file = repo_path / ".codemap.yml"
    if config_file.exists():
        try:
            import yaml

            with config_file.open("r") as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config is not None and isinstance(yaml_config, dict) and "commit" in yaml_config:
                commit_config = yaml_config["commit"]

                # Load LLM settings if available
                if "llm" in commit_config and isinstance(commit_config["llm"], dict):
                    llm_config = commit_config["llm"]

                    if "model" in llm_config:
                        config["model"] = llm_config["model"]

                    if "api_base" in llm_config:
                        config["api_base"] = llm_config["api_base"]

                    # Use the same API keys from commit configuration
                    # This ensures consistency between commit and PR features
                    provider = None
                    if "/" in config["model"]:
                        provider = config["model"].split("/")[0].lower()

                    if provider:
                        config_key = f"{provider}_api_key"
                        if config_key in llm_config:
                            config["api_key"] = llm_config[config_key]

                    # Also check for generic API key
                    if "api_key" in llm_config and not config["api_key"]:
                        config["api_key"] = llm_config["api_key"]

        except (ImportError, yaml.YAMLError, OSError) as e:
            logger.warning("Error loading config: %s", e)

    return config


@app.command(help="Create a new pull request")
def create(
    path: Annotated[
        Path | None,
        typer.Argument(
            help="Path to repository",
            exists=True,
        ),
    ] = None,
    branch_name: Annotated[
        str | None,
        typer.Option(
            "--branch",
            "-b",
            help="Branch name to use (will be created if it doesn't exist)",
        ),
    ] = None,
    base_branch: Annotated[
        str | None,
        typer.Option(
            "--base",
            help="Base branch for the PR (default: main or master)",
        ),
    ] = None,
    title: Annotated[
        str | None,
        typer.Option(
            "--title",
            "-t",
            help="PR title (generated from commits if not provided)",
        ),
    ] = None,
    description: Annotated[
        str | None,
        typer.Option(
            "--description",
            "-d",
            help="PR description (generated from commits if not provided)",
        ),
    ] = None,
    no_commit: Annotated[
        bool,
        typer.Option(
            "--no-commit",
            help="Don't commit changes before creating PR",
            is_flag=True,
        ),
    ] = False,
    force_push: Annotated[
        bool,
        typer.Option(
            "--force-push",
            "-f",
            help="Force push branch to remote",
            is_flag=True,
        ),
    ] = False,
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive",
            help="Run in non-interactive mode",
            is_flag=True,
        ),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="LLM model to use for PR content generation",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="API key for LLM provider",
        ),
    ] = None,
) -> int:
    """Create a new pull request.

    Args:
        path: Path to repository
        branch_name: Branch name to use
        base_branch: Base branch for the PR
        title: PR title
        description: PR description
        no_commit: Don't commit changes before creating PR
        force_push: Force push branch to remote
        non_interactive: Run in non-interactive mode
        model: LLM model to use for PR content generation
        api_key: API key for LLM provider

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    repo_path = validate_repo_path(path)
    if not repo_path:
        console.print("[red]Error:[/red] Not a valid Git repository")
        return 1

    # Load LLM config from .codemap.yml
    llm_config = _load_llm_config(repo_path)

    # Command line options take precedence over config file
    if not model:
        model = llm_config["model"]

    if not api_key:
        api_key = llm_config["api_key"]

    api_base = llm_config["api_base"]

    options = PROptions(
        repo_path=repo_path,
        branch_name=branch_name,
        base_branch=base_branch,
        title=title,
        description=description,
        commit_first=not no_commit,
        force_push=force_push,
        interactive=not non_interactive,
        model=model,
        api_key=api_key,
        api_base=api_base,
    )

    # Handle branch creation or selection
    branch = _handle_branch_creation(options)
    if not branch:
        console.print("[red]Branch creation/selection cancelled.[/red]")
        return 1

    # Handle commits if needed
    if options.commit_first and not _handle_commits(options):
        console.print("[red]Commit process failed or was cancelled.[/red]")
        return 1

    # Handle push
    if not _handle_push(options, branch):
        console.print("[red]Push failed or was cancelled.[/red]")
        return 1

    # Handle PR creation
    pr = _handle_pr_creation(options, branch)
    if not pr:
        console.print("[red]PR creation failed or was cancelled.[/red]")
        return 1

    return 0


@app.command(help="Update an existing pull request")
def update(
    pr_number: Annotated[
        int | None,
        typer.Argument(
            help="PR number to update (if not provided, will try to find PR for current branch)",
        ),
    ] = None,
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            "-p",
            help="Path to repository",
            exists=True,
        ),
    ] = None,
    title: Annotated[
        str | None,
        typer.Option(
            "--title",
            "-t",
            help="New PR title",
        ),
    ] = None,
    description: Annotated[
        str | None,
        typer.Option(
            "--description",
            "-d",
            help="New PR description",
        ),
    ] = None,
    no_commit: Annotated[
        bool,
        typer.Option(
            "--no-commit",
            help="Don't commit changes before updating PR",
            is_flag=True,
        ),
    ] = False,
    force_push: Annotated[
        bool,
        typer.Option(
            "--force-push",
            "-f",
            help="Force push branch to remote",
            is_flag=True,
        ),
    ] = False,
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive",
            help="Run in non-interactive mode",
            is_flag=True,
        ),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="LLM model to use for PR content generation",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="API key for LLM provider",
        ),
    ] = None,
) -> int:
    """Update an existing pull request.

    Args:
        pr_number: PR number to update
        path: Path to repository
        title: New PR title
        description: New PR description
        no_commit: Don't commit changes before updating PR
        force_push: Force push branch to remote
        non_interactive: Run in non-interactive mode
        model: LLM model to use for PR content generation
        api_key: API key for LLM provider

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    repo_path = validate_repo_path(path)
    if not repo_path:
        console.print("[red]Error:[/red] Not a valid Git repository")
        return 1

    # Load LLM config from .codemap.yml
    llm_config = _load_llm_config(repo_path)

    # Command line options take precedence over config file
    if not model:
        model = llm_config["model"]

    if not api_key:
        api_key = llm_config["api_key"]

    api_base = llm_config["api_base"]

    options = PROptions(
        repo_path=repo_path,
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

    # Get current branch
    current_branch = get_current_branch()

    # Find PR if number not provided
    pr = None
    if pr_number:
        # Try to get PR details
        try:
            import json
            import subprocess

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
                                console.print(f"[red]Error switching branches: {e}[/red]")
                                return 1
                    else:
                        console.print(
                            f"[red]PR #{pr.number} is for branch '{pr.branch}', but you're on "
                            f"'{current_branch}'.[/red]",
                        )
                        return 1
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as e:
            console.print(f"[red]Error getting PR details: {e}[/red]")
    else:
        # Try to find PR for current branch
        pr = get_existing_pr(current_branch)
        if not pr:
            console.print(f"[red]No PR found for branch '{current_branch}'.[/red]")
            return 1

    # Handle commits if needed
    if options.commit_first and not _handle_commits(options):
        console.print("[red]Commit process failed or was cancelled.[/red]")
        return 1

    # Handle push
    if not _handle_push(options, current_branch):
        console.print("[red]Push failed or was cancelled.[/red]")
        return 1

    # Handle PR update
    updated_pr = _handle_pr_update(options, pr)
    if not updated_pr:
        console.print("[red]PR update failed or was cancelled.[/red]")
        return 1

    return 0
