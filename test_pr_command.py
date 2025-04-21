#!/usr/bin/env python3
"""Test script for the PR command."""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from codemap.cli.pr import create as pr_create
from codemap.git.utils.pr_utils import (
    PullRequest,
    branch_exists,
    checkout_branch,
    create_branch,
    create_pull_request,
    generate_pr_description_from_commits,
    generate_pr_title_from_commits,
    get_commit_messages,
    get_current_branch,
    get_default_branch,
    get_existing_pr,
    push_branch,
    suggest_branch_name,
    update_pull_request,
)

# Test the PR utilities
print("Testing PR utilities...")

# Test branch name suggestion
commits = ["feat(api): Add new endpoint", "fix: Fix bug in authentication"]
branch_name = suggest_branch_name(commits)
print(f"Suggested branch name: {branch_name}")

# Test PR title generation
title = generate_pr_title_from_commits(commits)
print(f"Generated PR title: {title}")

# Test PR description generation
description = generate_pr_description_from_commits(commits)
print(f"Generated PR description:\n{description}")

print("PR utilities tests completed successfully!")

print("\nAll tests completed successfully!")