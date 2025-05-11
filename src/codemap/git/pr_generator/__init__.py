"""
PR generation package for CodeMap.

This package provides modules for generating and managing pull requests.

"""

from codemap.git.pr_generator.decorators import git_operation
from codemap.git.pr_generator.generator import PRGenerator
from codemap.git.pr_generator.prompts import PR_DESCRIPTION_PROMPT, PR_TITLE_PROMPT, format_commits_for_prompt
from codemap.git.pr_generator.schemas import BranchType, PRContent, PullRequest, WorkflowStrategySchema
from codemap.git.pr_generator.strategies import (
	GitFlowStrategy,
	GitHubFlowStrategy,
	TrunkBasedStrategy,
	WorkflowStrategy,
	create_strategy,
)
from codemap.git.pr_generator.utils import (
	PRCreationError,
	create_pull_request,
	detect_branch_type,
	generate_pr_content_from_template,
	generate_pr_description_from_commits,
	generate_pr_description_with_llm,
	generate_pr_title_from_commits,
	generate_pr_title_with_llm,
	get_branch_description,
	get_default_branch,
	get_existing_pr,
	suggest_branch_name,
	update_pull_request,
)

__all__ = [
	# Prompts
	"PR_DESCRIPTION_PROMPT",
	"PR_TITLE_PROMPT",
	# Type definitions
	"BranchType",
	"GitFlowStrategy",
	"GitHubFlowStrategy",
	"PRContent",
	# Classes
	"PRCreationError",
	"PRGenerator",
	"PullRequest",
	"TrunkBasedStrategy",
	"WorkflowStrategy",
	"WorkflowStrategySchema",
	# Functions - PR creation/update
	"create_pull_request",
	# Strategy functions
	"create_strategy",
	"detect_branch_type",
	"format_commits_for_prompt",
	"generate_pr_content_from_template",
	"generate_pr_description_from_commits",
	"generate_pr_description_with_llm",
	# Functions - Content generation
	"generate_pr_title_from_commits",
	"generate_pr_title_with_llm",
	"get_branch_description",
	"get_default_branch",
	"get_existing_pr",
	# Decorators
	"git_operation",
	"suggest_branch_name",
	"update_pull_request",
]
