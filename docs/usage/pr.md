# Pull Requests (`pr`)

The `codemap pr` command helps you create and manage pull requests with ease. It integrates with the existing `codemap commit` command to provide a seamless workflow from code changes to pull request creation.

## PR Command Features

- Create branches with intelligent naming based on your current changes
- Support for multiple Git workflow strategies (GitHub Flow, GitFlow, Trunk-Based)
- Rich branch visualization with metadata and relationships
- Smart base branch selection based on branch type
- Automatic content generation for different PR types (feature, release, hotfix)
- Workflow-specific PR templates based on branch type
- Interactive PR content editing with previews
- Update existing PRs with new commits
- Configurable via `.codemap.yml` for team-wide settings (see [Configuration](configuration.md))

## PR Command Requirements

- Git repository with a remote named `origin`
- GitHub CLI (`gh`) installed for PR creation and management
- Valid GitHub authentication for the `gh` CLI

## Creating a PR

```bash
codemap pr create [OPTIONS]
# Or using the alias:
cm pr create [OPTIONS]
```

**Options:**

- `--branch`, `-b`: Target branch name
- `--type`, `-t`: Branch type (e.g., feature, release, hotfix, bugfix). Valid types depend on workflow strategy.
- `--base`: Base branch for the PR (defaults to repo default or workflow-defined default)
- `--title`: Pull request title
- `--desc`, `-d`: Pull request description (file path or text)
- `--no-commit`: Skip the commit process before creating PR
- `--force-push`, `-f`: Force push the branch
- `--workflow`, `-w`: Git workflow strategy (github-flow, gitflow, trunk-based). Overrides config (`pr.strategy`).
- `--non-interactive`: Run in non-interactive mode
- `--model`, `-m`: LLM model for content generation (overrides config `llm.model`).
- `--bypass-hooks`, `--no-verify`: Bypass git hooks with `--no-verify`
- `--api-base`: (Advanced) Custom API base URL for LLM
- `--api-key`: (Advanced) Custom API key for LLM
- `--verbose`, `-v`: Enable verbose logging

## Updating a PR

```bash
codemap pr update [OPTIONS]
# Or using the alias:
cm pr update [OPTIONS]
```

**Options:**

- `--pr`: PR number to update (required if not updating PR for current branch)
- `--title`: New PR title
- `--desc`, `-d`: New PR description (file path or text)
- `--force-push`, `-f`: Force push the branch (use with caution)
- `--workflow`, `-w`: Git workflow strategy (github-flow, gitflow, trunk-based)
- `--non-interactive`: Run in non-interactive mode
- `--model`, `-m`: LLM model for content generation
- `--bypass-hooks`, `--no-verify`: Bypass git hooks with `--no-verify`
- `--api-base`: (Advanced) Custom API base URL for LLM
- `--api-key`: (Advanced) Custom API key for LLM
- `--verbose`, `-v`: Enable verbose logging

/// warning
--no-commit is NOT an option for 'update'
///

## Notes
- `[PATH]` is not required; the command operates in the current repository by default.
- Advanced LLM options (`--api-base`, `--api-key`) are rarely needed unless using a custom or self-hosted LLM endpoint.

## Git Workflow Strategies

The PR command supports multiple Git workflow strategies:

1. **GitHub Flow** (default)
   - Simple, linear workflow
   - Feature branches merge directly to main
   
2. **GitFlow**
   - Feature branches → develop
   - Release branches → main
   - Hotfix branches → main (with back-merge to develop)
   
3. **Trunk-Based Development**
   - Short-lived feature branches
   - Emphasizes small, frequent PRs

## PR Template System

CodeMap includes a robust PR template system that automatically generates appropriate titles and descriptions based on the selected workflow strategy, branch type, and changes being made. See the [Configuration](configuration.md) page for details on customizing templates.

## Examples

```bash
# Create PR using workflow-specific templates (GitFlow)
codemap pr create --workflow gitflow --type feature

# Create PR with custom title but workflow-based description
codemap pr create --title "My Custom Title" --workflow trunk-based

# Override both the workflow template and use custom description
codemap pr create --desc "Custom description with **markdown** support"

# Non-interactive PR creation with defined template usage
codemap pr create --non-interactive --workflow gitflow --type release

# Update an existing PR by PR number
codemap pr update --pr 42 --title "Update PR Title"

# Bypass git hooks when creating or updating a PR
codemap pr create --bypass-hooks
codemap pr update --bypass-hooks
``` 