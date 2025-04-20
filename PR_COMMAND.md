# CodeMap PR Command

The `codemap pr` command helps you create and manage pull requests with ease. It integrates with the existing `codemap commit` command to provide a seamless workflow from code changes to pull request creation.

## Features

- Create a new branch by analyzing your changes
- Commit all changes using the pre-existing commit tools
- Push changes to origin
- Generate a PR with appropriate messages by analyzing commits
- Update existing PRs with new commits
- Interactive workflow with helpful prompts

## Requirements

- Git repository with a remote named `origin`
- GitHub CLI (`gh`) installed for PR creation and management
- Valid GitHub authentication for the `gh` CLI

## Usage

### Creating a PR

```bash
codemap pr create [PATH] [OPTIONS]
```

**Arguments:**
- `PATH`: Path to the repository (defaults to current directory)

**Options:**
- `--branch`, `-b`: Branch name to use (will be created if it doesn't exist)
- `--base`: Base branch for the PR (default: main or master)
- `--title`, `-t`: PR title (generated from commits if not provided)
- `--description`, `-d`: PR description (generated from commits if not provided)
- `--no-commit`: Don't commit changes before creating PR
- `--force-push`, `-f`: Force push branch to remote
- `--non-interactive`: Run in non-interactive mode
- `--model`, `-m`: LLM model to use for commit message generation
- `--api-key`: API key for LLM provider

### Updating a PR

```bash
codemap pr update [PR_NUMBER] [OPTIONS]
```

**Arguments:**
- `PR_NUMBER`: PR number to update (if not provided, will try to find PR for current branch)

**Options:**
- `--path`, `-p`: Path to repository
- `--title`, `-t`: New PR title
- `--description`, `-d`: New PR description
- `--no-commit`: Don't commit changes before updating PR
- `--force-push`, `-f`: Force push branch to remote
- `--non-interactive`: Run in non-interactive mode
- `--model`, `-m`: LLM model to use for commit message generation
- `--api-key`: API key for LLM provider

## Examples

### Create a new PR

```bash
# Interactive mode (recommended)
codemap pr create

# Specify a branch name
codemap pr create --branch feature-branch

# Create PR with custom title and description
codemap pr create --title "My Feature" --description "This PR adds a new feature"

# Create PR without committing changes
codemap pr create --no-commit
```

### Update an existing PR

```bash
# Update PR by number
codemap pr update 123

# Update PR for current branch
codemap pr update

# Update PR with new title
codemap pr update --title "Updated Feature"
```

## Workflow

The typical workflow with the `codemap pr` command is:

1. Make changes to your code
2. Run `codemap pr create`
3. Follow the interactive prompts to:
   - Create or select a branch
   - Commit your changes
   - Push to remote
   - Create a PR with generated title and description
4. Make additional changes
5. Run `codemap pr update` to add new commits and update the PR

## Integration with Commit Command

The PR command integrates with the existing `codemap commit` command, using the same LLM-powered commit message generation. This ensures consistent commit messages and PR descriptions.