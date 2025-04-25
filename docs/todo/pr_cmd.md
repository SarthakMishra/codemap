# PR Command Enhancement Plan

## Current Implementation Analysis

The current PR command implementation provides basic functionality for:
- Creating and updating pull requests
- Interactive branch creation or selection
- Commit handling
- Basic PR title and description generation using commit messages or LLM
- Support for interactive mode to review/edit PR titles and descriptions

## Limitations and Areas for Improvement

1. **Branch Selection UX**
   - Limited interactivity for target and base branch selection
   - No persistence of preferred branches for different workflows
   - No visualization of branch relationships

2. **Configuration**
   - Minimal integration with `.codemap.yml` for PR-specific settings
   - No ability to define default branches for different repository contexts
   - No configuration for different git branching strategies

3. **Release Management**
   - No built-in support for different git workflows (GitFlow, GitHub Flow, etc.)
   - No automated release note generation for branch-to-branch PRs
   - No tracking of changes across related branches

4. **Content Generation**
   - Limited capability to generate meaningful PR content when there are no commits to analyze
   - No differentiation between feature PRs and release PRs

## Enhancement Plan

### 1. Interactive Branch Selection Improvements

- **Rich Branch Visualization**
  - Display branch hierarchy and relationships using a tree visualization
  - Show branch age, last commit date, and commit counts
  - Indicate branch protection status (if available via GitHub API)

- **Smart Branch Selection**
  - Use `questionary` with autocomplete for branch selection
  - Add fuzzy search capability for large repositories
  - Group branches by type (feature, release, hotfix)
  - Show preview of unique commits in the selected branch

```python
# Example implementation
def select_branch(message, default=None, exclude=None):
    branches = get_all_branches()  # Returns dict with metadata
    exclude = exclude or []
    
    # Filter out excluded branches
    branches = {k: v for k, v in branches.items() if k not in exclude}
    
    # Format for questionary with rich info
    choices = [
        {
            "name": f"{name} ({meta['last_commit_date']}, {meta['commit_count']} commits)",
            "value": name
        }
        for name, meta in branches.items()
    ]
    
    return questionary.select(
        message,
        choices=choices,
        default=default,
        qmark="🔀"
    ).ask()
```

### 2. Configuration via `.codemap.yml`

Add a dedicated `pr` section to the configuration file to support PR-specific settings:

```yaml
# PR configuration
pr:
  # Default branches
  defaults:
    base_branch: main
    feature_prefix: "feature/"
    
  # Git workflow strategy
  strategy: github-flow  # Options: github-flow, gitflow, trunk-based
  
  # Branch mapping for different PR types
  branch_mapping:
    # For GitFlow
    feature: 
      base: develop
      prefix: "feature/"
    release:
      base: main
      prefix: "release/"
    hotfix:
      base: main
      prefix: "hotfix/"
      
  # Content generation
  generate:
    title_strategy: commits  # Options: commits, llm, branch-name
    description_strategy: llm  # Options: commits, llm, template
    
    # Template for PR descriptions
    description_template: |
      ## Changes
      {changes}
      
      ## Testing
      {testing_instructions}
      
      ## Screenshots
      {screenshots}
```

- **Implementation Priority**: Create a `PRConfigLoader` class extending the current `ConfigLoader` functionality to handle PR-specific settings.
- **Configuration Validation**: Include validation of PR configuration to ensure correct branch patterns and valid strategy names.

### 3. Git Workflow Strategy Support

Implement built-in support for common git workflows with appropriate branch management:

- **GitHub Flow**
  - Feature branches -> main
  - Simple, linear workflow

- **GitFlow**
  - Feature branches -> develop
  - Release branches -> main (with back-merge to develop)
  - Hotfix branches -> main (with back-merge to develop)

- **Trunk-Based Development**
  - Short-lived feature branches -> main
  - Emphasis on small, frequent PRs

```python
class WorkflowStrategy:
    """Base class for git workflow strategies."""
    
    def get_default_base(self, branch_type):
        """Get the default base branch for a given branch type."""
        raise NotImplementedError
        
    def suggest_branch_name(self, branch_type, description):
        """Suggest a branch name based on the workflow."""
        raise NotImplementedError
        
    def get_branch_types(self):
        """Get valid branch types for this workflow."""
        raise NotImplementedError


class GitFlowStrategy(WorkflowStrategy):
    """Implementation of GitFlow workflow strategy."""
    
    def get_default_base(self, branch_type):
        mapping = {
            "feature": "develop",
            "release": "main",
            "hotfix": "main",
            "bugfix": "develop"
        }
        return mapping.get(branch_type, "develop")
    
    # ... other implementations
```

### 4. Enhanced PR Content Generation

Improve content generation for different PR scenarios:

- **Feature Branch PRs**
  - Focus on the changes made
  - Include testing steps
  - Reference relevant issues

- **Release PRs (develop -> main)**
  - Aggregate changes from all included feature branches
  - Generate categorized release notes (features, bugfixes, etc.)
  - Include migration notes if applicable

- **Empty PRs**
  - Generate content based on branch naming and prior PRs
  - Include PR history for the branches involved
  - Allow template-based generation

```python
def generate_release_pr_content(base_branch, head_branch, config):
    """Generate content for a release PR (e.g., develop -> main)."""
    # Get all PRs merged to head_branch since last merge to base_branch
    merged_prs = get_merged_prs_since_last_release(base_branch, head_branch)
    
    # Categorize PRs
    categorized = categorize_prs(merged_prs)
    
    # Generate structured content
    return {
        "title": f"Release {suggest_version_from_changes(categorized)}",
        "description": generate_release_notes(categorized, config)
    }
```

### 5. Implementation Roadmap

#### Phase 1: Configuration and Structure
- Add PR configuration to `.codemap.yml`
- Create `PRConfigLoader` class
- Refactor existing PR command to use the configuration

#### Phase 2: Interactive Improvements
- Enhance branch selection with rich visualization
- Implement smart branch filtering and search
- Add branch relationship visualization

#### Phase 3: Workflow Strategies
- Implement workflow strategy classes for different git workflows
- Add branch mapping for different PR types
- Integrate workflow detection based on branch structure

#### Phase 4: Advanced Content Generation
- Implement specialized content generation for different PR types
- Add support for release note generation
- Create template-based content customization

#### Phase 5: Testing and Documentation
- Comprehensive testing of all workflows
- Documentation for each strategy
- Examples of different PR workflows

## Future Extensions

1. **Integration with Issue Trackers**
   - Automatically link to relevant issues
   - Update issue status when PR is created/merged

2. **CI/CD Integration**
   - Show build status in PR command
   - Run pre-PR checks and validations

3. **PR Analytics**
   - Track PR lifecycle metrics
   - Provide insights on PR patterns and bottlenecks

4. **Multi-Repository Support**
   - Manage PRs across multiple repositories
   - Handle dependencies between repositories
