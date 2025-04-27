"""PR template definitions for different workflow strategies."""

# Default PR Templates
DEFAULT_PR_TEMPLATE = {
	"title": "{branch_type}: {description}",
	"description": """## Description

{description}

## Changes

-

## Related Issues

-
""",
}

# GitHub Flow PR Templates
GITHUB_FLOW_PR_TEMPLATE = {
	"title": "{description}",
	"description": """## Description

{description}

## What does this PR do?

<!-- Please include a summary of the change and which issue is fixed. -->

## Changes

-

## Screenshots (if appropriate)

## Testing completed

- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

## Related Issues

<!-- Please link to any related issues here -->

- Closes #
""",
}

# Trunk-Based Development PR Templates
TRUNK_BASED_PR_TEMPLATE = {
	"title": "{description}",
	"description": """## Change Description

{description}

## Implementation

<!-- Briefly describe implementation details -->

-

## Test Plan

<!-- How was this tested? -->

- [ ] Unit tests added/updated
- [ ] Integration tested

## Rollout Plan

<!-- How should this be deployed? -->

- [ ] Can be deployed immediately
- [ ] Requires feature flag
- [ ] Requires data migration

## Related Issues

- Fixes #
""",
}

# GitFlow PR Templates by Branch Type
GITFLOW_PR_TEMPLATES = {
	"feature": {
		"title": "Feature: {description}",
		"description": """## Feature Description

{description}

## Implemented Changes

-

## Testing Performed

- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

## Related Issues

- Closes #
""",
	},
	"release": {
		"title": "Release {description}",
		"description": """## Release {description}

### Features

-

### Bug Fixes

-

### Breaking Changes

-

## Deployment Notes

-

## Testing Required

- [ ] Smoke tests
- [ ] Regression tests
- [ ] Performance tests
""",
	},
	"hotfix": {
		"title": "Hotfix: {description}",
		"description": """## Hotfix: {description}

### Issue Description

<!-- Describe the issue being fixed -->

### Fix Implementation

<!-- Describe how the issue was fixed -->

-

### Testing Performed

- [ ] Verified fix locally
- [ ] Added regression test

### Impact Analysis

- Affected components:
- Risk assessment:
""",
	},
	"bugfix": {
		"title": "Fix: {description}",
		"description": """## Bug Fix

### Issue Description

{description}

### Root Cause

<!-- What caused the bug? -->

### Fix Implementation

-

### Testing Performed

- [ ] Added test case that reproduces the bug
- [ ] Verified fix locally

### Related Issues

- Fixes #
""",
	},
}
