# Test Coverage Improvement Plan

This document identifies modules with less than 80% test coverage that should be prioritized for test improvements.

## High Priority (< 30% coverage)

| Module | Coverage | Missing Lines | Implementation |
|--------|----------|---------------|---------------|
| src/codemap/analyzer/processor.py | 95% | 15, 90, 93 | ✅ |
| src/codemap/cli/commit_cmd.py | 73% | Multiple sections | ✅ |
| src/codemap/cli/generate_cmd.py | 100% | None | ✅ |
| src/codemap/cli/init_cmd.py | 100% | None | ✅ |
| src/codemap/processor/storage/lance.py | 77% | Multiple sections | ✅ |

## Medium Priority (30-60% coverage)

| Module | Coverage | Missing Lines | Implementation |
|--------|----------|---------------|---------------|
| src/codemap/git/command.py | 87% | 33, 55, 206, 247, 249, 260-269, 277-280, 286, 306-307, 312-313, 347-349, 362-363 | ✅ |
| src/codemap/processor/storage/utils.py | 90% | 32, 119-125 | ✅ |
| src/codemap/git/message_generator.py | 50% | Multiple sections | ❌ |
| src/codemap/git/diff_splitter.py | 55% | Multiple sections | ❌ |
| src/codemap/cli/pr_cmd.py | 42% | Multiple sections | ❌ |

## Lower Priority (60-80% coverage)

| Module | Coverage | Missing Lines | Implementation |
|--------|----------|---------------|---------------|
| src/codemap/generators/markdown_generator.py | 65% | Multiple sections | ❌ |
| src/codemap/processor/chunking/tree_sitter.py | 64% | Multiple sections | ❌ |
| src/codemap/processor/embedding/generator.py | 66% | Multiple sections | ❌ |
| src/codemap/utils/cli_utils.py | 95% | 15, 111-112 | ✅ |
| src/codemap/processor/analysis/tree_sitter/analyzer.py | 78% | Multiple sections | ❌ |
| src/codemap/processor/analysis/tree_sitter/languages/python.py | 74% | Multiple sections | ❌ |
| src/codemap/utils/git_utils.py | 58% | Multiple sections | ❌ |
| src/codemap/git/interactive.py | 75% | Multiple sections | ❌ |

## Progress Summary

- **Major Improvements**: 
  - analyzer/processor.py: 25% → 95% ✅
  - cli/commit_cmd.py: 29% → 73% ✅
  - cli/generate_cmd.py: 25% → 100% ✅
  - cli/init_cmd.py: 23% → 100% ✅
  - processor/storage/lance.py: 19% → 77% ✅
  - processor/storage/utils.py: 34% → 90% ✅
  - git/command.py: 48% → 87% ✅
  - utils/cli_utils.py: 77% → 95% ✅

- **Overall Coverage**: 74% (up from previous 68%)

## Recommended Next Steps

1. Focus on remaining medium-priority modules:
   - Git message generator in `git/message_generator.py` (50%)
   - Git diff splitter in `git/diff_splitter.py` (55%)
   - PR command implementation in `cli/pr_cmd.py` (42%)

2. Continue improving lower-priority modules, especially:
   - `generators/markdown_generator.py` (65%)
   - `processor/chunking/tree_sitter.py` (64%)
   - `processor/embedding/generator.py` (66%)
   - `processor/analysis/tree_sitter/languages/python.py` (74%)

3. Write focused tests that cover:
   - Major code paths
   - Edge cases
   - Error conditions

4. Use mocking where appropriate to isolate unit tests from external dependencies

5. Regularly run coverage reports (`task test`) to track progress 