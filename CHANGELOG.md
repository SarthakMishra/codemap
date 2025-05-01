# CHANGELOG


## v2.1.0-next.1 (2025-05-01)

### Chores

- Bump project version to 2.0.0 in pyproject.toml
  ([`22192c4`](https://github.com/SarthakMishra/codemap/commit/22192c4344e4106a97d4ebd020db40c827cd3087))


## v2.0.0 (2025-05-01)

### Features

- Add build and pyproject-hooks dependencies, update CodeMap version to 1.0.0
  ([`d033caa`](https://github.com/SarthakMishra/codemap/commit/d033caaa475861eb65e7d68ebdc6e65c6c6cddfc))

This commit introduces the 'build' and 'pyproject-hooks' packages as dependencies, and updates the
  version of CodeMap to 1.0.0. The build dependency is necessary for project packaging and building.

- **config**: Add project source and changelog to pyproject.toml
  ([`32bec04`](https://github.com/SarthakMishra/codemap/commit/32bec047b6a0a13175ee982abaa53b48cb7532bf))

- **project**: Bump version to 1.0.0 and add build dependency
  ([`5b2cd6c`](https://github.com/SarthakMishra/codemap/commit/5b2cd6c12cc3145ced9ce753fbc0ec7e09edbdda))

Bump the project version to 1.0.0 and add build dependency to support building and packaging the
  project.

BREAKING CHANGE: version bump to 1.0.0


## v1.0.0 (2025-05-01)


## v1.0.0-next.1 (2025-05-01)

### Bug Fixes

- Add dotenv loading to cli_entry.py for proper environment variable handling
  ([`112a85b`](https://github.com/SarthakMishra/codemap/commit/112a85bac64b0adc5cdda14dd7da72f1dc82e598))

- Add required token_limit to test configuration
  ([`07e8a58`](https://github.com/SarthakMishra/codemap/commit/07e8a58813fe51b51a10ffcb571d006820330209))

- Add type stubs for daemon and lockfile
  ([`bc6127c`](https://github.com/SarthakMishra/codemap/commit/bc6127c52534924a7aef7a71311c1eb0895e51f6))

- Correct documentation output path handling
  ([`eda7719`](https://github.com/SarthakMishra/codemap/commit/eda77194c7339246127e5a37e969a06727ea2e6c))

- Improve dependency checking
  ([`e6897ab`](https://github.com/SarthakMishra/codemap/commit/e6897abbd1cb5fd773ce87515a24b3de531f1b1c))

- Add user prompts when Python or pip is not found

- Remove automatic pip installation

- Add confirmation before proceeding with installation

- Improve file extension detection and output directory selection
  ([`51f59fb`](https://github.com/SarthakMishra/codemap/commit/51f59fb55c36e60395f2c47fde0f4c7aa387b65e))

- Added file extension detection using Pygments lexers instead of hardcoding Python files only

- Fixed issue with parent directory selection for output files

- Updated ConfigLoader to properly detect and use config file locations

- Improved CLI to handle file paths more robustly

- Improve input handling in scripts
  ([`051bc3a`](https://github.com/SarthakMishra/codemap/commit/051bc3a2a65573c5aaef24b79204b3e358d80f47))

- Add read_input function to handle both piped and direct input

- Fix script paths in error messages

- Remove problematic tty redirection

- Improve interactive input handling
  ([`7025e3b`](https://github.com/SarthakMishra/codemap/commit/7025e3b835011d0decb29525c36f6fc4ef94af51))

- Add proper tty redirection for interactive input

- Remove unnecessary pipe checks

- Simplify script execution flow

- Improve pattern matching for .venv and directory exclusions
  ([`3d73491`](https://github.com/SarthakMishra/codemap/commit/3d7349105c0651c2957df7553b8dcce2d9bd8363))

- Add better handling for .dot patterns (like .venv)

- Support directory patterns with and without trailing slash

- Add common default patterns to gitignore

- Fix pattern matching for files in subdirectories

- Resolve CLI entry point conflicts and fix import errors by renaming cli.py to cli_entry.py
  ([`a51422a`](https://github.com/SarthakMishra/codemap/commit/a51422a0ba2181035ea634e91fa94c57e2b57cad))

- Resolve linting errors in markdown generator
  ([`a2dad69`](https://github.com/SarthakMishra/codemap/commit/a2dad69e368be848587342980584c475fd44c005))

- Fix FBT001/FBT002: Make is_last a keyword-only argument

- Fix TRY300: Restructure try-except block for better error handling

- Improve code organization in _add_path_to_tree method

- Restore default token_limit to 1000 in DEFAULT_CONFIG
  ([`e7246ad`](https://github.com/SarthakMishra/codemap/commit/e7246adb47e6d970a0cdb13aef680d6a69762f25))

- Simplify installation process
  ([`fde335f`](https://github.com/SarthakMishra/codemap/commit/fde335fe39ba44d75b228da83519e6a508e3c504))

- Use bash -s to maintain interactivity with piped input

- Update installation commands to single-line format

- Remove two-step installation process

- Suppress type-checking lint for Path import as it's used at runtime
  ([`af83808`](https://github.com/SarthakMishra/codemap/commit/af8380806772e30e591fae936b62f7de9b6e5802))

- Update default token_limit to match .codemap.yml
  ([`afd1e47`](https://github.com/SarthakMishra/codemap/commit/afd1e47518b3cb25a3e4256829a93205549420ac))

- Update exclude_patterns to use .venv instead of venv
  ([`ac62995`](https://github.com/SarthakMishra/codemap/commit/ac62995740a2400d16e6fa45a657c4991fded7e4))

- Update repository URLs to correct GitHub path
  ([`8f45bfe`](https://github.com/SarthakMishra/codemap/commit/8f45bfed1004baf2b959706bec9b00c9530a4dda))

- Update repository URLs from sarthakagrawal927/code-map to SarthakMishra/codemap

- Fix URLs in install and upgrade scripts to prevent authentication prompts

- Update tree-sitter initialization to use CLI commands for building language library
  ([`f7af30d`](https://github.com/SarthakMishra/codemap/commit/f7af30d6fd69feb83ab4798bf7a42ce7f5d10943))

- **cli**: Handle errors during branch relation checks
  ([`ed3d13d`](https://github.com/SarthakMishra/codemap/commit/ed3d13d5cd59589b68b3e422094460b12fdd01b7))

Previously, errors during branch relation checks could lead to the program crashing. This change
  adds error handling to ensure that the program can recover and provide a meaningful fallback.

- **cli**: Handle typer.Exit exceptions properly
  ([`997c994`](https://github.com/SarthakMishra/codemap/commit/997c99430cbfa8e2e2d60fcb36e703551c36ea13))

- **cli**: Remove non-interactive argument due to deprecation
  ([`8333054`](https://github.com/SarthakMishra/codemap/commit/8333054b01304d8458d5c82a13f76073f92318d3))

- **cli**: Remove redundant arguments in CLI functions
  ([`768e34c`](https://github.com/SarthakMishra/codemap/commit/768e34c8a7d6409a28eb2867af7a76e6cd1de1d0))

- **cli**: Remove unnecessary git check
  ([`721686a`](https://github.com/SarthakMishra/codemap/commit/721686ae6efdd0912548f46a1a06872239a64e5e))

- **cli**: Respect Git staging rules
  ([`071fed3`](https://github.com/SarthakMishra/codemap/commit/071fed36df36214e8d237a7774231f4f94a8f340))

- **cli**: Update default LLM model to gpt-4o-mini
  ([`6436e27`](https://github.com/SarthakMishra/codemap/commit/6436e272bead7a18b2a292856438b6ea6c85d024))

- **cli): remove unused import and mock GitWrapper feat(cli**: Add ut..
  ([`35ea0cf`](https://github.com/SarthakMishra/codemap/commit/35ea0cf444a5e954490d1c871ba7abd47ff782df))

- **cli, update.provider**: Update default model for OpenRouter and ad..
  ([`833f34d`](https://github.com/SarthakMishra/codemap/commit/833f34d42d9ba60cba9b045deb8db520b306b939))

- **cli/pr**: Add AI-generated PR title and description support
  ([`093e31a`](https://github.com/SarthakMishra/codemap/commit/093e31acc023c83f067b62eea8dd56eeb6a53b2d))

- **cli_app**: Change log level to debug for env load messages
  ([`b440c73`](https://github.com/SarthakMishra/codemap/commit/b440c73875c06ec4dba8dafff833fc62a9e9e316))

- **cli_entry**: Handle command run error properly
  ([`c0b3688`](https://github.com/SarthakMishra/codemap/commit/c0b3688fb5738ce08eb37756b982971449f971fe))

- **cli_entry**: Set OPENAI_API_KEY from command line option
  ([`705784d`](https://github.com/SarthakMishra/codemap/commit/705784db3728ca29f43b014c74e774e1e314381d))

- **codemap/cli**: Handle specific exceptions in CLI entry point
  ([`f267511`](https://github.com/SarthakMishra/codemap/commit/f2675118dad37b0ae2af1fa25b683a00c1dd03be))

- **codemap/cli_entry**: Handle specific exceptions in cli_entry
  ([`95b2134`](https://github.com/SarthakMishra/codemap/commit/95b2134c8c7b0b27abf5b5b9903f1bbb91ec997c))

- **codemap/git**: Add check for ignore_hooks
  ([`aaea856`](https://github.com/SarthakMishra/codemap/commit/aaea856b77990450687ca94cf7bed2638cd28be6))

- **codemap/git**: Remove useless GitWrapper class
  ([`0e50637`](https://github.com/SarthakMishra/codemap/commit/0e506377a2783101cbf9c7815dee5ee956fc7d77))

- **commit**: Automatically generate commit messages
  ([`b633c6e`](https://github.com/SarthakMishra/codemap/commit/b633c6e5a97d8e4b0dd28fa3140bcd49ea39d730))

- **commit**: Handle abort and error states in commit command
  ([`9bced54`](https://github.com/SarthakMishra/codemap/commit/9bced544509379566c60ad20c6d729c16ecae253))

- **commit**: Update commit command configuration
  ([`13b0a4a`](https://github.com/SarthakMishra/codemap/commit/13b0a4a1ff64786e56207e88f8b5598f54ad3118))

- **commit**: Update commit handling logic
  ([`72c842f`](https://github.com/SarthakMishra/codemap/commit/72c842f18ac75b5bc099d1eae60c1d342d1c03aa))

- **commit_cmd**: Handle commit exceptions anonymously
  ([`1018e8f`](https://github.com/SarthakMishra/codemap/commit/1018e8f82a328e208bc6a19d2556f2cdbb17a84f))

- **commit_cmd**: Handle unexpected command run errors
  ([`48ad2ec`](https://github.com/SarthakMishra/codemap/commit/48ad2ec41d286f5d6d02671e20db9305f0232100))

- **config**: Remove default config file
  ([`876a3f4`](https://github.com/SarthakMishra/codemap/commit/876a3f4f9d5e6ae9c8479c984101903245a7017c))

- **diff_splitter**: Filter out non-existent and untracked files
  ([`1883992`](https://github.com/SarthakMishra/codemap/commit/1883992dae76fe712fe10eef5a50a2ab490ca1c4))

- **diff_splitter**: Handle already staged deletions
  ([`f7d3ff9`](https://github.com/SarthakMishra/codemap/commit/f7d3ff94e2215aa6acf83b7c6477d4a7490a3690))

- **diff_splitter**: Handle untracked files in diff content check
  ([`a0bae87`](https://github.com/SarthakMishra/codemap/commit/a0bae87e0a296cdb9db6b2f115738f817498c613))

- **diffsplitter**: Add embedding model availability checks
  ([`6477be9`](https://github.com/SarthakMishra/codemap/commit/6477be974ca798ecac38f3be706bef9e19996e91))

- **docs**: Add bug reporting template
  ([`bbc799b`](https://github.com/SarthakMishra/codemap/commit/bbc799b09c4ba6fa3bdfa6083792bb23d90e2089))

- **env**: Remove OpenRouter API key example
  ([`ea225f7`](https://github.com/SarthakMishra/codemap/commit/ea225f7278ec1ab3ac8ef281ff548c2a0e6f3b08))

- **file_filters**: Enhance dot pattern matching logic and fix(git_..
  ([`da2ceb7`](https://github.com/SarthakMishra/codemap/commit/da2ceb7ceb00de763b4179c8a1ef01c49065cee7))

- **git_utils**: Handle bytes output in git status
  ([`637c503`](https://github.com/SarthakMishra/codemap/commit/637c503a36ed39d89b1c76153ec594e5732df7fa))

- **llm_utils**: Resolve circular import issue
  ([`4fd91e4`](https://github.com/SarthakMishra/codemap/commit/4fd91e495d5f4addd60a0879cdcd4a6bd8cdc45a))

- **log**: Handle exception in environment info logging
  ([`c361aae`](https://github.com/SarthakMishra/codemap/commit/c361aaeb285c58aceb19a5a1f2e0b356c57585a3))

The log_environment_info function now properly handles exceptions during logging, preventing
  potential crashes.

- **markdown_generator**: Ensure file_path is Path object before relat..
  ([`7a549f3`](https://github.com/SarthakMishra/codemap/commit/7a549f3926234d903f1798fc7cb87809fe8b3da7))

- **mpr**: Use LLM for PR title and description
  ([`e7c3ea5`](https://github.com/SarthakMishra/codemap/commit/e7c3ea5c8b58c786e866732b658b2b05a38e4ba7))

- **parser**: Prevent array parsing issue when multiple spaces are contained in string
  ([`6ecfb0d`](https://github.com/SarthakMishra/codemap/commit/6ecfb0d2f773abfa0eaf3244f6244815546c24c1))

Introduce a request id and a reference to latest request. Dismiss incoming responses other than from
  latest request.

- **pr_cmd**: Fix function parameter order in pr creation
  ([`6c30230`](https://github.com/SarthakMishra/codemap/commit/6c3023057c130f0c165607102b6c0001e5f76a31))

- **pr_generator**: Handle GitHub CLI errors and unexpected output
  ([`2e3f392`](https://github.com/SarthakMishra/codemap/commit/2e3f39292103195404d44b1c0b5ec6cd006f7d4f))

Previously, the code did not properly handle errors from the GitHub CLI or unexpected output
  formats. This change introduces improved error handling for CLI errors, JSON decoding issues, and
  cases where the GitHub CLI command outputs unexpected data. Error messages are now more
  informative, and specific exceptions are caught to provide better debugging information.

- **pr_utils**: Handle Git errors and edge cases in branch checks
  ([`b56e4a1`](https://github.com/SarthakMishra/codemap/commit/b56e4a1bf7eb045a2844e7c4ccb92af1c30d2b4f))

The function branch_exists now handles empty branch names and Git errors more robustly. It checks
  for local and remote branches separately and handles exceptions without failing immediately. The
  function get_branch_relation uses full ref names for branches when checking their existence and
  relation.

- **readme**: Update badge links for tests and code coverage
  ([`57b924a`](https://github.com/SarthakMishra/codemap/commit/57b924ab83959dd443ae5707c6ed7555f4adbb5e))

- **Taskfile**: Update linter tasks
  ([`7391bad`](https://github.com/SarthakMishra/codemap/commit/7391bad2664ddc02217afe613db884a716ec2ae3))

- **tests**: Improve code embedding test cases
  ([`952f97f`](https://github.com/SarthakMishra/codemap/commit/952f97f6fde046f44bd71f8389fd7207f5c886ae))

- **tests**: Remove shebang from test_llm_integration.py
  ([`10b1c86`](https://github.com/SarthakMishra/codemap/commit/10b1c86bd2815378ccc044e17853eb4132044766))

- **tests**: Update mock confirm_abort return value
  ([`16fee4e`](https://github.com/SarthakMishra/codemap/commit/16fee4e859ff398bd0f5e180bbe6c6d3a5b2192b))

- **tests**: Use utils module for Git utilities
  ([`0da6102`](https://github.com/SarthakMishra/codemap/commit/0da6102772169c16c69a133bdadb60340c275231))

- **utils**: Clean up deleted utilities for Codemap
  ([`8b6a431`](https://github.com/SarthakMishra/codemap/commit/8b6a431549c0224f51f2b3544c85a141bdac6171))

- **utils**: Filter out invalid filenames in commit
  ([`ece7fe4`](https://github.com/SarthakMishra/codemap/commit/ece7fe4d26ac62459754525b0b66f652d13822fb))

- **utils**: Update git utils import path
  ([`7b7b0df`](https://github.com/SarthakMishra/codemap/commit/7b7b0df1ce907e95dd2f2ae19afcfc68745a1efa))

### Build System

- Refine semantic-release configuration
  ([`1eed9d7`](https://github.com/SarthakMishra/codemap/commit/1eed9d70abd7bd45c5c16f65f4500dd24cc455a1))

- Update dependencies Update uv.lock to include new dependency..
  ([`c2e6208`](https://github.com/SarthakMishra/codemap/commit/c2e62080b9cd4edbf6829f1ca4baa1a6842725e6))

- Update dependencies lockfile
  ([`e8b77c0`](https://github.com/SarthakMishra/codemap/commit/e8b77c074f58dd993f874dea37b336a6eccaea30))

- Update dev dependencies Add isort to dev dependencies to imp..
  ([`a833614`](https://github.com/SarthakMishra/codemap/commit/a833614abb06f5bd67373e633f2e52a87565c475))

- Update project version to 0.3.4
  ([`da3f16c`](https://github.com/SarthakMishra/codemap/commit/da3f16cfb59ae19272c43aea7dbd1d07bc463272))

- Update project version to 0.3.6
  ([`bbb6ce7`](https://github.com/SarthakMishra/codemap/commit/bbb6ce7eb026e7b9759d27aaea0dc400dbe69966))

- Update project version to 0.3.8
  ([`bc3dda2`](https://github.com/SarthakMishra/codemap/commit/bc3dda24c69397c104612b6ab4d6d27e1f747944))

- **deps**: Update dependencies
  ([`01434c7`](https://github.com/SarthakMishra/codemap/commit/01434c72163ac751d9f70320025d62ba2374c6c4))

- **deps**: Update dependencies
  ([`1bc10a7`](https://github.com/SarthakMishra/codemap/commit/1bc10a74037b6e81835faa7b8d28a7dfb1c28726))

- **deps**: Update dependencies
  ([`869a402`](https://github.com/SarthakMishra/codemap/commit/869a402e65d4468b6762820ac9c5b948449f99a3))

- **pyproject.toml**: Bump version to 0.3.7
  ([`2ee03e5`](https://github.com/SarthakMishra/codemap/commit/2ee03e53f7990adf81eda731502961e9d1f8db4f))

- **release**: Bump version to 0.3.10
  ([`74e1d3c`](https://github.com/SarthakMishra/codemap/commit/74e1d3c14d25f97a67fa392f603ff7a6d554547b))

- **release**: Bump version to 0.3.9
  ([`7eb0906`](https://github.com/SarthakMishra/codemap/commit/7eb090621b6f0f05ad7191b074770e8d3869d3ca))

- **uv.lock**: Update codemap to 0.3.9
  ([`b64a232`](https://github.com/SarthakMishra/codemap/commit/b64a232963eef13d6c3814b0cee7ba3e2c60280f))

- **uv.lock**: Update codemap version to 0.3.4
  ([`b02be1b`](https://github.com/SarthakMishra/codemap/commit/b02be1b2440d64a0ab9b86ed8e3c709de84d6e31))

- **uv.lock**: Update codemap version to 0.3.5
  ([`21dad04`](https://github.com/SarthakMishra/codemap/commit/21dad04cce771303a72b0cdfe1a0f3e7b68511a8))

- **uv.lock**: Update codemap version to 0.3.8
  ([`1d8f8a9`](https://github.com/SarthakMishra/codemap/commit/1d8f8a9676af416059e59ea2ffb04bbf61026c37))

- **uv.lock**: Update codemap version to 0.4.0
  ([`30bea1f`](https://github.com/SarthakMishra/codemap/commit/30bea1fc27c8c7e56fa0e436eaf9f2d2c21f6b51))

### Chores

- Add gitignore for codemap cache
  ([`c4eaa27`](https://github.com/SarthakMishra/codemap/commit/c4eaa27dee7dac8b98edb5e31e1baa739d94c4c9))

- Add pytest configuration
  ([`83ef3b7`](https://github.com/SarthakMishra/codemap/commit/83ef3b77eeba04178852ebb870c554abec96147c))

- Bump version to 0.2.0 in pyproject.toml
  ([`eccbfeb`](https://github.com/SarthakMishra/codemap/commit/eccbfeb3674ea9eca409f65ba7646cbf0919e786))

- Bump version to 0.3.1 in pyproject.toml
  ([`3985462`](https://github.com/SarthakMishra/codemap/commit/39854620556c10699f3be9aa62e18275f1aba269))

- Bump version to 0.3.2 in pyproject.toml
  ([`1133a1b`](https://github.com/SarthakMishra/codemap/commit/1133a1bd8fce775316e79fb413b8964ca66583e4))

- Change documentation output directory to examples
  ([`6fa9279`](https://github.com/SarthakMishra/codemap/commit/6fa927920f9fd52d26e0701732e4fe85b5567d91))

- Fix .gitignore pattern for codemap cache
  ([`8066b8a`](https://github.com/SarthakMishra/codemap/commit/8066b8a6ee62ec806946a81d94d230bd12c12210))

Remove trailing slash from .codemap_cache pattern to match both file and directory

- Fix for workflow does not contain permissions issue
  ([`a000796`](https://github.com/SarthakMishra/codemap/commit/a00079664dce8e5b236785c18f56023e34d28e9f))

Co-authored-by: Copilot Autofix powered by AI
  <62310815+github-advanced-security[bot]@users.noreply.github.com>

- Refactored tests to match the new module structure
  ([`ac21459`](https://github.com/SarthakMishra/codemap/commit/ac21459531fcc6626af8e0dfb9cff862b82cbbdd))

- Remove example documentation file
  ([`1f23bff`](https://github.com/SarthakMishra/codemap/commit/1f23bfffc1a5a95950a9ea0ca162e955c6a986e6))

- Remove unused code
  ([`c933b45`](https://github.com/SarthakMishra/codemap/commit/c933b458dea75bc3743dff35b90999a24198d823))

- Remove unused files
  ([`fd70a17`](https://github.com/SarthakMishra/codemap/commit/fd70a171068043b57044891af41c74befd128510))

- Update .gitignore
  ([`63d4414`](https://github.com/SarthakMishra/codemap/commit/63d44148ed520e06635a29ba847412d35c412440))

- Update commit lint rules Add commit lint rules to match .comm..
  ([`f3e5dbc`](https://github.com/SarthakMishra/codemap/commit/f3e5dbc70b5592593a3ddf47eb67596f9782bf2a))

- Update configuration to disable token limits and content length
  ([`f6f7027`](https://github.com/SarthakMishra/codemap/commit/f6f702747ae560a9b363f4469fb628a256518028))

- Update files in root
  ([`1027eda`](https://github.com/SarthakMishra/codemap/commit/1027eda208ebe6e7120a1cc83bb82a9f11e929d3))

- Update files in src/codemap/cli
  ([`8295050`](https://github.com/SarthakMishra/codemap/commit/8295050f9c1661231c4ae2e8bc736d711f214554))

- Update files in src/codemap/utils
  ([`fe7d25c`](https://github.com/SarthakMishra/codemap/commit/fe7d25ce22445a9d5ae8c23202d537916be0af7c))

- Update gitignore and codemap
  ([`26c2892`](https://github.com/SarthakMishra/codemap/commit/26c28929130f3c1eeb3d56a859d81b5ba9e97c6e))

- Update gitignore to exclude model files
  ([`3041f47`](https://github.com/SarthakMishra/codemap/commit/3041f470cced47f735726c56cea924c9dd626c6c))

- Update pre-commit hooks and ruff settings
  ([`0160666`](https://github.com/SarthakMishra/codemap/commit/0160666096566b9cefb73d08424f61eb670fff80))

- Update src/codemap/analyzer/processor.py
  ([`83f39cc`](https://github.com/SarthakMishra/codemap/commit/83f39cc27d51c4f77aeee9579496feaed0ac4b89))

- Update Taskfile.yml
  ([`d3446fd`](https://github.com/SarthakMishra/codemap/commit/d3446fd97ed169793effc674d76dfd93d3a632bd))

- **.gitignore**: Add env.local to ignore list
  ([`91516d9`](https://github.com/SarthakMishra/codemap/commit/91516d9a07a3a5da1237c8e03c5b55d811a70f4a))

- **analyzer**: Remove code analysis modules
  ([`5c578f8`](https://github.com/SarthakMishra/codemap/commit/5c578f85c0ae6675b777cf8504cf357509a4a313))

The code analysis modules were removed due to refactoring needs.

- **ci**: Add format task to full CI pipeline
  ([`64d1185`](https://github.com/SarthakMishra/codemap/commit/64d118581210bf28d0339382642a92245ae226ae))

- **ci**: Add git setup and PYTHONPATH env
  ([`d75c706`](https://github.com/SarthakMishra/codemap/commit/d75c706fd8c83880beecccda059ecabc0a84fc22))

- **ci**: Remove redundant Git setup from CI workflow
  ([`8b8cd54`](https://github.com/SarthakMishra/codemap/commit/8b8cd5464dc6c98f3d75b90e75031208c6310e6b))

- **ci**: Set env vars for venv activation
  ([`07b69ec`](https://github.com/SarthakMishra/codemap/commit/07b69ec8f02b257524a49ae48dd64a9103f8e694))

- **ci**: Update test workflow to use task and uv
  ([`281eccb`](https://github.com/SarthakMishra/codemap/commit/281eccb8648ed630878f8bd15ca119427813332d))

- **cli**: Remove daemon command and related functionality
  ([`e1f07eb`](https://github.com/SarthakMishra/codemap/commit/e1f07eb8f73a7f6e0e0791bb43e9210953d0adf1))

The daemon command and related functionality have been removed from the codebase. This change
  affects the CLI and related modules.

BREAKING CHANGE: daemon command and functionality removed

- **codemap**: Remove file watcher module
  ([`4905bda`](https://github.com/SarthakMishra/codemap/commit/4905bdaea01657c62fd2206b4429b824932f7dfb))

- **codemap**: Remove unused git commit module
  ([`bce6d47`](https://github.com/SarthakMishra/codemap/commit/bce6d47e6494abc8fae3e1ea5db12459956b6200))

- **codemap**: Update package version
  ([`718704a`](https://github.com/SarthakMishra/codemap/commit/718704a63741f74dc85014d34a34f853b9d41de8))

- **codemap**: Update TODO list for codemap processor
  ([`8e6f883`](https://github.com/SarthakMishra/codemap/commit/8e6f883075f09da0ff426386f6872ad9c8a1a49d))

The TODO list for the codemap processor has been updated to reflect the following changes: - let the
  commit command operate on git diffs as it's already doing. No need to integrate with processor -
  remove lsp entirely too complicated to maintain. - use tree-sitter to generate metadata on the
  file level not on the chunk level, cache this data and keep updating it every time any codemap cmd
  is called. - remove the git analysis to make the data even simpler.

- **config**: Add bypass hooks option to codemap.yml docs(readme):..
  ([`de734f4`](https://github.com/SarthakMishra/codemap/commit/de734f4b493e2160d2680c7efbf1717f22d42394))

- **config**: Add bypass_hooks option to commit strategy
  ([`83fbff0`](https://github.com/SarthakMishra/codemap/commit/83fbff05d02687ba79ab9087f710168944beb0ff))

- **config**: Add commitlint configuration file
  ([`9a946bc`](https://github.com/SarthakMishra/codemap/commit/9a946bc0020d5e5739256c16892e2e1f1b93d9e3))

- **config**: Add commitlint to pre-commit conf
  ([`3d295df`](https://github.com/SarthakMishra/codemap/commit/3d295df453ba8e2635322d51e9be0abe7c5e0429))

- **config**: Add pre-commit configuration file
  ([`7d086c5`](https://github.com/SarthakMishra/codemap/commit/7d086c5417a73f36d366393a161233d3b18f5a5c))

- **config**: Change hotfix base branch from 'main' to 'dev'
  ([`4ac1f05`](https://github.com/SarthakMishra/codemap/commit/4ac1f0584bc489f8b8fbe5ff0889217cf1e71887))

Updated the base branch for hotfixes to 'dev' to align with the current development workflow.

- **config**: Update codemap.yml configuration
  ([`75f535b`](https://github.com/SarthakMishra/codemap/commit/75f535b59cf7fb06ee1aa6841026336ee9d1fa03))

- **config**: Update default base branch setting
  ([`1f44833`](https://github.com/SarthakMishra/codemap/commit/1f4483344f531da00c99b757c0edfc4a0d59f172))

The default base branch setting was updated from 'dev' to 'None' to reflect changes in the
  development workflow.

- **config**: Update ruff linting configuration
  ([`5f712a0`](https://github.com/SarthakMishra/codemap/commit/5f712a03ea1f09327ef96b12335435283298c985))

Allow unused arguments in strategy interface methods

- **config**: Update token limit and model settings
  ([`08c5d2b`](https://github.com/SarthakMishra/codemap/commit/08c5d2bf32441416a5a31426d37e13e4c11e2a4b))

- **daemon**: Remove daemon module
  ([`4bb753f`](https://github.com/SarthakMishra/codemap/commit/4bb753fc273b7562943388678e200561cefcad32))

The daemon module has been removed, including its API server, client, and service components.

- **deployment**: Remove supervisor configuration files
  ([`109644c`](https://github.com/SarthakMishra/codemap/commit/109644c016cf4fd80618fc7c02575379018638ce))

The codemap.conf and install.sh files are no longer needed and have been removed.

- **deployment**: Remove systemd service files
  ([`b1f2385`](https://github.com/SarthakMishra/codemap/commit/b1f238568e0ebed240c774c02d710ee427b90371))

The codemap.service file and install.sh script have been removed from the deployment/systemd
  directory.

- **deps**: Add interrogate dependency
  ([`1ab182b`](https://github.com/SarthakMishra/codemap/commit/1ab182bc94a6e4c5dd814e24d084a911bde13727))

- **deps**: Add interrogate dependency
  ([`836fc9c`](https://github.com/SarthakMishra/codemap/commit/836fc9cf7c04305f303b18f6164dfcdb3c5481c6))

- **deps**: Add pyright dependency
  ([`eff74e6`](https://github.com/SarthakMishra/codemap/commit/eff74e6ad5675e45dbff3b22f0ae4085d4a7fc5e))

- **deps**: Add watchdog dependency
  ([`38a5990`](https://github.com/SarthakMishra/codemap/commit/38a5990ddab40ab06fa680a04710ff2d6e44baa2))

- **deps**: Add watchdog dependency
  ([`802db83`](https://github.com/SarthakMishra/codemap/commit/802db83868c7babf8dbe1cc5950df96f2ebea081))

- **deps**: Remove unused files
  ([`948bd9f`](https://github.com/SarthakMishra/codemap/commit/948bd9f281a3a4e15a7877e646bf12393b53c8a4))

- **deps**: Update codemap to 0.4.2
  ([`a6dc694`](https://github.com/SarthakMishra/codemap/commit/a6dc694a872bf61a11920e2ed272772c5b7af4de))

- **deps**: Update codemap to 0.4.3
  ([`81a59b8`](https://github.com/SarthakMishra/codemap/commit/81a59b88860a08bae49008d780664f5536797d0d))

- **deps**: Update codemap version to 0.4.4 and remove unused git settings
  ([`6826aa1`](https://github.com/SarthakMishra/codemap/commit/6826aa1686cb338bd932cc10ab2839bb90742779))

- **deps**: Update dependencies
  ([`dd3fca1`](https://github.com/SarthakMishra/codemap/commit/dd3fca188969439154dfc089f54cdb7bc3f0ac32))

This commit updates the dependencies in the uv.lock file to the latest versions.

- **deps**: Update dependencies
  ([`917b31f`](https://github.com/SarthakMishra/codemap/commit/917b31fe88f940fb9a0cfa7dc4a6853d7e950831))

- **deps**: Update dependencies
  ([`26ae412`](https://github.com/SarthakMishra/codemap/commit/26ae412dad1ecfebe91fda2bd40b44ef65dc522a))

- **deps**: Update dependencies
  ([`6235aca`](https://github.com/SarthakMishra/codemap/commit/6235acad1100dd1aaa0b188ec2787f56b643b49b))

- **deps**: Update dependencies
  ([`1085f09`](https://github.com/SarthakMishra/codemap/commit/1085f092dff75d439f19c778f6dad8d41416c245))

- **deps**: Update dependencies
  ([`d745175`](https://github.com/SarthakMishra/codemap/commit/d745175d97851cc36a2fbaa50966d2e43d49efae))

- **deps**: Update dependencies
  ([`110a113`](https://github.com/SarthakMishra/codemap/commit/110a11320521077458662551913a4629b6d8a08b))

- **deps**: Update dependencies
  ([`2701d0b`](https://github.com/SarthakMishra/codemap/commit/2701d0b66e598639f19ca308a01a293f110c5835))

- **deps**: Update dependencies
  ([`7a3906e`](https://github.com/SarthakMishra/codemap/commit/7a3906e82f6a713de3287ee58daa80cd53ce0955))

- **deps**: Update dependencies
  ([`97f9b50`](https://github.com/SarthakMishra/codemap/commit/97f9b50aacb4401252f9d21f2024efa51dd0d027))

- **deps**: Update git settings
  ([`3b78304`](https://github.com/SarthakMishra/codemap/commit/3b783043de99c90c865f15269f853de2c63d6a4d))

- **deps**: Update litellm dependency version
  ([`d753dcb`](https://github.com/SarthakMishra/codemap/commit/d753dcbe40b337d8ec6cee297bd9f30f0a81c5b0))

- **deps**: Update litellm dependency version
  ([`ec5b8f4`](https://github.com/SarthakMishra/codemap/commit/ec5b8f454c6e039316f9862c50b6d9f17bdd6932))

- **deps**: Update project metadata and dependencies
  ([`24ffe9c`](https://github.com/SarthakMishra/codemap/commit/24ffe9cae7d03ff6b8491a77345050b237414181))

- **docs**: Remove markdown generator
  ([`e108194`](https://github.com/SarthakMishra/codemap/commit/e108194c563a98922d482ac9076817a2bc35cf3c))

The markdown generator has been removed from the codebase.

- **docs**: Remove outdated deployment documentation
  ([`8590ec8`](https://github.com/SarthakMishra/codemap/commit/8590ec8a98d29b66b5056b385b4c467961623b6f))

- **docs**: Remove outdated documentation files
  ([`ce9bdfa`](https://github.com/SarthakMishra/codemap/commit/ce9bdfa8d7594848b43030bf26f813ea0231ebe6))

- **env**: Add environment variables example file
  ([`e4f7f79`](https://github.com/SarthakMishra/codemap/commit/e4f7f79304af8afa9fabc4e03093e73b565abbd6))

- **gitignore**: Add todo directory to ignore list
  ([`e0ccb35`](https://github.com/SarthakMishra/codemap/commit/e0ccb35d0dfdd443fe848d1657e2320b877b124f))

- **languages**: Remove language config module
  ([`0f1db12`](https://github.com/SarthakMishra/codemap/commit/0f1db120af13bea8bcc868c0f20a26359ad79b14))

- **llm**: Silence http and litellm logs
  ([`d12cac6`](https://github.com/SarthakMishra/codemap/commit/d12cac64104d85caf45a75b859b2cacac0d915c1))

- **lsp**: Remove unused LSP code
  ([`6b31d10`](https://github.com/SarthakMishra/codemap/commit/6b31d100557d8351ca2cd457e79cec13b8be103f))

The LSP (Language Server Protocol) integration was partially removed, and some files became
  obsolete. This commit removes the remaining unused LSP code.

- **pre-commit**: Remove pytest hook config
  ([`fb139f1`](https://github.com/SarthakMishra/codemap/commit/fb139f14bd6c8b7d75f3fefcedf25d050714bb6a))

- **pre-commit**: Simplify hook configurations
  ([`fe8863e`](https://github.com/SarthakMishra/codemap/commit/fe8863e93e7a49f7ffe4f0f60d420c89df89e6df))

- **pre-commit**: Update pre-commit hooks to run on staged files only
  ([`3cb984e`](https://github.com/SarthakMishra/codemap/commit/3cb984e56ef368e521027a8b36627e09d6dee81f))

Previously, pre-commit hooks were run on all files. This change updates the pre-commit task to only
  run on staged Python files, improving performance and reducing unnecessary checks.

- **pre-commit**: Update pytest config to run on commit stage
  ([`196bc5f`](https://github.com/SarthakMishra/codemap/commit/196bc5f58c0967d769a1002e448b8499e21201da))

- **readme**: Update badges and add Codacy and pre-commit badges; enhance CI workflow to upload
  coverage reports to Codacy
  ([`5906f03`](https://github.com/SarthakMishra/codemap/commit/5906f030e72ca5feeb9adf2ed41ea08591f5fbf3))

- **Taskfile**: Add platform-specific vars and cmds
  ([`53e9d2b`](https://github.com/SarthakMishra/codemap/commit/53e9d2bbeedf24b8a86870ba6fd7e6e6ae8ab32c))

- **taskfile**: Add pre-commit and pre-push hooks
  ([`9b6854c`](https://github.com/SarthakMishra/codemap/commit/9b6854c60edcc4136d5c904012c65e6b0e8edbb6))

- **taskfile**: Add uv run prefix to radon and test cmds
  ([`f9ee92c`](https://github.com/SarthakMishra/codemap/commit/f9ee92c20d1c83108d63950dfd32a1865affcc2e))

- **taskfile**: Remove format task from CI checks
  ([`2acc958`](https://github.com/SarthakMishra/codemap/commit/2acc9582ab25339911ea584320ddc8b99dcbb509))

- **taskfile**: Update isort command
  ([`d81c246`](https://github.com/SarthakMishra/codemap/commit/d81c246fd24e682ee491f5449c865ae3892f32a5))

- **taskfile**: Update pre-commit task to use ruff format
  ([`b753d99`](https://github.com/SarthakMishra/codemap/commit/b753d99246d58e744a5ac9d44843028dbc5dca2b))

- **testing**: Add new test markers
  ([`5d29240`](https://github.com/SarthakMishra/codemap/commit/5d2924030755f18cb54960c396724706263999b0))

- **testing**: Add new test markers
  ([`f799daa`](https://github.com/SarthakMishra/codemap/commit/f799daa5607a38a7b4d724297037410e3617bf88))

- **testing**: Add path_sensitive test marker
  ([`9a328ef`](https://github.com/SarthakMishra/codemap/commit/9a328efaf6dcf81020f9bece2422d3442d162c90))

- **tests**: Add base and helper test classes and fixtures
  ([`82b350c`](https://github.com/SarthakMishra/codemap/commit/82b350cb4a389308e7dedaa433ca97406cb9b724))

- **tests**: Remove deleted test files
  ([`e1898ba`](https://github.com/SarthakMishra/codemap/commit/e1898ba98be4d12c81dc7b882c1b94eff5c6ef80))

The following test files were deleted: - tests/processor/analysis/test_lsp_analyzer.py -
  tests/processor/analysis/test_lsp_integration.py - tests/processor/chunking/__init__.py -
  tests/processor/chunking/test_regexp_chunker.py -
  tests/processor/chunking/test_semantic_chunking.py

- **tests**: Remove deleted test files
  ([`424c0bf`](https://github.com/SarthakMishra/codemap/commit/424c0bfe19459863f39924dfed563c4e9a49d1e2))

- **tests**: Remove test file
  ([`bb48114`](https://github.com/SarthakMishra/codemap/commit/bb48114b1a3d4512f24080f83e5f8012197a74ce))

- **tests**: Remove test TODO file
  ([`9a5b873`](https://github.com/SarthakMishra/codemap/commit/9a5b873a139b2870b67c0d67e3af35809776e60c))

- **tests**: Remove unused test files
  ([`94b1cbf`](https://github.com/SarthakMishra/codemap/commit/94b1cbf81ad0131e1a6e14adf0b52f67077a81ef))

The test files were no longer needed and were taking up space.

- **tests**: Remove unused test files
  ([`e1bbfe5`](https://github.com/SarthakMishra/codemap/commit/e1bbfe56a05018aa7784d203d9186d1e128c3a18))

Removed test files for analyzer processor and tree parser

- **tests**: Update test coverage information
  ([`ccb1186`](https://github.com/SarthakMishra/codemap/commit/ccb1186fd556b16af28ad49cf1f0e15487611b05))

- **typings**: Add type stub file for pandas tests
  ([`8651fde`](https://github.com/SarthakMishra/codemap/commit/8651fdebc03622c4a09649c4e7b0daf6e2cebb38))

- **versions**: Bump to 0.4.3
  ([`7bd10bb`](https://github.com/SarthakMishra/codemap/commit/7bd10bb98f656f23526cc4ce72a3268ed3ef119a))

- **vscode**: Add VSCode settings file
  ([`01adf5b`](https://github.com/SarthakMishra/codemap/commit/01adf5b05a97d42d68cc28f69c9666dc34f4a193))

- **vscode**: Update python analysis extra paths
  ([`fd14cf3`](https://github.com/SarthakMishra/codemap/commit/fd14cf3069a5e939dea567f2e4e06586e27e3f5d))

- **vscode**: Update python interpreter path to python3
  ([`9dfcd2d`](https://github.com/SarthakMishra/codemap/commit/9dfcd2d566912ff368e6e1d7236ed16f9813fdd6))

- **vscode**: Update settings.json Update VSCode settings to inclu..
  ([`91a39f8`](https://github.com/SarthakMishra/codemap/commit/91a39f85b229f2149caf9382ce3c91d3e91bd274))

### Code Style

- **cli**: Add missing linter pragma
  ([`10be9b9`](https://github.com/SarthakMishra/codemap/commit/10be9b9db71d2f88241c3b71b57fba1bf890d020))

- **cli_entry**: Remove unnecessary noqa comment
  ([`deaa99c`](https://github.com/SarthakMishra/codemap/commit/deaa99cc6c7f7d7650e3af90d3a96d57321c2ddd))

- **config**: Update ruff lint ignores
  ([`d626f0d`](https://github.com/SarthakMishra/codemap/commit/d626f0d6e4a0d4f8ccc76d642020fd99403028fd))

- **linting**: Update ruff lint ignores
  ([`7b4317f`](https://github.com/SarthakMishra/codemap/commit/7b4317f2a3d30f8dbc42456478b92c4662cf593d))

The commit updates the ruff lint ignores in pyproject.toml to include additional error codes.

- **linting**: Update ruff lint ignores
  ([`22eb4e7`](https://github.com/SarthakMishra/codemap/commit/22eb4e7e311fb18681a560ccae06f13c98c9bcbc))

- **linting**: Update ruff lint ignores for cli main
  ([`65ca084`](https://github.com/SarthakMishra/codemap/commit/65ca0842c0355a6c226d03c404f180f0217e6832))

- **linting**: Update ruff linting configuration
  ([`0a414a1`](https://github.com/SarthakMishra/codemap/commit/0a414a1afdde69acf44d5beff99833a38cec88c2))

- **logging**: Refactor logging to handle exceptions
  ([`96d2214`](https://github.com/SarthakMishra/codemap/commit/96d2214d704287ca6b6475283cfb193442e81ce6))

The logging functionality has been refactored to properly handle exceptions. This change improves
  the robustness of the logging mechanism.

- **parser**: Update tests according to PEP8 style refactor(code_p..
  ([`d2cda8f`](https://github.com/SarthakMishra/codemap/commit/d2cda8f1a5fa1b455fe2a4ea0ef1bdf648ccda9d))

- **pyproject**: Update linter ignores
  ([`6f29f13`](https://github.com/SarthakMishra/codemap/commit/6f29f13f071ff3db95c1b04c9a16098e143ed49d))

Add G004 ignore for logging f-string

- **pyproject**: Update ruff lint ignores
  ([`bc056f6`](https://github.com/SarthakMishra/codemap/commit/bc056f6e39cb6863e4604a6373eba8d7f290e838))

- **tests**: Update test code to remove unnecessary noqa comments
  ([`8c5ae08`](https://github.com/SarthakMishra/codemap/commit/8c5ae0871cda172d51aabad5d8683aef42ce3013))

The test code had several lines with noqa comments that were not necessary. These comments were
  removed to clean up the code.

### Continuous Integration

- Add format task to CI commands
  ([`33cca7e`](https://github.com/SarthakMishra/codemap/commit/33cca7e994d4af9af663d3fdc332575aff8a00dd))

- Add lint and release workflows
  ([`8c6e700`](https://github.com/SarthakMishra/codemap/commit/8c6e700c5c8ace669e7555491f32f7d3b73d57c5))

- Add dedicated lint workflow to separate linting from tests\n- Add semantic release workflow for
  automated versioning and PyPI releases\n- Configure release workflow to respect branch-based
  release strategies

- Added more linter and formatters to conmfigs
  ([`ea453e9`](https://github.com/SarthakMishra/codemap/commit/ea453e94c1d1c2f538c063af3270de17c7e2c3bd))

- Added pyright
  ([`71790e2`](https://github.com/SarthakMishra/codemap/commit/71790e2f1f2ce7e89347d8fc5542be7b610454c6))

- Depend on tests and lint workflows before releasing
  ([`95913cb`](https://github.com/SarthakMishra/codemap/commit/95913cb1f8649c450cba3763c896de706a21ce5d))

- Enable VCS release in GitHub Actions workflow
  ([`bc377d4`](https://github.com/SarthakMishra/codemap/commit/bc377d4f97b80f5ed68a994f8ecb83d534827625))

- Remove check for required workflows in release job
  ([`9cc9207`](https://github.com/SarthakMishra/codemap/commit/9cc92078ca56edd60178295c356077e7861ba87d))

- Run release workflows on pull requests
  ([`9ee91fd`](https://github.com/SarthakMishra/codemap/commit/9ee91fd99402f501f1620499cb5ad4042d000c11))

- **pre-commit**: Added pre-commit hooks
  ([`5818317`](https://github.com/SarthakMishra/codemap/commit/58183171e47c70f849fd56a285380bd3f97ff656))

- **workflow**: Unify ci task runs across os
  ([`dd6a1f9`](https://github.com/SarthakMishra/codemap/commit/dd6a1f96018080225a6b90b000741a75bd8d5e24))

- **workflows**: Add tests workflow for github actions
  ([`4172388`](https://github.com/SarthakMishra/codemap/commit/4172388cdf79b31ee788b10c48fa05a4728d82d4))

- **workflows**: Remove windows from test matrix
  ([`ea8091d`](https://github.com/SarthakMishra/codemap/commit/ea8091d1fbdaeb93c2fafac2b15048d178e38456))

- **workflows**: Restrict test workflow triggers to src and tests directories
  ([`e425eb0`](https://github.com/SarthakMishra/codemap/commit/e425eb01678a2d89b2e9440197f89169446a000b))

- **workflows**: Update test workflow to use ruff, pylint, and pyright
  ([`b8aafee`](https://github.com/SarthakMishra/codemap/commit/b8aafee49bbe91ea51aea863c9d99f563cdc70c6))

### Documentation

- Add lint and PyPI badges to README
  ([`e1c9fd1`](https://github.com/SarthakMishra/codemap/commit/e1c9fd11f101ac85a80062dd0c90019245b0417b))

- Add Lint workflow status badge\n- Add PyPI version badge using shields.io\n- Improve badge
  positioning for better readability

- Add tree flag documentation to README
  ([`7fd0154`](https://github.com/SarthakMishra/codemap/commit/7fd01547ba6e27660213908de67aa5a92e228b86))

- Generate example docs using the tool
  ([`6bfaecc`](https://github.com/SarthakMishra/codemap/commit/6bfaecc335a0e668406a29ded45fc4338aec9b7a))

- Remove PR_COMMAND.md and update README.md
  ([`afb8dca`](https://github.com/SarthakMishra/codemap/commit/afb8dca861b1bd30445fb6aea11956a98a525776))

- Remove todo files (chore): remove old todos fix is not right..
  ([`66318ed`](https://github.com/SarthakMishra/codemap/commit/66318ed14886cd710eb32c0adca5eb2bcad43d94))

- Update CONTRIBUTING.md with semantic release process
  ([`61b849f`](https://github.com/SarthakMishra/codemap/commit/61b849f2bab1c62a72fec98df586d101bc90cbc8))

- Add release process section explaining automated releases\n- Update branching strategy to align
  with semantic-release\n- Update commit message guidelines with version impact details\n- Clarify
  workflow examples for release preparation and hotfixes

- Update documentation and configuration for ERD feature
  ([`85533cb`](https://github.com/SarthakMishra/codemap/commit/85533cbd2b175408eec3ea02bcf3dbcabfad059d))

- Update installation and upgrade scripts with correct repository URL
  ([`ee9658e`](https://github.com/SarthakMishra/codemap/commit/ee9658eb89b8cf160e950920f58f34a0adf070ca))

- Replace 'code-map' with 'codemap' in curl installation commands - Update GitHub repository URL to
  use full refs/heads/main path - Ensure consistent URL format for install, upgrade, and uninstall
  scripts

- Update installation instructions for PyPI
  ([`06c59e3`](https://github.com/SarthakMishra/codemap/commit/06c59e355f360ea70634f37549ee4dd5e7106e58))

- Update README and gitignore with new configuration options
  ([`efdb901`](https://github.com/SarthakMishra/codemap/commit/efdb90147ece33584166bf432b0f759b2af88e72))

- Update README with diff splitting strategies information
  ([`e5d37ae`](https://github.com/SarthakMishra/codemap/commit/e5d37ae9287d6cc828bddd509922189f78d3bd89))

- Update README with simplified installation instructions
  ([`fd68e1b`](https://github.com/SarthakMishra/codemap/commit/fd68e1b8fbce0f8e9cf3c04c971d6d3b2bd9a4c6))

- Replace complex installation scripts with pipx-based installation - Remove custom install,
  upgrade, and uninstall scripts - Simplify installation, upgrade, and uninstall process - Update
  documentation to reflect new installation method

- Update README.md content
  ([`cef8fc2`](https://github.com/SarthakMishra/codemap/commit/cef8fc2ddfda3aa045615083262b101e3b706c58))

- Update uninstall command to use curl
  ([`9ebc216`](https://github.com/SarthakMishra/codemap/commit/9ebc21689a7dfb519ad3dcb575aa4aa13de2e1b1))

Change uninstall instructions to use curl command instead of local script execution

- **.github**: Add Code of Conduct and Contributing guidelines
  ([`37da742`](https://github.com/SarthakMishra/codemap/commit/37da742750b45b2dee83184737b6b135f07189b4))

- **.github**: Add Git Flow diagram to contributing guide
  ([`b1b8ac7`](https://github.com/SarthakMishra/codemap/commit/b1b8ac7b5d4d5665c78e1ac817274734f6bfa91e))

- **build**: Update build setup for tests
  ([`b324308`](https://github.com/SarthakMishra/codemap/commit/b3243085296be81fe7fbc1a1fff21778bf1ba8a6))

- **cli**: Add alias note to help messages
  ([`5c1a158`](https://github.com/SarthakMishra/codemap/commit/5c1a158ab9d13209b1f61576ae588a8c0738eff4))

The command-line interface now displays a note about the 'cm' alias in the help messages.

- **codemap**: Update TODO comments
  ([`81f5759`](https://github.com/SarthakMishra/codemap/commit/81f57595271d9c7afee6b0a0eef83f395c5a5b96))

- **codemap**: Update TODO list with completed tasks
  ([`eef5369`](https://github.com/SarthakMishra/codemap/commit/eef536960a4ef0258fd1ee80087540bed82f050f))

Marked several tasks as completed in the TODO list, reflecting the current status of the project.

- **codemap/processor**: Add README.md for CodeMap Processor Module
  ([`bb57dbc`](https://github.com/SarthakMishra/codemap/commit/bb57dbcc03880e9637cfc6f1db3a331d38811de6))

This commit introduces documentation for the CodeMap Processor Module, which is a core component of
  the CodeMap project responsible for processing source code repositories.

- **contributing**: Fix typos in guidelines text
  ([`83b6bec`](https://github.com/SarthakMishra/codemap/commit/83b6bec50966df8433210d1f4f580a9372774832))

- **deployment**: Add deployment documentation for CodeMap daemon
  ([`a339cca`](https://github.com/SarthakMishra/codemap/commit/a339cca5c26cd9e4261edf67cba1041a3b3615f3))

This commit introduces documentation for deploying the CodeMap daemon using systemd and Supervisor.
  The documentation covers installation, management commands, and configuration options for both
  deployment methods.

- **embedding**: Update module documentation
  ([`fcdd294`](https://github.com/SarthakMishra/codemap/commit/fcdd2941c80d06a6384d10fb44737eeb3f958770))

- **models**: Add model documentation
  ([`8c86f4e`](https://github.com/SarthakMishra/codemap/commit/8c86f4e501009227c0779c829ca463ad93676bb1))

- **processor**: Add TODO file for processor module
  ([`f44f138`](https://github.com/SarthakMishra/codemap/commit/f44f138334149886125fbf28cf2deddaac9e9e05))

- **processor**: Update module documentation
  ([`9a28978`](https://github.com/SarthakMishra/codemap/commit/9a28978db1bdc44b9a4affcde80f0e6d2ae44574))

- **processor**: Update TODO comments
  ([`28374b1`](https://github.com/SarthakMishra/codemap/commit/28374b14ecca0a69de890da5945a390305eb0d55))

- **processor**: Update TODO comments for chunking and git metadata
  ([`817cb5b`](https://github.com/SarthakMishra/codemap/commit/817cb5b86a760b31b4c171194abd35966b88119f))

- **readme**: Add badges and update intro
  ([`e0bc768`](https://github.com/SarthakMishra/codemap/commit/e0bc7682157d50504c45d876e07518401f88decd))

- **readme**: Add commit feature documentation
  ([`b2ceffc`](https://github.com/SarthakMishra/codemap/commit/b2ceffc0c4c9b127e667819e2acba0655c77be52))

- **readme**: Add contribution guidelines and code of conduct links
  ([`1e5e762`](https://github.com/SarthakMishra/codemap/commit/1e5e762e9f9f85a394351203716fa88074d49d00))

- **readme**: Add installation note and warning about Git commit behavior
  ([`f13f48f`](https://github.com/SarthakMishra/codemap/commit/f13f48ffd4a884feb0da9735cf70da78599c76ba))

- **readme**: Add known issue notice for commit command
  ([`bc99760`](https://github.com/SarthakMishra/codemap/commit/bc99760b23ae9937acc19e307300b770316fc3fb))

- **readme**: Add platform support note
  ([`1e6ba22`](https://github.com/SarthakMishra/codemap/commit/1e6ba2210e8c772ed847118b02ea37f26cdf69a5))

- **README**: Add semantic chunking documentation
  ([`3f4192c`](https://github.com/SarthakMishra/codemap/commit/3f4192cce794fc067aa22f079f32e4a38f77109c))

- **readme**: Change license badge from Apache 2.0 to MIT
  ([`6eb0860`](https://github.com/SarthakMishra/codemap/commit/6eb0860e09a649e53c18a9d2663ad1ca21a9cdc5))

- **readme**: Clarify caution note regarding active development and potential breaking changes
  ([`7cb0fb5`](https://github.com/SarthakMishra/codemap/commit/7cb0fb5703f74fda9e8ef65451b4370f9794c2e1))

- **readme**: Enhance documentation with updated features, command options, and configuration
  details
  ([`9226b9c`](https://github.com/SarthakMishra/codemap/commit/9226b9cd325f6cd55b6397d37ee0bc8ee1813d12))

- **README**: Remove optional dependencies section for semantic chunking
  ([`40ee24b`](https://github.com/SarthakMishra/codemap/commit/40ee24b3f7b5a1dc615da52f3f866d27b6b6c628))

- **readme**: Update Git hooks compatibility
  ([`8a042cc`](https://github.com/SarthakMishra/codemap/commit/8a042cc710211be444d56ca533ca7712becc6063))

- **README**: Update installation instructions
  ([`f21b14d`](https://github.com/SarthakMishra/codemap/commit/f21b14dbb4aa70a36ddf68f394e65bc0f1c3938a))

The installation instructions have been updated to reflect the new recommended method using pipx.
  The manual installation method using pip is also provided.

- **readme**: Update module documentation strings
  ([`47bf28a`](https://github.com/SarthakMishra/codemap/commit/47bf28ae38be2bbf1154e15588dc62bf593ed1c2))

- **readme**: Update Python version badge and add status badge
  ([`3f60a74`](https://github.com/SarthakMishra/codemap/commit/3f60a74477aaa22e54c6551cf9a5e954dca99665))

- **rules**: Add best practices and terminal guidelines
  ([`33ae753`](https://github.com/SarthakMishra/codemap/commit/33ae753b0b3d8d5f67fcb060f4b68c88cae4c4c9))

- **rules**: Add documentation guidelines
  ([`f5e44bb`](https://github.com/SarthakMishra/codemap/commit/f5e44bb36588123e8ebccb554b5b1a0d3fccf970))

- **template**: Add pull request template
  ([`3b69257`](https://github.com/SarthakMishra/codemap/commit/3b69257e1bc464ebd75bc6e76c6ab6ec911605fb))

- **terminal**: Update terminal documentation with new package management commands
  ([`9bd0823`](https://github.com/SarthakMishra/codemap/commit/9bd082352c06cec7e258fb2463b8abd4a964e353))

Added documentation for installing and removing packages with uv, including dev packages and custom
  dependency groups.

- **test**: Update test configuration
  ([`470da7b`](https://github.com/SarthakMishra/codemap/commit/470da7b0f3b4a19c9e986ca84aceaf0099ce1b89))

Add gen: Generator related tests for codemap.gen module

- **testing**: Add test categories and helper info
  ([`f750a8b`](https://github.com/SarthakMishra/codemap/commit/f750a8bb2c85f1e7187787e7f809f69bdc7569a3))

- **testing**: Add testing standards and best practices guide
  ([`355af7b`](https://github.com/SarthakMishra/codemap/commit/355af7b1da116ee944ee6ecc10ffc1407bc2f08c))

- **todo**: Add development environment enhancement plan
  ([`5d65c5d`](https://github.com/SarthakMishra/codemap/commit/5d65c5d0caec78ae4263d7d91d9af57c1618a2bb))

- **todo**: Add licensing comments to dev notes
  ([`11a94cf`](https://github.com/SarthakMishra/codemap/commit/11a94cff722b50f78d92420408afc77f6a54a263))

- **todo**: Future implementation plans for all cmds
  ([`2a8baaf`](https://github.com/SarthakMishra/codemap/commit/2a8baaf02db3b83fad99c515be60a1b386a738af))

- **todo**: Update todo list
  ([`9a2e7d3`](https://github.com/SarthakMishra/codemap/commit/9a2e7d37068c42aa96d2a84d49b429ebca101d72))

Several items were completed in the todo list, reflecting progress on the processor pipeline and
  related features.

- **utils**: Drop unused imports
  ([`89df3ea`](https://github.com/SarthakMishra/codemap/commit/89df3eac21d54999165828518feef13abeedc483))

- **utils**: Update PR utils with title and description generation
  ([`a382c44`](https://github.com/SarthakMishra/codemap/commit/a382c44d737731ff7a435aac8d6072d5ecf28af0))

### Features

- Add commit linter package This commit introduces a new package..
  ([`34f10f6`](https://github.com/SarthakMishra/codemap/commit/34f10f62fc4e4a2ff61ccce5706c0eb850a1bc1c))

- Add ERD generator with tests and fixtures
  ([`40317c6`](https://github.com/SarthakMishra/codemap/commit/40317c6e554cfc575f9a6c32f52ca3a2cf3497fb))

- Add function to commit command
  ([`9d77910`](https://github.com/SarthakMishra/codemap/commit/9d77910f363eba410e596af708566c8618e2198f))

- Add language-specific configurations and handlers for tree-sitter analysis
  ([`781fda7`](https://github.com/SarthakMishra/codemap/commit/781fda7956a6f05d29190eadaa24971da1007130))

- Add PR command for generating and managing pull requests
  ([`77fb553`](https://github.com/SarthakMishra/codemap/commit/77fb553359bc178e764ee918d74fe2ff05e2803e))

- Add tree flag to generate command
  ([`a5b7dd2`](https://github.com/SarthakMishra/codemap/commit/a5b7dd242adac2c50c4c816626c84ae408687b8d))

- Add uninstall script
  ([`43a89bb`](https://github.com/SarthakMishra/codemap/commit/43a89bb1bce44930cf9d0ec4e64ae5d7ec6b0db4))

- Add script to safely remove CodeMap and its dependencies

- Clean up configuration, cache, and documentation files

- Remove build artifacts and virtual environment

- Add upgrade script and update README
  ([`b7b669b`](https://github.com/SarthakMishra/codemap/commit/b7b669bba54311b43e336073a2f47af0107d6027))

- Add upgrade.sh script for updating existing installations

- Update README with upgrade instructions and versioning info

- Add configuration backup functionality during upgrades

- Improve installation documentation

- Added type stubs for requests_unixsocket
  ([`8f373ce`](https://github.com/SarthakMishra/codemap/commit/8f373ced157cd60affa48e16a83c3a35d0798bca))

- Bump version to 0.4.0
  ([`381de75`](https://github.com/SarthakMishra/codemap/commit/381de75c7bc83bd7c50237830d3ca04ddd35684f))

- Implement file filtering with exclude_patterns and gitignore
  ([`19b7d95`](https://github.com/SarthakMishra/codemap/commit/19b7d95d4ec0db1be4486cee437565988656f829))

- Improve markdown documentation formatting
  ([`88132ca`](https://github.com/SarthakMishra/codemap/commit/88132cafa15e833a94651c9ba1461103a297e85e))

- Add checkboxes to show file inclusion status

- Improve tree visualization with better line drawing

- Add two empty lines before major sections for better readability

- Fix directory checkbox status based on included files

- Include file content in parser output
  ([`fe338bc`](https://github.com/SarthakMishra/codemap/commit/fe338bcbfe6baa11c2d8ebcf540c734ad9ad933f))

- Add content field to all parser return values

- Include original content even when parsing fails

- Ensure content is preserved for error cases

- Integrate ERD generation into CLI and update configuration
  ([`616d9d3`](https://github.com/SarthakMishra/codemap/commit/616d9d3f21bd75368a6d5ba30a069cb8fb51efc3))

- Update configuration with output directory and timestamp settings
  ([`87fb0c5`](https://github.com/SarthakMishra/codemap/commit/87fb0c590c6634453757a99e22cc3f87c8cfd36f))

- **.codemap**: Increase token limit to 20000
  ([`601e179`](https://github.com/SarthakMishra/codemap/commit/601e179250fbe8b46fb823a5a05aa3dd528e59b5))

- **analysis**: Add import extraction for languages
  ([`7bbcc7c`](https://github.com/SarthakMishra/codemap/commit/7bbcc7c0ad02c360781edb3193d0704f5574468e))

- **analysis**: Add tree-sitter based code analysis module
  ([`0bee68e`](https://github.com/SarthakMishra/codemap/commit/0bee68e459e878d7a5180e54da7b72cd273a3581))

- **analyzer**: Extract dependencies from import statements
  ([`9c4b616`](https://github.com/SarthakMishra/codemap/commit/9c4b616c969ccc37c5745c371c4a13c5ca68c46c))

- **analyzer): add parser modules or feat(analyzer**: Extend __init..
  ([`ad33c57`](https://github.com/SarthakMishra/codemap/commit/ad33c573848ad517947e8a0d538e371645e74d43))

- **api**: Add support for Unix socket connections
  ([`e9d507c`](https://github.com/SarthakMishra/codemap/commit/e9d507ceb332548ccf3a9f16c2db867e0e3abd6d))

This change introduces the ability to use Unix sockets for inter-process communication between the
  daemon client and the API server. The client now checks for a socket path in the configuration and
  uses it if available. The API server has been updated to listen on the specified socket path when
  configured to do so.

- **build**: Add pytest-cov to dev dependencies
  ([`9ff0dc2`](https://github.com/SarthakMishra/codemap/commit/9ff0dc2e7ea6b36e322dceda69a7fe611969b4ec))

- **ci**: Add CI task to run checks and tests
  ([`628e7e1`](https://github.com/SarthakMishra/codemap/commit/628e7e1ed04f4901b5e2ac50f54cd19541ac89f7))

- **ci**: Add docs coverage check task
  ([`c569785`](https://github.com/SarthakMishra/codemap/commit/c569785b11abc792a0882b7e91a88fa7c2783212))

- **cli**: Add commit command for generating conventional commits
  ([`11e7be5`](https://github.com/SarthakMishra/codemap/commit/11e7be58cc5f773500f622146a19b3577bb7e319))

- **cli**: Add configuration wizard to init command
  ([`d7573fe`](https://github.com/SarthakMishra/codemap/commit/d7573fef298b89291a0e8e61cfebf8967e7e0516))

The init command now includes a guided configuration wizard to help users set up global and
  repository-specific settings. This includes configuring the daemon, LLM API, and storage options.

- **cli**: Add daemon command
  ([`81f6853`](https://github.com/SarthakMishra/codemap/commit/81f685372c21a3c370160b0495b24cb89c20abcf))

The daemon command provides a set of subcommands for managing the CodeMap daemon, including
  starting, stopping, and viewing logs.

- **cli**: Add daemon command to cli app
  ([`6fb183e`](https://github.com/SarthakMishra/codemap/commit/6fb183e598b2433059be20ce9adab7f526eed94d))

This commit introduces a new command to the cli app, allowing users to start the daemon. The daemon
  command is added to the list of available commands and can be executed using the 'daemon' keyword.

- **cli**: Add functionality for intelligent Git commits with AI-assisted message generation
  ([`24371f2`](https://github.com/SarthakMishra/codemap/commit/24371f22313afaff85ef4a6c731abab48749449d))

The tool analyzes your changes, splits them into logical chunks, and generates meaningful commit
  messages using LLMs.

- **cli**: Add intelligent Git commit message generation
  ([`441daf5`](https://github.com/SarthakMishra/codemap/commit/441daf5e16dc12ef24968e234a144687c18d7869))

This feature introduces AI-assisted message generation for Git commits. The tool analyzes changes,
  splits them into logical chunks, and generates meaningful commit messages using large language
  models.

- **cli**: Add intelligent Git commit message generation
  ([`35ad7cc`](https://github.com/SarthakMishra/codemap/commit/35ad7cc2ad9ceffc2c02e7d1820f0027d8f424e4))

This feature introduces AI-assisted message generation for Git commits. The tool analyzes changes,
  splits them into logical chunks, and generates meaningful commit messages using large language
  models.

- **cli**: Add interactive base branch selection
  ([`3c35a35`](https://github.com/SarthakMishra/codemap/commit/3c35a35cce7045abc52a9240053a8996a017ce84))

Introduce an interactive base branch selection feature when creating a PR. This change allows users
  to be prompted for the base branch when the `--interactive` flag is used and no base branch is
  provided via the CLI.

- **cli**: Add loading indicator for branch analysis
  ([`0bb9fe6`](https://github.com/SarthakMishra/codemap/commit/0bb9fe6c6020c1d61de48e6fcb03c1130046c797))

- **cli**: Add options for initial scan and daemon service
  ([`a364819`](https://github.com/SarthakMishra/codemap/commit/a3648199af159285abcaa03aff1ea064fa46129a))

This commit introduces new options to the init command for performing an initial repository scan and
  starting the daemon service after initialization.

BREAKING CHANGE: The command now accepts new options and performs additional actions after
  initialization.

- **cli**: Add package management commands
  ([`09efb38`](https://github.com/SarthakMishra/codemap/commit/09efb38df9f8436a6a5a7660e49fd33b1b83d0e5))

This commit introduces package management functionality to the CodeMap CLI, including commands for
  updating, version information, uninstallation, and system information.

BREAKING CHANGE: none

- **cli**: Add standardized error and warning display functions
  ([`d762e6f`](https://github.com/SarthakMishra/codemap/commit/d762e6fc5e78b0027b228f153e504e7b6efe360e))

Introduce functions to display error and warning summaries with standardized formatting, including
  titles, dividers, and detailed messages.

- **cli**: Handle changes with commit.py and pr.py
  ([`e1e5e3d`](https://github.com/SarthakMishra/codemap/commit/e1e5e3d6b4df065eba8eee616ffc7d8a31038a2c))

- **cli**: Handle staged files during commit and edit actions
  ([`a3fc63d`](https://github.com/SarthakMishra/codemap/commit/a3fc63d0f8707b26be028a6a718f5c448462d82b))

- **cli**: Improve commit message display with rich panels
  ([`fa49015`](https://github.com/SarthakMishra/codemap/commit/fa49015e80b0ae7b87acd7035bb8320762d2db02))

- **cli**: Load model and api key from config file
  ([`ec13c81`](https://github.com/SarthakMishra/codemap/commit/ec13c81c59519ba93c586d463939b888b0f10f0b))

- **cli**: Refactor cli directory structure and config management
  ([`105238c`](https://github.com/SarthakMishra/codemap/commit/105238c39e832475789adbf4cbc74ab8a5def13e))

Refactor cli to use new directory and config managers. This change improves the organization and
  maintainability of the cli codebase.

BREAKING CHANGE: cli directory structure and config management have been refactored

- **cli_app**: Remove option flags as default False
  ([`e0975ea`](https://github.com/SarthakMishra/codemap/commit/e0975ea7fc47cbaeb2feeb761c3dc48ebc9275b1))

- **cli_utils**: Add standardized progress indicator
  ([`bb01344`](https://github.com/SarthakMishra/codemap/commit/bb013442268bd658dc9078c49576e0572569d520))

Introduce a context manager for a standardized progress indicator that supports different styles
  uniformly, including spinner, progress bar, and step-by-step progress.

- **codemap**: Add CLI entry point for Python module
  ([`2ee1550`](https://github.com/SarthakMishra/codemap/commit/2ee1550ac5baec87f24dc71cb9cf08afed326a5d))

- **codemap**: Add code chunking module with syntax-based strategy
  ([`a410e78`](https://github.com/SarthakMishra/codemap/commit/a410e78a59b9a101c6d72a227fee54a6b771713d))

- **codemap**: Add code documentation generation based on LOD
  ([`dba8e69`](https://github.com/SarthakMishra/codemap/commit/dba8e69ac217aed1c0879c17123b0feeaea8636a))

This feature introduces a new capability to generate code documentation based on the Level of Detail
  (LOD) entities. It allows for the creation of markdown documentation from processed LOD entities,
  including headers, repository information, table of contents, and code documentation grouped by
  file.

- **codemap**: Add commit message generator module
  ([`3655cd7`](https://github.com/SarthakMishra/codemap/commit/3655cd787c25da17ecff76f85614148a210509d0))

- **codemap**: Add file filtering utilities
  ([`d6296b0`](https://github.com/SarthakMishra/codemap/commit/d6296b08164d844c1df1a4c15939ec5532095674))

- **codemap**: Add.coveragerc configuration
  ([`945fa0f`](https://github.com/SarthakMishra/codemap/commit/945fa0fdc815ed7cfe5c788a79f64b9952939b5b))

- **codemap**: Update package description and version
  ([`8849505`](https://github.com/SarthakMishra/codemap/commit/884950521b66ab8c20aaa901b0586a8bb24c0f76))

The package description has been updated to reflect its new purpose as an AI-powered developer
  toolkit. The version has also been updated to 0.1.0, indicating a significant change in the
  package's functionality.

- **codemap/cli**: Add CLI package for CodeMap
  ([`6258098`](https://github.com/SarthakMishra/codemap/commit/62580989519f90e64781bd352ff9d840a8b453e7))

- **codemap/cli**: Add main CLI module for CodeMap initialization and documentation generation
  ([`96e1a0a`](https://github.com/SarthakMishra/codemap/commit/96e1a0ac72bb73dee9bb433c3dfeab3f40aa4640))

- **codemap/cli/commit**: Add command for generating conventional commits
  ([`f1eea12`](https://github.com/SarthakMishra/codemap/commit/f1eea1265d41e0908fceaecd0b485f498b13c056))

- **codemap/cli_entry**: Add CLI entry point implementation
  ([`f2a8d44`](https://github.com/SarthakMishra/codemap/commit/f2a8d443f8525f46b11c3fa31425b112563e3911))

- **codemap/commit**: Add commit feature package
  ([`b159fe5`](https://github.com/SarthakMishra/codemap/commit/b159fe54691fde6fcaff450374ab2b5762947c6c))

- **codemap/commit/command**: Add commit command implementation
  ([`eceb6c3`](https://github.com/SarthakMishra/codemap/commit/eceb6c3454a67f3bebe28d1977253b368041e904))

- **codemap/commit/diff_splitter**: Add diff splitting utilities
  ([`8bef1e2`](https://github.com/SarthakMishra/codemap/commit/8bef1e210249ff22f7a5212361766af539a52d3e))

- **codemap/commit/interactive**: Add interactive commit interface
  ([`5f69e85`](https://github.com/SarthakMishra/codemap/commit/5f69e8502a3e728d1228ff5ea6c2a8ae3d31d22a))

- **codemap/git**: Add change detection for Git repository
  ([`6e3ee59`](https://github.com/SarthakMishra/codemap/commit/6e3ee59b54176ba723c974864d4958d3f51a16c2))

- **codemap/git**: Add Git functionality module
  ([`95c8077`](https://github.com/SarthakMishra/codemap/commit/95c8077e5061367967cb682864d1f70f4504aba0))

- **codemap/git**: Add Git wrapper for CodeMap
  ([`274fd33`](https://github.com/SarthakMishra/codemap/commit/274fd3358575a64082b19d9b76242193a4b88166))

- **codemap/git**: Add support for untracked files in GitDiff
  ([`d157c1f`](https://github.com/SarthakMishra/codemap/commit/d157c1fb55ab40e28c24dc77c9ce30e6f963244a))

- **codemap/utils**: Update codemap.utils imports from codemap.git to..
  ([`905a59a`](https://github.com/SarthakMishra/codemap/commit/905a59ac10c64b6069f4ca77067d1211ba20175f))

- **codemap/utils/git_utils**: Add function to get untracked files
  ([`3dbaa80`](https://github.com/SarthakMishra/codemap/commit/3dbaa80a7fcff1c79dde9ab9235c0597b8709fdf))

- **codemap/utils/git_utils**: Add function to unstage files
  ([`e3a0af1`](https://github.com/SarthakMishra/codemap/commit/e3a0af16df1d60528dc3781a7039a889287f8658))

- **codemap/utils/git_utils**: Add Git utilities module
  ([`1b2f17a`](https://github.com/SarthakMishra/codemap/commit/1b2f17a4a3960a763394ee96af3e059787d984ca))

- **command**: Replace utils import point
  ([`37743ec`](https://github.com/SarthakMishra/codemap/commit/37743ec7ca86e984b72c64fc6359fd2b5b3db040))

- **commit**: Add AI-assisted message generation for intelligent Git commits
  ([`c55cb4a`](https://github.com/SarthakMishra/codemap/commit/c55cb4ab562a8e544c56f7f941a6e9788db7db53))

This feature introduces a new capability to analyze code changes, split them into logical chunks,
  and generate meaningful commit messages using large language models.

- **commit**: Add detection for untracked files
  ([`bcdfc13`](https://github.com/SarthakMishra/codemap/commit/bcdfc13d2481ea05f8d42e6bdb0c510a29313dec))

- **commit**: Add intelligent Git commit message generation
  ([`b6b4b19`](https://github.com/SarthakMishra/codemap/commit/b6b4b1914dc908ce91b6819ba16f52dc8376d013))

This feature introduces AI-assisted message generation for Git commits. The tool analyzes changes,
  splits them into logical chunks, and generates meaningful commit messages using large language
  models.

- **commit_generator**: Add commit message generation package for CodeMap
  ([`a45988f`](https://github.com/SarthakMishra/codemap/commit/a45988fe191cbaef9e615015341c47bf91af5e7c))

This package provides modules for generating commit messages using LLMs.

BREAKING CHANGE: none

- **commit_generator**: Add intelligent Git commit message generation
  ([`6e9b085`](https://github.com/SarthakMishra/codemap/commit/6e9b0853e10a4d83476d784486a2300c9f975172))

This feature introduces AI-assisted message generation for Git commits. The tool analyzes changes,
  splits them into logical chunks, and generates meaningful commit messages using large language
  models.

- **commit_generator**: Store current branch at initialization and restore it after execution
  ([`83b8090`](https://github.com/SarthakMishra/codemap/commit/83b8090a0c6ac1e9f23e0588359343c70b58cbd0))

This change ensures that the original branch is restored after the commit command execution,
  preventing unexpected branch switching.

- **config**: Add commit feature configuration
  ([`28b114d`](https://github.com/SarthakMishra/codemap/commit/28b114dc8ad17b9e9d1931ff9303aa25a7da6c85))

- **config**: Add commit feature configuration to codemap.yml
  ([`f59a692`](https://github.com/SarthakMishra/codemap/commit/f59a6927176b42b23a8d6cbfa5ecfea1f2c3aa04))

- **config**: Add configuration management utilities
  ([`dd7fb21`](https://github.com/SarthakMishra/codemap/commit/dd7fb21bcf87e82e6bf7b0ff46f1fd80bb9c8432))

This commit introduces a new configuration management system, including utilities for loading,
  updating, and managing configuration across different scopes. The system provides a flexible and
  extensible way to handle configuration data.

- **config**: Add entity relationship graph and tree output
  ([`69a3c03`](https://github.com/SarthakMishra/codemap/commit/69a3c03b26b08e50dd8a0630216477fc7d1b3883))

The entity relationship graph provides a visual representation of the code structure, while the tree
  output displays the directory hierarchy.

- **config**: Add gen configuration options
  ([`0ee22dc`](https://github.com/SarthakMishra/codemap/commit/0ee22dc8f8aaeb3a8e0160571a2e9eb9d5bba694))

Added configuration options for the gen command, including token limit, gitignore usage, output
  directory, and more.

- **config**: Add get_commit_hooks method
  ([`1a90b3f`](https://github.com/SarthakMishra/codemap/commit/1a90b3faf72bcc2f27e4ac32a999fc6e634332a6))

- **config**: Add ignore settings for Python analysis
  ([`a9c11fd`](https://github.com/SarthakMishra/codemap/commit/a9c11fdf2dbb38e658327ec51b9dc40e568558e0))

Add ignore settings for Python analysis to exclude vendor, scripts, and typings directories

- **config**: Add Mermaid diagram configuration options
  ([`245ba2c`](https://github.com/SarthakMishra/codemap/commit/245ba2c7d8f70eee12b78f1982e0945f79d5e6b9))

This commit introduces configuration options for generating Mermaid diagrams, including entity types
  to include, relationships to show, legend visibility, and removal of unconnected nodes.

- **config**: Add pull request configuration
  ([`e14f9c3`](https://github.com/SarthakMishra/codemap/commit/e14f9c337811fb5b38ca6b3efa02f552db7521d6))

The pull request configuration has been added to the codemap config file. This includes default
  branch settings, Git workflow strategy, branch mapping for different PR types, and content
  generation settings.

- **config**: Add pull request configuration options
  ([`6135aa7`](https://github.com/SarthakMishra/codemap/commit/6135aa7d5ab47977b14d6ad2b20a1c156c04469d))

This commit introduces new configuration options for pull requests, including default branch
  settings, Git workflow strategy, branch mapping, and content generation settings.

- **config**: Add repo root and llm config to ConfigLoader
  ([`d1e7bd5`](https://github.com/SarthakMishra/codemap/commit/d1e7bd5871ddf892d7a9db1a49eb23baee9eae25))

- **config**: Add semantic chunking configuration
  ([`60edf14`](https://github.com/SarthakMishra/codemap/commit/60edf14275ec54e5d0badb9cc21558db8212da47))

- **config**: Update codemap configuration options
  ([`24eb512`](https://github.com/SarthakMishra/codemap/commit/24eb51230845449a9fe6eee8c25101a4a1ccf689))

The configuration file has been updated to include additional options for code generation, commit
  features, and server settings.

BREAKING CHANGE: some configuration options have been changed

- **config**: Update config loader to handle invalid values and nonexistent files
  ([`f25fb72`](https://github.com/SarthakMishra/codemap/commit/f25fb7236b6277e354bfeb8047e47f5d4a5429ec))

The config loader has been updated to no longer validate types during loading, instead merging
  values as-is. It also no longer raises exceptions for nonexistent config files, instead using
  default config values.

- **config**: Update default base branch
  ([`3aca438`](https://github.com/SarthakMishra/codemap/commit/3aca4386db8c21c25e4132981de71c6ba36149f4))

Change default base branch from None to dev for better development workflow

- **config**: Update default base branch and feature prefix
  ([`76cf865`](https://github.com/SarthakMishra/codemap/commit/76cf865741483f5bd2892a46bec09914b15fc0fa))

The default base branch has been updated from 'main' to 'dev' and the feature prefix has been
  updated from 'feature/' to 'feat/' to align with conventional naming conventions.

- **config**: Update default config with new logging and caching settings
  ([`1457645`](https://github.com/SarthakMishra/codemap/commit/1457645adccc00c5c5eb48983f9dc148bbaad22f))

The default configuration has been updated to reflect new logging and caching settings. The logging
  configuration now points to new directories for pid and log files. Additionally, a cache directory
  has been specified.

- **config**: Update embedding model and dependencies
  ([`cb8d763`](https://github.com/SarthakMishra/codemap/commit/cb8d7631cb97418ff38b9d0879ae0d876b43cbed))

Update pyproject.toml to include model2vec dependencies and change embedding model to
  sarthak1/Qodo-Embed-M-1-1.5B-M2V-Distilled

BREAKING CHANGE: This change updates the embedding model and adds new dependencies

- **config_loader**: Add method to get default base branch
  ([`77507c6`](https://github.com/SarthakMishra/codemap/commit/77507c691faa9df4b9c1b52746cdefdfc80316e6))

- **config_loader**: Update default workflow strategy
  ([`1794cc6`](https://github.com/SarthakMishra/codemap/commit/1794cc68ed6ab5016e4c95c2ab51389e37025573))

- **daemon**: Add CodeMap daemon with API server and client
  ([`d67bc1a`](https://github.com/SarthakMishra/codemap/commit/d67bc1af397bbed14c5b3ac53d4863b34ef635a3))

This commit introduces the CodeMap daemon, which provides a background service for running CodeMap.
  The daemon includes an HTTP API server for remote interaction and a client library for
  communicating with the daemon. The API allows for executing tasks, checking status, and managing
  jobs.

- **dependencies**: Add missing dependencies for CLI functionality
  ([`2a62991`](https://github.com/SarthakMishra/codemap/commit/2a6299190d10aebd69e60acbb84318eecc83a404))

- **dependencies**: Add requests-unixsocket dependency
  ([`3095481`](https://github.com/SarthakMishra/codemap/commit/3095481cde59ca2b933ef0c466004eaa92549696))

Add requests-unixsocket to the list of dependencies to enable Unix socket support for requests.

- **dependencies**: Add required dependencies for project
  ([`e9ef0c4`](https://github.com/SarthakMishra/codemap/commit/e9ef0c453e2814ca0949bf26f8746f8de24e2563))

- **dependencies**: Add sentence-transformers library
  ([`24d3acb`](https://github.com/SarthakMishra/codemap/commit/24d3acb73bd2a5f85ace6eb2cd6b47540eb1865d))

- **dependency**: Add requests-unixsocket dependency
  ([`a63793a`](https://github.com/SarthakMishra/codemap/commit/a63793acf0e20ea6bcb7522e54392ad97ad666da))

- **deployment**: Add supervisor configuration and installation script
  ([`92283cb`](https://github.com/SarthakMishra/codemap/commit/92283cb0660cdee777ad3f213e58da56e0afdc81))

This commit introduces a new feature to deploy CodeMap with supervisor configuration. The
  installation script checks for supervisor installation, creates necessary log directories, and
  handles different supervisor configuration locations.

- **deployment**: Add systemd service and installation script
  ([`10ac1c6`](https://github.com/SarthakMishra/codemap/commit/10ac1c6b8bb2ab19057ac59fbb2d1ba72cdc29d6))

This commit introduces a systemd service file for the CodeMap daemon and a corresponding
  installation script. The service file defines a simple service that runs the CodeMap daemon, while
  the installation script copies the service file to the user's systemd directory, reloads the
  systemd user configuration, and provides instructions for enabling and starting the service.

- **diff_splitter**: Add diff splitting functionality
  ([`9429be0`](https://github.com/SarthakMishra/codemap/commit/9429be06b7c80f920e5757757f124b6124f79490))

This commit introduces a new feature to split Git diffs into logical chunks. The diff splitting
  package provides utilities for splitting Git diffs into logical chunks.

- **embedding**: Add cache directory configuration to embedding generator
  ([`da6d0d3`](https://github.com/SarthakMishra/codemap/commit/da6d0d3693d2dbe92baba425dc57065e5d79c1f0))

The cache directory is now configurable through the EmbeddingGenerator class. If no cache directory
  is provided, the default cache directory from the directory manager will be used. The cache
  directory is created if it does not exist.

- **embedding**: Add embedding generator and models
  ([`15d0fb6`](https://github.com/SarthakMishra/codemap/commit/15d0fb6030c61373267342a1be129b619736085b))

- **embeddings**: Add type stub files for embedding functions
  ([`dedb8fa`](https://github.com/SarthakMishra/codemap/commit/dedb8fa9a32cd0bec8c8a2e053242da09152674d))

- **file_utils**: Add function to read file content with error handling
  ([`2769e30`](https://github.com/SarthakMishra/codemap/commit/2769e308873f24c0acf9e5a8684404d5bf4303cb))

- **gen**: Add code documentation generation module
  ([`02461ea`](https://github.com/SarthakMishra/codemap/commit/02461ea7630ffc924dd598f6291472a366d5acc9))

This module provides functionality for generating LLM-optimized code context and human-readable
  documentation. It includes features such as semantic code compression, directory tree generation,
  and support for multiple documentation formats.

- **gen**: Add entity relationship diagram and code documentation generation
  ([`eb89977`](https://github.com/SarthakMishra/codemap/commit/eb89977381cc63d0e7f8bf73d3bea617fb0a4c64))

This feature adds the ability to generate entity relationship diagrams and code documentation based
  on the Level of Detail (LOD) configuration. It includes a Mermaid diagram for entity relationships
  and recursive documentation formatting.

- **gen**: Add support for mermaid configuration
  ([`ae52cb9`](https://github.com/SarthakMishra/codemap/commit/ae52cb9bce22bf905d0c73531c109c257648d472))

This commit adds support for mermaid configuration, including entity types, relationships, legend
  visibility, and unconnected node removal.

- **git**: Add git metadata analysis module
  ([`62f6568`](https://github.com/SarthakMishra/codemap/commit/62f6568160b95e1dc6659a3d8bce6afb5a6d5d8a))

- **git-utils**: Add git utilities module
  ([`3996054`](https://github.com/SarthakMishra/codemap/commit/39960546f545894460f49c8f5343b002f03d36fa))

- **git_utils**: Add functions for selective staging and stashing
  ([`c7d927b`](https://github.com/SarthakMishra/codemap/commit/c7d927be64fd1b25c11ef0cc7b4dd786e85a904e))

- **git_utils**: Add validate_repo_path function
  ([`f5b3046`](https://github.com/SarthakMishra/codemap/commit/f5b3046ae13db2568ab313fb2a0b486efcf099ab))

- **gui**: Update tests to match refactored llm_utils
  ([`875bd38`](https://github.com/SarthakMishra/codemap/commit/875bd38fa4acabb0703c8dc8fb3eaba2c4b4bfad))

- **install**: Add easy install script and update documentation
  ([`2e295e2`](https://github.com/SarthakMishra/codemap/commit/2e295e25d7ed5d8de538ed1d189b076a1620a7cc))

The installation process has been streamlined with the addition of an easy install script. This
  script allows users to quickly and easily install CodeMap. Additionally, the documentation has
  been updated to reflect these changes and provide clear instructions on how to install, update,
  and uninstall CodeMap.

- **lang**: Add initial language configurations
  ([`852955e`](https://github.com/SarthakMishra/codemap/commit/852955e139d5525c4142004f8fb6fdfc940eecfa))

- **languages**: Add initial language configurations for JavaScript,..
  ([`51cbd47`](https://github.com/SarthakMishra/codemap/commit/51cbd4796d1568845fee2f6a1dff8a5edb2b476c))

- **llm**: Add LLM module for CodeMap
  ([`e4a74e8`](https://github.com/SarthakMishra/codemap/commit/e4a74e8237b49c4ee4b5cdc85a6b36b467db33f1))

This commit introduces a new module for interacting with language models, providing a unified
  interface for various LLM services. The module includes a client for making API calls,
  configuration management, and utility functions for loading templates and generating text.

- **lock**: Add missing dependencies for coverage and pytest-cov
  ([`1da85e8`](https://github.com/SarthakMishra/codemap/commit/1da85e80e4437cdefef5b8ec7e928d54b52f6c27))

- **lod**: Implement Level of Detail (LOD) generation for code analysis
  ([`994d203`](https://github.com/SarthakMishra/codemap/commit/994d20318fd0ab1a6eec53c11bfd0dba31c9e6bf))

This change introduces a new feature to generate different levels of detail from source code using
  tree-sitter analysis. The LOD approach provides a hierarchical view of code, from high-level
  entity names to detailed implementations.

- **lsp**: Add LSP analyzer implementation
  ([`20494b7`](https://github.com/SarthakMishra/codemap/commit/20494b74066bd74b9a6299f25238875f49e9c9a7))

- **message_generator**: Add commit linting and regeneration The com..
  ([`f3f784b`](https://github.com/SarthakMishra/codemap/commit/f3f784bc423e150e373ff4a8902be148a0e7ff81))

- **models**: Optimize Qodo-Embed-1-1.5B model using Model2Vec
  ([`3292f5d`](https://github.com/SarthakMishra/codemap/commit/3292f5d50bb6ddd4c52b80e1b23bb905e24169e1))

- **multilspy**: Add MultiLSPy integration documentation for LSP usage
  ([`b4ac109`](https://github.com/SarthakMishra/codemap/commit/b4ac10910880fc0d72b577677ca5f5abe45ee49b))

- **pipeline**: Add LSP analysis support
  ([`5f0dfc4`](https://github.com/SarthakMishra/codemap/commit/5f0dfc49d954a9fc6f5dc41432707d4b3d3cbbfb))

- **pr-generator**: Add PR generation package for CodeMap
  ([`7fdf2aa`](https://github.com/SarthakMishra/codemap/commit/7fdf2aa2bcaa21afd18eb38b120f2a34e28a08c4))

This package provides modules for generating and managing pull requests.

- **pr_cmd**: Add interactive mode for PR title and description editing
  ([`1134f8a`](https://github.com/SarthakMishra/codemap/commit/1134f8a5aa891b1f716becf8c5ad9883e89bf64f))

In interactive mode, the user is now prompted to review and edit the PR title and description before
  creation. This includes displaying the title and description in panels and asking for confirmation
  to edit. If editing is chosen, the user can either manually edit the text or regenerate the
  description using an LLM.

- **pr_generator**: Add JSON output to create_pull_request function
  ([`9c66e38`](https://github.com/SarthakMishra/codemap/commit/9c66e3844660191602898ea6eb852500876d3bd3))

The create_pull_request function now includes a --json option to output the PR number, URL, title,
  body, and headRefName.

- **pr_generator**: Introduce PRCreationError for handling pull request creation and update failures
  ([`7f62f24`](https://github.com/SarthakMishra/codemap/commit/7f62f248e5da28d2c64efed00404de2775ff0146))

The PRCreationError class was added to handle exceptions during pull request creation and updates.
  This change improves error handling and provides more informative error messages.

- **pr_utils**: Add check for branch existence in get_branch_relation
  ([`93a96aa`](https://github.com/SarthakMishra/codemap/commit/93a96aa2de9d36a64fbc0198a1321be9b3ec38aa))

The current implementation of get_branch_relation assumes that both branches exist. However, this is
  not always the case. This change adds a check to see if both branches exist before trying to
  determine if one is an ancestor of the other.

- **processor**: Add initial code processing pipeline
  ([`77425b8`](https://github.com/SarthakMishra/codemap/commit/77425b8815c6cecf29fcd73634e834dd6a5fcfcb))

- **processor**: Add initialize_processor function and update pipeline
  ([`b5de168`](https://github.com/SarthakMishra/codemap/commit/b5de168b302d081b828774962fdc75b80d0c5d24))

This commit introduces a new function `initialize_processor` to set up the processing pipeline with
  the appropriate directory structure for storing embeddings and vector databases. It also updates
  the `ProcessingPipeline` class to use the directory manager and register the project.

- **pyproject**: Add ignore rule for TC001
  ([`951ae40`](https://github.com/SarthakMishra/codemap/commit/951ae40bb4aa579ba54276c4258f2baab4e851b9))

- **pyproject**: Bump version to 0.4.1
  ([`a3babb7`](https://github.com/SarthakMishra/codemap/commit/a3babb7030f1afd708fe00d60d3865b16158966b))

- **pyproject**: Remove pre-commit hook
  ([`7fa348f`](https://github.com/SarthakMishra/codemap/commit/7fa348f3a2379db57587b6e4ad4a4e99f1f079f3))

- **storage**: Add default configuration for LanceDB storage backend
  ([`3bca60a`](https://github.com/SarthakMishra/codemap/commit/3bca60a432dfcb87b38cd1a454f34c1023aaee1e))

This commit introduces a new method `from_config` to create a storage configuration using the
  application's configured directories. It also adds a `create_default` method to create a
  LanceDBStorage instance with default configuration.

- **storage**: Add LanceDB storage backend implementation
  ([`3c214a9`](https://github.com/SarthakMishra/codemap/commit/3c214a96288aa75898dafa82205c3ca111e8d925))

- **storage**: Add LSP metadata storage and retrieval
  ([`5bd2729`](https://github.com/SarthakMishra/codemap/commit/5bd2729e9d702db279587fa1e73804206ef87e59))

- **Taskfile**: Add lint:fix-unsafe task to pre-commit hooks
  ([`3030c78`](https://github.com/SarthakMishra/codemap/commit/3030c781b57e4cffdfd1b5e5f016d0044641cb0d))

- **Taskfile**: Add uv task automation
  ([`4868c85`](https://github.com/SarthakMishra/codemap/commit/4868c852a2092cae4a29027a5c0611e7c08e7a56))

- **test**: Add test suite for commit feature
  ([`405995b`](https://github.com/SarthakMishra/codemap/commit/405995beefb7a493aa04a71173713637f6435bbe))

- **test**: Remove test script
  ([`142678a`](https://github.com/SarthakMishra/codemap/commit/142678a18f46695bfd5b3ac5d712f4ccb39525e8))

- **testing**: Add storage and cli test utilities
  ([`71979f3`](https://github.com/SarthakMishra/codemap/commit/71979f31d4688bb3019bdd9ac6d55f8d11a52507))

- **tests**: Add test files for git utilities
  ([`987cd26`](https://github.com/SarthakMishra/codemap/commit/987cd26bb2cd229fa46a4110530ae45f18d064d0))

This commit adds several test files for git utility functions, including test cases for commit
  command generation, Git hook handling, and various git utility functions.

- **tests**: Remove test files
  ([`09fbdcd`](https://github.com/SarthakMishra/codemap/commit/09fbdcda7ae33fb2120a8adbe8cb02ae8b272a16))

Removed test files from the tests directory to clean up the repository.

- **tests**: Update test files to follow commit conventions
  ([`6d6fe0d`](https://github.com/SarthakMishra/codemap/commit/6d6fe0d9063ca3dd01186c5b88e019e093273e27))

- **tests**: Update tests for Pull Request utilities
  ([`b6eea3e`](https://github.com/SarthakMishra/codemap/commit/b6eea3ed16160291459b27137aaceb37e6b6b973))

- **tests/utils**: Add tests for PR workflow strategies and templates
  ([`3ecfbf1`](https://github.com/SarthakMishra/codemap/commit/3ecfbf14d325d2830da4704044311a874999b94e))

This commit introduces tests for various PR workflow strategies including GitHub Flow, GitFlow, and
  Trunk-Based strategies. It also includes tests for PR template functionality and utility functions
  related to PR operations.

- **tree-sitter**: Add tree-sitter based code analysis
  ([`babfafa`](https://github.com/SarthakMishra/codemap/commit/babfafa8ab2befd8030c7762fa3810bcc54cb929))

This module provides functionality for analyzing source code using tree-sitter. It extracts
  structure and semantic information from code files in various programming languages.

- **typescript**: Add TypeScript-specific syntax handling logic
  ([`161fa5d`](https://github.com/SarthakMishra/codemap/commit/161fa5d73b557249a23cc6a48eb063037f7dcba2))

This commit introduces TypeScript-specific syntax handling logic, including entity type
  determination, name extraction, body node retrieval, and children processing. It also updates the
  configuration loader to handle TypeScript-specific settings.

- **typing**: Multilspy type stub
  ([`5b06f69`](https://github.com/SarthakMishra/codemap/commit/5b06f69ddd9933c56425c69c87d6b329bddc99b8))

- **typing**: Pandas type stub
  ([`c807021`](https://github.com/SarthakMishra/codemap/commit/c807021f3874d0056833eb66987d8a765d6540b7))

- **typing**: Pyarrow type stub
  ([`ae6ac5a`](https://github.com/SarthakMishra/codemap/commit/ae6ac5aabb0989187e64ea8238548914427da816))

- **typings**: Add pandas API type stubs
  ([`4732cd4`](https://github.com/SarthakMishra/codemap/commit/4732cd41b7519d70453f6f6e55c927fa05ed6432))

- **typings**: Add type stub file for pandas arrays
  ([`23cfeb3`](https://github.com/SarthakMishra/codemap/commit/23cfeb31046a4bb94e3e2039c062c7beb191863c))

- **typings**: Add type stub files for lancedb integrations
  ([`dd0ddc4`](https://github.com/SarthakMishra/codemap/commit/dd0ddc46c33b39a39940cb4401a7b05359e1f31c))

- **typings**: Add type stub files for lancedb rerankers
  ([`2a42184`](https://github.com/SarthakMishra/codemap/commit/2a421847c51f62746ec14996a91fcb16f782499e))

- **typings**: Add type stub files for pandas config module
  ([`c89e1ef`](https://github.com/SarthakMishra/codemap/commit/c89e1ef9b466b6e8128758cc861bb93af3be87a5))

- **typings**: Add type stub files for pandas module
  ([`0506492`](https://github.com/SarthakMishra/codemap/commit/0506492e1df7b75524ff225c13d3de7698520e32))

- **typings**: Add type stub files generated by pyright
  ([`78cbbba`](https://github.com/SarthakMishra/codemap/commit/78cbbba9dceb7dd28bbadf569b152c8baf53ccb4))

- **typings**: Add type stubs for DataFrame interchange protocol
  ([`94d2947`](https://github.com/SarthakMishra/codemap/commit/94d29473fe39cd4a54e8627bec343a37cbfb9e60))

- **typings**: Add type stubs for pandas API
  ([`f73c6b6`](https://github.com/SarthakMishra/codemap/commit/f73c6b6c980d4727b4f7f9c4340f873223517206))

- **typings**: Add type stubs for pandas API types
  ([`0f23bff`](https://github.com/SarthakMishra/codemap/commit/0f23bffbbfaee432d7039bccbf3fcef53f8d31cc))

- **typings**: Add type stubs for pandas errors module
  ([`8e8fe32`](https://github.com/SarthakMishra/codemap/commit/8e8fe3218d11acd2b2712ec45c0773218e9203bf))

- **typings**: Add type stubs for pandas extensions API
  ([`3f98742`](https://github.com/SarthakMishra/codemap/commit/3f9874225917db3410e991b2d4d5da140158ac5d))

- **typings**: Add type stubs for pandas indexers
  ([`b90c039`](https://github.com/SarthakMishra/codemap/commit/b90c0393befaad1adfc48f199bebd6e4d3bd2de2))

- **typings**: Add type stubs for pandas.util.version module
  ([`e9a0c1f`](https://github.com/SarthakMishra/codemap/commit/e9a0c1f6bb2c419e1a1af24182eebd5db0abb44b))

- **typings**: Add type stubs for pyperclip module
  ([`b644ec8`](https://github.com/SarthakMishra/codemap/commit/b644ec820d1440bf2db33971041df6327779ddbc))

- **typings**: Add type stubs for remote module
  ([`de30868`](https://github.com/SarthakMishra/codemap/commit/de30868e180579e72983802593a82dadace66243))

- **typings**: Add type-stubs for model2vec
  ([`6ee8b59`](https://github.com/SarthakMishra/codemap/commit/6ee8b59065a6cd572f7a6a50520b3bc255cb236a))

- **utils**: Add generative text function
  ([`4a896a1`](https://github.com/SarthakMishra/codemap/commit/4a896a1299f789c6d841d2713c61d1131bd5f1f8))

- **utils**: Add imports
  ([`4d53cd9`](https://github.com/SarthakMishra/codemap/commit/4d53cd9699f889cb754a597fbbb6ae6286ae29c4))

- **utils**: Add LLMErrors and loading spinner utilities
  ([`6b5bae6`](https://github.com/SarthakMishra/codemap/commit/6b5bae68891f63b665f50a7148e9fa81b90d7fd1))

- **utils**: Add package utilities for management and updates
  ([`2ab4478`](https://github.com/SarthakMishra/codemap/commit/2ab447819c78c76f1cb638f2c344c9bf2d9ae0f8))

This commit introduces a set of utility functions for package management and updates. These
  functions enable checking for updates, updating the package using pip, uninstalling the package,
  and retrieving system information for debugging purposes.

- **utils**: Add path utilities for handling file system operations
  ([`2c7823b`](https://github.com/SarthakMishra/codemap/commit/2c7823bb1e198946629677108c256f2fd6002964))

- **utils**: Add repository path validation function
  ([`785684d`](https://github.com/SarthakMishra/codemap/commit/785684d66630b21de2a0d02eda095499a73d32b9))

- **utils**: Implement workflow strategies and PR utilities
  ([`da318ea`](https://github.com/SarthakMishra/codemap/commit/da318ea4b558d1b4c60bd7262a5012e4f1ef7a6b))

Add implementations for GitHub Flow, GitFlow, and Trunk-Based Development strategies. Introduce
  functions for branch name suggestion, PR content generation, and release notes creation.

- **utils**: Update import paths for utils
  ([`e7b7f61`](https://github.com/SarthakMishra/codemap/commit/e7b7f6103a40eef8621ef439bd7dbc2e2aec9d02))

- **uv.lock**: Update dependencies
  ([`afe8bb0`](https://github.com/SarthakMishra/codemap/commit/afe8bb0ea547063988fb96e51db0a7a6e3058faf))

- **watcher**: Add file system watcher implementation
  ([`0130631`](https://github.com/SarthakMishra/codemap/commit/0130631a7e94f326b31b2f02f0a6db74b7347d06))

### Refactoring

- Codemap update LLM model and doc
  ([`deab28f`](https://github.com/SarthakMishra/codemap/commit/deab28f7953d1c7009d80b4f6adb44a609d7656f))

- Improve file tree generation and sorting
  ([`ea0cb91`](https://github.com/SarthakMishra/codemap/commit/ea0cb917f86d0fafac2b3c76a9b1e7877a8c6947))

- Add TreeState dataclass to encapsulate tree generation state

- Add MAX_TREE_DEPTH constant to prevent infinite recursion

- Improve file sorting with stable secondary sort

- Add proper handling of file inclusion in tree view

- Improve markdown generation and escaping logic
  ([`1fc7ac2`](https://github.com/SarthakMishra/codemap/commit/1fc7ac291689076c0c4bd7f576811585bc144868))

- Add language-specific code block formatting

- Simplify markdown escaping to only handle inline formatting chars

- Improve file documentation generation with better structure

- Fix escaping in docstrings while preserving code blocks

- Improve semantic commit strategy and make it the only option
  ([`c8f9806`](https://github.com/SarthakMishra/codemap/commit/c8f980688ce074ecb1c654d4f2a52f3e6fc22e7c))

- Refactor commit workflow for better maintainability
  ([`74f17e4`](https://github.com/SarthakMishra/codemap/commit/74f17e421a40d40885af0772be71206b1a977f6d))

- Refactor semantic splitting logic for better performance and readability
  ([`6380187`](https://github.com/SarthakMishra/codemap/commit/63801870384e1abde8b45f21c2b9a5bb6c3130ed))

- Refactor: remove unused imports
  ([`1f84e91`](https://github.com/SarthakMishra/codemap/commit/1f84e91694a2a3af273c5abe90f7226e8614c307))

- Remove ERD implementation and fix code quality issues
  ([`6f09d58`](https://github.com/SarthakMishra/codemap/commit/6f09d58ad41f3643c6a0eac51a7f07ee682c0608))

- Update type annotations to use Python 3.10+ syntax
  ([`ac9dd01`](https://github.com/SarthakMishra/codemap/commit/ac9dd01155659e0fb4b4b7d9faa6fb6f4cf9f738))

- Use Path.cwd() instead of os.getcwd()
  ([`d0c25aa`](https://github.com/SarthakMishra/codemap/commit/d0c25aafd85e11cea353e10431d7c4d96f0052f3))

- **.codemap.yml**: Update commit strategy to semantic
  ([`89efebd`](https://github.com/SarthakMishra/codemap/commit/89efebd9c85b0d934ca3ab51faaabe0dbf1b8b00))

- **analyzer**: Extract file filtering logic into separate class
  ([`76bf314`](https://github.com/SarthakMishra/codemap/commit/76bf314a04cae95d89d61e042c90b2ac8d2868b0))

- **analyzer**: Update docstrings for clarity
  ([`001a173`](https://github.com/SarthakMishra/codemap/commit/001a1736d04a1301342d7a18a2a1296ef4a7524d))

- **analyzer**: Update docstrings for clarity
  ([`81642f0`](https://github.com/SarthakMishra/codemap/commit/81642f09913f7898859eee0fc9d60c5d55fb4ed5))

- **chunking**: Add dependency extraction fallback
  ([`91edff5`](https://github.com/SarthakMishra/codemap/commit/91edff5543df784d7eb0db1b28cba70a5de396dc))

- **chunking**: Add hash and equality methods to Chunk class
  ([`cbcb8ae`](https://github.com/SarthakMishra/codemap/commit/cbcb8aeee28316f23b20e976d5dfac74cf0982d6))

- **chunking**: Replace syntax.py with tree_sitter.py and add reg..
  ([`fe0e3ad`](https://github.com/SarthakMishra/codemap/commit/fe0e3ad3b43808dd0fe8988cbe969312d4ef4982))

- **cli**: Add model availability checks and loading spinners
  ([`e9a0993`](https://github.com/SarthakMishra/codemap/commit/e9a099305f60800064b3c86f954ef2346685f78a))

- **cli**: Improve imports organization and formatting
  ([`ed91d08`](https://github.com/SarthakMishra/codemap/commit/ed91d08c83b71d157d20c1038afacf8ae809722c))

- **cli**: Refactor base branch selection logic
  ([`b148864`](https://github.com/SarthakMishra/codemap/commit/b148864b01f1be224c178d82fd68f29ca5971f32))

Refactor base branch selection to improve handling of interactive and non-interactive modes. Ensure
  base branch is determined correctly and handle cases where it cannot be automatically determined.

- **cli**: Refactor cli command generation
  ([`3d82d07`](https://github.com/SarthakMishra/codemap/commit/3d82d071334b7d6e91d0e03302425043a7477563))

- **cli**: Refactor cli module imports
  ([`ec29ea4`](https://github.com/SarthakMishra/codemap/commit/ec29ea4059e6776a5b8aaaf23a739ff3976b76e1))

The cli module imports were refactored to be more consistent and follow best practices.

- **cli**: Refactor cli module imports
  ([`a141d2c`](https://github.com/SarthakMishra/codemap/commit/a141d2c3ac463e7830a7c0921fd468ee502add59))

The cli_app module was removed and its contents were moved to cli module

- **cli**: Refactor commit command processing logic
  ([`bf9fea5`](https://github.com/SarthakMishra/codemap/commit/bf9fea504c80e985e550d9c7c6d33c0042ae93e6))

Reorganized the commit command processing logic to improve readability and maintainability. Updated
  the handling of commit chunks, including the processing of staged and unstaged changes. Improved
  error handling and messaging for large files.

- **cli**: Refactor commit message generation and add PR generation
  ([`c3dacdf`](https://github.com/SarthakMishra/codemap/commit/c3dacdf3e2219413e4e22e1e42688b01519bb2a7))

Refactor commit message generation to use new module structure and add pull request generation
  features.

- **cli**: Refactor daemon command implementation
  ([`0bdadd9`](https://github.com/SarthakMishra/codemap/commit/0bdadd9aa89a147e1c0e6ccfbf1b6e92afdcc08a))

This commit refactors the daemon command implementation to improve code organization and
  readability. It introduces a new structure for handling daemon commands, making it easier to add
  or modify commands in the future.

- **cli**: Refactor embedding config and cache initialization
  ([`2add263`](https://github.com/SarthakMishra/codemap/commit/2add2634f31a3a4b82c9aace27b5bbb02cde2ff4))

The embedding cache directory and configuration were refactored to improve code readability and
  maintainability. The changes include extracting the embedding model, dimensions, and batch size
  from the processor configuration.

- **cli**: Refactor PR command to improve branch handling and validation
  ([`4ec50ef`](https://github.com/SarthakMishra/codemap/commit/4ec50efc430a1a8d08d07433ea8f5d3db9d9f70f))

This commit refactors the PR command to enhance branch creation and validation. It introduces a more
  robust workflow strategy, improves interactive branch selection, and adds validation for workflow
  strategies.

- **cli**: Refactor PR creation and branch handling
  ([`b45d4dc`](https://github.com/SarthakMishra/codemap/commit/b45d4dc76ce8ba1c464bed2fe68e77e6a159736a))

Refactor PR creation and branch handling to improve interactive mode and template usage.

- **cli**: Remove cli_types module
  ([`a4ac581`](https://github.com/SarthakMishra/codemap/commit/a4ac581f429a7a336b0cb274e8584e64a8f98bda))

- **cli**: Rename generate command to gen
  ([`415ece3`](https://github.com/SarthakMishra/codemap/commit/415ece30516e0aa58092ff3e1dddca4f25ee9a90))

The generate command has been renamed to gen for brevity and consistency. This change affects the
  command line interface and related documentation.

- **cli**: Replace loading spinners with progress indicators
  ([`2f1eba4`](https://github.com/SarthakMishra/codemap/commit/2f1eba44df129d0b0e198a0cf79a63e6f35a5f72))

The commit replaces loading spinners with progress indicators in various CLI commands to improve
  user experience and provide more detailed feedback.

- **cli**: Simplify code and add comments
  ([`9f20bb5`](https://github.com/SarthakMishra/codemap/commit/9f20bb51d1a52bc82ec32320f473aeb842f6d673))

- **cli**: Simplify imports and handle errors
  ([`e6414bd`](https://github.com/SarthakMishra/codemap/commit/e6414bd3a9c663ebde091b2f3111ad5594aaecf8))

- **cli**: Simplify loading spinner and validate repo path imports
  ([`b27de57`](https://github.com/SarthakMishra/codemap/commit/b27de57ff5620f12033f62a463df1cdc788410f9))

- **cli**: Simplify message generator setup
  ([`6db1236`](https://github.com/SarthakMishra/codemap/commit/6db12368d4e1e9fc02a85048129da0c7b3e6db62))

- **cli**: Update error handling and warnings
  ([`6a6fd8f`](https://github.com/SarthakMishra/codemap/commit/6a6fd8f4c7cd95731eda71636bceeca318ca5c52))

Improved error messages and warnings for better user experience. Centralized error handling and
  added more informative messages.

- **cli**: Update type checking for commit command
  ([`a3616a0`](https://github.com/SarthakMishra/codemap/commit/a3616a0bd3cf940183066fb06f5865775c4f6b32))

- **cli_app**: Update daemon command import and registration
  ([`9c7d48c`](https://github.com/SarthakMishra/codemap/commit/9c7d48c04b446041adb4fe0458114ab029444602))

The daemon command has been refactored to use the add_typer method for registration, and its import
  has been updated to reflect the change.

- **cli_app**: Update dotenv loading logic
  ([`663bf85`](https://github.com/SarthakMishra/codemap/commit/663bf85ba575646e39038254ac0f607bf6587fb7))

- **codemap**: Extract embedding config and cache dir creation
  ([`6eaf84b`](https://github.com/SarthakMishra/codemap/commit/6eaf84b4634c8a9f83d211acc93c14f39c10fea5))

The code changes extract the creation of embedding cache directory and embedding configuration into
  separate functions for better modularity and reusability.

- **codemap**: Extract embedding config and cache dir creation
  ([`40e1d24`](https://github.com/SarthakMishra/codemap/commit/40e1d249f763d56e3c13658a23b1f58284340c7d))

The creation of embedding cache directory and configuration were previously scattered across
  multiple functions. This change consolidates these into a single location to improve
  maintainability and readability.

- **codemap**: Extract embedding config and cache dir creation
  ([`4059a68`](https://github.com/SarthakMishra/codemap/commit/4059a68263e3f31a26c522163b1522eb8e4006ec))

The code for creating the embedding cache directory and configuring the embedding settings was
  duplicated across multiple functions. This change extracts these into separate functions to
  improve maintainability and reduce duplication.

- **codemap**: Improve code readability with consistent formatting
  ([`cbd98fc`](https://github.com/SarthakMishra/codemap/commit/cbd98fc7823f211d048121e532cd6eb09976a1c3))

- **codemap**: Improve mermaid diagram generation and styling
  ([`13fbf99`](https://github.com/SarthakMishra/codemap/commit/13fbf996e59c28537ceb02fa11f02aadb482f00d))

Refactor the CodeMapGenerator class to improve the generation of mermaid diagrams. This includes
  enhancing the rendering logic, adding support for subgraphs, and improving the styling of nodes
  and edges.

- **codemap**: Move GitMetadata to analysis module
  ([`597c0c9`](https://github.com/SarthakMishra/codemap/commit/597c0c913efd0094a1fdd1288620c646085fc55f))

- **codemap**: Refactor CodeMapGenerator for improved mermaid diagram rendering
  ([`a02ed37`](https://github.com/SarthakMishra/codemap/commit/a02ed3745cfc43df98d3b5ea0e4cc3b2b8215d1d))

The CodeMapGenerator class has been refactored to improve the rendering of mermaid diagrams. This
  includes changes to the way nodes and edges are processed, as well as the addition of new features
  such as the ability to filter out unconnected nodes and subgraphs.

- **codemap**: Refactor embedding config and cache initialization
  ([`0817257`](https://github.com/SarthakMishra/codemap/commit/08172577981cd9430e4c2e01ad569fd518e8ebae))

The commit refactors the embedding configuration and cache initialization across multiple functions.
  It introduces a more consistent approach to setting up the embedding model, dimensions, and batch
  size. The changes aim to improve code readability and maintainability by reducing duplication.

- **codemap**: Refactor syntax handlers to extract function calls
  ([`b18bea5`](https://github.com/SarthakMishra/codemap/commit/b18bea5e45b39e723cdd15e281fcf9c7dd29e286))

The commit refactors the syntax handlers for JavaScript and Python to extract function calls within
  a node's scope. This change aims to improve code readability and maintainability by providing a
  more efficient way of identifying function calls.

- **codemap**: Remove base language config module fix is not su..
  ([`4b0bbc7`](https://github.com/SarthakMishra/codemap/commit/4b0bbc7a207671b25de2e9db18db66d9d0d2533a))

- **codemap**: Simplify model loading and improve error handling
  ([`527ed47`](https://github.com/SarthakMishra/codemap/commit/527ed47e59b7a184c4b97544b398b3995a3e827d))

- **codemap**: Simplify regex patterns
  ([`1549857`](https://github.com/SarthakMishra/codemap/commit/154985711ba6cee20c9ab41f08837b217db33894))

- **codemap**: Update CLI entry and config settings
  ([`eede626`](https://github.com/SarthakMishra/codemap/commit/eede62625312a113c6362a207e300fcd48106357))

- **codemap**: Update gitignore pattern matching logic
  ([`e6217c3`](https://github.com/SarthakMishra/codemap/commit/e6217c39e3636b787f2f0ee9ef8c2d512d2e94f3))

- **codemap**: Update import paths for consistency
  ([`4dddb8e`](https://github.com/SarthakMishra/codemap/commit/4dddb8e8d4c78c788edc3dc937a3fbec4b50519d))

- **codemap**: Update Mermaid diagram generation
  ([`7f44bee`](https://github.com/SarthakMishra/codemap/commit/7f44bee52ccb5b44bfde7e36c212d311fcee5334))

Improve performance and structure of Mermaid diagram generation

* Enhance node and edge definitions * Add support for subgraphs and nested modules * Optimize
  rendering and filtering of nodes and edges

BREAKING CHANGE: new Mermaid diagram layout and styling

- **codemap.commit.command**: Update default model to gpt-4o-mini
  ([`0f7960c`](https://github.com/SarthakMishra/codemap/commit/0f7960c7e24fe129d783d3dcd65f0bdfa1589a3e))

- **codemap.git.commit**: Move commit feature to git module
  ([`05d4db5`](https://github.com/SarthakMishra/codemap/commit/05d4db5d40c73316d7827afc58fc11164cbb9a54))

- **codemap/cli/commit**: Improve handling of staged and untracked files
  ([`b5bf1e0`](https://github.com/SarthakMishra/codemap/commit/b5bf1e0a84a5c35bd0bd5ec90497a521f6cf7c48))

- **codemap/cli/commit**: Update import paths for consistency
  ([`76d2d2f`](https://github.com/SarthakMishra/codemap/commit/76d2d2f5a2972105b22c4a86fa763d7251a5c207))

- **codemap/cli_entry**: Simplify commit command imports
  ([`af4c79c`](https://github.com/SarthakMishra/codemap/commit/af4c79c33bcef403e6a246e9ca5b352d6902b547))

- **codemap/commit/diff_splitter.py**: Enhance semantic diff splitting with code structure analysis
  ([`9ba4fcb`](https://github.com/SarthakMishra/codemap/commit/9ba4fcb49bc14dba601a5c4752267d906a747698))

- **codemap/commit/interactive**: Migrate to questionary for interactive prompts
  ([`a5d3aaf`](https://github.com/SarthakMishra/codemap/commit/a5d3aaf4a5fb37a24d3408bff1f808c1a7758fd3))

- **codemap/commit/message_generator**: Update default model to gpt-4o-mini
  ([`0a214d2`](https://github.com/SarthakMishra/codemap/commit/0a214d259109f7f4862b1a6656b056a2f85abbde))

- **codemap/git**: Add ignore_hooks param to commit_only_specified_files
  ([`4716b48`](https://github.com/SarthakMishra/codemap/commit/4716b487af25f0536aa998d30b3912821cdeaa5a))

- **codemap/git/diff_splitter**: Update embedding model and improve large file handling
  ([`0910d48`](https://github.com/SarthakMishra/codemap/commit/0910d48a32c6b401c89f96117094efb127082c71))

The embedding model has been updated from 'Qodo/Qodo-Embed-1-1.5B' to
  'sarthak1/Qodo-Embed-M-1-1.5B-M2V-Distilled' to leverage improved performance and accuracy.
  Additionally, large file handling has been enhanced to prevent API payload size issues by skipping
  files exceeding size limits and providing appropriate warnings.

- **commit**: Add option to bypass git hooks
  ([`b69758f`](https://github.com/SarthakMishra/codemap/commit/b69758ff98121abab67395a01fb13288d5e4726a))

- **commit**: Ensure selective staging for chunk commits
  ([`20558d8`](https://github.com/SarthakMishra/codemap/commit/20558d869fa7afce2437aa40cde6df3d13b4a135))

- **commit**: Extract commit logic into separate method
  ([`00d6c7b`](https://github.com/SarthakMishra/codemap/commit/00d6c7bd871cec75b5c157c150a62c9537833ec5))

- **commit**: Handle exit in confirm_abort
  ([`7622e28`](https://github.com/SarthakMishra/codemap/commit/7622e28cdbd83dd745cbc0cf55f90bba702b178e))

- **commit**: Improve change analysis and processing flow
  ([`089e571`](https://github.com/SarthakMishra/codemap/commit/089e571d54f32e26d68964b0339f85d7d2f8a69a))

- **commit**: Improve commit message generation performance
  ([`cad64ca`](https://github.com/SarthakMishra/codemap/commit/cad64ca570028247db457daa5bc941f6730c1194))

- **commit**: Refactor commit workflow for better maintainability
  ([`5a67fe0`](https://github.com/SarthakMishra/codemap/commit/5a67fe0f5eb57125d705ac867b65a472a913d3ce))

Refactor the commit command to improve code organization and readability. This includes splitting
  the commit generation and processing into more manageable functions, and improving error handling.

- **commit**: Update DiffChunk to DiffChunkData conversion
  ([`3ca7f01`](https://github.com/SarthakMishra/codemap/commit/3ca7f018d7748ea1affc35d95923e4048724baa3))

- **commit/command**: Simplify git add and unstage logic
  ([`b393523`](https://github.com/SarthakMishra/codemap/commit/b39352344d41fb3311b6f216fe0a8fdd2d89e4ab))

- **commit_cmd**: Handle git errors and add bypass hooks option
  ([`f3d077f`](https://github.com/SarthakMishra/codemap/commit/f3d077ff2c06868347145efc1b2144333f06d14c))

- **commit_cmd**: Simplify command execution logic
  ([`b25cccf`](https://github.com/SarthakMishra/codemap/commit/b25cccfbaa5c5ce487e1f1f2eac24ea15634af14))

- **commit_cmd**: Simplify commit logic and add error handling
  ([`b01c8b3`](https://github.com/SarthakMishra/codemap/commit/b01c8b3e0e8ca26df74183e16203f78b562b284d))

- **commit_cmd**: Update DiffChunk to DiffChunkData conversion
  ([`4b0d94a`](https://github.com/SarthakMishra/codemap/commit/4b0d94a18091131d681d8c1706db6fab3c71eab0))

- **commit_cmd**: Update function documentation Update the docu..
  ([`417c41c`](https://github.com/SarthakMishra/codemap/commit/417c41c470aef1eb31589c867b432e7750de5bfa))

- **commit_linter**: Move default config values to central config file
  ([`06482ad`](https://github.com/SarthakMishra/codemap/commit/06482ada36171ab0440917fa19fce692940ffdbd))

Rather than hardcoding default values in the commit_linter/config.py file, they are now loaded from
  the central config.py file via ConfigLoader. This change allows for easier maintenance and updates
  of default values across the project.

- **commit_linter**: Update commit linter configuration
  ([`e6da2da`](https://github.com/SarthakMishra/codemap/commit/e6da2da05d32bdb964b08b7aedf05fb2bb9b22c2))

Updated the commit linter configuration to include new rules and improve existing ones.

- **config**: Extract embedding config to separate variable
  ([`29f0814`](https://github.com/SarthakMishra/codemap/commit/29f081493d065ef66fb0ab3e75e50fa1953a1807))

Extracted embedding configuration to a separate variable for better readability and maintainability.

- **config**: Simplify config loading and validation
  ([`46b1ee7`](https://github.com/SarthakMishra/codemap/commit/46b1ee726ff6aaba2be849491f17e07f3a59429a))

- **config_loader**: Refactor config loader for better structure and readability
  ([`8d9a474`](https://github.com/SarthakMishra/codemap/commit/8d9a474f84fb19c619228134963c4ad831261f89))

This commit refactors the ConfigLoader class to improve code organization and readability. Changes
  include splitting long methods into smaller ones, improving variable names, and adding type hints.

- **diff_splitter**: Handle deleted tracked files in git status
  ([`8e8be1b`](https://github.com/SarthakMishra/codemap/commit/8e8be1bbf2d80aee933deb9ca11151e07685a2aa))

- **diff_splitter**: Improve model loading with rich progress bar
  ([`fd26505`](https://github.com/SarthakMishra/codemap/commit/fd26505f845bf825e775cb92ff3e770f24c82699))

- **diff_splitter**: Skip file existence checks in test environments
  ([`13bae8b`](https://github.com/SarthakMishra/codemap/commit/13bae8b8bb75a7416d72b6447b3f1e7fc088a329))

- **diff_splitter**: Update embedding model and improve large file handling
  ([`b7823fb`](https://github.com/SarthakMishra/codemap/commit/b7823fba1d8ac6be9eae99834ab50db374b6ce18))

The commit updates the embedding model to 'sarthak1/Qodo-Embed-M-1-1.5B-M2V-Distilled' and enhances
  large file handling by skipping them during analysis and providing a warning.

- **embedding**: Rename model_used to model
  ([`d7ea12f`](https://github.com/SarthakMishra/codemap/commit/d7ea12f178f928a1f3bff3c0418fd703374adb33))

- **git**: Improve handling of deleted files in git status
  ([`eb28f81`](https://github.com/SarthakMishra/codemap/commit/eb28f816c573418a077b50139e673af7b7dc026c))

The current implementation of get_deleted_tracked_files and filter_valid_files functions have been
  refactored to improve performance and readability. The changes include simplifying the logic for
  parsing git status output and checking file existence.

- **git**: Refactor git utilities and remove unused code
  ([`3cabecf`](https://github.com/SarthakMishra/codemap/commit/3cabecf66a97c87bcfca5061760939b51a2a984e))

This commit refactors the git utilities to improve code organization and remove unused functions. It
  also simplifies the import structure and reduces redundancy.

- **git**: Simplify module imports and structure
  ([`7e23d4f`](https://github.com/SarthakMishra/codemap/commit/7e23d4f4902f22879f0d791a6ae77a4abdbfef13))

- **git_utils**: Enhance staging logic for existing and deleted files
  ([`5350990`](https://github.com/SarthakMishra/codemap/commit/535099076fa9e44debc5f3e4c76672292d0b6671))

- **git_utils**: Handle already staged deletions in git status
  ([`4900a9c`](https://github.com/SarthakMishra/codemap/commit/4900a9cb494985db9d1746014f197dc542261202))

- **git_utils**: Handle deleted tracked files in git status
  ([`0cb67ec`](https://github.com/SarthakMishra/codemap/commit/0cb67ec25d2cce8ea48419f7fd4b40511a428fb3))

- **git_utils**: Improve staging logic and error handling
  ([`b70a9c0`](https://github.com/SarthakMishra/codemap/commit/b70a9c01c89595fa15c2144408018ad621b364d3))

- **git_utils**: Simplify file staging and commit logic
  ([`82937ee`](https://github.com/SarthakMishra/codemap/commit/82937eed4de2c85b3fbf1e922427b2cf2fbc2c4f))

- **git_utils**: Skip file existence checks in test environments
  ([`0396192`](https://github.com/SarthakMishra/codemap/commit/03961920eaa68b8373e6da116b5165a51a331524))

- **git_utils**: Update git command handling for robustness
  ([`6dc50ad`](https://github.com/SarthakMishra/codemap/commit/6dc50ad5d2d4d4006d7360fbe3baa9536db25b27))

Improve error handling in run_git_command by adding a check parameter to control whether exceptions
  are raised. Enhance logging for better debugging, especially for expected failures in merge-base
  --is-ancestor checks.

- **git_utils**: Update type hints and security flags
  ([`36344ff`](https://github.com/SarthakMishra/codemap/commit/36344ff7a5a667a9ab2b0c93d05e90bf08f6ce42))

- **languages**: Update type hints and docstrings
  ([`71711a2`](https://github.com/SarthakMishra/codemap/commit/71711a23ced30ce383faa7c6417b2603940bccaf))

- **llm_utils**: Simplify DiffChunkData creation
  ([`9a1dded`](https://github.com/SarthakMishra/codemap/commit/9a1dded15abd734a8a190097914ad416cab0111d))

- **llm_utils**: Simplify imports and handle LLM errors The com..
  ([`4cdf599`](https://github.com/SarthakMishra/codemap/commit/4cdf599ad4162df258b73b97eac935c589fb5bee))

- **lod-generator**: Update LOD generation logic for better performance and readability
  ([`12a913c`](https://github.com/SarthakMishra/codemap/commit/12a913cd2715176dc1346d71bce537bb1d8b9f02))

Refactored LOD generation to improve performance and readability. Changes include optimizing the
  conversion of tree-sitter analysis to LOD format, enhancing the handling of entity metadata, and
  improving the processing pipeline for better error handling and logging.

- **log**: Change log levels to improve readability
  ([`fe6bb4f`](https://github.com/SarthakMishra/codemap/commit/fe6bb4ff72f972c16bfd469e8f500fc4eebd8596))

- **logging**: Adjust log levels for verbosity
  ([`ee9ec05`](https://github.com/SarthakMishra/codemap/commit/ee9ec05ecb744dfe1217e27c824082abf47c34b4))

- **logging**: Remove redundant rich logging imports
  ([`91f94c2`](https://github.com/SarthakMishra/codemap/commit/91f94c256a22a7310158b83f0bdfa0528a04ad9d))

- **logging**: Remove rich logging import
  ([`55925d8`](https://github.com/SarthakMishra/codemap/commit/55925d8c47066c72c5c656e7c07d4607e3e135af))

- **lsp**: Improve docstrings and comments
  ([`4d9d653`](https://github.com/SarthakMishra/codemap/commit/4d9d65374dc39a5bf46065c65d0eb736fb0febfd))

- **markdown_generator**: Simplify gitignore pattern matching logic
  ([`dd5e386`](https://github.com/SarthakMishra/codemap/commit/dd5e386773086a3049b76340b44a48a12e00c73e))

- **message_generator**: Adapt chunk access for TypedDict
  ([`4153e86`](https://github.com/SarthakMishra/codemap/commit/4153e86440a60b71789dca2bb15794c03068a3a0))

- **message_generator**: Improve LLM API error handling
  ([`5044937`](https://github.com/SarthakMishra/codemap/commit/5044937e6d4f7639f23a8e146881b285a7d7d26f))

- **message_generator**: Update commit message generation to follow conventional commits
  specification
  ([`bf14e7d`](https://github.com/SarthakMishra/codemap/commit/bf14e7ddfd535dbe7ad12ad595b9f9839773a672))

The commit message generation has been updated to strictly follow the conventional commits
  specification. This includes adhering to the format of <type>[optional scope]: <description> and
  ensuring that the description is a concise, imperative present tense summary of the specific code
  changes.

BREAKING CHANGE: The commit message format has been updated to ensure consistency and adherence to
  the conventional commits specification.

- **message_generator**: Update LLM API call to use JSON response format and add error handling
  ([`284a108`](https://github.com/SarthakMishra/codemap/commit/284a108b355d01f7b29b418eb81679ed720ed309))

This commit refactors the LLM API call to use a JSON response format and adds error handling for
  cases where the model does not support JSON format or parsing fails.

BREAKING CHANGE: This change may break existing functionality if the LLM API response format has
  changed

- **model**: Update default model to gpt-4o-mini
  ([`8bf3480`](https://github.com/SarthakMishra/codemap/commit/8bf348024d5be3170aa552df78020dae8220a14e))

- **pipeline**: Update pipeline with embedding and storage integr..
  ([`477c482`](https://github.com/SarthakMishra/codemap/commit/477c482a1ae5c376d248ee950ec21acf3d93a2d6))

- **pr**: Improve interactive base branch selection and pr creation
  ([`549254a`](https://github.com/SarthakMishra/codemap/commit/549254a0b7473d78853654812a88e03cf10198c2))

- **pr_cmd**: Improve default branch handling in PR creation
  ([`6548939`](https://github.com/SarthakMishra/codemap/commit/6548939020ec960361d0a35fc1ce99274b38ca05))

The previous implementation assumed the default branch always exists. This change checks if the
  default branch exists in the repository before adding it to the list of choices. If it doesn't
  exist, it uses the first available branch as a fallback.

- **pr_cmd**: Improve default branch selection logic
  ([`b3344f0`](https://github.com/SarthakMishra/codemap/commit/b3344f04f9cee20ed17b18022d6bb2e87287d8db))

The previous logic for selecting a default branch was improved to first check if the config default
  branch exists, then the git default branch, and finally fallback to the first available branch.
  The logic for adding the default branch to the list of choices was also updated to only add it if
  it exists.

- **pr_cmd**: Refactor PR creation logic for better error handling and base branch determination
  ([`523f499`](https://github.com/SarthakMishra/codemap/commit/523f499f9754e81e021a37716dbcf3598336ede1))

Moved interactive base branch selection inside the main try block. Improved error handling for Git
  errors during PR creation. Ensured base branch is determined before PR creation.

- **pr_cmd**: Reorder branch creation and commit handling
  ([`c64102b`](https://github.com/SarthakMishra/codemap/commit/c64102b099246ba85ee17d6066bebe423a7545fe))

Reorder the steps in the pr_command function to handle branch creation/selection first, then handle
  commits if needed, and finally handle push. This change improves the logical flow of the function
  and makes it easier to understand.

- **pr_cmd**: Use config default branch if set, fallback to git default
  ([`ab94395`](https://github.com/SarthakMishra/codemap/commit/ab94395843eeb695fd659cb2ccafe27a1b3aa684))

Previously, the default branch was solely determined by git. Now, if a default branch is set in the
  config, it will be used. Otherwise, it falls back to the git default branch.

- **pr_utils**: Improve branch name suggestion logic
  ([`77b2d2b`](https://github.com/SarthakMishra/codemap/commit/77b2d2b39535ba44f4a1be101c3795dc157e00d3))

- **pr_utils**: Improve branch relation checks in get_branch_relation
  ([`2a1b4a6`](https://github.com/SarthakMishra/codemap/commit/2a1b4a6f01f5ca3b58ebbc2847199d1f45f9a882))

The get_branch_relation function was improved by adding additional checks to determine the
  relationship between two branches. This was achieved by trying both forward and reverse checks
  using git merge-base --is-ancestor to accurately determine if one branch is an ancestor of
  another.

- **pr_utils**: Simplify LLM prompt formatting
  ([`33abc7d`](https://github.com/SarthakMishra/codemap/commit/33abc7dc05869a746f457f981e4e89da0fbe6594))

- **processor**: Extract embedding config and cache dir creation
  ([`afbecb3`](https://github.com/SarthakMishra/codemap/commit/afbecb339268c737274be894d8d261e797b7d583))

The creation of embedding cache directory and embedding config was repeated in multiple functions.
  This change extracts them into separate lines for better maintainability and readability.

- **processor**: Extract embedding config and cache dir creation
  ([`f730bd6`](https://github.com/SarthakMishra/codemap/commit/f730bd616ac41a24f02c86dd18a552fd5853af11))

Moved embedding config creation and cache dir creation to separate lines for better readability and
  maintainability.

- **processor**: Remove git analysis and related code
  ([`75eb48a`](https://github.com/SarthakMishra/codemap/commit/75eb48a3a33c5bfa677fac0d6d4552628eaf4ad5))

The git analysis and related code have been removed to simplify the data and improve performance.
  This change affects the processor module and its functionality.

- **processor**: Update FileWatcher import location
  ([`e9f1c04`](https://github.com/SarthakMishra/codemap/commit/e9f1c04f8091816ce4d12d3e04ef784f5953a616))

- **pyproject**: Update codemap script reference
  ([`8c54b3f`](https://github.com/SarthakMishra/codemap/commit/8c54b3f76efd366333f98e0edee6bc33cbdd220f))

- **src/codemap/cli/main.py**: Add custom typer group for pr commands
  ([`e40b5f7`](https://github.com/SarthakMishra/codemap/commit/e40b5f7bebc197b3789da9f5262d46cf79a70154))

- **storage**: Add custom JSON encoder for CodeMap types
  ([`c7d5388`](https://github.com/SarthakMishra/codemap/commit/c7d53889bc0c75d40135d977d57c62de07bc4456))

- **storage**: Update documentation and formatting
  ([`96b595a`](https://github.com/SarthakMishra/codemap/commit/96b595a1426648d11f761285d96853b3703c94f3))

- **Taskfile**: Simplify task definitions
  ([`714ee1e`](https://github.com/SarthakMishra/codemap/commit/714ee1e1764ea7e85decf927ff464a90ab60a345))

- **test**: Simplify get_all_chunks function
  ([`9762baf`](https://github.com/SarthakMishra/codemap/commit/9762baf9d32997b82b443f299cb05f2d96d4195a))

- **test**: Update commit handler mock
  ([`f43aba9`](https://github.com/SarthakMishra/codemap/commit/f43aba9a44f8c676b240a4889b28568dfeef22e9))

- **test**: Update import paths for consistency
  ([`45d7109`](https://github.com/SarthakMishra/codemap/commit/45d71098a2c244a27c4725610e084274b60c3027))

- **test**: Update import statement
  ([`a07cb67`](https://github.com/SarthakMishra/codemap/commit/a07cb67442ed3d0e06d352caaf131ac87c2564fe))

- **test**: Update message generator imports
  ([`c830cf3`](https://github.com/SarthakMishra/codemap/commit/c830cf35c6166b737b5ca77cd9a84cfaf322add3))

- **test**: Update OpenAI model to gpt-4o-mini
  ([`8b58c41`](https://github.com/SarthakMishra/codemap/commit/8b58c41f37324667ce09f234e18cc9f95b287bdd))

- **test**: Update test_git_utils.py to use modern Python features
  ([`af45447`](https://github.com/SarthakMishra/codemap/commit/af454478f94976a059f6d532da6f1181bc118da9))

- **test**: Update test_llm_utils to use generate_message_with_li..
  ([`7f78eba`](https://github.com/SarthakMishra/codemap/commit/7f78eba4192e87fa51fad1a7970fc1cf03443237))

- **tests**: Remove caplog from tests without DB connection
  ([`6cc8d56`](https://github.com/SarthakMishra/codemap/commit/6cc8d56fcec22ade7b912d25128825a41322d607))

- **tests**: Update cli app import in test_pr_command.py
  ([`fee0072`](https://github.com/SarthakMishra/codemap/commit/fee0072503fb37d9b381bef2e5879e65ee4badf4))

- **tests**: Update mock parser and file filter usage
  ([`29c1ef1`](https://github.com/SarthakMishra/codemap/commit/29c1ef16d43df971f54e9168c9c8fd8a3a3b4514))

- **tests**: Update test cases for commit and PR generation
  ([`d9190a8`](https://github.com/SarthakMishra/codemap/commit/d9190a8a3975e2521a34d0b45dbadc8bab5f6175))

This commit refactors test cases to improve code coverage and readability.

- **tests**: Update test command initialization and progress handling
  ([`042a941`](https://github.com/SarthakMishra/codemap/commit/042a941054770e60cc164fe50817c435b6d049c8))

Refactor test commands to use updated progress indicator and initialization handling. This change
  affects test commands in test_generate_cmd.py and test_init_cmd.py, making them more robust and
  consistent with current progress indicator implementation.

- **tests**: Update test utilities and imports
  ([`f1f3811`](https://github.com/SarthakMishra/codemap/commit/f1f381101bbadfed76a53701d608ac4ebe54f759))

This commit refactors test utilities and imports to improve code organization and maintainability.

- **utils**: Improve handling of deleted files in git status
  ([`d0d0fe2`](https://github.com/SarthakMishra/codemap/commit/d0d0fe2d82b76fb6bb13d8cbea797eb5043b998a))

The current implementation of get_deleted_tracked_files and filter_valid_files functions have been
  reviewed and improved for better handling of deleted files in git status. The changes include
  improved logging and handling of git errors.

- **utils**: Move git utils to git package
  ([`60192b7`](https://github.com/SarthakMishra/codemap/commit/60192b798b15cf9e9af4da5e909f27e6adae02f2))

- **utils**: Remove unused utility modules
  ([`880a1b3`](https://github.com/SarthakMishra/codemap/commit/880a1b348068072e141cb20142b7f326ff0b8706))

- **utils**: Simplify __init__ module
  ([`9a682d8`](https://github.com/SarthakMishra/codemap/commit/9a682d884659afee4caee3f3c663dfcb69d71713))

The __init__.py file in the src/codemap/utils directory has been simplified by removing unnecessary
  imports and exports. This change improves the module's clarity and maintainability.

- **utils**: Simplify __init__.py exports and imports
  ([`585388b`](https://github.com/SarthakMishra/codemap/commit/585388b9f36c4c739f7581ed9cf14c3647321cfa))

- **watcher**: Improve docstrings and formatting
  ([`c0e889f`](https://github.com/SarthakMishra/codemap/commit/c0e889feb71addc625711a6cadcfb3dcc9b88a66))

- **watcher**: Relocate filewatcher module
  ([`0f8cccc`](https://github.com/SarthakMishra/codemap/commit/0f8ccccd31516a54c4b3659470d5d5edce799d51))

### Testing

- Add comprehensive tests for PR utilities to improve coverage
  ([`59408d1`](https://github.com/SarthakMishra/codemap/commit/59408d19e83a746530a626ce29a70dfeee01a03e))

- Add test cases for commit linter and message generation The co..
  ([`29f09b1`](https://github.com/SarthakMishra/codemap/commit/29f09b150483618e4da07c894c990b5a8cad5f5c))

- Add test cases for stage_files with deleted files
  ([`558cbe2`](https://github.com/SarthakMishra/codemap/commit/558cbe21947d67ce8e4e60b558e39e5f60cd4a24))

- Add test for split_by_file_implementation function
  ([`e7a7979`](https://github.com/SarthakMishra/codemap/commit/e7a79795fb3bb298971d980d91ebb268fa9a2cc9))

- Improve markdown generator tests
  ([`44e4c9b`](https://github.com/SarthakMishra/codemap/commit/44e4c9b7d70a2b27da526420ecae162d30384bf4))

- Add .git directory to mock repo root

- Create actual test files instead of just mocking

- Add tests for excluded files

- Fix file sorting test to check Details section only

- Improve test coverage for file exclusions
  ([`d4807f2`](https://github.com/SarthakMishra/codemap/commit/d4807f2936288f491b68ba7b34acc2cb4b6c495a))

- Add tests for file and directory checkbox states

- Add tests for excluded files and directories

- Add more default exclude patterns to config

- Improve test assertions with better error messages

- Improve test isolation and coverage
  ([`8c20a7f`](https://github.com/SarthakMishra/codemap/commit/8c20a7f7ff4f1f3fc76683fd4e5b93f10d491ffe))

- Fix test_default_config_loading to use temporary directory

- Update markdown escaping tests to be more comprehensive

- Add test cases for headings and code blocks

- Ensure tests don't depend on project config

- Skip git-dependent tests in CI environment
  ([`f9be4d2`](https://github.com/SarthakMishra/codemap/commit/f9be4d26457400c89a16e8d4957562a8fb16bc6d))

- Update workflows to split linting from testing\n- Add SKIP_GIT_TESTS environment variable to skip
  git-dependent tests\n- Update git-related test files with skipif markers\n- Fix test failures in
  CI environment where git isn't available

- Update files in tests
  ([`2e15556`](https://github.com/SarthakMishra/codemap/commit/2e1555636db382bdbae2e60c050ec4743b069082))

- Update test imports and fixtures for better type safety
  ([`ad8b263`](https://github.com/SarthakMishra/codemap/commit/ad8b2631e8321eb659d56c800dadad466acc1cef))

- Update test_respect_output_dir_from_config to use custom Path implementations
  ([`41c2571`](https://github.com/SarthakMishra/codemap/commit/41c25715d166b45af85bedb915aa5e103009da6b))

- Update tests for ERD generation and tree-sitter changes
  ([`d363392`](https://github.com/SarthakMishra/codemap/commit/d363392f06418ef3c4b32bde2d22fa5d72fdf1eb))

- **cli**: Add comprehensive tests for commit and generate commands
  ([`4c242ec`](https://github.com/SarthakMishra/codemap/commit/4c242ec13f1e6a48531a2dc422fdfe7fcf6289e0))

- **cli_utils**: Add test for specific loggers set to ERROR level
  ([`9b144a7`](https://github.com/SarthakMishra/codemap/commit/9b144a7ec7cf914a1326937970b67f2543581fb7))

- **commit**: Add test cases for bypass_hooks integration
  ([`1c97fd6`](https://github.com/SarthakMishra/codemap/commit/1c97fd69dd821cdfc9fafaa2cbb221285b123b99))

- **commit_lint**: Update commit linting rules for case validation
  ([`e4689f9`](https://github.com/SarthakMishra/codemap/commit/e4689f99923cf630bbd674c5db4675a9ba5aa162))

Updated the commit linting rules to include validation for commit message case. The validation now
  checks if the subject is in a specified case format.

- **config**: Add tests for commit hooks configuration
  ([`3b472b2`](https://github.com/SarthakMishra/codemap/commit/3b472b2312af9e0197be540877d001fa2a03c4f2))

- **init_cmd**: Add test fixtures and improve test coverage
  ([`36ad208`](https://github.com/SarthakMishra/codemap/commit/36ad208110ff86611421f9b73a0b7973f206cdc6))

This commit adds new test fixtures and improves test coverage for the init command.

- **lsp**: Add LSP analyzer tests
  ([`083c2e9`](https://github.com/SarthakMishra/codemap/commit/083c2e90c9d591a2f3e2f4ff1e056ed23ada68c3))

- **new**: Add tests for regexp, syntax chunkers
  ([`0aee56b`](https://github.com/SarthakMishra/codemap/commit/0aee56b574feb14d16f64c7d1a6fb9aebb3d3b44))

- **pr_command**: Add unit tests for PR creation and update
  ([`ca72b47`](https://github.com/SarthakMishra/codemap/commit/ca72b472e269c488d32aa4b8846687075f991157))

This commit introduces new test cases for the PR creation and update functionality. The tests cover
  various scenarios, including user interactions, PR generation, and description formatting.

- **pr_command**: Add unit tests for PR creation and update
  ([`ee9ddc8`](https://github.com/SarthakMishra/codemap/commit/ee9ddc840c332b08fba60edc4e1a34c40701110c))

This commit adds several unit tests to ensure the PR creation and update functionality works as
  expected. The tests cover various scenarios, including PR creation with branch selection, editing
  title and description, regenerating description with LLM, and error handling during PR creation.

- **pr_command**: Skip PR command integration tests
  ([`24eeb08`](https://github.com/SarthakMishra/codemap/commit/24eeb0841d765f92a7b5a346a62341fe8f8e1ca9))

These tests require a deeper mocking strategy to accurately test the PR command functionality.

- **pr_command**: Update test cases for branch and PR creation
  ([`9243b06`](https://github.com/SarthakMishra/codemap/commit/9243b06b6a028506a3832676ebc9bc57423a314b))

This commit updates the test cases in test_pr_command.py to better reflect the current functionality
  of branch and PR creation. It includes changes to test handle_branch_creation and
  handle_pr_creation functions, ensuring that the create_branch, checkout_branch, and create_pr
  functions are called as expected.

- **pr_command**: Update test cases for handling commits and PR generation
  ([`87617bc`](https://github.com/SarthakMishra/codemap/commit/87617bca70a8b2f4dd9e2584ed17ca2fe0b64ff3))

This commit updates the test cases for handling commits and PR generation to reflect changes in the
  code. The test cases now account for new functionality and edge cases.

- **pr_generator**: Update test cases for PR generator
  ([`fbbf77b`](https://github.com/SarthakMishra/codemap/commit/fbbf77b6ba084148265cdfcbcde13ba48d624131))

This commit updates the test cases for the PR generator to ensure it handles various scenarios
  correctly.

- **syntax_chunker**: Add tests for syntax-based code chunking
  ([`cdc10d7`](https://github.com/SarthakMishra/codemap/commit/cdc10d76e1967b67fb23e06ced685254d60388fc))

- **syntax_chunker**: Remove obsolete test file
  ([`b233415`](https://github.com/SarthakMishra/codemap/commit/b2334155adefaec056dc8424c730428e1c66cb00))

- **tests**: Add test coverage plan document and initial tests
  ([`7c02101`](https://github.com/SarthakMishra/codemap/commit/7c02101510f2bbd41f647aa96bba7c1947bf9a19))

- **tests**: Fix diff checks for commit message generation Here i..
  ([`2ec2dd3`](https://github.com/SarthakMishra/codemap/commit/2ec2dd36aa554621dfea79927333da3dde18d3fe))

- **unit**: Add markers to tests
  ([`9bfff64`](https://github.com/SarthakMishra/codemap/commit/9bfff64698a714fcebae8816ce2fe9555fe62fcb))

- **unit**: Add test for git commit hook error handling
  ([`ce843ac`](https://github.com/SarthakMishra/codemap/commit/ce843ac2f1deb89f4786470c181d624c5141364e))

- **unit**: Add tests for commit interactive UI
  ([`87baf3a`](https://github.com/SarthakMishra/codemap/commit/87baf3a8c5deeeb6904ee227fd72b43fc0c465f2))

- **unit**: Add tests for commit workflow logic
  ([`7c9d8e9`](https://github.com/SarthakMishra/codemap/commit/7c9d8e9618593941aa80590aebae89865c5adc86))

- **unit**: Add tests for file filters utility
  ([`408580a`](https://github.com/SarthakMishra/codemap/commit/408580ae66e0eada40e03f453d73aa9e28500160))

- **unit**: Add tests for Git metadata analyzer
  ([`f793ba8`](https://github.com/SarthakMishra/codemap/commit/f793ba86b1ccc64a97b2ef40da8dd64adc182ea4))

- **unit**: Add tests for message generator edge cases
  ([`277521d`](https://github.com/SarthakMishra/codemap/commit/277521d32d3317417e9c9408ec18106170c443a4))

- **unit**: Add unit tests for CLI initialization and commands
  ([`6868765`](https://github.com/SarthakMishra/codemap/commit/68687650ade18bc295c8f3ff3d6b55f82dd98209))

- **unit**: Update test case for abort action
  ([`cc441ed`](https://github.com/SarthakMishra/codemap/commit/cc441ed003eaabb3a9d1533fae04d49311101d46))

- **unit**: Update tests for split diff and mock git commands
  ([`ff5d02e`](https://github.com/SarthakMishra/codemap/commit/ff5d02e6da3affb370ec22ac7fffb3e36fee86d7))

- **unit): add path sensitive marks to tests fix(test**: Update asse..
  ([`15583f1`](https://github.com/SarthakMishra/codemap/commit/15583f10af6971516814545bb34cbec3922fdcfa))

- **utils**: Add unit tests for CLI and Git utilities
  ([`df75a8f`](https://github.com/SarthakMishra/codemap/commit/df75a8f24be0dcf26cae7f7ec16da508ecb6d572))

- **utils**: Add unit tests for determine_output_path function
  ([`7bb893c`](https://github.com/SarthakMishra/codemap/commit/7bb893c12baae1e699c6b9249fcccceefaa777ff))

- **utils**: Update git utils tests for better coverage
  ([`50d1a16`](https://github.com/SarthakMishra/codemap/commit/50d1a16c3ffa87f81775e490f5790c17548ff76c))

- **utils**: Update test cases and fixtures
  ([`797e28a`](https://github.com/SarthakMishra/codemap/commit/797e28ab25b9ccfb7433f6ca338962e6e4802d80))

- **watcher**: Add tests for file watcher module
  ([`526edfe`](https://github.com/SarthakMishra/codemap/commit/526edfe9c2959b69749c6ceaff8ccfe3029ab2e0))
