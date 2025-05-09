# CHANGELOG


## v0.3.0-rc.4 (2025-05-09)

### Bug Fixes

- Update codemap version
  ([`78a1299`](https://github.com/SarthakMishra/codemap/commit/78a1299cdf5ce95f415c27a807e710bdf91b06d3))

### Chores

- **release**: Remove pull_request trigger from release workflow
  ([`9259e8d`](https://github.com/SarthakMishra/codemap/commit/9259e8df4a73b69cb494acc38f91b4b5b961c27d))

### Refactoring

- **cli_utils**: Improve progress indicator management
  ([`1d12615`](https://github.com/SarthakMishra/codemap/commit/1d12615349fd55cf7de599492a9074128a1a1d85))

Enhance SpinnerState to track and display nested spinners and improve progress bar handling

- **codemap**: Remove progress updates
  ([`fbbaab8`](https://github.com/SarthakMishra/codemap/commit/fbbaab8f6fbb25f65d72bc876da610d1b3102efd))

- **docker_utils**: Update storage paths
  ([`5b932c7`](https://github.com/SarthakMishra/codemap/commit/5b932c75903494c4c8072bfaa7f395fcef32f039))

Change storage paths for qdrant and postgres

### Testing

- **tests**: Refactor progress indicator and spinner utilities
  ([`a1e4dc9`](https://github.com/SarthakMishra/codemap/commit/a1e4dc96456090bc0ca46888e41612890b5e194d))

Refactor progress indicator and spinner utilities for better isolation and testing


## v0.3.0-rc.3 (2025-05-09)

### Chores

- Update codemap version
  ([`10f6e2b`](https://github.com/SarthakMishra/codemap/commit/10f6e2bf865e5b8931aa3fe3607d715b690985ca))

- Update codemap version
  ([`c5079cb`](https://github.com/SarthakMishra/codemap/commit/c5079cb7696750e9ea34e05d6abd23ae15c6042c))

- Update docs config
  ([`9ef24bf`](https://github.com/SarthakMishra/codemap/commit/9ef24bfeb702902c997bf622aa4f0d26e74cb6b9))

- Update docs config
  ([`cfa4e0e`](https://github.com/SarthakMishra/codemap/commit/cfa4e0ea6d194543fc6d5f7f0e854f6701d45bc5))

### Continuous Integration

- **release**: Update workflow to trigger on specific paths for main and dev branches
  ([`12e6c16`](https://github.com/SarthakMishra/codemap/commit/12e6c160fe1dcf6d2d0a355e5cd5b7bd18062df5))

### Documentation

- **api**: Update config docs
  ([`11676a3`](https://github.com/SarthakMishra/codemap/commit/11676a3057535f954faff3c78e8ef5ce14a32b76))

- **api**: Update config docs
  ([`52c8a3f`](https://github.com/SarthakMishra/codemap/commit/52c8a3fd2b865dc35125cabfa79435c74c3d0347))

- **api**: Update index and config docs
  ([`1188652`](https://github.com/SarthakMishra/codemap/commit/11886529e5e3ed84aa37ad90ddbecf7af5051269))

- **api**: Update index and config docs
  ([`7d5ea20`](https://github.com/SarthakMishra/codemap/commit/7d5ea20e8606d8677cfd080381b2c970801f8839))

- **llm/rag**: Update documentation
  ([`dc0a0bc`](https://github.com/SarthakMishra/codemap/commit/dc0a0bc45db7e973f6552b280ba0d6f2aebf2b64))

- **llm/rag**: Update documentation
  ([`83e4c72`](https://github.com/SarthakMishra/codemap/commit/83e4c72076a5542879c280b7982c4eb8df6f3882))

### Features

- **cli**: Add config command
  ([`12ef1b4`](https://github.com/SarthakMishra/codemap/commit/12ef1b44ad1b119aca56c025d66f3bc6add6a9f5))


## v0.3.0-rc.2 (2025-05-09)

### Bug Fixes

- Downgrade log level for parser load failures
  ([`ae231fe`](https://github.com/SarthakMishra/codemap/commit/ae231fe63a256cd06c0eb67398a67412b86933db))

change logger warnings to debug for cases where parsers fail to load or languages cannot be
  determined

- Update distance threshold
  ([`40f4c4d`](https://github.com/SarthakMishra/codemap/commit/40f4c4df5dff843f9ce19b9a2f775417772fc328))

- Update llama model
  ([`f13d721`](https://github.com/SarthakMishra/codemap/commit/f13d721231ba1f9444d277b2272ed0495e4640c8))

- Update prompts for commit and PR generation
  ([`b82b41c`](https://github.com/SarthakMishra/codemap/commit/b82b41cb62bf7fd2f53c328dcbdae3a3f8465323))

Added system prompts for commit and PR generation to improve consistency and clarity

- **codemap**: Add move operation context and flag
  ([`7c1136b`](https://github.com/SarthakMishra/codemap/commit/7c1136b64e11cee52053476450ceb64592280e54))

Added a new context for move operations in commit_generator and a flag in diff_splitter to indicate
  file moves

- **command**: Remove await from llm client completion
  ([`2f3b89b`](https://github.com/SarthakMishra/codemap/commit/2f3b89b565e72af53ffe19758fad9292675a1274))

- **llm**: Add LLM API and client functionality
  ([`ff215dc`](https://github.com/SarthakMishra/codemap/commit/ff215dcd517e0bf99bff39a1f872f274855253ee))

- **tests**: Ensure tables are created before testing
  ([`ed2a715`](https://github.com/SarthakMishra/codemap/commit/ed2a715c1cbf6991d435272d2f567bc178d684a9))

Added create_db_and_tables call to setup test environment

- **tests**: Update import paths for rich print
  ([`cb4fea2`](https://github.com/SarthakMishra/codemap/commit/cb4fea2e384e1a31e70b0cb36dfca1486a70ee30))

Change the import path for rich_print from codemap.llm.rag.formatter to
  codemap.llm.rag.ask.formatter

### Chores

- Clean up utility module by removing unused imports and exports
  ([`4bab950`](https://github.com/SarthakMishra/codemap/commit/4bab950540cd4bf34c34e2c877156fd6b0a36b60))

- Remove module init
  ([`77260c6`](https://github.com/SarthakMishra/codemap/commit/77260c6de672868c4dc85b8017a0a9d90112adf5))

- Remove output file configuration
  ([`fc20c07`](https://github.com/SarthakMishra/codemap/commit/fc20c0715339224a2e210369a2ca694e131eef9f))

- Remove trailing newline
  ([`33cdd71`](https://github.com/SarthakMishra/codemap/commit/33cdd71c4c58af5f9d126337b255e0b158f42356))

- Remove unused dependencies
  ([`0b30f7f`](https://github.com/SarthakMishra/codemap/commit/0b30f7fa1e0e2db0fa9078be5ac27b30e0a92abf))

- Update lock file
  ([`a4419f1`](https://github.com/SarthakMishra/codemap/commit/a4419f12dae05f239d3671aad435e129cc854675))

- Update lock file
  ([`18587da`](https://github.com/SarthakMishra/codemap/commit/18587dab5cbc900c3b18fba48b4ea63b9b9fd1de))

- Update uv lock file
  ([`a96bf05`](https://github.com/SarthakMishra/codemap/commit/a96bf0593bde4288a096955b74fd7b268b0ab380))

- **config**: Remove diff splitter config
  ([`f5fcd9e`](https://github.com/SarthakMishra/codemap/commit/f5fcd9e0090f989da618ae6d6f098c94de1fe6a9))

- **llm/rag**: Add tests and formatting for ask response
  ([`ee3fa78`](https://github.com/SarthakMishra/codemap/commit/ee3fa78c15e6f579964f8d3a107d0b2ee5a350a6))

- **pyproject**: Update allowed_tags and modify minor/patch tags for release management
  ([`d41f412`](https://github.com/SarthakMishra/codemap/commit/d41f41232da169528b267c02f539b08bc1720bcf))

- **uv.lock**: Update uv lock file
  ([`f6396b3`](https://github.com/SarthakMishra/codemap/commit/f6396b3eba077e7a447a9ae61b926462d13b63bc))

- **uv.lock**: Update uv lock file
  ([`035fa24`](https://github.com/SarthakMishra/codemap/commit/035fa244313119f6358f9cac7fde411e3ee76859))

- **uv.lock**: Update uv lock file
  ([`a777cc1`](https://github.com/SarthakMishra/codemap/commit/a777cc15b131c6572c11cd1fcc52e353f4ee4bd8))

### Continuous Integration

- Update release workflow to use 'uv build' and 'uv publish' command
  ([`ff2ba91`](https://github.com/SarthakMishra/codemap/commit/ff2ba91edc45b71b65e4e8fbd68f49b32a70a4bb))

### Documentation

- Update installation instructions to use uv
  ([`03b239f`](https://github.com/SarthakMishra/codemap/commit/03b239fa5479ea1aced90b0e40bb432baaec38f4))

- Update installation section to recommend using uv instead of pipx
  ([`ce5d747`](https://github.com/SarthakMishra/codemap/commit/ce5d747eb387d4d750735f397ad8ccbd9677f5ff))

### Features

- Add codemap configuration file
  ([`04b4612`](https://github.com/SarthakMishra/codemap/commit/04b461291da20feb991b139dc8720ed8015b3f3c))

- Add device and model cache settings
  ([`8eaba47`](https://github.com/SarthakMishra/codemap/commit/8eaba471383641485e838941115ce73372a2353a))

- Add new field to pr command
  ([`c0cbac7`](https://github.com/SarthakMishra/codemap/commit/c0cbac7ccd6a3c716f54d3099714af77e97d5f62))

- Add pydantic-ai and nest-asyncio dependencies
  ([`a079899`](https://github.com/SarthakMishra/codemap/commit/a079899be6dea693a1a098e4670cb6c7031507c2))

- Add pylint plugin and nest_asyncio type stubs
  ([`85a965b`](https://github.com/SarthakMishra/codemap/commit/85a965bf0ca9d0eeefbeeb07eddbdcb24b8b0cd3))

- Add task to generate config keys and update pylint command
  ([`7e47514`](https://github.com/SarthakMishra/codemap/commit/7e47514c7dadfbc638cd1fd986fd26b9e7fbaced))

- Load configuration eagerly during initialization
  ([`f8975a8`](https://github.com/SarthakMishra/codemap/commit/f8975a8fdb9300ccee423c445e4171aa4cb193ea))

- Update multiple files and add new features
  ([`ef6589a`](https://github.com/SarthakMishra/codemap/commit/ef6589a6d682bc0aa0e9730cf37bf10e7190868d))

- **cli**: Add configuration command
  ([`683e271`](https://github.com/SarthakMishra/codemap/commit/683e2712a258c838f7a2ce5a926b6e2d68978abc))

Add a new configuration command to create a default .codemap.yml configuration file in the project
  root

- **cli**: Refactor cli commands and logging setup
  ([`05223b9`](https://github.com/SarthakMishra/codemap/commit/05223b95f1c69f6a0a635cb63229bff62a97c9de))

- **codemap**: Add embedding utils and voyage client
  ([`f0f4861`](https://github.com/SarthakMishra/codemap/commit/f0f48612cfdc23346c0f6b2117e87ab4c71e2626))

- **codemap**: Add entry point for running module
  ([`afee7d5`](https://github.com/SarthakMishra/codemap/commit/afee7d5d2c717bd3545a3e5b2f8f9740be1a3b6c))

- **config**: Add configuration loader
  ([`0f34b8c`](https://github.com/SarthakMishra/codemap/commit/0f34b8c77f7ed60b9c59e71d3654e240c51303b7))

introduce a new configuration loader to manage configuration settings

- **config**: Add initial config module
  ([`3e3ccd2`](https://github.com/SarthakMishra/codemap/commit/3e3ccd24224dee25746f18b8c70df48983a28e79))

- **config**: Update PR generate schema
  ([`57c9619`](https://github.com/SarthakMishra/codemap/commit/57c96195d07755584c4a6f8b618d447d534a74a0))

Changed title and description strategies to support LLM

- **utils**: Add get_absolute_path function and modify filter_valid_files to use repo_root
  ([`ff68ef2`](https://github.com/SarthakMishra/codemap/commit/ff68ef20d4a709ed51f6138265f3cc91067dea70))

### Refactoring

- Improve commit generator command
  ([`b52c169`](https://github.com/SarthakMishra/codemap/commit/b52c1692d7a2273b2a1f8e3eaf6416a8f138e85e))

- Refactor config and command to use repo root
  ([`c014363`](https://github.com/SarthakMishra/codemap/commit/c014363de6d8c769059e6a34e6eda53faf569d1d))

Update config schema to include repo root and modify command to use it

- Remove ai_models from this porject.
  ([`0f29a38`](https://github.com/SarthakMishra/codemap/commit/0f29a384287fe9a3adb5167df8a6c0ca0f9e7bba))

- **cli**: Improve cli command imports and add progress indicators
  ([`d503832`](https://github.com/SarthakMishra/codemap/commit/d50383219bd3e77a825dd834e03d5d8f9c1db3dd))

refactor cli command imports to use progress indicators for better user experience

- **cli**: Improve startup speed by deferring imports
  ([`f458f8a`](https://github.com/SarthakMishra/codemap/commit/f458f8a549abe49a83bd8f0d11aa22c910885683))

- **cli**: Refactor command registration
  ([`de30e79`](https://github.com/SarthakMishra/codemap/commit/de30e79b5e1d200b8139f38416a1214427de602b))

Refactor command registration to remove unused imports and commands

- **cli**: Remove main entry point and refactor cli commands
  ([`0ee3f5d`](https://github.com/SarthakMishra/codemap/commit/0ee3f5d775affd56e86617a7c8ea9e781afc6384))

refactored cli commands to improve code organization and readability

- **cli**: Replace loading spinner with progress indicator
  ([`9b0635d`](https://github.com/SarthakMishra/codemap/commit/9b0635dcb372322434882fc51afcdca8f5193f80))

- **cli**: Update argument types and add help messages
  ([`01fd7f1`](https://github.com/SarthakMishra/codemap/commit/01fd7f171e76ce9fbbc26fe3dfc808fad1a8562e))

- **cli**: Update ask command implementation
  ([`ea60ff6`](https://github.com/SarthakMishra/codemap/commit/ea60ff61cf5b9d8d6771a01fb803a34456601ee9))

- **cli**: Update import paths for AskCommand and print_ask_result
  ([`8eef037`](https://github.com/SarthakMishra/codemap/commit/8eef037bdf8a1a3029fd55ff1263336fcd43e8c0))

- **cli_utils**: Improve cli utility functions
  ([`7444e58`](https://github.com/SarthakMishra/codemap/commit/7444e58f6ce9a7b7b0eff6a46f1e2f218289880e))

- **codemap**: Improve path extraction and handling for renamed files
  ([`4e16b83`](https://github.com/SarthakMishra/codemap/commit/4e16b83947bce10c96615cbab03cd4860c644de6))

Enhance the robustness of path extraction from porcelain lines and handle renamed files more
  effectively

- **codemap**: Refactor code structure
  ([`9552b34`](https://github.com/SarthakMishra/codemap/commit/9552b3433f9151e157a3fc7ac0784c6105f69c98))

- **codemap**: Refactor pipeline and vector modules
  ([`77ff771`](https://github.com/SarthakMishra/codemap/commit/77ff771966aa7051b8ac5f3b745ea9b76d7941cd))

refactor pipeline and vector modules for better performance and readability

- **codemap**: Replace show_error and show_warning with logger exception and warning
  ([`c979925`](https://github.com/SarthakMishra/codemap/commit/c979925d3bee8b0671a00456f2fb00377a6e4c36))

Replaced show_error and show_warning functions with logger.exception and logger.warning to handle
  errors and warnings in a more standard way

- **codemap**: Simplify LLM response handling and prompt construction
  ([`fbf125d`](https://github.com/SarthakMishra/codemap/commit/fbf125dc45257cabeabf9ea422998981c04481ae))

- **codemap**: Update ConfigLoader instance creation
  ([`16f4eba`](https://github.com/SarthakMishra/codemap/commit/16f4eba00af7bc694d79b0b14bced83712d617cb))

- **command**: Filter out binary files during codebase processing
  ([`abdea67`](https://github.com/SarthakMishra/codemap/commit/abdea6751202dd07d510365c4d548db3fa330f44))

Added a check to skip binary files when generating LOD entities

- **commit_generator**: Refactor commit generator code
  ([`67e6d8c`](https://github.com/SarthakMishra/codemap/commit/67e6d8c0635bb723db24c4b081881d97a8c61bbf))

- **config**: Remove model settings
  ([`f3f8ba7`](https://github.com/SarthakMishra/codemap/commit/f3f8ba776c7ecd30a42350966715bb5e543c0045))

- **config_loader**: Refactor config loader classes
  ([`3c73102`](https://github.com/SarthakMishra/codemap/commit/3c7310227bdcaaa05c9d2e2048f0d1148f2827f5))

refactored ConfigError, ConfigFileNotFoundError, ConfigParsingError, and ConfigLoader classes

- **config_schema**: Replace Config class with model_config dictionary
  ([`ba13a44`](https://github.com/SarthakMishra/codemap/commit/ba13a44bb2d515b9cf91f9cbf083ca61f66db2e1))

Changed configuration for the application config from a class to a dictionary for better flexibility

- **db**: Update timestamp to use UTC timezone
  ([`af6a935`](https://github.com/SarthakMishra/codemap/commit/af6a9355ce7af99070ae96f0f5b848d56742b79e))

Changed the default factory for the timestamp field in the ChatHistory model to use
  datetime.now(UTC) instead of datetime.utcnow. Also, refactored the event loop configuration in the
  test fixtures to use pytest-asyncio's session-scoped event loops by default.

- **diff_splitter**: Improve diff splitting logic
  ([`83d8389`](https://github.com/SarthakMishra/codemap/commit/83d8389f03289d7650e699b9c3bd60fadd3d23cf))

add file move detection and create move chunks

- **diff_splitter**: Improve file filtering logic
  ([`d19be4b`](https://github.com/SarthakMishra/codemap/commit/d19be4bfacd62c40e1dcef3fedd3dc99da4a4993))

- **embedding_utils**: Lazy load voyageai clients
  ([`7a8bf21`](https://github.com/SarthakMishra/codemap/commit/7a8bf217946331b1f43949aea61b9715bf92ad13))

Moved import statements for voyageai clients inside functions to load them lazily, improving
  performance by avoiding unnecessary imports

- **generator**: Enhance LLM interaction by updating prompt handling
  ([`f958400`](https://github.com/SarthakMishra/codemap/commit/f9584004a12645e11932434ad1bac14313487464))

Replaced the generate_text method with a more structured completion method, incorporating a system
  prompt for improved context in commit message generation.

- **generator**: Improve move operation detection
  ([`b74e7e8`](https://github.com/SarthakMishra/codemap/commit/b74e7e8badb7f8a858b90623d406b5a3d22dfaf8))

added logic to identify moved file pairs and create context for move operations

- **llm**: Refactor LLM API calls to use pydantic-ai
  ([`53d9814`](https://github.com/SarthakMishra/codemap/commit/53d98149b7d7548cb9cdbbed941947e3d5e94e18))

Refactored call_llm_api function to use pydantic-ai for LLM API calls

- **path_utils**: Remove find_project_root function
  ([`cc5fba6`](https://github.com/SarthakMishra/codemap/commit/cc5fba6e1e04cc06c81917a64d9562530850d08b))

The function find_project_root has been removed from path_utils as it is no longer necessary

- **pr_cmd**: Make _handle_commits async and await commit_command run
  ([`7a97539`](https://github.com/SarthakMishra/codemap/commit/7a97539aadc1640836d00eb771bdacf92bcbbcc9))

Updated _handle_commits to be an async function and properly await the async run method of
  commit_command

- **pr_generator**: Refactor pr generator code
  ([`3339d04`](https://github.com/SarthakMishra/codemap/commit/3339d04d0a218d999940ff17cf7909d255b9f9f7))

- **tests**: Update commit message generator to return 5-tuple
  ([`6a9c5ac`](https://github.com/SarthakMishra/codemap/commit/6a9c5ac0ac1eace0f2392484421a284f03a13b0a))

Changed the return value of generate_message_with_linting to include is_json_error

### Testing

- **db**: Create tables before testing db interaction
  ([`446a82e`](https://github.com/SarthakMishra/codemap/commit/446a82ed94813f9f7e8cc77d8df66f952ce9e271))

- **db**: Skip db tests
  ([`fcb0a81`](https://github.com/SarthakMishra/codemap/commit/fcb0a8151121dcc714a32f0a07d245fa77f6722f))

- **db**: Skip db tests for create and get session
  ([`537ba01`](https://github.com/SarthakMishra/codemap/commit/537ba011dc2709952e51d623696ab33226752cad))

Add skip_db_tests decorator to test_create_db_and_tables, test_get_session, and
  test_get_session_rollback_on_error

- **embedder**: Update embedder to use mock sentence transformer
  ([`97b6a5d`](https://github.com/SarthakMishra/codemap/commit/97b6a5d5410688c60a0b8e9e82cd55b319f1f2f9))

Changed the embedder to use a mock sentence transformer for testing purposes

- **tests**: Update test cases
  ([`170d88d`](https://github.com/SarthakMishra/codemap/commit/170d88d9c0a6781eb2c0c6d700366cf6f84d31f5))

- **tests**: Update test files
  ([`8cf8f1a`](https://github.com/SarthakMishra/codemap/commit/8cf8f1a1e0575b37cff26053824dd33504c32867))

- **tests**: Update test_commit_cmd
  ([`fa75311`](https://github.com/SarthakMishra/codemap/commit/fa75311cc0b139c66c8030494036a78d5ba603f2))


## v0.3.0-rc.1 (2025-05-06)

### Chores

- Merge mkdocs override from feature branch
  ([`c1fc403`](https://github.com/SarthakMishra/codemap/commit/c1fc4030fae9d83d467c9b9cc8adaf114a58d1bc))

- Remove mkdocs.bak file and update version to 0.2.0
  ([`9a7fbd9`](https://github.com/SarthakMishra/codemap/commit/9a7fbd912bd3ec72065a88386d4aa5cc66433ec6))


## v0.2.0 (2025-05-06)

### Bug Fixes

- Resolve bug in login functionality
  ([`13c876a`](https://github.com/SarthakMishra/codemap/commit/13c876aa4e067461522b7cf39c0a810fe64e9893))

- Resolve merge conflicts in documentation
  ([`0a6064e`](https://github.com/SarthakMishra/codemap/commit/0a6064ecd178b2c96404d43f3f81f9c9b7d1650f))

- Resolve merge conflicts in documentation
  ([`dbd3b4b`](https://github.com/SarthakMishra/codemap/commit/dbd3b4b27a0467985e7de3abe48764ba0bbf91cc))

- Update dependencies to resolve vulnerability
  ([`acd6895`](https://github.com/SarthakMishra/codemap/commit/acd6895b59a04c79111fe036b006d052ee3c1c00))

- **gen**: Updated markdown formatting to use in-file links properly
  ([`549d518`](https://github.com/SarthakMishra/codemap/commit/549d518b25cbbdc70dccc3d389176fd7dcb71a03))

- **parser**: Resolve parsing issue with special characters
  ([`faf1e97`](https://github.com/SarthakMishra/codemap/commit/faf1e97506da779d706be677d4bfa0c6d3a90e6f))

- **parser**: Resolve syntax error in function calls
  ([`1d3bee3`](https://github.com/SarthakMishra/codemap/commit/1d3bee3b387f18bbf96be54ea8c4935f5667130e))

- **parser**: Resolve syntax error in parsing logic
  ([`e1deb94`](https://github.com/SarthakMishra/codemap/commit/e1deb94b69b4a64e9f969e381ded12b11e3f7ee6))

- **parser**: Resolve syntax error in query string
  ([`ec83aa2`](https://github.com/SarthakMishra/codemap/commit/ec83aa28deff450f7612a099d3db88ffeef309a2))

- **parser**: Resolve syntax error in query string
  ([`0343763`](https://github.com/SarthakMishra/codemap/commit/0343763f9c469785d87d46eb64ee089f916a10f8))

- **parser**: Resolve syntax error in variable declaration
  ([`2c12b48`](https://github.com/SarthakMishra/codemap/commit/2c12b48763e684e2888c30eb01d83d6107abe192))

- **path_utils**: Update error handling in get_cache_path
  ([`b621ee5`](https://github.com/SarthakMishra/codemap/commit/b621ee5c6bb39604193753669cc2b1c2bd3a531d))

The get_cache_path function now includes a comment about ConfigError that might be raised by
  ConfigLoader, improving code readability and maintainability.

### Chores

- Added interrogate badge to show docstring coverage
  ([`929a781`](https://github.com/SarthakMishra/codemap/commit/929a781d262df72d58c5080657540bf7166f6067))

- Added script for autogenerating docs for api from docstrings using mkdocs
  ([`9b4a0c9`](https://github.com/SarthakMishra/codemap/commit/9b4a0c9645e013d8151bd2b3ce3105a782681bfc))

- Merge mkdocs override from feature branch
  ([`ad1a108`](https://github.com/SarthakMishra/codemap/commit/ad1a108d55f92708750fdd29d2b4ffca51af6604))

- Remove .gitkeep file from docs/assets
  ([`039d831`](https://github.com/SarthakMishra/codemap/commit/039d83192e4a8566dd8c71b4ef13f243236778a6))

- Remove acknowledgements and redundant error handling
  ([`2802c7a`](https://github.com/SarthakMishra/codemap/commit/2802c7a95293ebc44c37926478937a632ccdafde))

- Remove interrogate badge svg
  ([`b9a6ef0`](https://github.com/SarthakMishra/codemap/commit/b9a6ef03723b8e881f2369c27eeafea6de1ba9d5))

- Remove mkdocs.bak file and update version to 0.2.0
  ([`7cdfd2a`](https://github.com/SarthakMishra/codemap/commit/7cdfd2a71bd0f8babf759f7c712edcf243d3b343))

- Removed unnecesary hooks
  ([`c7ce43c`](https://github.com/SarthakMishra/codemap/commit/c7ce43c855e73f806211a96a96af64bac074cd2e))

- Update .gitignore to include site directory for mkdocs and ass CNAME file
  ([`b294f5b`](https://github.com/SarthakMishra/codemap/commit/b294f5b0597adfc04b81534ca79a0f5c635ec438))

- Update dependencies and lock files
  ([`7232aaf`](https://github.com/SarthakMishra/codemap/commit/7232aaf7aabda3346ae601b343bd0aa8b6c6896d))

Updated pyproject.toml to include new dependencies and updated uv.lock to reflect the changes.

- Update image assets
  ([`cabbc10`](https://github.com/SarthakMishra/codemap/commit/cabbc1039568a1f3a156809ab6b73b78081a551c))

- Update Taskfile commands for vulture and interrogate tasks
  ([`9926f56`](https://github.com/SarthakMishra/codemap/commit/9926f5637888197ba8882c0af7ec67c3aebb3b0b))

- **config**: Remove commitlint config
  ([`0b7b18a`](https://github.com/SarthakMishra/codemap/commit/0b7b18a53f9b9bb054104cba508f7368f84255f3))

- **config**: Update code analysis level
  ([`410f0a6`](https://github.com/SarthakMishra/codemap/commit/410f0a60e3eaaed863aa3ec835eed72f26135804))

- **config**: Update commit strategy to semantic
  ([`802e93a`](https://github.com/SarthakMishra/codemap/commit/802e93a31ad703f1255a9a708736206a1bf18086))

- **dependencies**: Remove docformatter from dev dependencies
  ([`ed1e667`](https://github.com/SarthakMishra/codemap/commit/ed1e667c9501a0bc69e6c21dc37172581c6a9e90))

- **dev**: Add vulture to dev dependencies
  ([`985b99f`](https://github.com/SarthakMishra/codemap/commit/985b99fa523319d3fada2894ce19db18059acc78))

- **docs**: Update site url
  ([`a8d8a7d`](https://github.com/SarthakMishra/codemap/commit/a8d8a7d9c278270009ccbd1bf66a263478344462))

Updated the site URL in mkdocs.yml to point to the new domain

- **gen**: Chaged main heading to include rel path
  ([`16e9cba`](https://github.com/SarthakMishra/codemap/commit/16e9cbaa781dc7205abb6703f2df34035347860b))

- **lock**: Update lock file
  ([`1058cbd`](https://github.com/SarthakMishra/codemap/commit/1058cbdaa68c67b943d7ec899588f301f9f316f9))

- **pre-commit**: Update pre-commit checks to only run checks
  ([`4f41121`](https://github.com/SarthakMishra/codemap/commit/4f41121cf94e51d70215e2e3c116fc3d8db283fc))

Remove fix and format commands from pre-commit checks to prevent unnecessary file modifications

- **prompts**: Update prompt templates
  ([`0b086af`](https://github.com/SarthakMishra/codemap/commit/0b086afee46c044a6c7b4a9ce56545bc601d4e5b))

Added new prompt template for linting errors and updated existing templates for better clarity

- **Taskfile**: Remove docformatter task from formatting pipeline
  ([`7d48823`](https://github.com/SarthakMishra/codemap/commit/7d48823cba3f7a212b6146d3bb677e71a5219b08))

- **vscode**: Remove custom YAML tags for mermaid support
  ([`4fd71d1`](https://github.com/SarthakMishra/codemap/commit/4fd71d1f86131912e4802ad1f517e7ecd7f836ef))

### Code Style

- **clusterer**: Fix indentation in diffclusterer docstrings
  ([`b1ed6cb`](https://github.com/SarthakMishra/codemap/commit/b1ed6cb430928eb34551d037ccafc1a11141c9b3))

- **clusterer**: Fix indentation in diffclusterer docstrings
  ([`8b52503`](https://github.com/SarthakMishra/codemap/commit/8b525038cec43b38af51c0a8b120cee9bc396390))

- **resolver**: Fix docstring indentation
  ([`b9b53c7`](https://github.com/SarthakMishra/codemap/commit/b9b53c73d1ca50d658a9bc6c84c0292901b86306))

### Documentation

- Add mike for maintaining separate versions
  ([`1b0b8b6`](https://github.com/SarthakMishra/codemap/commit/1b0b8b6d2c55e2b0bac33985a18bec0ce87d1152))

- Added auto-generated api docs for multiple versions
  ([`28b0836`](https://github.com/SarthakMishra/codemap/commit/28b0836a8e6affb713f7bc537a40462768b7a355))

- Autogenerated api docs
  ([`3c7565d`](https://github.com/SarthakMishra/codemap/commit/3c7565d128e60d45ec633d32ab1d365193dbf5f2))

- Enhance docstrings across multiple modules for clarity and completeness.
  ([`bc2cc58`](https://github.com/SarthakMishra/codemap/commit/bc2cc58345226ed6b151004c9df5e858439e53e1))

- Remove contributing and installation guides
  ([`9a99f4e`](https://github.com/SarthakMishra/codemap/commit/9a99f4ed68429b0825a2fd0494a25ce80df16b2e))

- Remove documentation files
  ([`da7b6fe`](https://github.com/SarthakMishra/codemap/commit/da7b6fe3732ff02ca1ac1f2b4c247a3a3f29a45e))

- Remove processor documentation
  ([`9fe52be`](https://github.com/SarthakMishra/codemap/commit/9fe52bebae9034cf0049c876333638e550257972))

- Remove rag overview documentation
  ([`dda1f50`](https://github.com/SarthakMishra/codemap/commit/dda1f505c4e4e9ae8b9178c6adea4db40b88ebc4))

- Remove unused documentation files
  ([`38bf420`](https://github.com/SarthakMishra/codemap/commit/38bf420443730be4435105d44203ee54761fd5eb))

- Remove watcher overview documentation
  ([`b0d4f9a`](https://github.com/SarthakMishra/codemap/commit/b0d4f9a0865250e298947c03deb0054a339dc541))

- Reverted accidental delete of docs
  ([`df78c52`](https://github.com/SarthakMishra/codemap/commit/df78c52a1d1b8ad97f550365aa392128f03c11fc))

- **code-of-conduct**: Remove code of conduct document
  ([`bc77a14`](https://github.com/SarthakMishra/codemap/commit/bc77a146464469e799183229a0691a1f68f244ca))

- **db**: Remove database overview documentation
  ([`8b137b1`](https://github.com/SarthakMishra/codemap/commit/8b137b12eb73150a3ad37c804e1468df23c86bbf))

- **generate**: Remove generate markdown docs
  ([`721c159`](https://github.com/SarthakMishra/codemap/commit/721c159b20dd1ea4d26a543188a9527515dd6cfc))

- **mkdocs**: Update mkdocs configuration
  ([`40bdab5`](https://github.com/SarthakMishra/codemap/commit/40bdab5bbe145aebf5e9df0b1590a2e44d187958))

Added mkdocstrings plugin and updated markdown extensions. Improved theme configuration and added
  social links.

- **prompts.py**: Update prompts to request json output
  ([`564b356`](https://github.com/SarthakMishra/codemap/commit/564b3567e87816e32740804718552f419eaa4cec))

Changed the prompt to ask for a valid json object as output

- **readme**: Update readme with improved documentation and installation instructions
  ([`9c54ef6`](https://github.com/SarthakMishra/codemap/commit/9c54ef6a8c93136996d01b7fdd2476e275d02081))

The README has been updated to reflect the latest changes in CodeMap, including new installation
  options, improved documentation, and better explanations of the various features and commands.

### Features

- Add new functionality to handle user input
  ([`5c2026e`](https://github.com/SarthakMishra/codemap/commit/5c2026e091c1319939cbd43cc17478a0ae2114b5))

- Merge documentation from feature branch
  ([`b40c5ed`](https://github.com/SarthakMishra/codemap/commit/b40c5ed182426b5df237c30263009664e812b6c6))

- Merge documentation from feature branch
  ([`be813f7`](https://github.com/SarthakMishra/codemap/commit/be813f7bba120492443419a5f94f29319b5ebbab))

- **api**: Integrate message trimming for LLM API requests
  ([`f54639b`](https://github.com/SarthakMishra/codemap/commit/f54639b0f11a4558b0c0b39c9abf9a88e5a8043c))

- **batch-processor**: Add batch processing for semantic groups
  ([`06c6374`](https://github.com/SarthakMishra/codemap/commit/06c637424bff80963ba68fed7e55b4c5332d46fb))

Implement batch processing for multiple semantic groups using LiteLLM's batch_completion

- **cli_utils**: Add keyboard interrupt handling
  ([`34b6b1e`](https://github.com/SarthakMishra/codemap/commit/34b6b1e7ceebf12f4d6bf033bda5086ae5056342))

Improve error handling by adding a function to handle KeyboardInterrupt exceptions, providing a
  clean exit and user notification

- **codemap**: Add support for message format
  ([`3da6af8`](https://github.com/SarthakMishra/codemap/commit/3da6af84c1aeee770f88217c90d055b2d6199f95))

- **codemap**: Disable entity graph generation
  ([`7f04eeb`](https://github.com/SarthakMishra/codemap/commit/7f04eeb6105940ee8bc8402ac625c29bed5bd067))

The entity graph generation has been disabled to reduce computational overhead and improve overall
  performance. This change is expected to have a positive impact on the system's responsiveness.

- **codemap**: Update LLM configuration
  ([`3923e33`](https://github.com/SarthakMishra/codemap/commit/3923e33a9d97a1973eb0a9cc320433d65db743d6))

- **commit_linter**: Add create_linter function for config injection
  ([`35b4293`](https://github.com/SarthakMishra/codemap/commit/35b42939954e1555fa49b573a5be40daeb73b992))

Introduce a factory function to create a CommitLinter instance with proper configuration management

- **config**: Add llm config options
  ([`c614634`](https://github.com/SarthakMishra/codemap/commit/c6146348d0e6aaa93000d776564e137cf5c87ca0))

Added max_context_tokens and use_lod_context to the llm configuration

- **dependencies**: Add scikit-learn dependency
  ([`1f6b663`](https://github.com/SarthakMishra/codemap/commit/1f6b663e939935fa1ab3c8223dea98733cc5167a))

- **diff_splitter**: Make diffchunk hashable and comparable
  ([`b1e20c6`](https://github.com/SarthakMishra/codemap/commit/b1e20c6c3d5067d0a8dbf1778f47202c7809f8ed))

- **git**: Add untracked status to gitdiff
  ([`eeeaaa5`](https://github.com/SarthakMishra/codemap/commit/eeeaaa59847d80f88d3788b9aa75e7f86efcf065))

- **interactive**: Add bypass linter option
  ([`36974ed`](https://github.com/SarthakMishra/codemap/commit/36974ed082ab4fc86e7a6ebc4715b3e6bbdee012))

- **pr_generator**: Add workflow strategy class
  ([`14e3a9d`](https://github.com/SarthakMishra/codemap/commit/14e3a9d0f66537fc84ccc082aa2540e3ce23dea1))

- **Taskfile**: Add documentation coverage badge generation and update coverage command
  ([`f069ac9`](https://github.com/SarthakMishra/codemap/commit/f069ac92572787877b7a822db6faf11d62810289))

- **taskfile**: Add script directory to linter and formatter tasks
  ([`ac228b2`](https://github.com/SarthakMishra/codemap/commit/ac228b2546848da5ea18b259b8aa2562a1128110))

Added a new variable SCRIPT_DIR to the Taskfile and updated linter and formatter tasks to include
  the script directory. This change allows for more comprehensive code checks and formatting.

- **taskfile**: Add vulture task to find unused code
  ([`ce82ebf`](https://github.com/SarthakMishra/codemap/commit/ce82ebfa772ce03acff0fbfde70f92074b89e4d5))

- **utils**: Add support for simplified commit message format
  ([`4f1d27e`](https://github.com/SarthakMishra/codemap/commit/4f1d27e03bb1b5394f3d7432177ae49f35f0cce4))

Handle JSON responses with a 'commit_message' key

- **vscode**: Add mermaid custom tag support
  ([`a5b26fa`](https://github.com/SarthakMishra/codemap/commit/a5b26fa89dcd56e7eb21c67d5b831247f36015e0))

Added yaml.customTags configuration to enable mermaid custom tag support in vscode settings.

### Refactoring

- Remove commit linter docs and refactor commit generator
  ([`aaed54c`](https://github.com/SarthakMishra/codemap/commit/aaed54cb5f0d787ee1d91733c015b6ec666991a1))

removed commit linter documentation and refactored commit generator to improve performance

- **api**: Remove message trimming
  ([`e4c9b7b`](https://github.com/SarthakMishra/codemap/commit/e4c9b7b675550433b26657ddd855d93108eadb75))

The trim_messages function has been removed from the call_llm_api function to prevent unnecessary
  message modifications

- **cli**: Refactor cli commands
  ([`3fd7dba`](https://github.com/SarthakMishra/codemap/commit/3fd7dbab924b29cfb77ef569277c59a2be62ba07))

- **command**: Improve commit command workflow
  ([`c7999a0`](https://github.com/SarthakMishra/codemap/commit/c7999a0e2b2f24c54e02a9db188f1842156f3e77))

enhance error handling and user interaction for commit groups

- **command.py**: Improve chunk creation and error handling in commitcommand
  ([`989b571`](https://github.com/SarthakMishra/codemap/commit/989b5710efbda4139f2ab8f55430a965895cf20f))

Enhance fallback chunk creation, add logging, and handle edge cases for better robustness

- **commit_generator**: Improve commit command workflow by processing diffs individually
  ([`e603a2f`](https://github.com/SarthakMishra/codemap/commit/e603a2f3600df34ed8f5ad0259b270beab65967b))

Enhance _get_changes method to handle staged, unstaged, and untracked changes per file, and
  introduce constants for content truncation

- **commit_generator**: Refactor commit generator code
  ([`154ceee`](https://github.com/SarthakMishra/codemap/commit/154ceee193bde6ee2efe41e06828a4480a266645))

- **commit_generator**: Refactor commit generator command
  ([`42f6c59`](https://github.com/SarthakMishra/codemap/commit/42f6c59885507a9db98c77c330f68f243dabac1b))

- **commit_generator**: Refactor commit message generator to improve code quality
  ([`6204d75`](https://github.com/SarthakMishra/codemap/commit/6204d75836b16d406a65dbdff216e5dbff0b338b))

- **config**: Update config loader integration
  ([`e146433`](https://github.com/SarthakMishra/codemap/commit/e146433bf5c6a650028374b6fd5106eb5ebd730b))

Added ConfigLoader support to CommitLintConfig for dynamic configuration retrieval

- **diff_splitter**: Improve diff splitting logic
  ([`dedfbce`](https://github.com/SarthakMishra/codemap/commit/dedfbce62b27870dae0d1bb147ce1b6cba0623b9))

Enhance semantic split by using NLP and chunk detection, and handle untracked files

- **file_utils**: Remove is_text_file function
  ([`2b814bf`](https://github.com/SarthakMishra/codemap/commit/2b814bfed71b7e4508d5d1d2e59aece0854aab2c))

The is_text_file function was redundant and has been removed as it only called is_binary_file

- **generator**: Improve binary file detection and handling
  ([`73643b2`](https://github.com/SarthakMishra/codemap/commit/73643b290f027b900f5f3c37d61010a1a42b15c7))

add checks for common binary file extensions and improve diff content handling for binary files

- **linter**: Inject config loader into commitlinter
  ([`61e8996`](https://github.com/SarthakMishra/codemap/commit/61e89969809770b9cc0bf803ccb3ab62abdceb84))

use config_loader to get configuration and default types

- **splitter**: Improve diff splitting logic
  ([`7b444ba`](https://github.com/SarthakMishra/codemap/commit/7b444ba925e49a9b1552540910392e6907ba35a0))

added special handling for untracked files and restored original content for chunks

- **tests**: Remove test_generate_commit_message method
  ([`b531976`](https://github.com/SarthakMishra/codemap/commit/b531976230053918b59d9dcc445d147317324752))

- **update_api_docs**: Improve api reference section handling
  ([`8cd5bdc`](https://github.com/SarthakMishra/codemap/commit/8cd5bdc1335c83df3fbf6504c2958704a416ed54))

ensure home_section is a list before proceeding and add safer append logic

- **utils**: Remove batch_generate_completions function
  ([`7d0a5e6`](https://github.com/SarthakMishra/codemap/commit/7d0a5e6ec6176bf90c4914faa5b98717607527e3))

remove unused function to simplify codebase

- **utils**: Update commit linter to use dependency injection
  ([`5dafd33`](https://github.com/SarthakMishra/codemap/commit/5dafd33154c01587450fe2daad29b1efaab3d841))

Changed the commit linter to use a ConfigLoader instance for dependency injection, and improved
  error handling to return a failure message instead of blocking the process

- **utils**: Update file type checks
  ([`855d427`](https://github.com/SarthakMishra/codemap/commit/855d4270964d54a86056ecde8d60d02139ae20ba))

change is_text_file to is_binary_file for accurate file type checking

### Testing

- **diff_splitter**: Add test for splitting diff with untracked files
  ([`008e806`](https://github.com/SarthakMishra/codemap/commit/008e8060f19e84cfb255a11a4e6d29fd71d0a977))

This test case covers the handling of untracked files in the diff splitter

- **diff_splitter**: Update test strategies for untracked files
  ([`a0c1487`](https://github.com/SarthakMishra/codemap/commit/a0c148735ff536da35e28045216099a0ba435b87))

add is_untracked parameter to GitDiff and update assertions for chunk descriptions

- **git**: Add test cases for git commit and diff splitter
  ([`5332f29`](https://github.com/SarthakMishra/codemap/commit/5332f29cf1c9cccb0bc7084fdefa076f226c830d))

- **test_embedder**: Update test_embedder to check lines individually
  ([`a5c087b`](https://github.com/SarthakMishra/codemap/commit/a5c087b14d84cffdbaa7b1169a47268bc0ac56e4))

Changed the test_embedder function to check each line individually, preserving whitespace, for
  better diagnosis

- **tests**: Add init file for semantic grouping tests
  ([`c18e749`](https://github.com/SarthakMishra/codemap/commit/c18e74988b87d2c0835bc1d1473363fc2ef9c730))


## v0.2.0-rc.2 (2025-05-04)

### Chores

- Fix release configuration
  ([`2561087`](https://github.com/SarthakMishra/codemap/commit/256108740ca607ccab698b52d15894a0dd19e516))

- Remove CHANGELOG and reset version to 0.0.0
  ([`6b7e287`](https://github.com/SarthakMishra/codemap/commit/6b7e287140738edfe3b4183220a744151b79f441))

- Remove pymilvus and add tenacity to dependencies
  ([`d241719`](https://github.com/SarthakMishra/codemap/commit/d2417197afd65b78b8faa71f30b8559e046c9089))

- Removed unused deps
  ([`e716f3a`](https://github.com/SarthakMishra/codemap/commit/e716f3aa9f2038363c82ff3632ce9ba4829e16a3))

- Reset version to 0.0.0 and update changelog configuration
  ([`e76c20f`](https://github.com/SarthakMishra/codemap/commit/e76c20f664aca9f04553ff18622037032f2539b7))

- Revert to single-step release and enable debug logging
  ([`5972bb2`](https://github.com/SarthakMishra/codemap/commit/5972bb22f236faadb1a05e1a048888973686d2a1))

- Update deps
  ([`a2ca19a`](https://github.com/SarthakMishra/codemap/commit/a2ca19a65949fa6fcb877205936a1f068a5c9080))

- Update gitignore
  ([`0c31f2d`](https://github.com/SarthakMishra/codemap/commit/0c31f2d70a9e19f21af97d32009ffac76d5273e6))

- **config**: Update project configuration
  ([`7db018d`](https://github.com/SarthakMishra/codemap/commit/7db018d58961f3c9b786c7dc0d8a425456aa25d6))

- **dependencies**: Add kuzu package and update metadata
  ([`3ce2b11`](https://github.com/SarthakMishra/codemap/commit/3ce2b1182ac4a906597ec48a0fba164bd9185316))

- **mkdocs**: Update mkdocs configuration
  ([`d96469b`](https://github.com/SarthakMishra/codemap/commit/d96469b84be511a3f656c9bda1af9ecf1160241c))

Updated mkdocs.yml to remove unnecessary configuration and add support for mermaid diagrams. Removed
  unused markdown extensions and updated the theme configuration.

### Continuous Integration

- Deploy docs action
  ([`52755d9`](https://github.com/SarthakMishra/codemap/commit/52755d9f6c6670e2123c3f2411d4d935184ff0a9))

### Documentation

- **contributing**: Update contributing guidelines and code of conduct references
  ([`5bbacbf`](https://github.com/SarthakMishra/codemap/commit/5bbacbf2e0ccae5f9c308af151b394c1b3afe4eb))

Updated various documentation files to reflect changes in the contributing process and code of
  conduct. This includes changes to the development setup, guidelines, and references to other
  documentation files.

- **index**: Update documentation links and content
  ([`23fdc96`](https://github.com/SarthakMishra/codemap/commit/23fdc9602f73da9eb78916a297d2a9d1a93f6f8c))

Updated the index page to reflect changes in documentation structure and added new links to usage
  and contributing guides. Additionally, modified the installation page to use the new warning and
  tip formats.

- **usage**: Update documentation for commit and generate commands
  ([`bc2d1a7`](https://github.com/SarthakMishra/codemap/commit/bc2d1a74318968b480efa09aea5907c25e17038c))

Updated documentation for commit and generate commands to reflect changes in usage and
  configuration. Added details about the custom distilled model used for semantic analysis.

- **usage**: Update pr documentation
  ([`adf7b67`](https://github.com/SarthakMishra/codemap/commit/adf7b676433c6fff16dd403c963b8934cc95d0bb))

Added documentation for the 'cm pr create' and 'cm pr update' commands, including descriptions of
  arguments and options. Improved formatting and readability of the documentation.

### Features

- **codemap**: Enable entity graph inclusion
  ([`80f160c`](https://github.com/SarthakMishra/codemap/commit/80f160cfe5c6709f57c1a506b57e6360de975688))

Updated the codemap configuration to include entity relationship graphs in the output. This change
  enhances the documentation generated by the codemap tool, providing a more comprehensive
  understanding of the code structure.

- **deps**: Add multiple new dependencies for documentation and functionality
  ([`10269ab`](https://github.com/SarthakMishra/codemap/commit/10269abf0d75c4eaf1a849b55d4316a3f545f596))

This commit introduces several new packages including babel, backrefs, ghp-import, markdown,
  mergedeep, mkdocs, mkdocs-material, paginate, pymdown-extensions, and pyyaml-env-tag to enhance
  documentation capabilities and project functionality.

- **deps**: Add multiple new dependencies for documentation and functionality
  ([`3ee5daa`](https://github.com/SarthakMishra/codemap/commit/3ee5daa625edf70006980e934c10a9cc730dbf4f))

This commit introduces several new packages including babel, backrefs, ghp-import, markdown,
  mergedeep, mkdocs, mkdocs-material, paginate, pymdown-extensions, and pyyaml-env-tag to enhance
  documentation capabilities and project functionality.

- **docs**: Add comprehensive documentation for CodeMap features and usage
  ([`c2e0e4f`](https://github.com/SarthakMishra/codemap/commit/c2e0e4f22cc39b04ea38784fd63d451a2767fe27))

- **docs**: Add mkdocs-material dependency
  ([`d8e6c49`](https://github.com/SarthakMishra/codemap/commit/d8e6c49c96602a169a50cd3c0e7fad21b92e4d7f))

This commit adds mkdocs-material as a dependency for documentation generation.

- **docs**: Add mkdocs-material dependency
  ([`d80fef4`](https://github.com/SarthakMishra/codemap/commit/d80fef47885a6a77ad78118a70f1e22664be1342))

This commit adds the mkdocs-material dependency to the project.

- **docs**: Enhance mkdocs configuration with additional metadata and social links
  ([`5c45af1`](https://github.com/SarthakMishra/codemap/commit/5c45af195a429fed54b375afdd2e05019c6ba820))

This commit adds extra configuration to mkdocs, including status labels for new and deprecated
  items, as well as social media links for GitHub, Python, LinkedIn, and Twitter.


## v0.2.0-rc.1 (2025-05-04)

### Testing

- **database**: Skip database-dependent tests in ci environment
  ([`50ed567`](https://github.com/SarthakMishra/codemap/commit/50ed567554693bc4a5b778a4a375b7588052ab86))

Added environment variable SKIP_DB_TESTS to skip database-dependent tests in CI environments without
  PostgreSQL. Updated test_client.py and test_models.py to use the skip_db_tests marker from
  conftest.py.


## v0.1.0 (2025-05-02)


## v0.1.0-rc.1 (2025-05-02)

- Initial Release
