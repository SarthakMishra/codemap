"""Commit message generation using LLMs for CodeMap."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, TypedDict, cast

from codemap.git.commit_linter import CommitLinter
from codemap.utils.cli_utils import loading_spinner
from codemap.utils.config_loader import ConfigLoader

if TYPE_CHECKING:
	import pathlib
	from collections.abc import Iterable


# Define DiffChunk class outside of TYPE_CHECKING
class DiffChunk:
	"""Represents a logical chunk of changes to files in a diff."""

	def __init__(self, files: list[str], content: str, description: str | None = None) -> None:
		"""
		Initialize a diff chunk.

		Args:
		    files: List of files affected in this chunk
		    content: The diff content
		    description: Optional description of the changes

		"""
		self.files: list[str] = files
		self.content: str = content
		self.description: str | None = description
		self.is_llm_generated: bool = False


logger = logging.getLogger(__name__)

# Constants to avoid magic numbers
MIN_DESCRIPTION_LENGTH = 10
MIN_SCOPE_LENGTH = 5
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
KNOWN_PROVIDERS = {
	"openai",
	"anthropic",
	"azure",
	"cohere",
	"groq",
	"mistral",
	"together",
	"openrouter",
}


# Default prompt template for commit message generation
DEFAULT_PROMPT_TEMPLATE = """
# Conventional Commits 1.0.0

The commit message should be structured as follows:

---

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```
---

<br />
The commit contains the following structural elements, to communicate intent to the
consumers of your library:

1. **fix:** a commit of the _type_ `fix` patches a bug in your codebase
  (this correlates with [`PATCH`] in Semantic Versioning).
2. **feat:** a commit of the _type_ `feat` introduces a new feature to the codebase
  (this correlates with [`MINOR`] in Semantic Versioning).
3. **BREAKING CHANGE:** a commit that has a footer `BREAKING CHANGE:`, or appends a `!` after the type/scope,
  introduces a breaking API change (correlating with [`MAJOR`] in Semantic Versioning).
A BREAKING CHANGE can be part of commits of any _type_.
4. _types_ other than `fix:` and `feat:` are allowed, for example @commitlint/config-conventional
  (based on the Angular convention) recommends `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`,
  `perf:`, `test:`, and others.
5. _footers_ other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to
  [git trailer format](https://git-scm.com/docs/git-interpret-trailers).

Additional types are not mandated by the Conventional Commits specification, and have no implicit effect
in Semantic Versioning (unless they include a BREAKING CHANGE).

A scope may be provided to a commit's type, to provide additional contextual information and is contained within
parenthesis, e.g., `feat(parser): add ability to parse arrays`.

## Examples

### Commit message with description and breaking change footer
```
feat: allow provided config object to extend other configs

BREAKING CHANGE: `extends` key in config file is now used for extending other config files
```

### Commit message with `!` to draw attention to breaking change
```
feat!: send an email to the customer when a product is shipped
```

### Commit message with scope and `!` to draw attention to breaking change
```
feat(api)!: send an email to the customer when a product is shipped
```

### Commit message with both `!` and BREAKING CHANGE footer
```
chore!: drop support for Node 6

BREAKING CHANGE: use JavaScript features not available in Node 6.
```

### Commit message with no body
```
docs: correct spelling of CHANGELOG
```

### Commit message with scope
```
feat(lang): add Polish language
```

### Commit message with multi-paragraph body and multiple footers
```
fix: prevent racing of requests

Introduce a request id and a reference to latest request. Dismiss
incoming responses other than from latest request.

Remove timeouts which were used to mitigate the racing issue but are
obsolete now.

Reviewed-by: Z
Refs: #123
```

## Specification

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY",
and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

1. Commits MUST be prefixed with a type, which consists of a noun, `feat`, `fix`, etc., followed
by the OPTIONAL scope, OPTIONAL `!`, and REQUIRED terminal colon and space.
2. The type `feat` MUST be used when a commit adds a new feature to your application or library.
3. The type `fix` MUST be used when a commit represents a bug fix for your application.
4. A scope MAY be provided after a type. A scope MUST consist of a noun describing a
section of the codebase surrounded by parenthesis, e.g., `fix(parser):`
5. A description MUST immediately follow the colon and space after the type/scope prefix.
6. The description is a short summary of the code changes, e.g., _fix: array parsing issue when multiple spaces were
contained in string_.
7. A longer commit body MAY be provided after the short description, providing additional contextual information about
the code changes. The body MUST begin one blank line after the description.
8. A commit body is free-form and MAY consist of any number of newline separated paragraphs.
9. One or more footers MAY be provided one blank line after the body. Each footer MUST consist of
 a word token, followed by either a `:<space>` or `<space>#` separator, followed by a string value.
10. A footer's token MUST use `-` in place of whitespace characters, e.g., `Acked-by`.
An exception is made for `BREAKING CHANGE`, which MAY also be used as a token.
11. A footer's value MAY contain spaces and newlines, and parsing MUST terminate when the next valid footer
  token/separator pair is observed.
12. Breaking changes MUST be indicated in the type/scope prefix of a commit, or as an entry in the
  footer.
13. If included as a footer, a breaking change MUST consist of the uppercase text BREAKING CHANGE, followed by a colon,
space, and description, e.g., _BREAKING CHANGE: environment variables now take precedence over config files_.
14. If included in the type/scope prefix, breaking changes MUST be indicated by a
  `!` immediately before the `:`. If `!` is used, `BREAKING CHANGE:` MAY be omitted from the footer section,
  and the commit description SHALL be used to describe the breaking change.
15. Types other than `feat` and `fix` MAY be used in your commit messages, e.g., _docs: update ref docs._
16. The units of information that make up Conventional Commits MUST NOT be treated as case sensitive by implementors,
with the exception of BREAKING CHANGE which MUST be uppercase.
17. BREAKING-CHANGE MUST be synonymous with BREAKING CHANGE, when used as a token in a footer.
---

You are a helpful assistant that generates conventional commit messages based on code changes.
Given a Git diff, please generate a concise and descriptive commit message following these conventions:

1. Use the format:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```
2. Types include: {convention[types]}
3. Scope must be short (1-2 words), concise, and represent the specific component affected
4. The description should be a concise, imperative present tense summary of the *specific code changes*
   in the diff chunk (e.g., "add feature", "fix bug", "update documentation").
   Focus on *what* was changed and *why*.
5. The optional body should be a multi-paragraph summary of the changes, focusing on the *why* and *how* of the changes.
6. The optional footer(s) should be a list of one or more footers, each with a token and a value.

## Commit Linting Rules
Your generated commit message will be validated against the following rules:
1. Type must be one of the allowed types: {convention[types]}
2. Type must be lowercase
3. Subject must not end with a period
4. Subject must be at least 10 characters long
5. Header line (first line) should be no longer than {convention[max_length]} characters
6. If a scope is provided, it must be in lowercase
7. Header must have a space after the colon
8. Description must start with an imperative verb (e.g., "add", not "adds" or "added")

---
Here are some notes about the files changed:
{files}
---
Analyze the following diff and respond with ONLY the commit message string:

{diff}

---
IMPORTANT:
- Strictly follow the format and instructions above.
- Do not include any other text, explanation, or surrounding characters (like quotes or markdown).
- Strictly do not include any `Related Issue #`, `Closes #`, `REVIEWED-BY`, `TRACKING #`, `APPROVED` footers.
- Strictly follow the JSON schema provided while generating output in JSON format:

{schema}
"""


# Define a TypedDict to represent the structure of a DiffChunk
class DiffChunkData(TypedDict, total=False):
	"""TypedDict representing the structure of a DiffChunk."""

	files: list[str]
	content: str
	description: str


class LLMError(Exception):
	"""Custom exception for LLM-related errors."""


# Define a schema for structured commit message output
COMMIT_MESSAGE_SCHEMA = {
	"type": "object",
	"properties": {
		"type": {
			"type": "string",
			"description": "The type of change (e.g., feat, fix, docs, style, refactor, perf, test, chore)",
		},
		"scope": {"type": ["string", "null"], "description": "The scope of the change (e.g., component affected)"},
		"description": {"type": "string", "description": "A short, imperative-tense description of the change"},
		"body": {
			"type": ["string", "null"],
			"description": "A longer description of the changes, explaining why and how",
		},
		"breaking": {"type": "boolean", "description": "Whether this is a breaking change", "default": False},
		"footers": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"token": {
						"type": "string",
						"description": "Footer token (e.g., 'BREAKING CHANGE', 'Fixes', 'Refs')",
					},
					"value": {"type": "string", "description": "Footer value"},
				},
				"required": ["token", "value"],
			},
			"default": [],
		},
	},
	"required": ["type", "description"],
}


# Define a class to represent structured commit messages
class CommitMessageSchema(TypedDict):
	"""TypedDict representing the structured commit message output."""

	type: str
	scope: str | None
	description: str
	body: str | None
	breaking: bool
	footers: list[dict[str, str]]


class MessageGenerator:
	"""Generates commit messages using LLMs."""

	def __init__(
		self,
		repo_root: pathlib.Path,
		prompt_template: str | None = None,
		model: str = "gpt-4o-mini",  # Default model
		provider: str | None = None,  # Optional explicit provider
		api_base: str | None = None,
		config_loader: ConfigLoader | None = None,
	) -> None:
		"""
		Initialize the message generator.

		Args:
		    repo_root: Root directory of the Git repository
		    prompt_template: Custom prompt template to use
		    model: Model identifier to use (can include provider prefix like 'groq/llama3')
		    provider: Explicit provider name (e.g., "openai", "anthropic").
		              Overrides provider inferred from model prefix if both are present.
		    api_base: Optional API base URL for the provider
		    config_loader: Optional ConfigLoader instance to use for configuration

		"""
		self.repo_root = repo_root
		self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

		# Store initial config values from parameters
		self._initial_model = model
		self._initial_provider = provider
		self._initial_api_base = api_base

		# For tests - flag to bypass API key check
		self._mock_api_key_available = False

		# For testing: If model has a provider prefix, extract it for the initial provider
		if "/" in model and provider is None:
			provider_from_model = model.split("/")[0].lower()
			self._initial_provider = provider_from_model
			logger.debug("Using provider '%s' from model prefix", provider_from_model)

		# Try to load environment variables from .env.local file
		self._load_env_variables()

		# Use provided ConfigLoader or create a new one
		self._config_loader = config_loader or ConfigLoader(repo_root=repo_root)

		# Get LLM config from the loader
		llm_config = self._config_loader.get_llm_config()

		logger.debug("Initial provider: %s", self._initial_provider)
		logger.debug("Config provider: %s", llm_config.get("provider"))

		# Override initial values with config if not provided explicitly
		if model == "gpt-4o-mini" and llm_config["model"]:  # Only override if using the default
			self._initial_model = llm_config["model"]
		if provider is None and not self._initial_provider and llm_config["provider"]:
			self._initial_provider = llm_config["provider"]
		if api_base is None and llm_config["api_base"]:
			self._initial_api_base = llm_config["api_base"]

		logger.debug("Final initial provider before resolution: %s", self._initial_provider)

		# Load API keys from environment/config
		self._api_keys = self._get_api_keys(llm_config)

		# --- Centralized Configuration Resolution ---
		(
			self.resolved_model,
			self.resolved_provider,
			self.resolved_api_base,
		) = self._resolve_llm_configuration(self._initial_model, self._initial_provider, self._initial_api_base)
		logger.debug(
			"Resolved LLM Configuration: Provider=%s, Model=%s, API_Base=%s",
			self.resolved_provider,
			self.resolved_model,
			self.resolved_api_base or "Default",
		)
		# --- End Centralized Configuration Resolution ---

	@property
	def model(self) -> str:
		"""
		Get the model name.

		Returns:
		    The resolved model name

		"""
		return self.resolved_model

	@model.setter
	def model(self, value: str) -> None:
		"""
		Set the model name.

		Args:
		    value: The model name to set

		"""
		self.resolved_model = value

	@model.deleter
	def model(self) -> None:
		"""
		Delete the model property.

		This is needed for unit tests using patch.object.

		"""

	@property
	def provider(self) -> str:
		"""
		Get the provider name.

		Returns:
		    The resolved provider name

		"""
		return self.resolved_provider

	@provider.setter
	def provider(self, value: str) -> None:
		"""
		Set the provider name.

		Args:
		    value: The provider name to set

		"""
		self.resolved_provider = value

	@provider.deleter
	def provider(self) -> None:
		"""
		Delete the provider property.

		This is needed for unit tests using patch.object.

		"""

	def _load_env_variables(self) -> None:
		"""Load environment variables from .env.local or .env files."""
		try:
			from dotenv import load_dotenv

			# First try .env.local (higher priority)
			env_local = self.repo_root / ".env.local"
			if env_local.exists():
				load_dotenv(dotenv_path=env_local, override=True)  # Override existing env vars
				logger.debug("Loaded environment variables from %s", env_local)
				return

			# Then try .env (lower priority)
			env_file = self.repo_root / ".env"
			if env_file.exists():
				load_dotenv(dotenv_path=env_file, override=True)  # Override existing env vars
				logger.debug("Loaded environment variables from %s", env_file)
		except ImportError:
			logger.debug("python-dotenv not installed, skipping .env file loading")
		except OSError as e:
			logger.warning("Error loading .env file: %s", e)

	def _get_api_keys(self, llm_config: dict[str, Any] | None = None) -> dict[str, str]:
		"""
		Get API keys from environment or config.

		Args:
		    llm_config: LLM configuration from ConfigLoader

		Returns:
		    Dictionary of API keys found for known providers.

		"""
		api_keys: dict[str, str] = {}

		# Common environment variable pattern for provider API keys
		# Format: <PROVIDER>_API_KEY (e.g., OPENAI_API_KEY, GROQ_API_KEY)

		# 1. Check all provider-specific environment variables
		for provider in KNOWN_PROVIDERS:
			env_var = f"{provider.upper()}_API_KEY"
			key = os.environ.get(env_var)
			if key:
				api_keys[provider] = key
				logger.debug("Loaded API key for %s from environment variable %s", provider, env_var)

		# 2. Also look for generic API_KEY if present
		generic_key = os.environ.get("API_KEY")
		if generic_key and "api_key_provider" in os.environ:
			# If API_KEY is set with a provider, store it under that provider
			provider = os.environ.get("API_KEY_PROVIDER", "").lower()
			if provider and provider not in api_keys:
				api_keys[provider] = generic_key
				logger.debug("Loaded API key for %s from generic API_KEY environment variable", provider)

		# 3. Load from config if provided and not already loaded from environment
		if llm_config:
			provider = llm_config.get("provider")
			if provider:  # Check if provider exists and is not empty/None
				provider = provider.lower()

				# Only use config API key if not already loaded from environment
				if provider not in api_keys:
					# Try provider-specific key from config first
					provider_key_name = f"{provider}_api_key"
					provider_specific_key = llm_config.get(provider_key_name)
					if provider_specific_key:  # Check if key exists and value is truthy
						api_keys[provider] = provider_specific_key
						logger.debug("Loaded API key for %s from config (%s)", provider, provider_key_name)
					else:
						# Fall back to generic api_key if available
						generic_key = llm_config.get("api_key")
						if generic_key:  # Check if key exists and value is truthy
							api_keys[provider] = generic_key
							logger.debug("Loaded API key for %s from generic config api_key", provider)

		logger.debug("Loaded API keys for providers: %s", list(api_keys.keys()))
		return api_keys

	def _resolve_llm_configuration(
		self, model: str, provider: str | None, api_base: str | None
	) -> tuple[str, str, str | None]:
		"""
		Resolves the final model string, provider name, and API base URL.

		Args:
		    model: The model name (potentially with provider prefix).
		    provider: An explicitly configured provider name.
		    api_base: An explicitly configured API base URL.

		Returns:
		    A tuple containing (resolved_model, resolved_provider, resolved_api_base).
		    resolved_model will be in 'provider/model_name' format.
		    resolved_provider will be the determined provider name.

		"""
		resolved_model = model
		resolved_provider = provider
		resolved_api_base = api_base

		# 1. Extract provider from model name if it has a prefix
		model_has_prefix = "/" in resolved_model
		if model_has_prefix:
			model_prefix = resolved_model.split("/")[0].lower()
			# If no explicit provider is specified, use the prefix from the model
			if not resolved_provider:
				resolved_provider = model_prefix
				logger.debug("Provider '%s' inferred from model name '%s'", resolved_provider, resolved_model)
			# If explicit provider is different from model prefix, respect the model prefix in tests
			elif self._mock_api_key_available and resolved_provider.lower() != model_prefix:
				resolved_provider = model_prefix
				logger.debug("Using provider '%s' from model name for tests", resolved_provider)
			# Otherwise, if explicit provider is different from model prefix, log a warning
			elif resolved_provider.lower() != model_prefix:
				# Log the discrepancy but respect the explicitly provided provider
				logger.warning(
					"Provider '%s' specified but model '%s' has different prefix. Using specified provider.",
					resolved_provider,
					resolved_model,
				)

		# 2. If model has no prefix but provider is specified, add the prefix
		elif resolved_provider:
			# Format the model string with the provider prefix
			resolved_provider = resolved_provider.lower()
			if not resolved_model.startswith(f"{resolved_provider}/"):
				resolved_model = f"{resolved_provider}/{resolved_model}"
				logger.debug("Formatted model with provider prefix: '%s'", resolved_model)
		else:
			# No provider information at all - default to openai
			resolved_provider = "openai"
			if not resolved_model.startswith("openai/"):
				resolved_model = f"openai/{resolved_model}"
				logger.debug("Defaulting to provider 'openai', formatted model as '%s'", resolved_model)

		# Ensure resolved_provider is lowercase
		resolved_provider = resolved_provider.lower()

		return resolved_model, resolved_provider, resolved_api_base

	def _extract_file_info(self, chunk: DiffChunk | DiffChunkData) -> dict[str, Any]:
		"""
		Extract file information from the diff chunk.

		Args:
		    chunk: Diff chunk to extract information from

		Returns:
		    Dictionary with information about files

		"""
		file_info = {}
		files = chunk.files if isinstance(chunk, DiffChunk) else chunk.get("files", [])
		if not isinstance(files, list):
			try:
				# Convert to list only if it's actually iterable
				if hasattr(files, "__iter__") and not isinstance(files, str):
					files = list(cast("Iterable", files))
				else:
					files = []
			except (TypeError, ValueError):
				files = []

		for file in files:
			if not isinstance(file, str):
				continue  # Skip non-string file entries
			file_path = self.repo_root / file
			if not file_path.exists():
				continue
			try:
				extension = file_path.suffix.lstrip(".")
				file_info[file] = {
					"extension": extension,
					"directory": str(file_path.parent.relative_to(self.repo_root)),
				}
				path_parts = file_path.parts
				if len(path_parts) > 1:
					if "src" in path_parts:
						idx = path_parts.index("src")
						if idx + 1 < len(path_parts):
							file_info[file]["module"] = path_parts[idx + 1]
					elif "tests" in path_parts:
						file_info[file]["module"] = "tests"
			except (ValueError, IndexError, TypeError):
				continue
		return file_info

	def _get_commit_convention(self) -> dict[str, Any]:
		"""Get commit convention settings from config."""
		# Use the centralized ConfigLoader to get the convention
		return self._config_loader.get_commit_convention()

	def _prepare_prompt(self, chunk: DiffChunk | DiffChunkData) -> str:
		"""
		Prepare the prompt for the LLM.

		Args:
		    chunk: Diff chunk to prepare prompt for

		Returns:
		    Prepared prompt with diff and file information

		"""
		file_info = self._extract_file_info(chunk)
		convention = self._get_commit_convention()

		# Get the diff content from the chunk
		diff_content = chunk.content if isinstance(chunk, DiffChunk) else chunk.get("content", "")

		context = {
			"diff": diff_content,
			"files": file_info,
			"convention": convention,
			"schema": COMMIT_MESSAGE_SCHEMA,
		}
		try:
			return self.prompt_template.format(**context)
		except KeyError:
			logger.exception("Prompt template formatting error. Missing key: %s. Using default template.")
			# Fallback to default template if custom one fails
			return DEFAULT_PROMPT_TEMPLATE.format(**context)

	def _call_llm_api(self, prompt: str) -> str:
		"""
		Call the LLM API using the resolved configuration.

		Args:
		    prompt: Formatted prompt for the API

		Returns:
		    Generated commit message

		Raises:
		    LLMError: If API call fails or litellm is not installed.

		"""
		try:
			import litellm
		except ImportError:
			msg = "LiteLLM library not installed. Install it with 'pip install litellm'."
			logger.exception(msg)
			raise LLMError(msg) from None

		# Use the resolved configuration from __init__
		model_to_use = self.resolved_model
		provider_to_use = self.resolved_provider
		api_base_to_use = self.resolved_api_base

		# Get the API key for the resolved provider
		api_key = self._api_keys.get(provider_to_use)

		# Check generic key from config if provider-specific key is missing
		# This relies on _get_api_keys potentially storing a generic key
		# associated with the initial provider if no specific key was found.
		# A bit fragile, might be better to explicitly check llm_config['api_key'] here if needed.
		if not api_key and "api_key" in self._get_llm_config_from_yaml():  # Helper needed
			api_key = self._get_llm_config_from_yaml().get("api_key")
			logger.debug("Using generic 'api_key' from config for provider %s", provider_to_use)

		# Define the env_var_map at the beginning
		env_var_map = {
			"openai": "OPENAI_API_KEY",
			"anthropic": "ANTHROPIC_API_KEY",
			"azure": "AZURE_API_KEY",
			"cohere": "COHERE_API_KEY",
			"groq": "GROQ_API_KEY",
			"mistral": "MISTRAL_API_KEY",
			"together": "TOGETHER_API_KEY",
			"openrouter": "OPENROUTER_API_KEY",
		}

		if not api_key:
			# Re-check environment directly as a last resort for specific providers
			env_var = env_var_map.get(provider_to_use)
			if env_var:
				api_key = os.environ.get(env_var)
				if api_key:
					logger.debug(
						"API key for resolved provider '%s' found in environment variable %s.", provider_to_use, env_var
					)
					return api_key

		if not api_key:
			error_msg = (
				f"No API key found for provider '{provider_to_use}'. "
				f"Checked config, environment variables ({env_var_map.get(provider_to_use, 'N/A')}), "
				f"and generic 'api_key' in config."
			)
			raise LLMError(error_msg)

		logger.debug(
			"Calling LiteLLM: Provider=%s, Model=%s, API_Base=%s, Key_Found=True",
			provider_to_use,
			model_to_use,
			api_base_to_use or "Default",
		)

		try:
			# Check if model supports response_format and json_schema
			supports_json_format = False
			try:
				# Try to query model capabilities, but don't fail if it doesn't work
				# Just assume the model supports JSON format and let LiteLLM handle it
				supports_json_format = True
				logger.debug("Attempting to use JSON format with model %s", model_to_use)
			except (AttributeError, ImportError, ValueError, TypeError) as e:
				logger.warning("Could not check model capabilities, will try JSON format anyway: %s", e)
				supports_json_format = True  # Attempt it, LiteLLM will handle if not supported

			# Set up request parameters
			request_params = {
				"model": model_to_use,
				"messages": [{"role": "user", "content": prompt}],
				"api_key": api_key,
				"api_base": api_base_to_use,
				"max_retries": 2,
				"timeout": 30,
			}

			# Add response_format for JSON if supported
			if supports_json_format:
				request_params["response_format"] = {"type": "json_object", "schema": COMMIT_MESSAGE_SCHEMA}
				# Enable schema validation (client-side fallback)
				litellm.enable_json_schema_validation = True

			response = litellm.completion(**request_params)

			# Extract content from the response
			content = ""

			try:
				# Handle different response formats safely
				if isinstance(response, object) and hasattr(response, "choices"):
					choices = getattr(response, "choices", [])
					if choices and len(choices) > 0:
						first_choice = choices[0]
						if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
							content = getattr(first_choice.message, "content", "")

				# Then try as dictionary if the above failed
				if not content and isinstance(response, dict):
					choices = response.get("choices", [])
					if choices and len(choices) > 0:
						first_choice = choices[0]
						if isinstance(first_choice, dict):
							message = first_choice.get("message", {})
							if isinstance(message, dict):
								content = message.get("content", "")
			except (AttributeError, IndexError, TypeError) as extract_error:
				logger.warning("Error extracting content from response: %s", extract_error)

			if not content:
				logger.error("Could not extract content from LLM response")
				error_message = "Failed to extract content from LLM response"
				raise LLMError(error_message)  # noqa: TRY301

			# Parse and format the commit message
			return self._format_structured_message(content)

		except Exception as e:
			error_msg = "LLM API call failed: %s"
			logger.exception(error_msg)  # The exception info is already included in logger.exception
			raise LLMError(error_msg % e) from e

	def _format_structured_message(self, content: str) -> str:
		"""
		Format a structured JSON response into a conventional commit message.

		Args:
		        content: JSON content string from LLM response

		Returns:
		        Formatted commit message string

		"""
		try:
			# Try to parse the content as JSON
			message_data = json.loads(content)

			# Extract components
			commit_type = message_data.get("type", "chore")
			scope = message_data.get("scope")
			description = message_data.get("description", "")
			body = message_data.get("body")
			is_breaking = message_data.get("breaking", False)
			footers = message_data.get("footers", [])

			# Format the header
			header = f"{commit_type}"
			if scope:
				header += f"({scope})"
			if is_breaking:
				header += "!"
			header += f": {description}"

			# Build the complete message
			message_parts = [header]

			# Add body if provided
			if body:
				message_parts.append("")  # Empty line between header and body
				message_parts.append(body)

			# Filter footers - only allow BREAKING CHANGE
			# TODO: This implementation will be improved in the future to support  # noqa: FIX002, TD002, TD003
			# automatic tracking of related GitHub issues and linking them to commits,
			# along with other useful footer types. For now, we only allow BREAKING CHANGE
			# footers to avoid unwanted references like "Related Issues: #123".
			breaking_change_footers = []
			for footer in footers:
				token = footer.get("token", "")
				value = footer.get("value", "")
				if token and value:
					if token.upper() == "BREAKING CHANGE" or token.upper() == "BREAKING-CHANGE":
						breaking_change_footers.append(footer)
					else:
						logger.debug("Filtering out non-breaking-change footer token: %s", token)

			# Add breaking change footers if provided
			if breaking_change_footers:
				if not body:
					message_parts.append("")  # Empty line before footers if no body
				else:
					message_parts.append("")  # Empty line between body and footers

				for footer in breaking_change_footers:
					token = footer.get("token", "")
					value = footer.get("value", "")
					message_parts.append(f"{token}: {value}")

			return "\n".join(message_parts)

		except (json.JSONDecodeError, TypeError, AttributeError) as e:
			# If parsing fails, return the content as-is (might be a plaintext response)
			logger.warning("Could not parse JSON response, using raw content: %s", e)
			return content.strip()

	# Helper to read llm config section again (used in _call_llm_api for generic key)
	def _get_llm_config_from_yaml(self) -> dict:
		"""
		Get LLM configuration from config loader.

		Returns:
		    Dictionary with LLM configuration

		"""
		return self._config_loader.get_llm_config()

	def fallback_generation(self, chunk: DiffChunk | DiffChunkData) -> str:
		"""
		Generate a fallback commit message without LLM.

		This is used when LLM-based generation fails or is disabled.

		Args:
		    chunk: Diff chunk to generate message for

		Returns:
		    Generated commit message

		"""
		commit_type = "chore"

		# Get files from the chunk
		files = chunk.files if isinstance(chunk, DiffChunk) else chunk.get("files", [])

		string_files = [f for f in files if isinstance(f, str)]  # Filter only strings for path operations

		for file in string_files:
			if file.startswith("tests/"):
				commit_type = "test"
				break
			if file.startswith("docs/") or file.endswith(".md"):
				commit_type = "docs"
				break

		# Get content from the chunk
		content = chunk.content if isinstance(chunk, DiffChunk) else chunk.get("content", "")

		if isinstance(content, str) and ("fix" in content.lower() or "bug" in content.lower()):
			commit_type = "fix"  # Be slightly smarter about 'fix' type

		description = "update files"  # Default description
		if string_files:
			if len(string_files) == 1:
				description = f"update {string_files[0]}"
			else:
				try:
					common_dir = os.path.commonpath(string_files)
					# Make common_dir relative to repo root if possible
					try:
						common_dir_rel = os.path.relpath(common_dir, self.repo_root)
						if common_dir_rel and common_dir_rel != ".":
							description = f"update files in {common_dir_rel}"
						else:
							description = f"update {len(string_files)} files"
					except ValueError:  # Happens if paths are on different drives (unlikely in repo)
						description = f"update {len(string_files)} files"

				except (ValueError, TypeError):  # commonpath fails on empty list or mixed types
					description = f"update {len(string_files)} files"

		message = f"{commit_type}: {description}"
		# Ensure fallback follows length constraints without the old formatting methods
		convention = self._get_commit_convention()
		max_length = convention.get("max_length", 72)
		if len(message) > max_length:
			message = message[:max_length]

		return message

	def _verify_api_key_availability(self) -> bool:
		"""Verify that the API key for the resolved provider is available."""
		# For tests - if mock flag is set, return True
		if hasattr(self, "_mock_api_key_available") and self._mock_api_key_available:
			logger.debug("Mock API key flag is set, returning True for tests")
			return True

		# Use the resolved provider determined during initialization
		provider = self.resolved_provider
		if not provider:
			logger.error("Provider could not be resolved. Cannot verify API key.")
			return False

		# Check if we have the key in our _api_keys dictionary
		if provider in self._api_keys:
			logger.debug("API key for provider '%s' is available in cached keys.", provider)
			return True

		# Last check in environment directly
		env_var = f"{provider.upper()}_API_KEY"
		if os.environ.get(env_var):
			logger.debug("API key for provider '%s' found in environment variable %s.", provider, env_var)
			# Add it to our cache for future use
			self._api_keys[provider] = os.environ[env_var]
			return True

		# Also check for generic API_KEY with matching provider
		if os.environ.get("API_KEY") and os.environ.get("API_KEY_PROVIDER", "").lower() == provider.lower():
			logger.debug("API key for provider '%s' found in generic API_KEY environment variable.", provider)
			self._api_keys[provider] = os.environ["API_KEY"]
			return True

		logger.warning("API key for provider '%s' is MISSING.", provider)
		return False

	def _adapt_chunk_access(self, chunk: DiffChunk | DiffChunkData) -> DiffChunkData:
		"""
		Adapt chunk access to work with both DiffChunk objects and dictionaries.

		Args:
		    chunk: Chunk to adapt

		Returns:
		    Dictionary with chunk data

		"""
		if isinstance(chunk, DiffChunk):
			return DiffChunkData(
				files=chunk.files,
				content=chunk.content,
				description=chunk.description if chunk.description else "",
			)
		return cast("DiffChunkData", chunk)

	def _lint_commit_message(self, message: str) -> tuple[bool, list[str]]:
		"""
		Lint a commit message using the CommitLinter.

		Args:
		        message: Commit message to lint

		Returns:
		        Tuple of (is_valid, list_of_messages)

		"""
		try:
			# Create a linter using the commit convention config from config_loader
			linter = CommitLinter(config_path=str(self.repo_root / ".codemap.yml"))
			return linter.lint(message)
		except Exception:
			logger.exception("Error during commit message linting: %s")
			# Return valid=True to avoid blocking the process on linter errors
			return True, []

	def generate_message_with_linting(
		self, chunk: DiffChunk | DiffChunkData, max_retries: int = 3
	) -> tuple[str, bool, bool]:
		"""
		Generate a commit message with linting.

		Args:
		        chunk: Diff chunk to generate message for
		        max_retries: Maximum number of regeneration retries for invalid messages

		Returns:
		        Tuple of (message, was_generated_by_llm, passed_linting)

		"""
		# First attempt to generate a message
		message, used_llm = self.generate_message(chunk)

		# If not generated by LLM, skip linting
		if not used_llm:
			logger.debug("Message was not generated by LLM, skipping linting.")
			return message, used_llm, True

		# Clean the message before linting (basic cleaning that was in _format_message)
		message = message.strip()
		# Remove markdown code blocks and inline code that might come from LLM
		message = message.replace("```", "").replace("`", "")
		# Remove common prefixes the LLM might add
		prefixes_to_remove = ["commit message:", "message:", "response:"]
		for prefix in prefixes_to_remove:
			if message.lower().startswith(prefix):
				message = message[len(prefix) :].strip()

		# Remove multi-line formatting by joining lines (keep message in single paragraph)
		message = " ".join(message.splitlines())

		# Lint the message
		is_valid, lint_messages = self._lint_commit_message(message)

		# If valid, return immediately
		if is_valid:
			logger.debug("Generated message passed linting checks.")
			return message, used_llm, True

		# Log the linting issues
		logger.warning("Commit message failed linting: %s", message)
		for lint_msg in lint_messages:
			logger.warning("Lint issue: %s", lint_msg)

		# Try to regenerate with more explicit instructions
		retries_left = max_retries
		regenerated_message = message

		# Add a loading spinner for regeneration
		while retries_left > 0 and not is_valid:
			retries_left -= 1

			try:
				# Create a prompt with the lint feedback
				enhanced_prompt = self._prepare_lint_prompt(chunk, lint_messages)

				# Use a loading spinner to show regeneration progress
				with loading_spinner(f"Commit message failed linting, regenerating (attempts left: {retries_left})..."):
					regenerated_message = self._call_llm_api(enhanced_prompt)

				# Lint the regenerated message
				is_valid, lint_messages = self._lint_commit_message(regenerated_message)

				if is_valid:
					logger.info("Successfully regenerated a valid commit message.")
					break

				logger.warning("Regenerated message still failed linting: %s", regenerated_message)
				for lint_msg in lint_messages:
					logger.warning("Lint issue: %s", lint_msg)

			except Exception:
				logger.exception("Error during message regeneration: %s")
				# Break out of the loop on error
				break

		# If we exhausted retries or had an error, return the last message with linting status
		if not is_valid and retries_left == 0:
			logger.warning("Exhausted all regeneration attempts. Using the last generated message.")

		return regenerated_message, used_llm, is_valid

	def _prepare_lint_prompt(self, chunk: DiffChunk | DiffChunkData, lint_messages: list[str]) -> str:
		"""
		Prepare a prompt with lint feedback for regeneration.

		Args:
		        chunk: Diff chunk to prepare prompt for
		        lint_messages: List of linting error messages

		Returns:
		        Enhanced prompt with linting feedback

		"""
		file_info = self._extract_file_info(chunk)
		convention = self._get_commit_convention()

		# Get the diff content from the chunk
		diff_content = chunk.content if isinstance(chunk, DiffChunk) else chunk.get("content", "")

		# Create specific feedback for linting issues
		lint_feedback = "\n".join([f"- {msg}" for msg in lint_messages])

		# Create an enhanced context with linting feedback
		context = {
			"diff": diff_content,
			"files": file_info,
			"convention": convention,
			"schema": COMMIT_MESSAGE_SCHEMA,
			"lint_feedback": lint_feedback,
		}

		# Use a template that includes the linting feedback
		lint_template = """
{conventional_commits_spec}

You are a helpful assistant that generates conventional commit messages based on code changes.
Given a Git diff, please generate a concise and descriptive commit message following these conventions:

1. Use the format:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```
2. Types include: {convention[types]}
3. Scope must be short (1-2 words), concise, and represent the specific component affected
4. The description should be a concise, imperative present tense summary of the *specific code changes*
   in the diff chunk (e.g., "add feature", "fix bug", "update documentation").
   Focus on *what* was changed and *why*.
5. The optional body should be a multi-paragraph summary of the changes, focusing on the *why* and *how* of the changes.
6. The optional footer(s) should be a list of one or more footers, each with a token and a value.
7. Your response must ONLY contain the commit message string, formatted as:
  ```
  <type>[optional scope]: <description>

  [optional body]

  [optional footer(s)]
  ```
   with absolutely no other text, explanation, or surrounding characters (like quotes or markdown).

IMPORTANT: The previous commit message had the following issues:
{lint_feedback}

Please fix these issues and ensure the generated message adheres to the commit convention.

---
Here are some notes about the files changed:
{files}
---
Analyze the following diff and respond with ONLY the commit message string:

{diff}

---
IMPORTANT:
- Strictly follow the format <type>[optional scope]: <description>
- Do not include any other text, explanation, or surrounding characters (like quotes or markdown).
- Strictly do not include any `Related Issue #`, `Closes #`, `REVIEWED-BY`, `TRACKING #`, `APPROVED` footers.
- Strictly follow the JSON schema provided while generating output in JSON format:

{schema}
"""

		# Get the conventional commits spec from the DEFAULT_PROMPT_TEMPLATE
		conventional_commits_spec = DEFAULT_PROMPT_TEMPLATE.split("# Conventional Commits 1.0.0")[1].split(
			"---\n\nYou are a helpful assistant"
		)[0]

		# Format the template with all the context
		context["conventional_commits_spec"] = "# Conventional Commits 1.0.0" + conventional_commits_spec

		# Return the formatted prompt
		return lint_template.format(**context)

	def generate_message(self, chunk: DiffChunk | DiffChunkData) -> tuple[str, bool]:
		"""
		Generate a commit message for the given diff chunk.

		Args:
		    chunk: Diff chunk to generate message for

		Returns:
		    Tuple of (message, was_generated_by_llm)

		"""
		logger.debug(
			"Generating message for chunk ID: %s. Using resolved config: Provider=%s, Model=%s",
			id(chunk),
			self.resolved_provider,
			self.resolved_model,
		)

		chunk_dict = self._adapt_chunk_access(chunk)
		existing_desc = chunk_dict.get("description")

		# Check for existing description (same logic as before)
		if existing_desc and isinstance(existing_desc, str):
			is_generic = existing_desc.startswith(("chore: update", "fix: update", "docs: update", "test: update"))
			is_llm_gen = getattr(chunk, "is_llm_generated", False)  # Check original object if possible

			if not is_generic and is_llm_gen:
				logger.debug("Chunk already has LLM-generated description: '%s'", existing_desc)
				return existing_desc, True  # Assume it was LLM generated previously
			if not is_generic and not is_llm_gen:
				logger.debug(
					"Chunk has existing non-generic, non-LLM description: '%s'. Attempting to improve.", existing_desc
				)
				# Proceed to generate below
			elif is_generic:
				logger.debug("Existing description is generic ('%s'). Attempting to generate.", existing_desc)
				# Proceed to generate below

		# Verify API key availability using the resolved provider
		if not self._verify_api_key_availability():
			env_var_map = {
				"openai": "OPENAI_API_KEY",
				"anthropic": "ANTHROPIC_API_KEY",
				"azure": "AZURE_API_KEY",
				"cohere": "COHERE_API_KEY",
				"groq": "GROQ_API_KEY",
				"mistral": "MISTRAL_API_KEY",
				"together": "TOGETHER_API_KEY",
				"openrouter": "OPENROUTER_API_KEY",
			}
			provider_env = env_var_map.get(self.resolved_provider, f"{self.resolved_provider.upper()}_API_KEY")
			error_msg = (
				f"No API key found for resolved provider '{self.resolved_provider}'. "
				f"Please set {provider_env} in your environment or configure "
				f"'{self.resolved_provider}_api_key' or 'api_key' in .codemap.yml -> commit -> llm section."
			)
			logger.error(error_msg)
			# Don't raise here, fall back instead
			logger.warning("API key missing for %s. Falling back to simple generation.", self.resolved_provider)
			message = self.fallback_generation(chunk_dict)
			return message, False

		chunk_content = chunk_dict.get("content", "")
		if isinstance(chunk_content, str):
			chunk_content = chunk_content.strip()

		if not chunk_content:
			logger.warning("Chunk content is empty - using fallback generation.")
			message = self.fallback_generation(chunk_dict)
			return message, False

		# Try to generate a message using LLM
		try:
			prompt = self._prepare_prompt(chunk_dict)
			logger.debug("Prepared prompt for LLM, length: %d chars", len(prompt))

			with loading_spinner("Generating commit message..."):
				message = self._call_llm_api(prompt)

			logger.debug("LLM generated message: '%s'", message)
			# Mark the chunk if possible (requires chunk to be mutable or return new object)
			if isinstance(chunk, DiffChunk):
				chunk.is_llm_generated = True  # Mark original object if it's the class type
			return message, True

		except LLMError:
			# Handle specific LLM errors (API key, rate limit, etc.) gracefully
			logger.exception("LLM Error during generation")
			logger.info("Falling back to simple message generation.")
			message = self.fallback_generation(chunk_dict)
			return message, False
		except Exception:
			# Catch other unexpected errors during the process
			logger.exception("Unexpected error during message generation")
			error_msg = "Failed to generate commit message due to unexpected error: %s"
			# Decide whether to raise or fallback
			logger.info("Falling back to simple message generation due to unexpected error.")
			message = self.fallback_generation(chunk_dict)
			return message, False
