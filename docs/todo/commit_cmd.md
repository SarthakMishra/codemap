# Commit Command Enhancement Plan

## Current Implementation Analysis

The current commit command implementation provides:

- Git diff-based analysis of changes
- Multiple splitting strategies (file, hunk, semantic)
- LLM-based commit message generation
- Interactive commit flow
- Multiple LLM provider support
- Configuration via `.codemap.yml`

### Key Limitations

1. **Git Diff Analysis Issues**
   - Relies on text-based git diff parsing
   - Struggles with files containing diff-like content
   - Limited semantic context understanding
   - Splits changes based on file structure rather than true code relationships

2. **Git Hook Integration**
   - Currently bypasses git hooks with `--no-verify` flag
   - No proper hook lifecycle management
   - Lacks fallback mechanisms when hooks fail

3. **Performance Bottlenecks**
   - Embedding generation is slow for large repositories
   - No caching mechanism for embeddings
   - Uses large-scale embedding models that may be overkill

4. **Scope Management**
   - Limited support for scope identification in larger repositories
   - No intelligent scope inference from repository structure
   - Insufficient handling of monorepo structures

5. **Commit Message Quality**
   - No validation against commit conventions
   - No automatic fix/regeneration when messages violate conventions
   - Inconsistent formatting and style

## Enhancement Plan

### 1. Integrate Processing Pipeline for Code Analysis

Move from git diff-based analysis to the improved semantic processing pipeline:

```python
# Example integration with ProcessingPipeline
def analyze_changes(repo_path: Path, files: list[str]) -> list[Chunk]:
    """Analyze changes using the processing pipeline instead of git diff.
    
    Args:
        repo_path: Repository path
        files: Changed files to analyze
        
    Returns:
        List of code chunks with semantic relationships
    """
    # Create pipeline with appropriate configuration
    pipeline = ProcessingPipeline(
        repo_path=repo_path,
        # Use lightweight embedding config for performance
        embedding_config=EmbeddingConfig(
            model_name="Qodo/Qodo-Embed-1-1.5B-light", 
            # Enable caching
            cache_dir=repo_path / ".codemap" / "cache" / "embeddings"
        ),
        # Only analyze specified files
        ignored_patterns={"**/*", "!**/{" + ",".join(files) + "}"}
    )
    
    # Process files
    pipeline.batch_process(files)
    
    # Get processed chunks with semantic relationships
    chunks = pipeline.get_processed_chunks()
    
    # Group related chunks
    return group_related_chunks(chunks)
```

**Benefits:**
- Handles all file types correctly, avoiding diff-related issues
- Better semantic understanding of code relationships
- More accurate grouping of related changes
- LSP-enhanced code analysis
- Handles multi-language repositories better

### 2. Improved Git Hook Integration

Create a robust hook handling system:

```python
def run_with_hooks(files: list[str], message: str) -> bool:
    """Run git commit with proper hook handling.
    
    Args:
        files: Files to commit
        message: Commit message
        
    Returns:
        Success status
    """
    # 1. Run pre-commit hooks
    pre_commit_result = run_hooks("pre-commit", files)
    if not pre_commit_result.success:
        # Show hook output to user
        console.print(f"[yellow]Pre-commit hooks failed:[/yellow]\n{pre_commit_result.output}")
        
        # Ask if user wants to continue
        continue_anyway = questionary.confirm(
            "Pre-commit hooks failed. Continue anyway?", 
            default=False
        ).ask()
        
        if not continue_anyway:
            return False
    
    # 2. Create commit
    commit_result = create_commit(files, message)
    if not commit_result.success:
        return False
    
    # 3. Run post-commit hooks
    post_commit_result = run_hooks("post-commit")
    if not post_commit_result.success:
        console.print(f"[yellow]Post-commit hooks failed:[/yellow]\n{post_commit_result.output}")
        # Post-commit failures don't invalidate the commit
    
    return True

def run_hooks(hook_type: str, files: list[str] | None = None) -> HookResult:
    """Run git hooks of specified type with proper environment.
    
    Args:
        hook_type: Type of hook to run
        files: Files context for the hook
        
    Returns:
        Hook execution result
    """
    # Get hook path
    hook_path = get_repo_root() / ".git" / "hooks" / hook_type
    
    if not hook_path.exists() or not os.access(hook_path, os.X_OK):
        # No hook or not executable
        return HookResult(success=True, output="")
    
    # Set up environment
    env = os.environ.copy()
    if files:
        env["GIT_FILES"] = " ".join(files)
    
    # Run the hook in a subprocess
    try:
        result = subprocess.run(
            [str(hook_path)],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        return HookResult(
            success=result.returncode == 0,
            output=result.stdout + result.stderr
        )
    except Exception as e:
        return HookResult(success=False, output=str(e))
```

**Configuration options in `.codemap.yml`:**

```yaml
commit:
  hooks:
    # Whether to run git hooks
    enabled: true
    # Whether to prompt to continue on hook failure
    prompt_on_failure: true
    # Whether to bypass hooks with --no-verify when explicitly requested
    allow_bypass: true
```

**Benefits:**
- Proper hook lifecycle management
- User-friendly handling of hook failures
- Configurable behavior

### 3. Performance Optimizations for Embeddings

Implement caching and lightweight models:

```python
class CachedEmbeddingGenerator:
    """Embedding generator with caching support."""
    
    def __init__(
        self,
        config: EmbeddingConfig,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the embedding generator.
        
        Args:
            config: Embedding configuration
            cache_dir: Directory for caching embeddings
        """
        self.config = config
        self.cache_dir = cache_dir
        self.model = self._load_model()
        self.cache = self._init_cache()
    
    def _get_cache_key(self, content: str, metadata: dict) -> str:
        """Generate a cache key for the content.
        
        Args:
            content: Code content to embed
            metadata: Metadata for the content
            
        Returns:
            Cache key string
        """
        # Use hash of content + metadata hash as cache key
        content_hash = hashlib.md5(content.encode()).hexdigest()
        metadata_hash = hashlib.md5(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        return f"{content_hash}_{metadata_hash}"
    
    def generate_embedding(self, content: str, metadata: dict) -> np.ndarray:
        """Generate embedding with caching.
        
        Args:
            content: Code content to embed
            metadata: Metadata for the content
            
        Returns:
            Embedding vector
        """
        # Check cache first
        cache_key = self._get_cache_key(content, metadata)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate embedding
        embedding = self.model.encode(content)
        
        # Update cache
        self.cache[cache_key] = embedding
        self._save_cache()
        
        return embedding
```

Add configuration in `.codemap.yml`:

```yaml
commit:
  embedding:
    # Model to use for embeddings (can be a lightweight variant)
    model: "Qodo/Qodo-Embed-1-1.5B-light"
    # Whether to enable caching
    cache_enabled: true
    # Max cache size in MB
    cache_size_mb: 100
    # Cache directory (relative to repo)
    cache_dir: ".codemap/cache/embeddings"
```

**Lighter model alternatives:**
- `all-MiniLM-L6-v2` (small but effective for code similarity)
- `thenlper/gte-small` (balanced size/performance)
- Custom-trained, quantized model specific for code

**Benefits:**
- Significantly faster repeated operations
- Reduced memory usage
- Better performance on large repositories

### 4. Enhanced Scope Management for Large Repositories

Improve scope identification and management:

```python
class ScopeResolver:
    """Intelligent scope resolver for repositories."""
    
    def __init__(self, repo_path: Path, config: dict) -> None:
        """Initialize the scope resolver.
        
        Args:
            repo_path: Repository path
            config: Configuration dictionary
        """
        self.repo_path = repo_path
        self.config = config
        self.scopes = self._load_scopes()
        
    def _load_scopes(self) -> dict[str, ScopeDefinition]:
        """Load scope definitions from configuration.
        
        Returns:
            Dictionary of scope definitions
        """
        # Get configured scopes
        configured_scopes = self.config.get("convention", {}).get("scopes", [])
        
        # Create scope map
        scopes = {}
        
        if configured_scopes:
            # Use explicitly configured scopes
            for scope in configured_scopes:
                if isinstance(scope, str):
                    # Simple scope name
                    scopes[scope] = ScopeDefinition(
                        name=scope,
                        patterns=[f"**/{scope}/**"],
                        description=f"Changes in {scope}"
                    )
                elif isinstance(scope, dict):
                    # Detailed scope definition
                    scopes[scope["name"]] = ScopeDefinition(
                        name=scope["name"],
                        patterns=scope.get("patterns", [f"**/{scope['name']}/**"]),
                        description=scope.get("description", f"Changes in {scope['name']}")
                    )
        else:
            # Infer scopes from repository structure
            scopes = self._infer_scopes_from_structure()
            
        return scopes
    
    def _infer_scopes_from_structure(self) -> dict[str, ScopeDefinition]:
        """Infer scopes from repository structure.
        
        Returns:
            Dictionary of inferred scope definitions
        """
        scopes = {}
        
        # Check for package.json (monorepo detection)
        if (self.repo_path / "package.json").exists():
            try:
                with open(self.repo_path / "package.json") as f:
                    package_data = json.load(f)
                    
                # Check for workspaces (yarn/npm monorepo)
                if "workspaces" in package_data:
                    workspaces = package_data["workspaces"]
                    if isinstance(workspaces, list):
                        for workspace in workspaces:
                            # Extract scope name from pattern
                            scope_name = workspace.split("/")[0].strip("*{}")
                            if scope_name:
                                scopes[scope_name] = ScopeDefinition(
                                    name=scope_name,
                                    patterns=[workspace],
                                    description=f"Changes in {scope_name} workspace"
                                )
            except (json.JSONDecodeError, IOError):
                pass
                
        # Check for top-level directories in src/
        src_dir = self.repo_path / "src"
        if src_dir.exists() and src_dir.is_dir():
            for item in src_dir.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    scopes[item.name] = ScopeDefinition(
                        name=item.name,
                        patterns=[f"src/{item.name}/**"],
                        description=f"Changes in {item.name} module"
                    )
                    
        return scopes
    
    def resolve_scope(self, files: list[str]) -> str:
        """Resolve the most appropriate scope for a set of files.
        
        Args:
            files: List of changed files
            
        Returns:
            Resolved scope name
        """
        if not files:
            return ""
            
        # Count matches for each scope
        scope_matches = {name: 0 for name in self.scopes}
        
        for file in files:
            for scope_name, scope_def in self.scopes.items():
                for pattern in scope_def.patterns:
                    if fnmatch.fnmatch(file, pattern):
                        scope_matches[scope_name] += 1
                        break
                        
        # Find the scope with the most matches
        best_scope = max(scope_matches.items(), key=lambda x: x[1])
        
        # Return the best scope if it has any matches
        if best_scope[1] > 0:
            return best_scope[0]
            
        # Fallback: Try to extract scope from common directory
        try:
            common_dir = os.path.commonpath(files)
            if common_dir and common_dir != ".":
                # Extract the first directory component
                parts = Path(common_dir).parts
                if len(parts) > 0:
                    return parts[0]
        except ValueError:
            # Commonpath fails if files have no common directory
            pass
            
        return ""
```

Add configuration in `.codemap.yml`:

```yaml
commit:
  convention:
    # Detailed scope definitions
    scopes:
      - name: ui
        patterns: ["src/ui/**", "src/components/**"]
        description: "UI components"
      
      - name: api
        patterns: ["src/api/**", "server/**"]
        description: "API endpoints"
      
      # For monorepos, define package scopes
      - name: pkg1
        patterns: ["packages/pkg1/**"]
        
      - name: pkg2
        patterns: ["packages/pkg2/**"]
    
    # Default scope for unmatched files
    default_scope: "core"
    
    # Monorepo settings
    monorepo:
      enabled: true
      root_as_scope: false  # Whether to use root package name as a scope
```

**Benefits:**
- Better scope accuracy for complex repositories
- Support for monorepo structures
- Intelligent scope inference
- Customizable scope definitions

### 5. Commit Message Validation and Auto-correction

Implement validation against commit conventions:

```python
class CommitMessageValidator:
    """Validates and fixes commit messages against conventions."""
    
    def __init__(self, config: dict) -> None:
        """Initialize the validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.convention = config.get("convention", {})
        self.types = self.convention.get("types", [])
        self.max_length = self.convention.get("max_length", 72)
        
    def validate(self, message: str) -> tuple[bool, list[str]]:
        """Validate a commit message against conventions.
        
        Args:
            message: Commit message to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check format (type(scope): description)
        pattern = r"^([a-z]+)(\([a-z0-9_\-/]+\))?: .+"
        if not re.match(pattern, message):
            issues.append("Message does not follow the format: type(scope): description")
            
        # Check type
        match = re.match(r"^([a-z]+)", message)
        if match:
            commit_type = match.group(1)
            if commit_type not in self.types:
                issues.append(f"Invalid type '{commit_type}'. Must be one of: {', '.join(self.types)}")
                
        # Check length
        if len(message) > self.max_length:
            issues.append(f"Message exceeds maximum length of {self.max_length} characters")
            
        # Check for imperative mood
        description_match = re.match(r"^[a-z]+(\([a-z0-9_\-/]+\))?: (.+)$", message)
        if description_match:
            description = description_match.group(2)
            first_word = description.split()[0] if description.split() else ""
            
            # Crude check for common non-imperative verbs
            non_imperative = ["added", "adding", "fixed", "fixing", "updated", "updating"]
            if first_word.lower() in non_imperative:
                issues.append(f"Description should use imperative mood (e.g., 'add' not '{first_word}')")
                
        return len(issues) == 0, issues
        
    def fix(self, message: str, chunk_context: DiffChunk) -> str:
        """Fix common issues in commit messages.
        
        Args:
            message: Message to fix
            chunk_context: Context information for fixing
            
        Returns:
            Fixed message
        """
        # Extract parts
        type_match = re.match(r"^([a-z]+)", message)
        scope_match = re.match(r"^[a-z]+\(([a-z0-9_\-/]+)\)", message)
        desc_match = re.match(r"^[a-z]+(?:\([a-z0-9_\-/]+\))?: (.+)$", message)
        
        commit_type = type_match.group(1) if type_match else "chore"
        scope = scope_match.group(1) if scope_match else ""
        description = desc_match.group(1) if desc_match else message
        
        # Fix type if invalid
        if commit_type not in self.types:
            commit_type = self._infer_type_from_context(chunk_context)
            
        # Fix non-imperative mood
        description = self._fix_imperative_mood(description)
        
        # Reconstruct message
        if scope:
            fixed_message = f"{commit_type}({scope}): {description}"
        else:
            fixed_message = f"{commit_type}: {description}"
            
        # Ensure length limit
        if len(fixed_message) > self.max_length:
            # Truncate description
            avail_length = self.max_length - len(f"{commit_type}({scope}): ")
            description = description[:avail_length - 3] + "..."
            
            if scope:
                fixed_message = f"{commit_type}({scope}): {description}"
            else:
                fixed_message = f"{commit_type}: {description}"
                
        return fixed_message
        
    def _infer_type_from_context(self, chunk: DiffChunk) -> str:
        """Infer commit type from chunk context.
        
        Args:
            chunk: Diff chunk context
            
        Returns:
            Inferred commit type
        """
        # Basic type inference logic
        files = chunk.files
        content = chunk.content
        
        # Check for tests
        if any("test" in f or f.startswith("tests/") for f in files):
            return "test"
            
        # Check for docs
        if any(f.endswith(".md") or f.startswith("docs/") for f in files):
            return "docs"
            
        # Check content for clues
        if content:
            if "fix" in content.lower() or "bug" in content.lower():
                return "fix"
            if "feature" in content.lower() or "feat" in content.lower():
                return "feat"
                
        # Default
        return "chore"
        
    def _fix_imperative_mood(self, description: str) -> str:
        """Fix non-imperative mood in description.
        
        Args:
            description: Description to fix
            
        Returns:
            Fixed description
        """
        # Simple replacements for common non-imperative forms
        word_map = {
            "added": "add",
            "adding": "add",
            "fixed": "fix",
            "fixing": "fix",
            "updated": "update",
            "updating": "update",
            "removed": "remove",
            "removing": "remove",
            "implemented": "implement",
            "implementing": "implement",
            "changed": "change",
            "changing": "change",
        }
        
        words = description.split()
        if words and words[0].lower() in word_map:
            words[0] = word_map[words[0].lower()]
            return " ".join(words)
            
        return description
```

**Add validation hooks to message generation:**

```python
def generate_and_validate_message(chunk: DiffChunk, generator: MessageGenerator) -> str:
    """Generate, validate, and fix commit message if needed.
    
    Args:
        chunk: Diff chunk
        generator: Message generator
        
    Returns:
        Valid commit message
    """
    # Generate initial message
    message, _ = generator.generate_message(chunk)
    
    # Create validator
    validator = CommitMessageValidator(generator._config_loader.config)
    
    # Validate message
    is_valid, issues = validator.validate(message)
    
    # If valid, return as is
    if is_valid:
        return message
        
    # For minor issues, fix automatically
    if len(issues) <= 2 and not any("format" in issue for issue in issues):
        fixed_message = validator.fix(message, chunk)
        return fixed_message
        
    # For major issues, regenerate with more explicit instructions
    logger.warning("Generated message has validation issues: %s", issues)
    logger.info("Regenerating message with stricter format guidance")
    
    # Update prompt to emphasize format
    enhanced_prompt = f"""
    The previous commit message had these issues:
    {chr(10).join(f"- {issue}" for issue in issues)}
    
    Please generate a new commit message that strictly follows the format:
    <type>(<scope>): <description>
    
    Valid types: {', '.join(validator.types)}
    Make sure the description starts with an imperative verb.
    Total length must be under {validator.max_length} characters.
    """
    
    # Regenerate with enhanced prompt
    regenerated_message, _ = generator.generate_message_with_custom_prompt(
        chunk, 
        enhanced_prompt
    )
    
    # Validate again
    is_valid, remaining_issues = validator.validate(regenerated_message)
    
    # If still invalid, force fix
    if not is_valid:
        return validator.fix(regenerated_message, chunk)
        
    return regenerated_message
```

Add configuration in `.codemap.yml`:

```yaml
commit:
  validation:
    # Whether to enable validation
    enabled: true
    # Whether to auto-fix messages
    auto_fix: true
    # Whether to regenerate invalid messages
    regenerate_on_failure: true
    # Whether to apply strict validation
    strict_mode: false
```

**Benefits:**
- Consistent commit message quality
- Automatic fixing of common issues
- Improved adherence to project conventions
- Better git history

## Implementation Roadmap

### Phase 1: Processing Pipeline Integration
1. Create adapter between processor pipeline and commit workflow
2. Implement chunk grouping based on semantic relationships
3. Replace diff splitter with semantic processor
4. Update tests to work with new processing approach

### Phase 2: Performance Optimization
1. Implement embedding caching
2. Add support for lightweight models
3. Create configuration options for performance tuning
4. Add progress indicators for long-running operations

### Phase 3: Git Hook Handling
1. Develop hook execution framework
2. Add proper pre/post hook lifecycle
3. Create hook failure handling mechanisms
4. Update configuration options for hook behavior

### Phase 4: Scope Management
1. Implement ScopeResolver class
2. Add monorepo detection and support
3. Enhance scope inference from repository structure
4. Update configuration schema for scope definitions

### Phase 5: Commit Message Validation
1. Create CommitMessageValidator class
2. Integrate validation into generation workflow
3. Implement auto-correction for common issues
4. Add regeneration logic for invalid messages

## Expected Outcomes

The enhanced commit command will:

1. **Improve Reliability**
   - Eliminate issues with files containing diff-like content
   - Properly handle git hooks
   - Provide better error recovery

2. **Increase Performance**
   - Faster embedding generation through caching
   - Reduced memory usage with lightweight models
   - More responsive UI for large repositories

3. **Enhance User Experience**
   - More intelligent scope handling
   - Higher quality commit messages
   - Consistent formatting and style

4. **Support Larger Codebases**
   - Better monorepo support
   - Improved handling of complex repository structures
   - More efficient processing of large changes
