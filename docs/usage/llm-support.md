# LLM Provider Support

CodeMap supports multiple LLM providers through [LiteLLM](https://github.com/BerriAI/litellm).

You can specify the desired model using the `--model` option in the `commit` and `pr` commands, or set a default in the [Configuration](configuration.md).

## Examples

```bash
# Using OpenAI (default)
codemap commit --model openai/gpt-4o-mini
# Or using the alias:
cm commit --model openai/gpt-4o-mini

# Using Anthropic
codemap commit --model anthropic/claude-3-sonnet-20240229

# Using Groq (recommended for speed)
codemap commit --model groq/llama-3.1-8b-instant

# Using OpenRouter
codemap commit --model openrouter/meta-llama/llama-3-8b-instruct
```

## Environment Variables

The following environment variables are needed to authenticate with the respective LLM providers. You can set these in your system environment or place them in a `.env` or `.env.local` file in your project root.

```env
# LLM Provider API Keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here

# Optional: Custom API Base URLs (for proxies or self-hosted models)
OPENAI_API_BASE=your_custom_url
ANTHROPIC_API_BASE=your_custom_url
# ... add others as needed ...
``` 