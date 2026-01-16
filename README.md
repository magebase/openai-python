# skew-openai

Drop-in replacement for the OpenAI Python client with automatic cost optimization and telemetry.

## Installation

```bash
pip install skew-openai
```

## Usage

Change one line of code:

```python
# Before
from openai import OpenAI

# After
from skew_openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Everything works exactly the same!
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

That's it. No configuration needed.

## What It Does

### Telemetry (Always On)
- Tracks token usage, cost, and latency
- Privacy-preserving (no prompts sent by default)
- Zero performance impact (async)
- Never breaks your app (fail-safe)

### Cost Optimization (Opt-In)
When you enable policies in the SKEW dashboard:
- Automatic model downgrading for simple queries
- Retry storm suppression
- Semantic caching
- Token optimization

## Configuration

### Required
```bash
export SKEW_API_KEY=sk_live_...  # Get from dashboard.skew.ai
```

### Optional
```bash
export SKEW_PROXY_ENABLED=true  # Enable when policies require routing
export SKEW_BASE_URL=https://api.skew.ai/v1/openai  # Custom proxy URL
```

## Async Support

```python
from skew_openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Migration Path

1. **Install** - `pip install skew-openai`
2. **Replace import** - Change `from openai` to `from skew_openai`
3. **Set API key** - `export SKEW_API_KEY=sk_live_...`
4. **See savings** - Visit dashboard.skew.ai
5. **Enable policies** - When ready, `export SKEW_PROXY_ENABLED=true`

## Guarantees

✅ Drop-in replacement - works identically  
✅ No behavior changes without opt-in  
✅ Fail-safe - errors don't break your app  
✅ Reversible - uninstall anytime  
✅ Privacy-first - no prompts sent by default  

## License

MIT
