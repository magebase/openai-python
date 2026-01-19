# langmesh-openai

langmesh SDK for OpenAI - Telemetry and cost optimization without touching your traffic.

## Installation

```bash
pip install langmesh-openai
```

## Quick Start

**Start with the SDK. It observes only — langmesh will not touch your traffic.**

```python
from openai import OpenAI
from langmesh_openai import langmesh_wrap

# Your existing OpenAI client
openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Wrap with langmesh for telemetry (no behavior changes)
client = langmesh_wrap(openai, api_key=os.environ["langmesh_API_KEY"])

# Use exactly as before
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## SDK Guarantees

- ✅ **Zero behavior changes** - Your code works exactly as before
- ✅ **No request modification** - Requests pass through unchanged
- ✅ **Async telemetry** - Never blocks your requests
- ✅ **Fail-safe** - SDK errors never break your app
- ✅ **No raw prompts** - Only metadata sent by default
- ✅ **Removable in minutes** - Just unwrap the client

## Proxy Mode (Advanced)

When you enable policies that require enforcement, the SDK automatically routes through langmesh's proxy:

```python
client = langmesh_wrap(openai,
    api_key=os.environ["langmesh_API_KEY"],
    proxy_enabled=True  # Enable when policies require it
)
```

## Configuration

```python
client = langmesh_wrap(
    openai,
    api_key=os.environ["langmesh_API_KEY"],

    # Optional: Organization ID (auto-detected from key)
    org_id="org_xxx",

    # Optional: Project ID for grouping
    project_id="my-project",

    # Telemetry options
    telemetry_enabled=True,  # Default: True
    include_prompts=False,   # Default: False (privacy)
    sample_rate=1.0,         # Default: 1.0 (all requests)

    # Proxy options (for enforcement)
    proxy_enabled=False,     # Default: False
    fail_open=True,          # Default: True
)
```

## Unwrapping

```python
# Get the original OpenAI client
original_client = client.unwrap()

# Or pause telemetry
client.pause_telemetry()
# ... do something ...
client.resume_telemetry()
```

## License

MIT
