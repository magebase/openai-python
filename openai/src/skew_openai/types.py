"""
SKEW SDK Types
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


@dataclass
class TelemetryConfig:
    """Telemetry configuration"""
    enabled: bool = True
    include_prompts: bool = False
    include_responses: bool = False
    sample_rate: float = 1.0
    batch_size: int = 10
    flush_interval_seconds: float = 5.0
    endpoint: str = "https://api.skew.ai/v1/telemetry"


@dataclass
class ProxyConfig:
    """Proxy configuration"""
    enabled: bool = False
    fail_open: bool = True
    base_url: str = "https://api.skew.ai/v1/openai"
    timeout_seconds: float = 30.0


@dataclass
class SkewConfig:
    """Full SDK configuration"""
    api_key: str
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)


@dataclass
class TokenUsage:
    """Token usage stats"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class TelemetryRequest:
    """Telemetry request data"""
    request_id: str
    org_id: str
    endpoint: str
    model: str
    timestamp_start: str
    project_id: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


@dataclass
class TelemetryResponse:
    """Telemetry response data"""
    timestamp_end: str
    token_usage: TokenUsage
    cost_estimate_usd: float
    latency_ms: float
    error_class: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class TelemetryContext:
    """Telemetry context data"""
    sdk_language: str = "python"
    sdk_version: str = "1.0.0"
    openai_client_version: str = "unknown"
    call_lineage_id: Optional[str] = None
    prompt_hash: Optional[str] = None


@dataclass
class TelemetryPayload:
    """Complete telemetry payload"""
    request: TelemetryRequest
    response: TelemetryResponse
    context: TelemetryContext


# Model pricing (per 1M tokens)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "o1": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 3.0, "output": 12.0},
    "o3-mini": {"input": 1.1, "output": 4.4},
    "text-embedding-3-small": {"input": 0.02, "output": 0},
    "text-embedding-3-large": {"input": 0.13, "output": 0},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost estimate for a request"""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return (prompt_tokens + completion_tokens) * 0.00001
    
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
