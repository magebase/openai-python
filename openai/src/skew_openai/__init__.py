"""
SKEW SDK for OpenAI - Telemetry and cost optimization

Core Promise:
"SKEW will never touch user traffic unless explicitly configured to do so."
"""

from .wrapper import skew_wrap
from .telemetry import TelemetryClient
from .types import SkewConfig, TelemetryPayload

__version__ = "1.0.0"
__all__ = ["skew_wrap", "TelemetryClient", "SkewConfig", "TelemetryPayload"]
