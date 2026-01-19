"""
langmesh SDK for OpenAI - Telemetry and cost optimization

Core Promise:
"langmesh will never touch user traffic unless explicitly configured to do so."
"""

from .wrapper import langmesh_wrap
from .telemetry import TelemetryClient
from .types import langmeshConfig, TelemetryPayload

__version__ = "1.0.0"
__all__ = ["langmesh_wrap", "TelemetryClient", "langmeshConfig", "TelemetryPayload"]
