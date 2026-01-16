"""
SKEW-wrapped OpenAI client - Drop-in replacement

Usage:
    # Before
    from openai import OpenAI
    
    # After
    from skew_openai import OpenAI
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # Works exactly the same!
"""

import os
import time
import json
from typing import Any, Optional, Dict, List
from threading import Timer
import httpx
from openai import OpenAI as OriginalOpenAI, AsyncOpenAI as OriginalAsyncOpenAI
from openai.types.chat import ChatCompletion

__version__ = "1.0.0"

# Configuration from environment
SKEW_API_KEY = os.environ.get("SKEW_API_KEY", "")
SKEW_TELEMETRY_ENDPOINT = os.environ.get(
    "SKEW_TELEMETRY_ENDPOINT", "https://api.skew.ai/v1/telemetry"
)
SKEW_PROXY_ENABLED = os.environ.get("SKEW_PROXY_ENABLED", "").lower() == "true"
SKEW_BASE_URL = os.environ.get("SKEW_BASE_URL", "https://api.skew.ai/v1/openai")


class OpenAI(OriginalOpenAI):
    """SKEW-wrapped OpenAI client with telemetry and optional proxy routing"""

    def __init__(self, *args, **kwargs):
        self.skew_api_key = SKEW_API_KEY
        self.telemetry_enabled = bool(SKEW_API_KEY)
        self.telemetry_buffer: List[Dict[str, Any]] = []
        self.flush_timer: Optional[Timer] = None

        # If proxy is enabled, route through SKEW
        if SKEW_PROXY_ENABLED and SKEW_API_KEY:
            kwargs["base_url"] = SKEW_BASE_URL
            if "default_headers" not in kwargs:
                kwargs["default_headers"] = {}
            kwargs["default_headers"]["X-SKEW-API-Key"] = SKEW_API_KEY
            original_api_key = kwargs.get("api_key", "")
            kwargs["default_headers"]["X-SKEW-Original-API-Key"] = original_api_key

        super().__init__(*args, **kwargs)

        if self.telemetry_enabled:
            self._start_telemetry()

        # Wrap chat.completions.create
        self._wrap_chat_completions()

    def _wrap_chat_completions(self):
        """Wrap the chat completions create method with telemetry"""
        original_create = self.chat.completions.create

        def wrapped_create(*args, **kwargs):
            start_time = time.time()
            request_id = f"req_{int(time.time() * 1000)}_{os.urandom(4).hex()}"

            try:
                result: ChatCompletion = original_create(*args, **kwargs)
                end_time = time.time()

                # Collect telemetry
                if self.telemetry_enabled:
                    self._record_telemetry(
                        {
                            "request_id": request_id,
                            "timestamp_start": time.strftime(
                                "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(start_time)
                            ),
                            "timestamp_end": time.strftime(
                                "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(end_time)
                            ),
                            "model": kwargs.get("model", args[0] if args else "unknown"),
                            "endpoint": "chat.completions",
                            "latency_ms": int((end_time - start_time) * 1000),
                            "token_usage": {
                                "prompt_tokens": result.usage.prompt_tokens if result.usage else 0,
                                "completion_tokens": result.usage.completion_tokens if result.usage else 0,
                                "total_tokens": result.usage.total_tokens if result.usage else 0,
                            },
                            "cost_estimate_usd": self._estimate_cost(
                                kwargs.get("model", "gpt-4o"),
                                result.usage.prompt_tokens if result.usage else 0,
                                result.usage.completion_tokens if result.usage else 0,
                            ),
                            "status": "success",
                        }
                    )

                return result

            except Exception as error:
                end_time = time.time()

                # Record error telemetry
                if self.telemetry_enabled:
                    self._record_telemetry(
                        {
                            "request_id": request_id,
                            "timestamp_start": time.strftime(
                                "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(start_time)
                            ),
                            "timestamp_end": time.strftime(
                                "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(end_time)
                            ),
                            "model": kwargs.get("model", "unknown"),
                            "endpoint": "chat.completions",
                            "latency_ms": int((end_time - start_time) * 1000),
                            "token_usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                            },
                            "cost_estimate_usd": 0,
                            "status": "error",
                            "error_class": type(error).__name__,
                            "error_message": str(error),
                        }
                    )

                raise

        self.chat.completions.create = wrapped_create

    def _record_telemetry(self, event: Dict[str, Any]):
        """Add telemetry event to buffer"""
        self.telemetry_buffer.append(event)

        if len(self.telemetry_buffer) >= 10:
            self._flush_telemetry()

    def _flush_telemetry(self):
        """Send telemetry batch to SKEW"""
        if not self.telemetry_buffer:
            return

        batch = self.telemetry_buffer[:]
        self.telemetry_buffer = []

        try:
            httpx.post(
                SKEW_TELEMETRY_ENDPOINT,
                json={"events": batch},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.skew_api_key}",
                },
                timeout=5.0,
            )
        except Exception:
            # Silent drop - telemetry must never break user's app
            pass

    def _start_telemetry(self):
        """Start periodic telemetry flush"""

        def flush_periodically():
            self._flush_telemetry()
            self.flush_timer = Timer(5.0, flush_periodically)
            self.flush_timer.daemon = True
            self.flush_timer.start()

        self.flush_timer = Timer(5.0, flush_periodically)
        self.flush_timer.daemon = True
        self.flush_timer.start()

    def _estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Estimate request cost"""
        pricing = {
            "gpt-4o": {"input": 2.5, "output": 10.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        }

        model_pricing = pricing.get(model, {"input": 0.01, "output": 0.01})
        return (prompt_tokens / 1_000_000) * model_pricing["input"] + (
            completion_tokens / 1_000_000
        ) * model_pricing["output"]


# Async version
class AsyncOpenAI(OriginalAsyncOpenAI):
    """Async SKEW-wrapped OpenAI client"""

    def __init__(self, *args, **kwargs):
        self.skew_api_key = SKEW_API_KEY
        self.telemetry_enabled = bool(SKEW_API_KEY)
        self.telemetry_buffer: List[Dict[str, Any]] = []

        if SKEW_PROXY_ENABLED and SKEW_API_KEY:
            kwargs["base_url"] = SKEW_BASE_URL
            if "default_headers" not in kwargs:
                kwargs["default_headers"] = {}
            kwargs["default_headers"]["X-SKEW-API-Key"] = SKEW_API_KEY
            original_api_key = kwargs.get("api_key", "")
            kwargs["default_headers"]["X-SKEW-Original-API-Key"] = original_api_key

        super().__init__(*args, **kwargs)


__all__ = ["OpenAI", "AsyncOpenAI"]
