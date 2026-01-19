"""
langmesh OpenAI Wrapper

Wraps the OpenAI client with telemetry and optional proxy support.

Invariants:
- No behavior changes without explicit configuration
- Telemetry is async and non-blocking
- SDK errors never break user code
- Proxy only activates when explicitly enabled
"""

import functools
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TypeVar

from .types import (
    langmeshConfig,
    TelemetryConfig,
    ProxyConfig,
    TelemetryPayload,
    TelemetryRequest,
    TelemetryResponse,
    TelemetryContext,
    TokenUsage,
    calculate_cost,
)
from .telemetry import TelemetryClient, generate_request_id, hash_prompt

T = TypeVar("T")

SDK_VERSION = "1.0.0"


class langmeshWrapper:
    """Wrapper that adds telemetry and proxy support to OpenAI client"""
    
    def __init__(
        self,
        client: Any,
        api_key: str,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        telemetry_enabled: bool = True,
        include_prompts: bool = False,
        sample_rate: float = 1.0,
        proxy_enabled: bool = False,
        fail_open: bool = True,
    ):
        self._client = client
        self._config = langmeshConfig(
            api_key=api_key,
            org_id=org_id,
            project_id=project_id,
            telemetry=TelemetryConfig(
                enabled=telemetry_enabled,
                include_prompts=include_prompts,
                sample_rate=sample_rate,
            ),
            proxy=ProxyConfig(
                enabled=proxy_enabled,
                fail_open=fail_open,
            ),
        )
        self._telemetry_client = TelemetryClient(api_key, self._config.telemetry)
        self._proxy_active = proxy_enabled
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped client, wrapping namespaces"""
        attr = getattr(self._client, name)
        
        if hasattr(attr, "__call__"):
            return self._wrap_method(attr, name)
        elif hasattr(attr, "__dict__"):
            return _NamespaceWrapper(attr, self)
        
        return attr
    
    def _wrap_method(self, method: Callable, method_name: str) -> Callable:
        """Wrap a method with telemetry collection"""
        @functools.wraps(method)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            request_id = generate_request_id()
            timestamp_start = datetime.now(timezone.utc).isoformat()
            
            # Add proxy headers if enabled
            if self._proxy_active:
                headers = kwargs.get("extra_headers", {})
                headers.update({
                    "X-langmesh-API-Key": self._config.api_key,
                    "X-langmesh-Request-ID": request_id,
                    "X-langmesh-Org-ID": self._config.org_id or "",
                })
                kwargs["extra_headers"] = headers
            
            try:
                result = method(*args, **kwargs)
                
                # Send telemetry asynchronously
                self._send_telemetry_async(
                    request_id,
                    timestamp_start,
                    method_name,
                    kwargs,
                    result,
                    None,
                )
                
                return result
            except Exception as e:
                # Send error telemetry
                self._send_telemetry_async(
                    request_id,
                    timestamp_start,
                    method_name,
                    kwargs,
                    None,
                    e,
                )
                raise
        
        return wrapped
    
    def _send_telemetry_async(
        self,
        request_id: str,
        timestamp_start: str,
        method_name: str,
        params: dict,
        result: Any,
        error: Optional[Exception],
    ) -> None:
        """Send telemetry in background thread"""
        def send():
            try:
                self._build_and_send_telemetry(
                    request_id, timestamp_start, method_name, params, result, error
                )
            except Exception:
                pass  # Silent drop
        
        thread = threading.Thread(target=send, daemon=True)
        thread.start()
    
    def _build_and_send_telemetry(
        self,
        request_id: str,
        timestamp_start: str,
        method_name: str,
        params: dict,
        result: Any,
        error: Optional[Exception],
    ) -> None:
        """Build and send telemetry payload"""
        timestamp_end = datetime.now(timezone.utc).isoformat()
        
        start_time = datetime.fromisoformat(timestamp_start.replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(timestamp_end.replace("Z", "+00:00"))
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        model = params.get("model", "unknown")
        
        # Extract token usage
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        if result and hasattr(result, "usage") and result.usage:
            prompt_tokens = getattr(result.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(result.usage, "completion_tokens", 0) or 0
            total_tokens = getattr(result.usage, "total_tokens", 0) or prompt_tokens + completion_tokens
        
        request = TelemetryRequest(
            request_id=request_id,
            org_id=self._config.org_id or "",
            project_id=self._config.project_id,
            endpoint=self._method_to_endpoint(method_name),
            model=model,
            timestamp_start=timestamp_start,
            max_tokens=params.get("max_tokens"),
            temperature=params.get("temperature"),
        )
        
        response = TelemetryResponse(
            timestamp_end=timestamp_end,
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            cost_estimate_usd=calculate_cost(model, prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
        )
        
        if error:
            response.error_class = type(error).__name__
            response.error_message = str(error)
        
        context = TelemetryContext(
            sdk_language="python",
            sdk_version=SDK_VERSION,
        )
        
        # Add prompt hash if configured
        if not self._config.telemetry.include_prompts and "messages" in params:
            context.prompt_hash = hash_prompt(json.dumps(params["messages"]))
        
        payload = TelemetryPayload(request=request, response=response, context=context)
        self._telemetry_client.submit(payload)
    
    def _method_to_endpoint(self, method_name: str) -> str:
        """Map method name to OpenAI endpoint"""
        mapping = {
            "create": "chat.completions",
            "generate": "images.generate",
            "transcriptions": "audio.transcriptions",
            "translations": "audio.translations",
            "embeddings": "embeddings",
            "moderations": "moderations",
        }
        return mapping.get(method_name, "chat.completions")
    
    def unwrap(self) -> Any:
        """Get the original OpenAI client"""
        return self._client
    
    def pause_telemetry(self) -> None:
        """Pause telemetry collection"""
        self._telemetry_client.pause()
    
    def resume_telemetry(self) -> None:
        """Resume telemetry collection"""
        self._telemetry_client.resume()
    
    def flush_telemetry(self) -> None:
        """Flush pending telemetry"""
        self._telemetry_client.flush()
    
    def is_proxy_active(self) -> bool:
        """Check if proxy is currently active"""
        return self._proxy_active


class _NamespaceWrapper:
    """Wrapper for OpenAI client namespaces (chat, completions, etc.)"""
    
    def __init__(self, namespace: Any, parent: langmeshWrapper):
        self._namespace = namespace
        self._parent = parent
    
    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._namespace, name)
        
        if hasattr(attr, "__call__"):
            return self._parent._wrap_method(attr, name)
        elif hasattr(attr, "__dict__"):
            return _NamespaceWrapper(attr, self._parent)
        
        return attr


def langmesh_wrap(
    client: T,
    api_key: str,
    org_id: Optional[str] = None,
    project_id: Optional[str] = None,
    telemetry_enabled: bool = True,
    include_prompts: bool = False,
    sample_rate: float = 1.0,
    proxy_enabled: bool = False,
    fail_open: bool = True,
) -> T:
    """
    Wrap an OpenAI client with langmesh telemetry and optional proxy.
    
    Args:
        client: The OpenAI client to wrap
        api_key: langmesh API key
        org_id: Organization ID (optional, auto-detected from key)
        project_id: Project ID for grouping (optional)
        telemetry_enabled: Enable telemetry (default: True)
        include_prompts: Include raw prompts in telemetry (default: False)
        sample_rate: Telemetry sample rate 0-1 (default: 1.0)
        proxy_enabled: Enable proxy routing (default: False)
        fail_open: Fail open on proxy errors (default: True)
    
    Returns:
        Wrapped client that can be used exactly like the original
    """
    return langmeshWrapper(  # type: ignore
        client,
        api_key=api_key,
        org_id=org_id,
        project_id=project_id,
        telemetry_enabled=telemetry_enabled,
        include_prompts=include_prompts,
        sample_rate=sample_rate,
        proxy_enabled=proxy_enabled,
        fail_open=fail_open,
    )
