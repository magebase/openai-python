"""
langmesh Telemetry Client

Async, non-blocking telemetry submission
"""

import asyncio
import hashlib
import threading
import time
import uuid
from dataclasses import asdict
from typing import List, Optional
import json

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    import urllib.request
    import urllib.error

from .types import TelemetryConfig, TelemetryPayload


class TelemetryClient:
    """Telemetry client with batching and async submission"""
    
    def __init__(self, api_key: str, config: Optional[TelemetryConfig] = None):
        self.api_key = api_key
        self.config = config or TelemetryConfig()
        self._buffer: List[TelemetryPayload] = []
        self._buffer_lock = threading.Lock()
        self._paused = False
        self._flush_timer: Optional[threading.Timer] = None
        
        if self.config.enabled:
            self._start_flush_timer()
    
    def submit(self, payload: TelemetryPayload) -> None:
        """Submit telemetry - never blocks, never raises"""
        if not self.config.enabled or self._paused:
            return
        
        # Apply sampling
        import random
        if random.random() > self.config.sample_rate:
            return
        
        with self._buffer_lock:
            self._buffer.append(payload)
            
            if len(self._buffer) >= self.config.batch_size:
                self._flush_async()
    
    def flush(self) -> None:
        """Flush buffered telemetry synchronously"""
        with self._buffer_lock:
            if not self._buffer:
                return
            
            batch = self._buffer[:]
            self._buffer.clear()
        
        try:
            self._send_batch(batch)
        except Exception:
            # Silent drop - telemetry must never affect user
            pass
    
    def _flush_async(self) -> None:
        """Flush in background thread"""
        thread = threading.Thread(target=self.flush, daemon=True)
        thread.start()
    
    def _send_batch(self, batch: List[TelemetryPayload]) -> None:
        """Send batch to telemetry endpoint"""
        data = json.dumps({
            "events": [self._payload_to_dict(p) for p in batch]
        }).encode("utf-8")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-langmesh-SDK-Version": "1.0.0",
            "X-langmesh-SDK-Language": "python",
        }
        
        if HAS_HTTPX:
            with httpx.Client(timeout=5.0) as client:
                client.post(self.config.endpoint, content=data, headers=headers)
        else:
            req = urllib.request.Request(
                self.config.endpoint,
                data=data,
                headers=headers,
                method="POST"
            )
            try:
                urllib.request.urlopen(req, timeout=5)
            except urllib.error.URLError:
                pass
    
    def _payload_to_dict(self, payload: TelemetryPayload) -> dict:
        """Convert payload to dict for JSON serialization"""
        return {
            "request": {
                "requestId": payload.request.request_id,
                "orgId": payload.request.org_id,
                "projectId": payload.request.project_id,
                "endpoint": payload.request.endpoint,
                "model": payload.request.model,
                "maxTokens": payload.request.max_tokens,
                "temperature": payload.request.temperature,
                "timestampStart": payload.request.timestamp_start,
            },
            "response": {
                "timestampEnd": payload.response.timestamp_end,
                "tokenUsage": {
                    "promptTokens": payload.response.token_usage.prompt_tokens,
                    "completionTokens": payload.response.token_usage.completion_tokens,
                    "totalTokens": payload.response.token_usage.total_tokens,
                },
                "costEstimateUsd": payload.response.cost_estimate_usd,
                "latencyMs": payload.response.latency_ms,
                "errorClass": payload.response.error_class,
                "errorMessage": payload.response.error_message,
            },
            "context": {
                "sdkLanguage": payload.context.sdk_language,
                "sdkVersion": payload.context.sdk_version,
                "openaiClientVersion": payload.context.openai_client_version,
                "callLineageId": payload.context.call_lineage_id,
                "promptHash": payload.context.prompt_hash,
            },
        }
    
    def _start_flush_timer(self) -> None:
        """Start periodic flush timer"""
        def flush_and_reschedule():
            self.flush()
            if self.config.enabled and not self._paused:
                self._flush_timer = threading.Timer(
                    self.config.flush_interval_seconds,
                    flush_and_reschedule
                )
                self._flush_timer.daemon = True
                self._flush_timer.start()
        
        self._flush_timer = threading.Timer(
            self.config.flush_interval_seconds,
            flush_and_reschedule
        )
        self._flush_timer.daemon = True
        self._flush_timer.start()
    
    def pause(self) -> None:
        """Pause telemetry collection"""
        self._paused = True
    
    def resume(self) -> None:
        """Resume telemetry collection"""
        self._paused = False
    
    def destroy(self) -> None:
        """Cleanup resources"""
        if self._flush_timer:
            self._flush_timer.cancel()
        self.flush()


def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{int(time.time() * 1000)}_{uuid.uuid4().hex[:9]}"


def hash_prompt(prompt: str) -> str:
    """Hash prompt for deduplication without storing content"""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]
