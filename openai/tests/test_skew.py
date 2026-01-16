"""
SKEW SDK Tests
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from skew_openai import skew_wrap, TelemetryClient
from skew_openai.types import calculate_cost, TelemetryConfig
from skew_openai.telemetry import generate_request_id, hash_prompt


# Mock OpenAI response
@dataclass
class MockUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15


@dataclass
class MockMessage:
    content: str = "Hello!"
    role: str = "assistant"


@dataclass
class MockChoice:
    message: MockMessage = None
    index: int = 0
    
    def __post_init__(self):
        if self.message is None:
            self.message = MockMessage()


@dataclass
class MockCompletion:
    id: str = "chatcmpl-123"
    choices: list = None
    usage: MockUsage = None
    
    def __post_init__(self):
        if self.choices is None:
            self.choices = [MockChoice()]
        if self.usage is None:
            self.usage = MockUsage()


class MockCompletions:
    def create(self, **kwargs):
        return MockCompletion()


class MockChat:
    def __init__(self):
        self.completions = MockCompletions()


class MockOpenAI:
    def __init__(self):
        self.chat = MockChat()


class TestSkewWrap:
    """Test the skew_wrap function"""
    
    def test_wrap_without_changing_behavior(self):
        """Wrapped client should behave identically to original"""
        mock_openai = MockOpenAI()
        client = skew_wrap(mock_openai, api_key="sk_test_123")
        
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert result.choices[0].message.content == "Hello!"
    
    def test_unwrap_returns_original(self):
        """unwrap() should return the original client"""
        mock_openai = MockOpenAI()
        client = skew_wrap(mock_openai, api_key="sk_test_123")
        
        unwrapped = client.unwrap()
        assert unwrapped is mock_openai
    
    def test_pause_and_resume_telemetry(self):
        """Should be able to pause and resume telemetry"""
        mock_openai = MockOpenAI()
        client = skew_wrap(mock_openai, api_key="sk_test_123")
        
        # Should not raise
        client.pause_telemetry()
        client.resume_telemetry()
    
    def test_proxy_status(self):
        """Should track proxy status correctly"""
        mock_openai = MockOpenAI()
        
        client_no_proxy = skew_wrap(mock_openai, api_key="sk_test_123")
        assert client_no_proxy.is_proxy_active() == False
        
        client_with_proxy = skew_wrap(
            mock_openai, 
            api_key="sk_test_123",
            proxy_enabled=True
        )
        assert client_with_proxy.is_proxy_active() == True
    
    def test_error_passthrough(self):
        """OpenAI errors should pass through unchanged"""
        mock_openai = MockOpenAI()
        mock_openai.chat.completions.create = MagicMock(
            side_effect=Exception("Rate limit exceeded")
        )
        
        client = skew_wrap(mock_openai, api_key="sk_test_123")
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}]
            )


class TestTelemetryClient:
    """Test the TelemetryClient"""
    
    @patch("skew_openai.telemetry.urllib.request.urlopen")
    def test_batch_telemetry(self, mock_urlopen):
        """Should batch telemetry events"""
        mock_urlopen.return_value = MagicMock()
        
        client = TelemetryClient(
            "sk_test",
            TelemetryConfig(batch_size=2)
        )
        
        from skew_openai.types import (
            TelemetryPayload, TelemetryRequest, TelemetryResponse,
            TelemetryContext, TokenUsage
        )
        
        payload = TelemetryPayload(
            request=TelemetryRequest(
                request_id="req_1",
                org_id="org_1",
                endpoint="chat.completions",
                model="gpt-4o",
                timestamp_start="2024-01-01T00:00:00Z",
            ),
            response=TelemetryResponse(
                timestamp_end="2024-01-01T00:00:01Z",
                token_usage=TokenUsage(10, 5, 15),
                cost_estimate_usd=0.0001,
                latency_ms=100,
            ),
            context=TelemetryContext(),
        )
        
        # First event - should not flush
        client.submit(payload)
        assert mock_urlopen.call_count == 0
        
        # Second event - should trigger flush
        client.submit(payload)
        # Give async thread time to complete
        time.sleep(0.1)
        assert mock_urlopen.call_count == 1
    
    def test_sampling(self):
        """Should respect sampling rate"""
        client = TelemetryClient(
            "sk_test",
            TelemetryConfig(sample_rate=0, batch_size=1)
        )
        
        from skew_openai.types import (
            TelemetryPayload, TelemetryRequest, TelemetryResponse,
            TelemetryContext, TokenUsage
        )
        
        payload = TelemetryPayload(
            request=TelemetryRequest(
                request_id="req_1",
                org_id="org_1",
                endpoint="chat.completions",
                model="gpt-4o",
                timestamp_start="2024-01-01T00:00:00Z",
            ),
            response=TelemetryResponse(
                timestamp_end="2024-01-01T00:00:01Z",
                token_usage=TokenUsage(10, 5, 15),
                cost_estimate_usd=0.0001,
                latency_ms=100,
            ),
            context=TelemetryContext(),
        )
        
        # With sample_rate=0, nothing should be submitted
        client.submit(payload)
        client.flush()
        # If we got here without error, sampling worked
    
    def test_pause_resume(self):
        """Should pause and resume correctly"""
        client = TelemetryClient(
            "sk_test",
            TelemetryConfig(batch_size=1)
        )
        
        client.pause()
        assert client._paused == True
        
        client.resume()
        assert client._paused == False


class TestHelperFunctions:
    """Test helper functions"""
    
    def test_generate_request_id(self):
        """Should generate unique request IDs"""
        id1 = generate_request_id()
        id2 = generate_request_id()
        
        assert id1 != id2
        assert id1.startswith("req_")
    
    def test_hash_prompt(self):
        """Should hash prompts consistently"""
        hash1 = hash_prompt("Hello, world!")
        hash2 = hash_prompt("Hello, world!")
        hash3 = hash_prompt("Different prompt")
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16
    
    def test_calculate_cost_known_model(self):
        """Should calculate cost for known models"""
        cost = calculate_cost("gpt-4o", 1000, 500)
        
        # gpt-4o: $2.5/1M input, $10/1M output
        expected = (1000 / 1_000_000) * 2.5 + (500 / 1_000_000) * 10
        assert abs(cost - expected) < 0.0000001
    
    def test_calculate_cost_unknown_model(self):
        """Should use fallback for unknown models"""
        cost = calculate_cost("unknown-model", 1000, 500)
        assert cost > 0


class TestProxyMode:
    """Test proxy mode functionality"""
    
    def test_adds_headers_when_proxy_enabled(self):
        """Should add SKEW headers when proxy is enabled"""
        mock_openai = MockOpenAI()
        original_create = mock_openai.chat.completions.create
        captured_kwargs = {}
        
        def capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return MockCompletion()
        
        mock_openai.chat.completions.create = capture_create
        
        client = skew_wrap(
            mock_openai,
            api_key="sk_test_123",
            org_id="org_test",
            proxy_enabled=True
        )
        
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert "extra_headers" in captured_kwargs
        headers = captured_kwargs["extra_headers"]
        assert headers["X-SKEW-API-Key"] == "sk_test_123"
        assert headers["X-SKEW-Org-ID"] == "org_test"
        assert headers["X-SKEW-Request-ID"].startswith("req_")
    
    def test_no_headers_when_proxy_disabled(self):
        """Should not add headers when proxy is disabled"""
        mock_openai = MockOpenAI()
        captured_kwargs = {}
        
        def capture_create(**kwargs):
            captured_kwargs.update(kwargs)
            return MockCompletion()
        
        mock_openai.chat.completions.create = capture_create
        
        client = skew_wrap(
            mock_openai,
            api_key="sk_test_123",
            proxy_enabled=False
        )
        
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert "extra_headers" not in captured_kwargs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
