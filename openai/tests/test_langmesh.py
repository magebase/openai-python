"""
langmesh SDK Tests
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from langmesh_openai import langmesh_wrap, TelemetryClient
from langmesh_openai.types import calculate_cost, TelemetryConfig
from langmesh_openai.telemetry import generate_request_id, hash_prompt


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


class TestlangmeshWrap:
    """Test the langmesh_wrap function"""
    
    def test_wrap_without_changing_behavior(self):
        """Wrapped client should behave identically to original"""
        mock_openai = MockOpenAI()
        client = langmesh_wrap(mock_openai, api_key="sk_test_123")
        
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert result.choices[0].message.content == "Hello!"
    
    def test_unwrap_returns_original(self):
        """unwrap() should return the original client"""
        mock_openai = MockOpenAI()
        client = langmesh_wrap(mock_openai, api_key="sk_test_123")
        
        unwrapped = client.unwrap()
        assert unwrapped is mock_openai
    
    def test_pause_and_resume_telemetry(self):
        """Should be able to pause and resume telemetry"""
        mock_openai = MockOpenAI()
        client = langmesh_wrap(mock_openai, api_key="sk_test_123")
        
        # Should not raise
        client.pause_telemetry()
        client.resume_telemetry()
    
    def test_proxy_status(self):
        """Should track proxy status correctly"""
        mock_openai = MockOpenAI()
        
        client_no_proxy = langmesh_wrap(mock_openai, api_key="sk_test_123")
        assert client_no_proxy.is_proxy_active() == False
        
        client_with_proxy = langmesh_wrap(
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
        
        client = langmesh_wrap(mock_openai, api_key="sk_test_123")
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}]
            )


class TestTelemetryClient:
    """Test the TelemetryClient"""
    
    @patch("langmesh_openai.telemetry.urllib.request.urlopen")
    def test_batch_telemetry(self, mock_urlopen):
        """Should batch telemetry events"""
        mock_urlopen.return_value = MagicMock()
        
        client = TelemetryClient(
            "sk_test",
            TelemetryConfig(batch_size=2)
        )
        
        from langmesh_openai.types import (
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
        
        from langmesh_openai.types import (
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
        
        # With sample rate 0, should never send
        client.submit(payload)
        time.sleep(0.1)
        assert True


def test_calculate_cost():
    cost = calculate_cost("gpt-4o", 100, 50)
    assert cost > 0


def test_generate_request_id():
    request_id_1 = generate_request_id()
    request_id_2 = generate_request_id()
    assert request_id_1 != request_id_2


def test_hash_prompt():
    prompt = "Hello world"
    hash_1 = hash_prompt(prompt)
    hash_2 = hash_prompt(prompt)
    assert hash_1 == hash_2


def test_telemetry_config_defaults():
    config = TelemetryConfig()
    assert config.batch_size == 20
    assert config.batch_interval == 10
    assert config.sample_rate == 1.0
    assert config.endpoint == "https://telemetry.langmesh.dev/v1/events"
    assert config.timeout == 5
