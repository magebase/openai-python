import pytest
from unittest.mock import Mock, patch
from skew_openai import OpenAI


def test_client_creation():
    """Test that SKEW client can be created"""
    client = OpenAI(api_key="test-key")
    assert client is not None


def test_compatible_with_openai():
    """Test that SKEW client is compatible with OpenAI"""
    client = OpenAI(api_key="test-key")
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client.chat.completions, "create")


@patch("skew_openai.httpx.post")
def test_telemetry_collection(mock_post):
    """Test that telemetry is collected when SKEW_API_KEY is set"""
    with patch.dict("os.environ", {"SKEW_API_KEY": "test-skew-key"}):
        client = OpenAI(api_key="test-key")
        assert client.telemetry_enabled is True


def test_works_without_skew_key():
    """Test that client works without SKEW_API_KEY"""
    with patch.dict("os.environ", {"SKEW_API_KEY": ""}, clear=True):
        client = OpenAI(api_key="test-key")
        assert client is not None
