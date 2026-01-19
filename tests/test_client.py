import pytest
from unittest.mock import Mock, patch
from langmesh_openai import OpenAI


def test_client_creation():
    """Test that langmesh client can be created"""
    client = OpenAI(api_key="test-key")
    assert client is not None


def test_compatible_with_openai():
    """Test that langmesh client is compatible with OpenAI"""
    client = OpenAI(api_key="test-key")
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client.chat.completions, "create")


@patch("langmesh_openai.httpx.post")
def test_telemetry_collection(mock_post):
    """Test that telemetry is collected when langmesh_API_KEY is set"""
    with patch.dict("os.environ", {"langmesh_API_KEY": "test-langmesh-key"}):
        client = OpenAI(api_key="test-key")
        assert client.telemetry_enabled is True


def test_works_without_langmesh_key():
    """Test that client works without langmesh_API_KEY"""
    with patch.dict("os.environ", {"langmesh_API_KEY": ""}, clear=True):
        client = OpenAI(api_key="test-key")
        assert client is not None
