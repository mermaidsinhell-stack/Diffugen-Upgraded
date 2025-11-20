"""
Unit tests for MCP client
Tests retry logic, error handling, and API calls
"""

import pytest
import asyncio
from pathlib import Path
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import httpx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "langgraph_agent"))

from mcp_client import (
    retry_with_backoff,
    DiffuGenMCPClient
)


class TestRetryWithBackoff:
    """Test retry_with_backoff function"""

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        """Test function succeeds on first try"""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_with_backoff(success_func, max_retries=3)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_retry(self):
        """Test function succeeds after retries"""
        call_count = 0

        async def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection failed")
            return "success"

        result = await retry_with_backoff(
            eventually_succeeds,
            max_retries=3,
            base_delay=0.01  # Fast for testing
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test function fails after max retries"""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("Connection failed")

        with pytest.raises(httpx.ConnectError):
            await retry_with_backoff(
                always_fails,
                max_retries=3,
                base_delay=0.01
            )

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test exponential backoff delays"""
        import time
        call_times = []

        async def track_time():
            call_times.append(time.time())
            raise httpx.ConnectError("Connection failed")

        with pytest.raises(httpx.ConnectError):
            await retry_with_backoff(
                track_time,
                max_retries=3,
                base_delay=0.1,
                exponential_base=2.0
            )

        # Check delays between calls
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            assert 0.08 <= delay1 <= 0.15  # ~0.1s

        if len(call_times) >= 3:
            delay2 = call_times[2] - call_times[1]
            assert 0.18 <= delay2 <= 0.25  # ~0.2s

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test that non-retryable errors are not retried"""
        call_count = 0

        async def validation_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError):
            await retry_with_backoff(
                validation_error,
                max_retries=3,
                base_delay=0.01
            )

        # Should only be called once (not retried)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_error_retried(self):
        """Test that timeout errors are retried"""
        call_count = 0

        async def timeout_error():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ReadTimeout("Timeout")
            return "success"

        result = await retry_with_backoff(
            timeout_error,
            max_retries=3,
            base_delay=0.01
        )

        assert result == "success"
        assert call_count == 2


class TestDiffuGenMCPClient:
    """Test DiffuGenMCPClient functionality"""

    def test_initialization(self):
        """Test client initialization"""
        client = DiffuGenMCPClient(base_url="http://localhost:8080")

        assert client.base_url == "http://localhost:8080"
        assert client.timeout is not None

    @pytest.mark.asyncio
    async def test_generate_image_success(self):
        """Test successful image generation"""
        client = DiffuGenMCPClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "image_path": "/outputs/test.png",
            "model": "sdxl"
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await client.generate_image(
                prompt="a sunset",
                model="sdxl",
                width=1024,
                height=1024
            )

            assert result["success"] is True
            assert result["image_path"] == "/outputs/test.png"

    @pytest.mark.asyncio
    async def test_generate_image_with_lora_handling(self):
        """Test that LoRA is properly handled in kwargs"""
        client = DiffuGenMCPClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "image_path": "/outputs/test.png"
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.post = mock_post

            await client.generate_image(
                prompt="a sunset",
                model="sd15",
                lora="style_lora",
                negative_prompt="blur"
            )

            # Get the JSON payload that was sent
            call_args = mock_post.call_args
            json_payload = call_args[1]['json']

            # Verify LoRA and negative_prompt are NOT in the payload
            # (they should be filtered out)
            assert 'lora' not in json_payload
            assert 'negative_prompt' not in json_payload

    @pytest.mark.asyncio
    async def test_generate_image_timeout(self):
        """Test generation timeout handling"""
        client = DiffuGenMCPClient(base_url="http://localhost:8080")

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            result = await client.generate_image(
                prompt="a sunset",
                model="sdxl"
            )

            assert result["success"] is False
            assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_refine_image_success(self):
        """Test successful image refinement"""
        client = DiffuGenMCPClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "image_path": "/outputs/refined.png"
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await client.refine_image(
                prompt="improve quality",
                init_image_base64="base64data...",
                model="sdxl",
                strength=0.5
            )

            assert result["success"] is True
            assert "refined" in result["image_path"]

    @pytest.mark.asyncio
    async def test_get_loras_success(self):
        """Test getting LoRA list"""
        client = DiffuGenMCPClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "loras": ["style1", "style2", "character1"]
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            loras = await client.get_loras()

            assert len(loras) == 3
            assert "style1" in loras

    @pytest.mark.asyncio
    async def test_get_loras_error(self):
        """Test LoRA list error handling"""
        client = DiffuGenMCPClient(base_url="http://localhost:8080")

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Connection failed")
            )

            loras = await client.get_loras()

            assert loras == []

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test health check success"""
        client = DiffuGenMCPClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            is_healthy = await client.health_check()

            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure"""
        client = DiffuGenMCPClient(base_url="http://localhost:8080")

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Connection failed")
            )

            is_healthy = await client.health_check()

            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_retry_on_network_error(self):
        """Test that network errors trigger retry"""
        client = DiffuGenMCPClient(base_url="http://localhost:8080")

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count < 2:
                raise httpx.ConnectError("Connection failed")

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True, "image_path": "/test.png"}
            return mock_response

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=mock_post
            )

            result = await client.generate_image(
                prompt="test",
                model="sdxl"
            )

            # Should succeed after retry
            assert result["success"] is True
            assert call_count == 2


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_generation_flow(self):
        """Test full generation flow with all parameters"""
        client = DiffuGenMCPClient(base_url="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "image_path": "/outputs/generated.png",
            "model": "sdxl",
            "prompt": "a sunset",
            "width": 1024,
            "height": 1024
        }

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await client.generate_image(
                prompt="a sunset",
                model="sdxl",
                width=1024,
                height=1024,
                steps=30,
                cfg_scale=7.5,
                sampling_method="euler_a",
                return_base64=True,
                negative_prompt="blur"
            )

            assert result["success"] is True
            assert result["model"] == "sdxl"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
