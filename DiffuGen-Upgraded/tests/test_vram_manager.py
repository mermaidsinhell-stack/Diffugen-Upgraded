"""
Unit tests for VRAM orchestration
Tests VRAMOrchestrator, phase management, and model loading/unloading
"""

import pytest
import asyncio
from pathlib import Path
import sys
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "langgraph_agent"))

from vram_manager import (
    VRAMOrchestrator,
    VRAMPhase,
    VRAMRequest,
    requires_llm_phase,
    requires_diffusion_phase
)


class TestVRAMRequest:
    """Test VRAMRequest dataclass"""

    def test_creation(self):
        """Test basic creation"""
        request = VRAMRequest(
            phase=VRAMPhase.LLM,
            priority=1
        )

        assert request.phase == VRAMPhase.LLM
        assert request.priority == 1
        assert request.created_at is not None

    def test_default_timestamp(self):
        """Test automatic timestamp creation"""
        request = VRAMRequest(phase=VRAMPhase.DIFFUSION)

        assert request.created_at is not None
        assert request.priority == 0


class TestVRAMOrchestrator:
    """Test VRAMOrchestrator functionality"""

    def test_initialization_enabled(self):
        """Test initialization with orchestration enabled"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            enable_orchestration=True,
            ollama_model="qwen2.5:latest"
        )

        assert orchestrator.enable_orchestration is True
        assert orchestrator.ollama_model == "qwen2.5:latest"
        assert orchestrator.current_phase == VRAMPhase.IDLE
        assert orchestrator.is_switching is False

    def test_initialization_disabled(self):
        """Test initialization with orchestration disabled"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            enable_orchestration=False
        )

        assert orchestrator.enable_orchestration is False

    @pytest.mark.asyncio
    async def test_check_vllm_health_success(self):
        """Test vLLM health check success"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080"
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            is_healthy = await orchestrator.check_vllm_health()

            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_check_vllm_health_failure(self):
        """Test vLLM health check failure"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080"
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 500

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            is_healthy = await orchestrator.check_vllm_health()

            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_check_diffugen_health_success(self):
        """Test DiffuGen health check success"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080"
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            is_healthy = await orchestrator.check_diffugen_health()

            assert is_healthy is True

    @pytest.mark.asyncio
    async def test_unload_ollama_model(self):
        """Test Ollama model unloading"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            ollama_model="qwen2.5:latest",
            switch_delay=0.1  # Fast for testing
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            success = await orchestrator._unload_ollama_model()

            assert success is True

            # Verify API was called with keep_alive=0
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            assert call_args is not None
            json_data = call_args[1]['json']
            assert json_data['keep_alive'] == 0
            assert json_data['model'] == "qwen2.5:latest"

    @pytest.mark.asyncio
    async def test_load_ollama_model(self):
        """Test Ollama model loading"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            ollama_model="qwen2.5:latest"
        )

        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200

            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            success = await orchestrator._load_ollama_model()

            assert success is True

            # Verify API was called with keep_alive=5m
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            assert call_args is not None
            json_data = call_args[1]['json']
            assert json_data['keep_alive'] == "5m"
            assert json_data['model'] == "qwen2.5:latest"

    @pytest.mark.asyncio
    async def test_prepare_for_llm_phase(self):
        """Test LLM phase preparation"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            enable_orchestration=True,
            switch_delay=0.1
        )

        with patch.object(orchestrator, 'check_vllm_health', return_value=False):
            with patch.object(orchestrator, '_load_ollama_model', return_value=True):
                await orchestrator.prepare_for_llm_phase()

                assert orchestrator.current_phase == VRAMPhase.LLM
                assert orchestrator.stats["llm_requests"] == 1
                assert orchestrator.stats["switches"] == 1

    @pytest.mark.asyncio
    async def test_prepare_for_llm_phase_already_ready(self):
        """Test LLM phase when already in LLM phase"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            enable_orchestration=True
        )

        orchestrator.current_phase = VRAMPhase.LLM

        with patch.object(orchestrator, '_load_ollama_model') as mock_load:
            await orchestrator.prepare_for_llm_phase()

            # Should not attempt to load again
            mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_prepare_for_diffusion_phase(self):
        """Test Diffusion phase preparation"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            enable_orchestration=True,
            switch_delay=0.1
        )

        orchestrator.current_phase = VRAMPhase.LLM  # Start in LLM phase

        with patch.object(orchestrator, '_unload_ollama_model', return_value=True):
            await orchestrator.prepare_for_diffusion_phase()

            assert orchestrator.current_phase == VRAMPhase.DIFFUSION
            assert orchestrator.stats["diffusion_requests"] == 1
            assert orchestrator.stats["switches"] == 1

    @pytest.mark.asyncio
    async def test_phase_switching_with_lock(self):
        """Test that phase switching uses locking"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            enable_orchestration=True,
            switch_delay=0.1
        )

        with patch.object(orchestrator, 'check_vllm_health', return_value=False):
            with patch.object(orchestrator, '_load_ollama_model', return_value=True):
                with patch.object(orchestrator, '_unload_ollama_model', return_value=True):
                    # Start two phase switches concurrently
                    task1 = asyncio.create_task(orchestrator.prepare_for_llm_phase())
                    task2 = asyncio.create_task(orchestrator.prepare_for_diffusion_phase())

                    await asyncio.gather(task1, task2)

                    # Both should complete without error
                    # Lock ensures they don't interfere

    @pytest.mark.asyncio
    async def test_force_idle(self):
        """Test forcing IDLE state"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            switch_delay=0.1
        )

        orchestrator.current_phase = VRAMPhase.LLM

        with patch.object(orchestrator, '_unload_ollama_model', return_value=True):
            await orchestrator.force_idle()

            assert orchestrator.current_phase == VRAMPhase.IDLE

    def test_get_vram_status(self):
        """Test VRAM status reporting"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            enable_orchestration=True,
            ollama_model="qwen2.5:latest"
        )

        orchestrator.current_phase = VRAMPhase.LLM
        orchestrator.stats["switches"] = 5
        orchestrator.stats["llm_requests"] = 10
        orchestrator.stats["total_switch_time"] = 25.0

        status = orchestrator.get_vram_status()

        assert status["current_phase"] == "llm"
        assert status["orchestration_enabled"] is True
        assert status["orchestration_implemented"] is True
        assert status["ollama_model"] == "qwen2.5:latest"
        assert status["statistics"]["total_switches"] == 5
        assert status["statistics"]["llm_requests"] == 10
        assert status["statistics"]["avg_switch_time"] == 5.0

    @pytest.mark.asyncio
    async def test_orchestration_disabled(self):
        """Test behavior with orchestration disabled"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            enable_orchestration=False
        )

        # Should do nothing
        await orchestrator.prepare_for_llm_phase()
        await orchestrator.prepare_for_diffusion_phase()

        # No stats should be recorded
        assert orchestrator.stats["switches"] == 0


class TestDecorators:
    """Test phase requirement decorators"""

    @pytest.mark.asyncio
    async def test_requires_llm_phase_decorator(self):
        """Test @requires_llm_phase decorator"""

        class MockService:
            def __init__(self):
                self.vram_manager = VRAMOrchestrator(
                    vllm_base_url="http://localhost:11434/v1",
                    diffugen_base_url="http://localhost:8080",
                    enable_orchestration=True,
                    switch_delay=0.1
                )

            @requires_llm_phase
            async def llm_operation(self):
                return "LLM result"

        service = MockService()

        with patch.object(service.vram_manager, 'prepare_for_llm_phase') as mock_prepare:
            result = await service.llm_operation()

            # Should have called prepare
            mock_prepare.assert_called_once()
            assert result == "LLM result"

    @pytest.mark.asyncio
    async def test_requires_diffusion_phase_decorator(self):
        """Test @requires_diffusion_phase decorator"""

        class MockService:
            def __init__(self):
                self.vram_manager = VRAMOrchestrator(
                    vllm_base_url="http://localhost:11434/v1",
                    diffugen_base_url="http://localhost:8080",
                    enable_orchestration=True,
                    switch_delay=0.1
                )

            @requires_diffusion_phase
            async def diffusion_operation(self):
                return "Diffusion result"

        service = MockService()

        with patch.object(service.vram_manager, 'prepare_for_diffusion_phase') as mock_prepare:
            result = await service.diffusion_operation()

            # Should have called prepare
            mock_prepare.assert_called_once()
            assert result == "Diffusion result"


class TestStatistics:
    """Test statistics tracking"""

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test that statistics are properly tracked"""
        orchestrator = VRAMOrchestrator(
            vllm_base_url="http://localhost:11434/v1",
            diffugen_base_url="http://localhost:8080",
            enable_orchestration=True,
            switch_delay=0.01  # Fast for testing
        )

        with patch.object(orchestrator, 'check_vllm_health', return_value=False):
            with patch.object(orchestrator, '_load_ollama_model', return_value=True):
                with patch.object(orchestrator, '_unload_ollama_model', return_value=True):
                    # Switch phases multiple times
                    await orchestrator.prepare_for_llm_phase()
                    await orchestrator.prepare_for_diffusion_phase()
                    await orchestrator.prepare_for_llm_phase()

                    stats = orchestrator.stats

                    assert stats["switches"] == 3
                    assert stats["llm_requests"] == 2
                    assert stats["diffusion_requests"] == 1
                    assert stats["total_switch_time"] > 0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
