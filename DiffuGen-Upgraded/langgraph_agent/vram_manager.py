"""
VRAM Orchestration Manager
Handles explicit model loading/unloading to fit LLM + Diffusion in 8GB VRAM
"""

import os
import logging
import asyncio
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class VRAMOrchestrator:
    """
    Manages VRAM allocation between vLLM (Qwen3) and DiffuGen (Stable Diffusion)

    8GB VRAM Budget:
    - Qwen3-8B-AWQ: ~4GB
    - Stable Diffusion: ~4GB
    - Strategy: Never run both simultaneously
    """

    def __init__(
        self,
        vllm_base_url: str,
        diffugen_base_url: str,
        enable_orchestration: bool = True
    ):
        self.vllm_base_url = vllm_base_url.rstrip('/')
        self.diffugen_base_url = diffugen_base_url.rstrip('/')
        self.enable_orchestration = enable_orchestration

        self.vllm_loaded = True  # vLLM starts loaded by default
        self.diffugen_loaded = False  # DiffuGen only loads on-demand

        logger.info(f"VRAM Orchestrator initialized (orchestration={'ON' if enable_orchestration else 'OFF'})")
        logger.info(f"vLLM: {self.vllm_base_url}")
        logger.info(f"DiffuGen: {self.diffugen_base_url}")

    async def check_vllm_health(self) -> bool:
        """Check if Ollama is responsive"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try Ollama's API endpoint (remove /v1 from base URL for health check)
                base = self.vllm_base_url.replace('/v1', '')
                response = await client.get(f"{base}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    async def check_diffugen_health(self) -> bool:
        """Check if DiffuGen MCP is responsive"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # DiffuGen doesn't have /health, so check if it's running
                # We can't make an actual MCP call here, just check connectivity
                response = await client.get(
                    f"{self.diffugen_base_url}/",
                    follow_redirects=True
                )
                # Even a 404 means it's alive
                return response.status_code in [200, 404, 405]
        except Exception as e:
            logger.warning(f"DiffuGen health check failed: {e}")
            return False

    async def prepare_for_llm_phase(self):
        """
        Ensure vLLM is loaded and DiffuGen is unloaded
        Phase 1: Planning, Critique, Reasoning
        """
        if not self.enable_orchestration:
            logger.debug("VRAM orchestration disabled, skipping prepare_for_llm_phase")
            return

        logger.info("=== VRAM Phase Transition: LLM ===")

        # Unload DiffuGen if needed
        if self.diffugen_loaded:
            await self.unload_diffugen()

        # Ensure vLLM is loaded
        if not self.vllm_loaded:
            await self.load_vllm()

    async def prepare_for_diffusion_phase(self):
        """
        Ensure DiffuGen is loaded and vLLM is unloaded
        Phase 2: Image Generation
        """
        if not self.enable_orchestration:
            logger.debug("VRAM orchestration disabled, skipping prepare_for_diffusion_phase")
            return

        logger.info("=== VRAM Phase Transition: DIFFUSION ===")

        # Unload vLLM to free ~4GB
        if self.vllm_loaded:
            await self.unload_vllm()

        # Load DiffuGen (stable-diffusion.cpp loads models on-demand, so this is lightweight)
        if not self.diffugen_loaded:
            await self.load_diffugen()

    async def unload_vllm(self):
        """
        Unload vLLM from VRAM

        NOTE: vLLM doesn't have a built-in unload API yet.
        Workaround options:
        1. Stop/start the container (slow but works)
        2. Use CUDA memory management tricks
        3. Accept that vLLM stays loaded (less optimal)

        For now, we'll mark it as unloaded and add a sleep to simulate
        In production, you'd implement one of the above strategies.
        """
        logger.warning("vLLM unload requested - vLLM doesn't support dynamic unloading")
        logger.info("Consider: docker stop diffugen-vllm (manual VRAM release)")

        # Placeholder for future implementation
        # TODO: Implement actual VRAM release strategy
        # Option 1: Container stop/start
        # Option 2: CUDA IPC tricks
        # Option 3: Multiple vLLM instances with routing

        self.vllm_loaded = False
        await asyncio.sleep(0.1)  # Placeholder

    async def load_vllm(self):
        """Load vLLM back into VRAM"""
        logger.info("vLLM load requested - waiting for service to be ready")

        # Placeholder for future implementation
        # TODO: Implement actual model loading
        # This would restart the container or trigger model loading

        # Wait for vLLM to be healthy
        for _ in range(30):  # 30 second timeout
            if await self.check_vllm_health():
                logger.info("vLLM is loaded and healthy")
                self.vllm_loaded = True
                return
            await asyncio.sleep(1)

        raise RuntimeError("Failed to load vLLM - service not responding")

    async def unload_diffugen(self):
        """
        Unload DiffuGen from VRAM

        stable-diffusion.cpp loads models on-demand, so we just mark it unloaded
        The actual VRAM is freed when the generation process completes
        """
        logger.info("DiffuGen marked as unloaded (VRAM freed after generation)")
        self.diffugen_loaded = False
        await asyncio.sleep(0.1)  # Ensure generation process has exited

    async def load_diffugen(self):
        """
        Prepare DiffuGen for image generation

        stable-diffusion.cpp loads models on first use, so this is mostly
        a marker that we're entering the diffusion phase
        """
        logger.info("DiffuGen prepared for generation (model loads on-demand)")
        self.diffugen_loaded = True
        await asyncio.sleep(0.1)

    def get_vram_status(self) -> dict:
        """Get current VRAM allocation status"""
        return {
            "vllm_loaded": self.vllm_loaded,
            "diffugen_loaded": self.diffugen_loaded,
            "current_phase": "llm" if self.vllm_loaded else "diffusion",
            "orchestration_enabled": self.enable_orchestration
        }


# Decorator for automatic VRAM management
def requires_llm_phase(func):
    """Decorator to ensure LLM is loaded before execution"""
    async def wrapper(self, *args, **kwargs):
        await self.vram_manager.prepare_for_llm_phase()
        return await func(self, *args, **kwargs)
    return wrapper


def requires_diffusion_phase(func):
    """Decorator to ensure Diffusion is loaded before execution"""
    async def wrapper(self, *args, **kwargs):
        await self.vram_manager.prepare_for_diffusion_phase()
        return await func(self, *args, **kwargs)
    return wrapper
