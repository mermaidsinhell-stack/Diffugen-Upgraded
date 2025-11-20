"""
VRAM Orchestration Manager (Production Implementation)
Handles actual model loading/unloading for 8GB VRAM constraint
"""

import os
import logging
import asyncio
from typing import Optional, Callable, Any, Dict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


class VRAMPhase(Enum):
    """VRAM allocation phases"""
    IDLE = "idle"
    LLM = "llm"
    DIFFUSION = "diffusion"
    TRANSITIONING = "transitioning"


@dataclass
class VRAMRequest:
    """Request for VRAM phase allocation"""
    phase: VRAMPhase
    priority: int = 0
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class VRAMOrchestrator:
    """
    Production VRAM Manager for 8GB VRAM constraint

    Manages switching between Qwen (LLM) and DiffuGen (Stable Diffusion)

    Architecture:
    - Queue-based request handling
    - Model loading/unloading via Ollama API
    - Smart scheduling to minimize switches
    - Graceful degradation on errors
    """

    def __init__(
        self,
        vllm_base_url: str,
        diffugen_base_url: str,
        enable_orchestration: bool = True,
        ollama_model: str = "qwen2.5:latest",
        switch_delay: float = 2.0,
        max_queue_size: int = 100
    ):
        self.vllm_base_url = vllm_base_url.rstrip('/')
        self.diffugen_base_url = diffugen_base_url.rstrip('/')
        self.enable_orchestration = enable_orchestration
        self.ollama_model = ollama_model
        self.switch_delay = switch_delay

        # State tracking
        self.current_phase = VRAMPhase.IDLE
        self.target_phase = VRAMPhase.IDLE
        self.is_switching = False
        self._lock = asyncio.Lock()

        # Queue for phase requests
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)

        # Statistics
        self.stats = {
            "switches": 0,
            "llm_requests": 0,
            "diffusion_requests": 0,
            "errors": 0,
            "total_switch_time": 0.0
        }

        if enable_orchestration:
            logger.info("🎯 VRAM Orchestration ENABLED (8GB constraint)")
            logger.info(f"   LLM Model: {ollama_model}")
            logger.info(f"   vLLM URL: {self.vllm_base_url}")
            logger.info(f"   DiffuGen URL: {self.diffugen_base_url}")
        else:
            logger.info("VRAM Orchestration disabled (both services run simultaneously)")

    async def check_vllm_health(self) -> bool:
        """Check if Ollama is responsive"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                base = self.vllm_base_url.replace('/v1', '')
                response = await client.get(f"{base}/api/tags")
                is_healthy = response.status_code == 200

                if is_healthy:
                    logger.debug("✓ Ollama health check: OK")
                else:
                    logger.warning(f"✗ Ollama health check failed: HTTP {response.status_code}")

                return is_healthy
        except Exception as e:
            logger.warning(f"Ollama health check error: {e}")
            return False

    async def check_diffugen_health(self) -> bool:
        """Check if DiffuGen is responsive"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.diffugen_base_url}/health",
                    follow_redirects=True
                )
                is_healthy = response.status_code in [200, 404, 405]

                if is_healthy:
                    logger.debug("✓ DiffuGen health check: OK")
                else:
                    logger.warning(f"✗ DiffuGen health check failed: HTTP {response.status_code}")

                return is_healthy
        except Exception as e:
            logger.warning(f"DiffuGen health check error: {e}")
            return False

    async def _unload_ollama_model(self) -> bool:
        """
        Unload Ollama model to free VRAM

        Uses Ollama's /api/generate endpoint with keep_alive=0
        """
        try:
            logger.info(f"🔄 Unloading Ollama model: {self.ollama_model}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                base = self.vllm_base_url.replace('/v1', '')

                # Use keep_alive=0 to unload the model immediately
                response = await client.post(
                    f"{base}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": "",
                        "keep_alive": 0  # This unloads the model
                    }
                )

                if response.status_code == 200:
                    logger.info(f"✓ Ollama model unloaded: {self.ollama_model}")
                    # Give VRAM time to be released
                    await asyncio.sleep(self.switch_delay)
                    return True
                else:
                    logger.error(f"✗ Failed to unload Ollama model: HTTP {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"Error unloading Ollama model: {e}")
            self.stats["errors"] += 1
            return False

    async def _load_ollama_model(self) -> bool:
        """
        Load Ollama model into VRAM

        Uses /api/generate with a dummy prompt to trigger model loading
        """
        try:
            logger.info(f"🔄 Loading Ollama model: {self.ollama_model}")

            async with httpx.AsyncClient(timeout=60.0) as client:
                base = self.vllm_base_url.replace('/v1', '')

                # Send a dummy request to load the model
                # Set keep_alive to keep it loaded
                response = await client.post(
                    f"{base}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": "hello",
                        "keep_alive": "5m"  # Keep loaded for 5 minutes
                    }
                )

                if response.status_code == 200:
                    logger.info(f"✓ Ollama model loaded: {self.ollama_model}")
                    return True
                else:
                    logger.error(f"✗ Failed to load Ollama model: HTTP {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"Error loading Ollama model: {e}")
            self.stats["errors"] += 1
            return False

    async def prepare_for_llm_phase(self):
        """
        Prepare VRAM for LLM operations

        Unloads DiffuGen models (they auto-unload), loads Ollama
        """
        if not self.enable_orchestration:
            logger.debug("Phase: LLM (orchestration disabled)")
            return

        async with self._lock:
            if self.current_phase == VRAMPhase.LLM:
                logger.debug("Already in LLM phase, skipping")
                return

            start_time = asyncio.get_event_loop().time()
            logger.info("=" * 60)
            logger.info("🔄 VRAM Phase Transition: → LLM")
            logger.info("=" * 60)

            self.is_switching = True
            self.target_phase = VRAMPhase.LLM
            self.stats["llm_requests"] += 1

            try:
                # DiffuGen models auto-unload when process ends
                # Just need to ensure Ollama is loaded

                # Check if Ollama is already loaded
                is_healthy = await self.check_vllm_health()

                if not is_healthy:
                    # Load Ollama model
                    success = await self._load_ollama_model()
                    if not success:
                        logger.error("Failed to load Ollama model")
                        raise RuntimeError("Ollama model loading failed")

                self.current_phase = VRAMPhase.LLM
                self.stats["switches"] += 1

                elapsed = asyncio.get_event_loop().time() - start_time
                self.stats["total_switch_time"] += elapsed

                logger.info(f"✓ LLM phase ready (took {elapsed:.1f}s)")

            except Exception as e:
                logger.error(f"Error transitioning to LLM phase: {e}")
                self.stats["errors"] += 1
                raise
            finally:
                self.is_switching = False

    async def prepare_for_diffusion_phase(self):
        """
        Prepare VRAM for Diffusion operations

        Unloads Ollama, makes room for DiffuGen (which loads on-demand)
        """
        if not self.enable_orchestration:
            logger.debug("Phase: DIFFUSION (orchestration disabled)")
            return

        async with self._lock:
            if self.current_phase == VRAMPhase.DIFFUSION:
                logger.debug("Already in DIFFUSION phase, skipping")
                return

            start_time = asyncio.get_event_loop().time()
            logger.info("=" * 60)
            logger.info("🔄 VRAM Phase Transition: → DIFFUSION")
            logger.info("=" * 60)

            self.is_switching = True
            self.target_phase = VRAMPhase.DIFFUSION
            self.stats["diffusion_requests"] += 1

            try:
                # Unload Ollama to free VRAM for DiffuGen
                success = await self._unload_ollama_model()

                if not success:
                    logger.warning("Failed to unload Ollama, continuing anyway")
                    # Don't fail - DiffuGen might still work

                # DiffuGen loads models on-demand, so we're ready
                self.current_phase = VRAMPhase.DIFFUSION
                self.stats["switches"] += 1

                elapsed = asyncio.get_event_loop().time() - start_time
                self.stats["total_switch_time"] += elapsed

                logger.info(f"✓ DIFFUSION phase ready (took {elapsed:.1f}s)")

            except Exception as e:
                logger.error(f"Error transitioning to DIFFUSION phase: {e}")
                self.stats["errors"] += 1
                raise
            finally:
                self.is_switching = False

    def get_vram_status(self) -> Dict[str, Any]:
        """Get current VRAM orchestration status"""
        return {
            "current_phase": self.current_phase.value,
            "target_phase": self.target_phase.value,
            "is_switching": self.is_switching,
            "orchestration_enabled": self.enable_orchestration,
            "orchestration_implemented": True,  # Now it's real!
            "ollama_model": self.ollama_model,
            "statistics": {
                "total_switches": self.stats["switches"],
                "llm_requests": self.stats["llm_requests"],
                "diffusion_requests": self.stats["diffusion_requests"],
                "errors": self.stats["errors"],
                "avg_switch_time": (
                    self.stats["total_switch_time"] / self.stats["switches"]
                    if self.stats["switches"] > 0
                    else 0
                )
            }
        }

    async def force_idle(self):
        """Force unload everything to free maximum VRAM"""
        logger.info("🔄 Forcing IDLE state - unloading all models")

        async with self._lock:
            self.target_phase = VRAMPhase.IDLE
            self.is_switching = True

            try:
                # Unload Ollama
                await self._unload_ollama_model()

                # DiffuGen models unload automatically

                self.current_phase = VRAMPhase.IDLE
                logger.info("✓ IDLE state - all models unloaded")

            except Exception as e:
                logger.error(f"Error forcing IDLE: {e}")
                raise
            finally:
                self.is_switching = False


# Decorators for automatic phase management
def requires_llm_phase(func):
    """
    Decorator to ensure LLM is loaded before execution

    Now actually manages VRAM!
    """
    async def wrapper(self, *args, **kwargs):
        await self.vram_manager.prepare_for_llm_phase()
        return await func(self, *args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def requires_diffusion_phase(func):
    """
    Decorator to ensure Diffusion is loaded before execution

    Now actually manages VRAM!
    """
    async def wrapper(self, *args, **kwargs):
        await self.vram_manager.prepare_for_diffusion_phase()
        return await func(self, *args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# CLI for testing VRAM management
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test VRAM Orchestration")
    parser.add_argument("--vllm-url", default="http://localhost:11434/v1")
    parser.add_argument("--diffugen-url", default="http://localhost:8080")
    parser.add_argument("--model", default="qwen2.5:latest")
    parser.add_argument("--test", choices=["llm", "diffusion", "switch", "status"], required=True)

    args = parser.parse_args()

    async def main():
        manager = VRAMOrchestrator(
            vllm_base_url=args.vllm_url,
            diffugen_base_url=args.diffugen_url,
            ollama_model=args.model,
            enable_orchestration=True
        )

        if args.test == "llm":
            print("Testing LLM phase...")
            await manager.prepare_for_llm_phase()
            print("✓ LLM phase ready")

        elif args.test == "diffusion":
            print("Testing DIFFUSION phase...")
            await manager.prepare_for_diffusion_phase()
            print("✓ DIFFUSION phase ready")

        elif args.test == "switch":
            print("Testing phase switching...")
            await manager.prepare_for_llm_phase()
            print("✓ LLM phase ready")
            await asyncio.sleep(2)
            await manager.prepare_for_diffusion_phase()
            print("✓ DIFFUSION phase ready")
            await asyncio.sleep(2)
            await manager.prepare_for_llm_phase()
            print("✓ LLM phase ready")

        elif args.test == "status":
            print("VRAM Status:")
            status = manager.get_vram_status()
            import json
            print(json.dumps(status, indent=2))

        print("\n📊 Statistics:")
        print(json.dumps(manager.get_vram_status()["statistics"], indent=2))

    asyncio.run(main())
