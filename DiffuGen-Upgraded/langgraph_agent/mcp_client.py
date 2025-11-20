"""
DiffuGen MCP Client
Wrapper for calling DiffuGen MCP tools from LangGraph
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0
):
    """
    Retry a function with exponential backoff

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff

    Returns:
        Function result if successful

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func()
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries} retry attempts failed")
                raise
        except Exception as e:
            # Don't retry on other exceptions (e.g., validation errors)
            logger.error(f"Non-retryable error: {e}")
            raise

    # Should never reach here, but just in case
    raise last_exception


class DiffuGenMCPClient:
    """
    Client for DiffuGen MCP Server

    Provides clean async interface for tool calls
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.timeout = httpx.Timeout(300.0, connect=10.0)  # 5min for generation

    async def refine_image(
        self,
        prompt: str,
        init_image_base64: str,
        model: str = "sd15",
        strength: float = 0.5,
        width: Optional[int] = None,
        height: Optional[int] = None,
        return_base64: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call refine_image tool with base64 injection

        This is THE CRITICAL METHOD that enables the magic workflow!
        """
        payload = {
            "prompt": prompt,
            "init_image_base64": init_image_base64,  # ← ACTUAL BASE64 DATA
            "model": model,
            "strength": strength,
            "return_base64": return_base64,
        }

        if width:
            payload["width"] = width
        if height:
            payload["height"] = height

        # Add any additional parameters
        payload.update(kwargs)

        logger.info(f"MCP Call: refine_image(prompt='{prompt[:50]}...', base64_len={len(init_image_base64)})")

        async def _make_request():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/refine_image",
                    json=payload
                )

                result = response.json()

                if response.status_code == 200 and result.get("success"):
                    logger.info(f"✓ refine_image succeeded: {result.get('image_path')}")
                else:
                    logger.error(f"✗ refine_image failed: {result.get('error')}")

                return result

        try:
            return await retry_with_backoff(_make_request, max_retries=2)
        except httpx.TimeoutException:
            logger.error("refine_image timed out after all retries")
            return {
                "success": False,
                "error": "Generation timed out after 5 minutes"
            }
        except Exception as e:
            logger.error(f"refine_image exception: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"MCP call failed: {str(e)}"
            }

    async def generate_image(
        self,
        prompt: str,
        model: str = "sd15",
        width: int = 512,
        height: int = 512,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        return_base64: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call generate_stable_diffusion_image tool
        """
        payload = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "return_base64": return_base64,
        }

        if steps:
            payload["steps"] = steps
        if cfg_scale:
            payload["cfg_scale"] = cfg_scale

        # Handle LoRA properly - it's already in the prompt with <lora:name:weight> syntax
        # Remove it from kwargs to avoid duplication
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['lora', 'negative_prompt']}

        # Add any additional parameters
        payload.update(clean_kwargs)

        logger.info(f"MCP Call: generate_image(prompt='{prompt[:50]}...', model={model})")

        async def _make_request():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/generate/stable",
                    json=payload
                )

                result = response.json()

                if response.status_code == 200 and result.get("success"):
                    logger.info(f"✓ generate_image succeeded: {result.get('image_path')}")
                else:
                    logger.error(f"✗ generate_image failed: {result.get('error')}")

                return result

        try:
            return await retry_with_backoff(_make_request, max_retries=2)
        except httpx.TimeoutException:
            logger.error("generate_image timed out after all retries")
            return {
                "success": False,
                "error": "Generation timed out after 5 minutes"
            }
        except Exception as e:
            logger.error(f"generate_image exception: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"MCP call failed: {str(e)}"
            }

    async def get_loras(self) -> list[str]:
        """Get list of available LoRA models"""
        logger.info("MCP Call: get_loras")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/loras")
                if response.status_code == 200:
                    loras = response.json().get("loras", [])
                    logger.info(f"✓ get_loras succeeded: {len(loras)} found")
                    return loras
                else:
                    logger.error(f"✗ get_loras failed with status {response.status_code}")
                    return []
        except Exception as e:
            logger.error(f"get_loras exception: {e}", exc_info=True)
            return []

    async def health_check(self) -> bool:
        """Check if DiffuGen MCP is responsive"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/")
                return response.status_code in [200, 404, 405]
        except Exception:
            return False
