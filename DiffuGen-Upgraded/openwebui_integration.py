"""
OpenWebUI Native Integration Layer
Implements OpenAI-compatible API for seamless Open WebUI integration
"""

import asyncio
import base64
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import httpx

# Import from existing modules
from diffugen_core import (
    encode_image_to_base64,
    handle_image_input
)
from streaming import (
    ProgressTracker,
    StreamingQueue,
    ProgressPhase,
    create_sse_message
)

logger = logging.getLogger(__name__)


# ============================================================================
# OpenAI-Compatible Models
# ============================================================================

class OpenAIImageGenerationRequest(BaseModel):
    """OpenAI /v1/images/generations compatible request"""
    prompt: str = Field(..., description="Text prompt for image generation")
    model: Optional[str] = Field(None, description="Model to use")
    n: int = Field(1, ge=1, le=10, description="Number of images to generate")
    size: str = Field("1024x1024", description="Image size (e.g., '512x512', '1024x1024')")
    response_format: Literal["url", "b64_json"] = Field("url", description="Response format")
    user: Optional[str] = Field(None, description="User identifier")

    # Extended parameters (DiffuGen-specific, optional)
    negative_prompt: Optional[str] = Field(None)
    steps: Optional[int] = Field(None)
    cfg_scale: Optional[float] = Field(None)
    sampling_method: Optional[str] = Field(None)
    seed: Optional[int] = Field(None)
    lora: Optional[str] = Field(None)


class OpenAIImageEditRequest(BaseModel):
    """OpenAI /v1/images/edits compatible request"""
    image: str = Field(..., description="Base64 encoded image to edit")
    prompt: str = Field(..., description="Text prompt for editing")
    mask: Optional[str] = Field(None, description="Base64 encoded mask")
    model: Optional[str] = Field(None)
    n: int = Field(1, ge=1, le=10)
    size: Optional[str] = Field(None)
    response_format: Literal["url", "b64_json"] = Field("url")
    user: Optional[str] = Field(None)

    # Extended parameters
    strength: float = Field(0.5, ge=0.0, le=1.0)
    negative_prompt: Optional[str] = Field(None)
    steps: Optional[int] = Field(None)
    cfg_scale: Optional[float] = Field(None)


class OpenAIImageURL(BaseModel):
    """Image URL response"""
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class OpenAIImageResponse(BaseModel):
    """OpenAI /v1/images response"""
    created: int
    data: List[OpenAIImageURL]


class OpenAIModel(BaseModel):
    """OpenAI model info"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "diffugen"
    permission: List = []
    root: Optional[str] = None
    parent: Optional[str] = None


class OpenAIModelsResponse(BaseModel):
    """OpenAI /v1/models response"""
    object: str = "list"
    data: List[OpenAIModel]


class OpenAIError(BaseModel):
    """OpenAI error response"""
    error: Dict[str, Any]


# ============================================================================
# OpenWebUI Integration Service
# ============================================================================

class OpenWebUIIntegrationService:
    """
    Service layer for OpenWebUI integration

    Translates between OpenAI API format and DiffuGen internal format
    """

    def __init__(
        self,
        diffugen_base_url: str = "http://diffugen-mcp:8080",
        serve_path: str = "/images",
        public_url: Optional[str] = None
    ):
        self.diffugen_base_url = diffugen_base_url.rstrip('/')
        self.serve_path = serve_path
        self.public_url = public_url
        self.client = httpx.AsyncClient(timeout=300.0)

        # Model registry
        self.models = {
            "stable-diffusion-xl": {
                "internal_name": "sdxl",
                "capabilities": ["text2img", "img2img"],
                "max_size": (1024, 1024),
                "default_size": (1024, 1024)
            },
            "stable-diffusion-3": {
                "internal_name": "sd3",
                "capabilities": ["text2img", "img2img"],
                "max_size": (1024, 1024),
                "default_size": (1024, 1024)
            },
            "stable-diffusion-1.5": {
                "internal_name": "sd15",
                "capabilities": ["text2img", "img2img", "lora"],
                "max_size": (768, 768),
                "default_size": (512, 512)
            },
            "revanimated": {
                "internal_name": "revanimated",
                "capabilities": ["text2img", "img2img", "lora"],
                "max_size": (768, 768),
                "default_size": (512, 512)
            },
            "flux-schnell": {
                "internal_name": "flux-schnell",
                "capabilities": ["text2img"],
                "max_size": (1024, 1024),
                "default_size": (1024, 1024)
            },
            "flux-dev": {
                "internal_name": "flux-dev",
                "capabilities": ["text2img"],
                "max_size": (1024, 1024),
                "default_size": (1024, 1024)
            }
        }

    def parse_size(self, size_str: str) -> tuple[int, int]:
        """Parse size string like '1024x1024' into (width, height)"""
        try:
            parts = size_str.lower().split('x')
            if len(parts) != 2:
                raise ValueError("Invalid size format")
            return (int(parts[0]), int(parts[1]))
        except Exception:
            logger.warning(f"Invalid size '{size_str}', using default 512x512")
            return (512, 512)

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get model information"""
        return self.models.get(model_id)

    def list_models(self) -> List[OpenAIModel]:
        """List all available models in OpenAI format"""
        models = []
        for model_id, info in self.models.items():
            models.append(OpenAIModel(
                id=model_id,
                created=int(datetime(2024, 1, 1).timestamp()),
                root=info["internal_name"]
            ))
        return models

    async def generate_image(
        self,
        request: OpenAIImageGenerationRequest,
        base_url: str
    ) -> OpenAIImageResponse:
        """
        Generate image using DiffuGen backend

        Args:
            request: OpenAI-compatible request
            base_url: Base URL for image serving

        Returns:
            OpenAI-compatible response
        """
        try:
            # Resolve model
            model = request.model or "stable-diffusion-xl"
            model_info = self.get_model_info(model)

            if not model_info:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model}' not found. Available models: {', '.join(self.models.keys())}"
                )

            internal_model = model_info["internal_name"]
            width, height = self.parse_size(request.size)

            # Validate size
            max_w, max_h = model_info["max_size"]
            if width > max_w or height > max_h:
                raise HTTPException(
                    status_code=400,
                    detail=f"Size {request.size} exceeds maximum {max_w}x{max_h} for model {model}"
                )

            # Build DiffuGen request
            diffugen_req = {
                "prompt": request.prompt,
                "model": internal_model,
                "width": width,
                "height": height,
                "seed": request.seed or -1,
                "return_base64": request.response_format == "b64_json"
            }

            # Add optional parameters
            if request.negative_prompt:
                diffugen_req["negative_prompt"] = request.negative_prompt
            if request.steps:
                diffugen_req["steps"] = request.steps
            if request.cfg_scale:
                diffugen_req["cfg_scale"] = request.cfg_scale
            if request.sampling_method:
                diffugen_req["sampling_method"] = request.sampling_method
            if request.lora:
                diffugen_req["lora"] = request.lora

            # Call DiffuGen
            logger.info(f"Generating image: model={internal_model}, size={width}x{height}")

            endpoint = "/generate/flux" if "flux" in internal_model else "/generate/stable"

            response = await self.client.post(
                f"{self.diffugen_base_url}{endpoint}",
                json=diffugen_req,
                timeout=300.0
            )

            if response.status_code != 200:
                error_detail = response.json().get("error", "Unknown error")
                raise HTTPException(
                    status_code=500,
                    detail=f"Image generation failed: {error_detail}"
                )

            result = response.json()

            # Build OpenAI response
            images = []

            if request.response_format == "b64_json":
                images.append(OpenAIImageURL(
                    b64_json=result.get("image_base64"),
                    revised_prompt=request.prompt
                ))
            else:
                # Build URL
                image_url = result.get("image_url")
                if not image_url and result.get("image_path"):
                    # Construct URL from path
                    filename = Path(result["image_path"]).name
                    image_url = f"{base_url}{self.serve_path}/{filename}"

                images.append(OpenAIImageURL(
                    url=image_url,
                    revised_prompt=request.prompt
                ))

            return OpenAIImageResponse(
                created=int(time.time()),
                data=images
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating image: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    async def edit_image(
        self,
        request: OpenAIImageEditRequest,
        base_url: str
    ) -> OpenAIImageResponse:
        """
        Edit/refine image using img2img

        Args:
            request: OpenAI-compatible edit request
            base_url: Base URL for image serving

        Returns:
            OpenAI-compatible response
        """
        try:
            # Resolve model
            model = request.model or "stable-diffusion-xl"
            model_info = self.get_model_info(model)

            if not model_info:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model}' not found"
                )

            if "img2img" not in model_info["capabilities"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model}' does not support image editing"
                )

            internal_model = model_info["internal_name"]

            # Build DiffuGen request
            diffugen_req = {
                "prompt": request.prompt,
                "model": internal_model,
                "init_image_base64": request.image,
                "strength": request.strength,
                "seed": -1,
                "return_base64": request.response_format == "b64_json"
            }

            if request.size:
                width, height = self.parse_size(request.size)
                diffugen_req["width"] = width
                diffugen_req["height"] = height

            if request.negative_prompt:
                diffugen_req["negative_prompt"] = request.negative_prompt
            if request.steps:
                diffugen_req["steps"] = request.steps
            if request.cfg_scale:
                diffugen_req["cfg_scale"] = request.cfg_scale

            # Call DiffuGen
            logger.info(f"Editing image: model={internal_model}, strength={request.strength}")

            response = await self.client.post(
                f"{self.diffugen_base_url}/generate/stable",
                json=diffugen_req,
                timeout=300.0
            )

            if response.status_code != 200:
                error_detail = response.json().get("error", "Unknown error")
                raise HTTPException(
                    status_code=500,
                    detail=f"Image editing failed: {error_detail}"
                )

            result = response.json()

            # Build OpenAI response
            images = []

            if request.response_format == "b64_json":
                images.append(OpenAIImageURL(
                    b64_json=result.get("image_base64"),
                    revised_prompt=request.prompt
                ))
            else:
                image_url = result.get("image_url")
                if not image_url and result.get("image_path"):
                    filename = Path(result["image_path"]).name
                    image_url = f"{base_url}{self.serve_path}/{filename}"

                images.append(OpenAIImageURL(
                    url=image_url,
                    revised_prompt=request.prompt
                ))

            return OpenAIImageResponse(
                created=int(time.time()),
                data=images
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error editing image: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    async def generate_image_stream(
        self,
        request: OpenAIImageGenerationRequest,
        base_url: str
    ):
        """
        Generate image with streaming progress (SSE)

        Args:
            request: OpenAI-compatible request
            base_url: Base URL for image serving

        Yields:
            SSE formatted progress updates
        """
        streaming_queue = StreamingQueue()

        try:
            # Resolve model
            model = request.model or "stable-diffusion-xl"
            model_info = self.get_model_info(model)

            if not model_info:
                error_msg = f"Model '{model}' not found. Available models: {', '.join(self.models.keys())}"
                streaming_queue.put(
                    ProgressUpdate(
                        phase=ProgressPhase.ERROR,
                        progress=0,
                        message=error_msg
                    )
                )
                streaming_queue.mark_done()
                return

            internal_model = model_info["internal_name"]
            width, height = self.parse_size(request.size)

            # Validate size
            max_w, max_h = model_info["max_size"]
            if width > max_w or height > max_h:
                error_msg = f"Size {request.size} exceeds maximum {max_w}x{max_h} for model {model}"
                streaming_queue.put(
                    ProgressUpdate(
                        phase=ProgressPhase.ERROR,
                        progress=0,
                        message=error_msg
                    )
                )
                streaming_queue.mark_done()
                return

            # Build DiffuGen request
            diffugen_req = {
                "prompt": request.prompt,
                "model": internal_model,
                "width": width,
                "height": height,
                "seed": request.seed or -1,
                "return_base64": request.response_format == "b64_json"
            }

            # Add optional parameters
            if request.negative_prompt:
                diffugen_req["negative_prompt"] = request.negative_prompt
            if request.steps:
                diffugen_req["steps"] = request.steps
            if request.cfg_scale:
                diffugen_req["cfg_scale"] = request.cfg_scale
            if request.sampling_method:
                diffugen_req["sampling_method"] = request.sampling_method
            if request.lora:
                diffugen_req["lora"] = request.lora

            # Call DiffuGen streaming endpoint
            logger.info(f"Streaming generation: model={internal_model}, size={width}x{height}")

            endpoint = "/generate/flux/stream" if "flux" in internal_model else "/generate/stable/stream"

            # Stream from DiffuGen
            async with self.client.stream(
                "POST",
                f"{self.diffugen_base_url}{endpoint}",
                json=diffugen_req,
                timeout=300.0
            ) as response:
                if response.status_code != 200:
                    error_detail = "Streaming generation failed"
                    streaming_queue.put(
                        ProgressUpdate(
                            phase=ProgressPhase.ERROR,
                            progress=0,
                            message=error_detail
                        )
                    )
                    streaming_queue.mark_done()
                    return

                # Forward SSE events
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        try:
                            data_str = line[5:].strip()
                            data = json.loads(data_str)

                            # Forward progress update
                            if "phase" in data:
                                from streaming import ProgressUpdate
                                update = ProgressUpdate(
                                    phase=ProgressPhase(data["phase"]),
                                    progress=data["progress"],
                                    message=data["message"],
                                    step=data.get("step"),
                                    total_steps=data.get("total_steps"),
                                    eta_seconds=data.get("eta_seconds"),
                                    timestamp=data.get("timestamp")
                                )
                                streaming_queue.put(update)

                        except Exception as e:
                            logger.error(f"Error parsing SSE data: {e}")

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}", exc_info=True)
            streaming_queue.put(
                ProgressUpdate(
                    phase=ProgressPhase.ERROR,
                    progress=0,
                    message=f"Internal server error: {str(e)}"
                )
            )
        finally:
            streaming_queue.mark_done()

    async def close(self):
        """Clean up resources"""
        await self.client.aclose()


# ============================================================================
# OpenWebUI-Compatible Endpoints
# ============================================================================

def create_openwebui_router(
    service: OpenWebUIIntegrationService
) -> FastAPI:
    """
    Create FastAPI router with OpenWebUI-compatible endpoints

    This provides the /v1/images/generations endpoint that OpenWebUI expects
    """
    router = FastAPI(
        title="DiffuGen OpenWebUI Integration",
        description="OpenAI-compatible image generation API",
        version="1.0.0"
    )

    @router.get("/v1/models", response_model=OpenAIModelsResponse)
    async def list_models():
        """List available models (OpenAI compatible)"""
        models = service.list_models()
        return OpenAIModelsResponse(data=models)

    @router.post("/v1/images/generations", response_model=OpenAIImageResponse)
    async def generate_images(request: OpenAIImageGenerationRequest, background_tasks: BackgroundTasks):
        """
        Generate images from text prompts (OpenAI compatible)

        This is the main endpoint OpenWebUI uses for image generation
        """
        # Get base URL from request (FastAPI doesn't have direct access without Request object)
        # We'll use the configured public URL or construct from service config
        base_url = service.public_url or "http://localhost:8000"

        return await service.generate_image(request, base_url)

    @router.post("/v1/images/edits", response_model=OpenAIImageResponse)
    async def edit_images(request: OpenAIImageEditRequest):
        """
        Edit/refine images (OpenAI compatible)

        OpenWebUI uses this for img2img operations
        """
        base_url = service.public_url or "http://localhost:8000"
        return await service.edit_image(request, base_url)

    @router.post(
        "/v1/images/generations/stream",
        response_class=StreamingResponse
    )
    async def generate_images_stream(
        request: OpenAIImageGenerationRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Generate images with real-time progress updates (SSE)

        Returns Server-Sent Events stream with progress updates.
        Compatible with OpenAI's streaming format.
        """
        base_url = service.public_url or "http://localhost:8000"

        # Create streaming queue
        streaming_queue = StreamingQueue()

        # Start streaming generation in background
        async def stream_generation():
            await service.generate_image_stream(request, base_url)

        background_tasks.add_task(stream_generation)

        # Return SSE stream
        from streaming import sse_generator
        return StreamingResponse(
            sse_generator(streaming_queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "service": "openwebui-integration"}

    return router


# ============================================================================
# Utility Functions
# ============================================================================

def create_openai_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None
) -> JSONResponse:
    """Create OpenAI-compatible error response"""
    error_obj = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code
        }
    }
    return JSONResponse(content=error_obj, status_code=400)
