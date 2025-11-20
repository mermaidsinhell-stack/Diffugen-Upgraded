"""
DiffuGen OpenAPI Server
Professional REST API for Open WebUI integration with Stable Diffusion and Flux models
"""

from fastapi import FastAPI, HTTPException, Request, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union, Any
import os
import sys
import json
import time
import re
import argparse
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import gc
import uuid
import base64
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import DiffuGen core functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from diffugen import (
    generate_stable_diffusion_image,
    generate_flux_image,
    load_config as load_diffugen_config,
    sd_cpp_path as default_sd_cpp_path,
    _model_paths
)
from streaming import (
    ProgressTracker,
    StreamingQueue,
    ProgressPhase,
    sse_generator
)
from intelligent_workflow import (
    IntelligentWorkflow,
    GenerationParameters,
    ConversationContext
)


# ============================================================================
# Configuration Management
# ============================================================================

def load_openapi_config() -> Dict:
    """Load and merge OpenAPI server configuration"""
    config_file = Path(__file__).parent / "openapi_config.json"

    # Default configuration
    config = {
        "server": {"host": "0.0.0.0", "port": 5199, "debug": False},
        "paths": {
            "sd_cpp_path": default_sd_cpp_path,
            "models_dir": None,
            "output_dir": "outputs",
            "lora_model_dir": "/app/loras"
        },
        "cors": {
            "allow_origins": ["*"],
            "allow_methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["*"]
        },
        "rate_limiting": {"rate": "60/minute", "enabled": True},
        "images": {"serve_path": "/images", "cache_control": "no-cache"}
    }

    # Load from file if exists
    if config_file.exists():
        try:
            with open(config_file) as f:
                file_config = json.load(f)
                config.update(file_config)
                logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Error loading config file: {e}, using defaults")

    # Environment variable overrides
    env_overrides = {
        "DIFFUGEN_OPENAPI_PORT": ("server", "port", int),
        "SD_CPP_PATH": ("paths", "sd_cpp_path", str),
        "DIFFUGEN_OUTPUT_DIR": ("paths", "output_dir", str),
        "DIFFUGEN_LORA_MODEL_DIR": ("paths", "lora_model_dir", str),
        "DIFFUGEN_CORS_ORIGINS": ("cors", "allow_origins", lambda x: x.split(",")),
        "DIFFUGEN_RATE_LIMIT": ("rate_limiting", "rate", str),
    }

    for env_var, (section, key, converter) in env_overrides.items():
        if env_var in os.environ:
            try:
                config[section][key] = converter(os.environ[env_var])
                logger.info(f"Applied env override: {env_var}")
            except Exception as e:
                logger.error(f"Invalid value for {env_var}: {e}")

    return config


# Load configuration
config = load_openapi_config()

# Initialize paths
SD_CPP_PATH = Path(config["paths"]["sd_cpp_path"])
DEFAULT_OUTPUT_DIR = Path(config["paths"]["output_dir"])

# Create output directory
try:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["DIFFUGEN_OUTPUT_DIR"] = str(DEFAULT_OUTPUT_DIR)
    logger.info(f"Output directory: {DEFAULT_OUTPUT_DIR.absolute()}")
except Exception as e:
    logger.error(f"Failed to create output directory: {e}")
    sys.exit(1)


# ============================================================================
# Helper Functions
# ============================================================================

def validate_model(model: str, model_type: str) -> str:
    """
    Validate and normalize model name

    Args:
        model: Model name to validate
        model_type: Expected type ('flux' or 'stable')

    Returns:
        Normalized model name

    Raises:
        HTTPException: If model is invalid
    """
    model = model.lower()

    flux_models = ["flux-schnell", "flux-dev"]
    stable_models = ["sd15", "sdxl", "sd3", "revanimated", "oia"]

    if model_type == "flux":
        if model not in flux_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Flux model '{model}'. Valid options: {', '.join(flux_models)}"
            )
    elif model_type == "stable":
        if model not in stable_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid SD model '{model}'. Valid options: {', '.join(stable_models)}"
            )

    return model


def wait_for_image_file(image_path: Path, max_retries: int = 5, delay: float = 0.5) -> bool:
    """
    Wait for image file to be available and readable

    Args:
        image_path: Path to image file
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds

    Returns:
        True if file is available, False otherwise
    """
    for attempt in range(max_retries):
        if (image_path.exists() and
            os.access(str(image_path), os.R_OK) and
            image_path.stat().st_size > 0):
            return True

        logger.debug(f"Waiting for image file (attempt {attempt + 1}/{max_retries})")
        time.sleep(delay)

    return False


def encode_image_base64(image_path: Path) -> str:
    """Encode image file to base64 string"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to encode image: {e}")


def build_image_url(base_url: str, image_path: Path, serve_path: str) -> str:
    """Build full image URL with cache-busting timestamp"""
    timestamp = int(time.time())
    filename = image_path.name
    return f"{base_url.rstrip('/')}{serve_path}/{filename}?t={timestamp}"


def build_markdown_response(result: Dict, image_url: str, character: str = None) -> str:
    """Build markdown-formatted generation response"""
    parts = [
        "Here's the image you requested:\n",
        f"![Image]({image_url})\n",
        "**Generation Details:**",
        f"- Model: {result['model']}",
        f"- Prompt: {result['prompt']}",
        f"- Resolution: {result['width']}x{result['height']} pixels",
        f"- Steps: {result['steps']}",
        f"- CFG Scale: {result['cfg_scale']}",
        f"- Sampling Method: {result['sampling_method']}",
        f"- Seed: {result['seed'] if result['seed'] != -1 else 'random'}"
    ]

    if character:
        parts.append(f"- Character: {character}")

    if result.get('negative_prompt'):
        parts.append(f"- Negative Prompt: {result['negative_prompt']}")

    return "\n".join(parts)


# ============================================================================
# Rate Limiting Middleware
# ============================================================================

class RateLimitMiddleware:
    """Simple in-memory rate limiting middleware"""

    def __init__(self, app, rate_limit: str = "60/minute", enabled: bool = True):
        self.app = app
        self.enabled = enabled
        self.requests = defaultdict(list)

        # Parse rate limit
        match = re.match(r"(\d+)/(\w+)", rate_limit)
        if not match:
            raise ValueError(f"Invalid rate limit format: {rate_limit}")

        self.max_requests = int(match.group(1))
        timeunit = match.group(2).lower()

        time_units = {
            "second": 1, "minute": 60, "hour": 3600, "day": 86400
        }
        self.window_seconds = time_units.get(timeunit, 60)

    def _clean_old_requests(self, key: str) -> List[float]:
        """Remove requests outside the time window"""
        now = time.time()
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < self.window_seconds
        ]
        return self.requests[key]

    async def __call__(self, scope, receive, send):
        if not self.enabled or scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope=scope, receive=receive)
        key = request.client.host

        requests = self._clean_old_requests(key)

        if len(requests) >= self.max_requests:
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"x-rate-limit-limit", str(self.max_requests).encode()),
                    (b"x-rate-limit-remaining", b"0"),
                ]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.max_requests} requests per {self.window_seconds} seconds"
                }).encode()
            })
            return

        # Record request
        self.requests[key].append(time.time())

        # Add rate limit headers
        async def wrapped_send(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                message["headers"].extend([
                    (b"x-rate-limit-limit", str(self.max_requests).encode()),
                    (b"x-rate-limit-remaining", str(max(0, self.max_requests - len(requests))).encode()),
                ])
            await send(message)

        await self.app(scope, receive, wrapped_send)


# ============================================================================
# FastAPI Application Setup
# ============================================================================

# Create app with middlewares
middlewares = []

# CORS middleware
middlewares.append({
    "middleware_class": CORSMiddleware,
    "allow_origins": config["cors"]["allow_origins"],
    "allow_credentials": True,
    "allow_methods": config["cors"]["allow_methods"],
    "allow_headers": config["cors"]["allow_headers"],
})

app = FastAPI(
    title="DiffuGen",
    description="AI Image Generation API using Stable Diffusion and Flux models",
    version="1.0.0",
    openapi_tags=[
        {"name": "Image Generation", "description": "Generate images using various AI models"},
        {"name": "System", "description": "System information and health checks"},
        {"name": "Models", "description": "Model information and management"}
    ]
)

# Add middlewares
for middleware_config in middlewares:
    middleware_class = middleware_config.pop("middleware_class")
    app.add_middleware(middleware_class, **middleware_config)

# Add rate limiting if enabled
if config.get("rate_limiting", {}).get("enabled", True):
    app.add_middleware(
        RateLimitMiddleware,
        rate_limit=config["rate_limiting"]["rate"],
        enabled=True
    )

# Mount static files for serving images
app.mount(
    config["images"]["serve_path"],
    StaticFiles(directory=str(DEFAULT_OUTPUT_DIR.absolute()), check_dir=True),
    name="images"
)
logger.info(f"Serving images from {DEFAULT_OUTPUT_DIR.absolute()} at {config['images']['serve_path']}")


# Cache control middleware
@app.middleware("http")
async def add_cache_control(request: Request, call_next):
    response = await call_next(request)

    if request.url.path.startswith(config["images"]["serve_path"]):
        response.headers["Cache-Control"] = config["images"]["cache_control"]
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response


# ============================================================================
# Request/Response Models
# ============================================================================

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    model: Optional[str] = Field(None, description="Model to use (e.g., 'sdxl', 'flux-schnell')")
    lora: Optional[str] = Field(None, description="LoRA model to use")
    width: Optional[int] = Field(None, ge=64, le=2048)
    height: Optional[int] = Field(None, ge=64, le=2048)
    steps: Optional[int] = Field(None, ge=1, le=150)
    cfg_scale: Optional[float] = Field(None, ge=1.0, le=20.0)
    seed: Optional[int] = Field(-1, description="Random seed (-1 for random)")
    sampling_method: Optional[str] = Field(None)
    negative_prompt: Optional[str] = Field("", description="Negative prompt")
    output_dir: Optional[str] = Field(None)
    return_base64: bool = Field(True, description="Return image as base64")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "a beautiful sunset over mountains",
                "model": "sdxl",
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "cfg_scale": 7.5,
                "seed": -1,
                "sampling_method": "euler_a",
                "negative_prompt": "blur, low quality"
            }
        }


class ImageGenerationResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    markdown_response: str
    model: Optional[str] = None
    prompt: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    image_base64: Optional[str] = None


# ============================================================================
# System Endpoints
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/system", tags=["System"])
async def system_info():
    """Get system information"""
    return {
        "python_version": sys.version,
        "sd_cpp_path": str(SD_CPP_PATH),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "available_models": list(_model_paths.keys()),
        "timestamp": datetime.now().isoformat(),
        "platform": sys.platform
    }


@app.get("/models", tags=["Models"])
async def list_models():
    """List available models and default parameters"""
    try:
        diffugen_config = load_diffugen_config()
        return {
            "models": {
                "flux": ["flux-schnell", "flux-dev"],
                "stable_diffusion": ["sdxl", "sd3", "sd15", "revanimated", "oia"]
            },
            "default_params": diffugen_config.get("default_params", {
                "width": 512,
                "height": 512,
                "steps": 20,
                "cfg_scale": 7.0,
                "sampling_method": "euler"
            })
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/loras", tags=["Models"])
async def list_loras():
    """List available LoRA models"""
    lora_dir = Path(config["paths"]["lora_model_dir"])
    if not lora_dir.is_dir():
        return {"loras": []}

    loras = [
        f.stem for f in lora_dir.iterdir()
        if f.suffix in [".safetensors", ".bin", ".pt"]
    ]
    return {"loras": loras}


@app.get("/images", tags=["Models"])
async def list_images():
    """List all generated images"""
    try:
        DEFAULT_OUTPUT_DIR.mkdir(exist_ok=True)

        images = []
        for file in DEFAULT_OUTPUT_DIR.glob("*.[jp][pn][g]"):
            if not file.exists() or not os.access(str(file), os.R_OK):
                continue

            images.append({
                "filename": file.name,
                "path": f"{config['images']['serve_path']}/{file.name}",
                "created": datetime.fromtimestamp(file.stat().st_ctime).isoformat()
            })

        # Also check for .jpeg files
        for file in DEFAULT_OUTPUT_DIR.glob("*.jpeg"):
            if not file.exists() or not os.access(str(file), os.R_OK):
                continue

            images.append({
                "filename": file.name,
                "path": f"{config['images']['serve_path']}/{file.name}",
                "created": datetime.fromtimestamp(file.stat().st_ctime).isoformat()
            })

        logger.info(f"Found {len(images)} images")
        return {"images": images}

    except Exception as e:
        logger.error(f"Error listing images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Image Generation Endpoints
# ============================================================================

def process_generation_result(
    result: Dict,
    request: ImageGenerationRequest,
    req: Request,
    character: str = None
) -> ImageGenerationResponse:
    """
    Process generation result into standard response format

    Args:
        result: Result dictionary from generation function
        request: Original generation request
        req: FastAPI request object
        character: Optional character name

    Returns:
        ImageGenerationResponse

    Raises:
        HTTPException: If result indicates failure or image not found
    """
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        logger.error(f"Generation failed: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    # Verify image file exists
    image_path = Path(result["image_path"])

    if not wait_for_image_file(image_path):
        error_msg = f"Generated image not found or not readable: {image_path}"
        logger.error(error_msg)
        logger.error(f"Files in output dir: {list(DEFAULT_OUTPUT_DIR.glob('*'))}")
        raise HTTPException(status_code=500, detail=error_msg)

    # Build image URL
    image_url = build_image_url(str(req.base_url), image_path, config['images']['serve_path'])

    # Build markdown response
    markdown_response = build_markdown_response(result, image_url, character)

    # Encode to base64 if requested
    image_base64 = None
    if request.return_base64:
        try:
            image_base64 = encode_image_base64(image_path)
            logger.info(f"Encoded image to base64 ({len(image_base64)} chars)")
        except Exception as e:
            logger.warning(f"Failed to encode image to base64: {e}")

    return ImageGenerationResponse(
        success=True,
        image_path=str(image_path),
        image_url=image_url,
        markdown_response=markdown_response,
        model=result["model"],
        prompt=result["prompt"],
        parameters={
            "width": result["width"],
            "height": result["height"],
            "steps": result["steps"],
            "cfg_scale": result["cfg_scale"],
            "seed": result["seed"],
            "sampling_method": result["sampling_method"],
            "negative_prompt": result.get("negative_prompt", "")
        },
        image_base64=image_base64
    )


@app.post(
    "/generate/stable",
    response_model=ImageGenerationResponse,
    tags=["Image Generation"],
    summary="Generate with Stable Diffusion"
)
async def generate_stable_image(request: ImageGenerationRequest, req: Request):
    """Generate image using Stable Diffusion models (SDXL, SD3, SD15)"""
    try:
        # Default to sd15 if no model specified
        model = request.model or "sd15"

        # Validate model
        model = validate_model(model, "stable")

        # Force random seed for fresh generation
        request.seed = -1

        # Prepare LoRA-enhanced prompt if needed
        prompt = request.prompt
        if request.lora:
            prompt = f"<lora:{request.lora}:1.0> {prompt}"

        # Call generation function
        result = generate_stable_diffusion_image(
            prompt=prompt,
            model=model,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            sampling_method=request.sampling_method,
            negative_prompt=request.negative_prompt,
            output_dir=str(DEFAULT_OUTPUT_DIR),
            lora_model_dir=config.get("paths", {}).get("lora_model_dir")
        )

        return process_generation_result(result, request, req)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_stable_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/generate/flux",
    response_model=ImageGenerationResponse,
    tags=["Image Generation"],
    summary="Generate with Flux Models"
)
async def generate_flux_image_endpoint(request: ImageGenerationRequest, req: Request):
    """Generate image using Flux models (flux-schnell, flux-dev)"""
    try:
        # Default to flux-schnell if no model specified
        model = request.model or "flux-schnell"

        # Validate model
        model = validate_model(model, "flux")

        # Force random seed for fresh generation
        request.seed = -1

        # Call generation function
        result = generate_flux_image(
            prompt=request.prompt,
            model=model,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
            sampling_method=request.sampling_method,
            output_dir=str(DEFAULT_OUTPUT_DIR)
        )

        return process_generation_result(result, request, req)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_flux_image_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/generate",
    response_model=ImageGenerationResponse,
    tags=["Image Generation"],
    summary="Generate Image (Unified)"
)
async def generate_image(request: ImageGenerationRequest, req: Request):
    """
    Unified generation endpoint - automatically routes to appropriate model type
    """
    try:
        # Force random seed
        request.seed = -1

        # Determine model and route to appropriate endpoint
        if request.model:
            if request.model.lower().startswith("flux-"):
                return await generate_flux_image_endpoint(request, req)
            else:
                return await generate_stable_image(request, req)
        else:
            # Use default model from config
            default_model = config.get("default_model", "flux-schnell")
            request.model = default_model

            if default_model.startswith("flux-"):
                return await generate_flux_image_endpoint(request, req)
            else:
                return await generate_stable_image(request, req)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Streaming Endpoints (SSE Support)
# ============================================================================

async def run_generation_with_streaming(
    generation_func: callable,
    streaming_queue: StreamingQueue,
    **kwargs
):
    """
    Run generation in background thread with streaming progress

    Args:
        generation_func: Generation function to call
        streaming_queue: Queue for progress updates
        **kwargs: Arguments for generation function
    """
    tracker = ProgressTracker(
        total_steps=kwargs.get('steps', 20),
        callback=streaming_queue.put
    )

    try:
        # Phase: Initializing
        tracker.update_phase(ProgressPhase.INITIALIZING, "Initializing generation...")
        await asyncio.sleep(0.1)

        # Phase: Loading model
        tracker.update_phase(ProgressPhase.LOADING_MODEL, "Loading model...")
        await asyncio.sleep(0.5)

        # Phase: Preparing
        tracker.update_phase(ProgressPhase.PREPARING, "Preparing generation parameters...")
        await asyncio.sleep(0.2)

        # Phase: Generating (run in thread to avoid blocking)
        tracker.update_phase(ProgressPhase.GENERATING, "Generating image...")

        # Simulate step-by-step progress
        # Since we can't get real progress from sd.cpp, we estimate
        total_steps = kwargs.get('steps', 20)

        # Run generation in executor
        loop = asyncio.get_event_loop()

        # Create a wrapper that updates progress
        async def generate_with_progress():
            # Start generation in thread
            generation_task = loop.run_in_executor(
                None,
                generation_func,
                **kwargs
            )

            # Simulate progress while generating
            for step in range(1, total_steps + 1):
                tracker.update_step(step)
                await asyncio.sleep(0.3)  # Rough estimate per step

                # Check if generation is done
                if generation_task.done():
                    break

            # Wait for actual completion
            result = await generation_task
            return result

        result = await generate_with_progress()

        # Phase: Post-processing
        tracker.update_phase(ProgressPhase.POST_PROCESSING, "Post-processing image...")
        await asyncio.sleep(0.5)

        # Complete
        if result.get("success"):
            tracker.complete(f"Generation complete: {result.get('image_path')}")
        else:
            tracker.error(result.get("error", "Unknown error"))

    except Exception as e:
        logger.error(f"Error in streaming generation: {e}", exc_info=True)
        tracker.error(str(e))
    finally:
        streaming_queue.mark_done()


@app.post(
    "/generate/stable/stream",
    tags=["Image Generation"],
    summary="Generate with Stable Diffusion (Streaming)",
    response_class=StreamingResponse
)
async def generate_stable_image_stream(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate image using Stable Diffusion with real-time progress updates (SSE)

    Returns Server-Sent Events stream with progress updates.
    Client should listen for 'progress' and 'done' events.
    """
    try:
        # Validate model
        model = request.model or "sd15"
        model = validate_model(model, "stable")

        # Prepare generation kwargs
        gen_kwargs = {
            "prompt": request.prompt if not request.lora else f"<lora:{request.lora}:1.0> {request.prompt}",
            "model": model,
            "width": request.width,
            "height": request.height,
            "steps": request.steps or 20,
            "cfg_scale": request.cfg_scale,
            "seed": -1,
            "sampling_method": request.sampling_method,
            "negative_prompt": request.negative_prompt,
            "output_dir": str(DEFAULT_OUTPUT_DIR),
            "lora_model_dir": config.get("paths", {}).get("lora_model_dir")
        }

        # Create streaming queue
        streaming_queue = StreamingQueue()

        # Start generation in background
        background_tasks.add_task(
            run_generation_with_streaming,
            generate_stable_diffusion_image,
            streaming_queue,
            **gen_kwargs
        )

        # Return SSE stream
        return StreamingResponse(
            sse_generator(streaming_queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting streaming generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/generate/flux/stream",
    tags=["Image Generation"],
    summary="Generate with Flux (Streaming)",
    response_class=StreamingResponse
)
async def generate_flux_image_stream(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate image using Flux with real-time progress updates (SSE)

    Returns Server-Sent Events stream with progress updates.
    """
    try:
        # Validate model
        model = request.model or "flux-schnell"
        model = validate_model(model, "flux")

        # Prepare generation kwargs
        gen_kwargs = {
            "prompt": request.prompt,
            "model": model,
            "width": request.width,
            "height": request.height,
            "steps": request.steps or 4,
            "cfg_scale": request.cfg_scale,
            "seed": -1,
            "sampling_method": request.sampling_method,
            "output_dir": str(DEFAULT_OUTPUT_DIR)
        }

        # Create streaming queue
        streaming_queue = StreamingQueue()

        # Start generation in background
        background_tasks.add_task(
            run_generation_with_streaming,
            generate_flux_image,
            streaming_queue,
            **gen_kwargs
        )

        # Return SSE stream
        return StreamingResponse(
            sse_generator(streaming_queue),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting streaming generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/generate/stream",
    tags=["Image Generation"],
    summary="Generate Image (Unified, Streaming)",
    response_class=StreamingResponse
)
async def generate_image_stream(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Unified streaming generation endpoint with real-time progress

    Automatically routes to Flux or Stable Diffusion based on model.
    Returns Server-Sent Events stream.
    """
    try:
        # Determine model type and route
        if request.model:
            if request.model.lower().startswith("flux-"):
                return await generate_flux_image_stream(request, background_tasks)
            else:
                return await generate_stable_image_stream(request, background_tasks)
        else:
            # Use default model
            default_model = config.get("default_model", "flux-schnell")
            request.model = default_model

            if default_model.startswith("flux-"):
                return await generate_flux_image_stream(request, background_tasks)
            else:
                return await generate_stable_image_stream(request, background_tasks)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in unified streaming generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Conversational / Intelligent Generation (Storybook Mode)
# ============================================================================

# Initialize intelligent workflow
intelligent_workflow = IntelligentWorkflow(
    diffugen_base_url="http://localhost:8080",  # Self-reference
    llm_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
)


class ConversationalRequest(BaseModel):
    """Request for conversational generation"""
    session_id: str = Field(..., description="Conversation session ID")
    message: str = Field(..., description="User's natural language request")


class ConversationalResponse(BaseModel):
    """Response from conversational generation"""
    success: bool
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    explanation: str
    parameters: Optional[Dict[str, Any]] = None
    adjustments_applied: Optional[Dict[str, Any]] = None
    safety_warnings: Optional[List[str]] = None
    intent: Optional[str] = None
    error: Optional[str] = None


@app.post(
    "/conversational/generate",
    response_model=ConversationalResponse,
    tags=["Conversational Generation"],
    summary="Conversational Image Generation for Storybooks"
)
async def conversational_generate(request: ConversationalRequest, req: Request):
    """
    Intelligent conversational image generation for children's storybooks.

    Uses natural language to iteratively refine images with LLM-powered parameter tuning.

    Examples:
    - "Create a friendly dragon in a castle"
    - "Make it brighter and more colorful"
    - "Less detailed, more cartoon-like"
    - "Make the dragon bigger"

    The system:
    - Analyzes your natural language request using Qwen LLM
    - Intelligently adjusts parameters (steps, cfg_scale, etc.)
    - Maintains conversation context for iterative refinement
    - Ensures child-appropriate content
    - Remembers what you liked/disliked
    """
    try:
        # Process message through intelligent workflow
        result = await intelligent_workflow.process_message(
            session_id=request.session_id,
            user_message=request.message
        )

        # Build image URL if path exists
        if result.get("image_path") and not result.get("image_url"):
            image_path = Path(result["image_path"])
            result["image_url"] = build_image_url(
                str(req.base_url),
                image_path,
                config['images']['serve_path']
            )

        return ConversationalResponse(**result)

    except Exception as e:
        logger.error(f"Error in conversational generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/conversational/session/{session_id}",
    tags=["Conversational Generation"],
    summary="Get Conversation Session"
)
async def get_conversation_session(session_id: str):
    """
    Get conversation session history and context
    """
    try:
        session = intelligent_workflow.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "message_count": len(session.messages),
            "generation_count": len(session.generation_history),
            "current_parameters": session.current_parameters.to_dict() if session.current_parameters else None,
            "recent_messages": session.messages[-5:],  # Last 5 messages
            "style_preferences": session.style_preferences,
            "character_references": session.character_references
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/conversational/session",
    tags=["Conversational Generation"],
    summary="Create New Conversation Session"
)
async def create_conversation_session():
    """
    Create a new conversation session
    """
    try:
        session_id = str(uuid.uuid4())
        context = intelligent_workflow.create_session(session_id)

        return {
            "session_id": session_id,
            "created_at": context.created_at,
            "message": "New conversation session created. Start by describing what you want to generate!"
        }

    except Exception as e:
        logger.error(f"Error creating session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/conversational/session/{session_id}",
    tags=["Conversational Generation"],
    summary="Delete Conversation Session"
)
async def delete_conversation_session(session_id: str):
    """
    Delete a conversation session
    """
    try:
        if session_id in intelligent_workflow.conversations:
            del intelligent_workflow.conversations[session_id]
            return {"success": True, "message": "Session deleted"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/conversational/sessions",
    tags=["Conversational Generation"],
    summary="List All Sessions"
)
async def list_conversation_sessions():
    """
    List all active conversation sessions
    """
    try:
        sessions = []
        for session_id, context in intelligent_workflow.conversations.items():
            sessions.append({
                "session_id": session_id,
                "created_at": context.created_at,
                "message_count": len(context.messages),
                "generation_count": len(context.generation_history),
                "last_activity": context.messages[-1]["timestamp"] if context.messages else context.created_at
            })

        return {"sessions": sessions, "total": len(sessions)}

    except Exception as e:
        logger.error(f"Error listing sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LoRA Training Endpoints
# ============================================================================

@app.post(
    "/character/train-lora",
    tags=["Character Management"],
    summary="Train LoRA for Character"
)
async def train_character_lora(request: Request):
    """
    Train a LoRA model for perfect character consistency

    Request body:
    {
        "character_name": "Spark",
        "num_additional_images": 10,
        "epochs": 10
    }
    """
    try:
        data = await request.json()
        character_name = data.get("character_name")

        if not character_name:
            raise HTTPException(status_code=400, detail="character_name is required")

        # Get character
        character = intelligent_workflow.character_engine.library.get_character(character_name)
        if not character:
            raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")

        # Train LoRA
        success, lora_path, error = await intelligent_workflow.character_engine.train_character_lora(
            character=character,
            num_additional_images=data.get("num_additional_images", 10),
            epochs=data.get("epochs", 10)
        )

        if success:
            return {
                "success": True,
                "character_name": character_name,
                "lora_path": lora_path,
                "message": f"LoRA training complete for {character_name}"
            }
        else:
            raise HTTPException(status_code=500, detail=error or "LoRA training failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training LoRA: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/character/{character_name}/lora-status",
    tags=["Character Management"],
    summary="Get LoRA Training Status"
)
async def get_lora_status(character_name: str):
    """Get LoRA training status for a character"""
    try:
        character = intelligent_workflow.character_engine.library.get_character(character_name)
        if not character:
            raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")

        return {
            "character_name": character_name,
            "has_lora": character.has_lora,
            "lora_path": character.lora_path,
            "lora_weight": character.lora_weight
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting LoRA status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Style Locking Endpoints
# ============================================================================

@app.post(
    "/style/lock",
    tags=["Style Management"],
    summary="Lock Art Style"
)
async def lock_style(request: Request):
    """
    Lock an art style for a session

    Request body:
    {
        "session_id": "session-123",
        "style_name": "watercolor_soft"
    }
    """
    try:
        data = await request.json()
        session_id = data.get("session_id")
        style_name = data.get("style_name")

        if not session_id or not style_name:
            raise HTTPException(status_code=400, detail="session_id and style_name are required")

        # Lock style
        success = intelligent_workflow.style_manager.lock_style(session_id, style_name)

        if success:
            # Update session context
            if session_id in intelligent_workflow.conversations:
                intelligent_workflow.conversations[session_id].locked_style = style_name

            style = intelligent_workflow.style_manager.get_locked_style(session_id)
            return {
                "success": True,
                "style_name": style.name,
                "style_description": style.description,
                "message": f"Locked style to '{style.name}'"
            }
        else:
            available_styles = list(intelligent_workflow.style_manager.library.styles.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Style '{style_name}' not found. Available: {', '.join(available_styles)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error locking style: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/style/unlock",
    tags=["Style Management"],
    summary="Unlock Art Style"
)
async def unlock_style(request: Request):
    """
    Unlock art style for a session

    Request body:
    {
        "session_id": "session-123"
    }
    """
    try:
        data = await request.json()
        session_id = data.get("session_id")

        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        # Unlock style
        intelligent_workflow.style_manager.unlock_style(session_id)

        # Update session context
        if session_id in intelligent_workflow.conversations:
            intelligent_workflow.conversations[session_id].locked_style = None

        return {
            "success": True,
            "message": "Style lock removed"
        }

    except Exception as e:
        logger.error(f"Error unlocking style: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/style/list",
    tags=["Style Management"],
    summary="List Available Styles"
)
async def list_styles():
    """List all available art styles"""
    try:
        styles = intelligent_workflow.style_manager.library.list_styles()

        return {
            "styles": [
                {
                    "name": style.name,
                    "description": style.description,
                    "technique": style.technique,
                    "color_palette": style.color_palette,
                    "mood": style.mood
                }
                for style in styles
            ],
            "total": len(styles)
        }

    except Exception as e:
        logger.error(f"Error listing styles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Batch Generation Endpoints
# ============================================================================

@app.post(
    "/batch/create",
    tags=["Batch Generation"],
    summary="Create Batch Generation Job"
)
async def create_batch(request: Request):
    """
    Create a batch generation job

    Request body:
    {
        "session_id": "session-123",
        "scene_descriptions": ["castle", "forest", "beach"],
        "character_name": "Spark",
        "style_name": "watercolor_soft",
        "use_lora": true
    }
    """
    try:
        data = await request.json()
        session_id = data.get("session_id")
        scene_descriptions = data.get("scene_descriptions", [])

        if not session_id or not scene_descriptions:
            raise HTTPException(status_code=400, detail="session_id and scene_descriptions are required")

        # Create batch
        batch = await intelligent_workflow.batch_manager.create_batch(
            session_id=session_id,
            scene_descriptions=scene_descriptions,
            character_name=data.get("character_name"),
            style_name=data.get("style_name"),
            use_lora=data.get("use_lora", False)
        )

        # Update session context
        if session_id in intelligent_workflow.conversations:
            intelligent_workflow.conversations[session_id].active_batch_id = batch.batch_id

        # Start execution
        asyncio.create_task(intelligent_workflow.batch_manager.execute_batch(batch.batch_id))

        return {
            "success": True,
            "batch_id": batch.batch_id,
            "total_scenes": batch.total_scenes,
            "message": f"Started batch generation of {batch.total_scenes} scenes"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/batch/{batch_id}/status",
    tags=["Batch Generation"],
    summary="Get Batch Status"
)
async def get_batch_status(batch_id: str):
    """Get status of a batch generation job"""
    try:
        batch = intelligent_workflow.batch_manager.get_batch(batch_id)

        if not batch:
            raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")

        return batch.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/batch/{batch_id}/cancel",
    tags=["Batch Generation"],
    summary="Cancel Batch Job"
)
async def cancel_batch(batch_id: str):
    """Cancel a batch generation job"""
    try:
        success = intelligent_workflow.batch_manager.cancel_batch(batch_id)

        if success:
            return {
                "success": True,
                "message": f"Batch {batch_id} cancelled"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Character Relationship Endpoints
# ============================================================================

@app.post(
    "/relationship/add",
    tags=["Character Relationships"],
    summary="Add Character Relationship"
)
async def add_relationship(request: Request):
    """
    Add a relationship between two characters

    Request body:
    {
        "character_a": "Spark",
        "character_b": "Whiskers",
        "relationship_type": "friend",
        "description": "best friends"
    }
    """
    try:
        from character_relationships import RelationType

        data = await request.json()
        char_a = data.get("character_a")
        char_b = data.get("character_b")
        rel_type_str = data.get("relationship_type", "friend")

        if not char_a or not char_b:
            raise HTTPException(status_code=400, detail="character_a and character_b are required")

        # Map string to RelationType
        rel_type_map = {
            "friend": RelationType.FRIEND,
            "sibling": RelationType.SIBLING,
            "companion": RelationType.COMPANION,
            "rival": RelationType.RIVAL,
            "mentor": RelationType.MENTOR,
            "student": RelationType.STUDENT,
            "teammate": RelationType.TEAMMATE,
            "family": RelationType.FAMILY,
            "parent": RelationType.PARENT,
            "child": RelationType.CHILD
        }

        rel_type = rel_type_map.get(rel_type_str.lower(), RelationType.FRIEND)

        # Add relationship
        relationship = intelligent_workflow.relationship_graph.add_relationship(
            character_a=char_a,
            character_b=char_b,
            relationship_type=rel_type,
            description=data.get("description", "")
        )

        return {
            "success": True,
            "character_a": char_a,
            "character_b": char_b,
            "relationship_type": rel_type.value,
            "message": relationship.get_relationship_description()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding relationship: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/relationship/{character_name}",
    tags=["Character Relationships"],
    summary="Get Character Relationships"
)
async def get_relationships(character_name: str):
    """Get all relationships for a character"""
    try:
        relationships = intelligent_workflow.relationship_graph.get_relationships(character_name)

        return {
            "character_name": character_name,
            "relationships": [
                {
                    "related_character": rel.character_b,
                    "relationship_type": rel.relationship_type.value,
                    "description": rel.get_relationship_description()
                }
                for rel in relationships
            ],
            "total": len(relationships)
        }

    except Exception as e:
        logger.error(f"Error getting relationships: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="DiffuGen OpenAPI Server")
    parser.add_argument("--host", type=str, help="Host to bind")
    parser.add_argument("--port", type=int, help="Port to bind")
    parser.add_argument("--config", type=str, help="Custom config file")
    args = parser.parse_args()

    # Override config with command line args
    host = args.host or config["server"]["host"]
    port = args.port or config["server"]["port"]

    # Load custom config if specified
    if args.config:
        try:
            with open(args.config) as f:
                custom_config = json.load(f)
                config.update(custom_config)
                logger.info(f"Loaded custom configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading custom config: {e}")

    logger.info(f"Starting DiffuGen OpenAPI server at http://{host}:{port}")
    logger.info(f"Documentation: http://{host}:{port}/docs")
    logger.info(f"Serving images at {host}:{port}{config['images']['serve_path']}")

    uvicorn.run(app, host=host, port=port)
