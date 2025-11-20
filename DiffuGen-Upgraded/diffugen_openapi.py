"""
DiffuGen OpenAPI Server
Professional REST API for Open WebUI integration with Stable Diffusion and Flux models
"""

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union, Any
import os
import sys
import json
import time
import re
import argparse
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
