"""
LangGraph Orchestrator Server for DiffuGen
Production-grade agentic system for image generation and refinement
"""

import os
import base64
import logging
from typing import Optional, Dict, Any
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from workflow import create_image_generation_workflow
from vram_manager import VRAMOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="DiffuGen Agentic Orchestrator",
    description="Self-refining image generation agent with VRAM optimization",
    version="2.0.0"
)

# Initialize VRAM manager
vram_manager = VRAMOrchestrator(
    vllm_base_url=os.getenv("VLLM_API_BASE", "http://vllm:5000/v1"),
    diffugen_base_url=os.getenv("DIFFUGEN_MCP_BASE", "http://diffugen-mcp:8080"),
    enable_orchestration=os.getenv("ENABLE_VRAM_ORCHESTRATION", "true").lower() == "true"
)

# Initialize workflow
workflow = create_image_generation_workflow(vram_manager)


class GenerateRequest(BaseModel):
    """Request for text-to-image generation"""
    prompt: str = Field(..., description="User's image generation request")
    model: Optional[str] = Field("sd15", description="Model to use")
    width: Optional[int] = Field(512, ge=256, le=2048)
    height: Optional[int] = Field(512, ge=256, le=2048)
    enable_critique: bool = Field(True, description="Enable self-refinement loop")
    max_iterations: int = Field(2, ge=1, le=5, description="Max refinement iterations")


class RefineRequest(BaseModel):
    """Request for image refinement"""
    prompt: str = Field(..., description="Refinement instructions")
    model: Optional[str] = Field("sd15", description="Model to use")
    strength: float = Field(0.5, ge=0.0, le=1.0)
    enable_critique: bool = Field(True, description="Enable self-refinement loop")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check vLLM connectivity
        vllm_healthy = await vram_manager.check_vllm_health()

        # Check DiffuGen connectivity
        diffugen_healthy = await vram_manager.check_diffugen_health()

        return {
            "status": "healthy" if (vllm_healthy and diffugen_healthy) else "degraded",
            "services": {
                "vllm": "healthy" if vllm_healthy else "unhealthy",
                "diffugen": "healthy" if diffugen_healthy else "unhealthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/generate")
async def generate_image(request: GenerateRequest):
    """
    Generate an image from text prompt

    This endpoint triggers the full agentic workflow:
    1. Prompt refinement (if enabled)
    2. VRAM-optimized generation
    3. Self-critique loop (if enabled)
    4. Returns the final image as base64
    """
    try:
        logger.info(f"Generate request: {request.prompt[:100]}...")

        # Run the workflow
        result = await workflow.run({
            "user_input": request.prompt,
            "task_type": "generate",
            "parameters": {
                "model": request.model,
                "width": request.width,
                "height": request.height,
            },
            "enable_critique": request.enable_critique,
            "max_iterations": request.max_iterations
        })

        return {
            "success": True,
            "image_base64": result["final_image_base64"],
            "image_path": result["image_path"],
            "refined_prompt": result["final_prompt"],
            "iterations": result["iteration_count"],
            "critique_history": result.get("critique_history", [])
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refine")
async def refine_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    model: Optional[str] = Form("sd15"),
    strength: float = Form(0.5),
    enable_critique: bool = Form(True)
):
    """
    Refine an uploaded image

    THE MAGIC WORKFLOW - This is what you wanted!
    1. User uploads image
    2. Server encodes to base64 (using System RAM)
    3. LangGraph injects base64 into tool calls
    4. DiffuGen processes it
    5. Returns refined image as base64

    NO FILE PATHS. NO MANUAL SAVES. PURE MAGIC.
    """
    try:
        logger.info(f"Refine request: {prompt[:100]}...")

        # Read uploaded image into memory
        image_bytes = await image.read()

        # Encode to base64 (THIS SOLVES THE BASE64 BLOCKER)
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Detect format
        if image_bytes.startswith(b'\xff\xd8'):
            mime_type = "image/jpeg"
        elif image_bytes.startswith(b'\x89PNG'):
            mime_type = "image/png"
        elif image_bytes.startswith(b'RIFF'):
            mime_type = "image/webp"
        else:
            mime_type = "image/png"

        # Create data URI
        image_data_uri = f"data:{mime_type};base64,{image_base64}"

        logger.info(f"Encoded image: {len(image_bytes)} bytes → {len(image_base64)} base64 chars")

        # Run the workflow with injected base64
        result = await workflow.run({
            "user_input": prompt,
            "task_type": "refine",
            "init_image_base64": image_data_uri,  # ← INJECTION POINT
            "parameters": {
                "model": model,
                "strength": strength,
            },
            "enable_critique": enable_critique,
            "max_iterations": 2
        })

        return {
            "success": True,
            "image_base64": result["final_image_base64"],
            "image_path": result["image_path"],
            "refined_prompt": result["final_prompt"],
            "iterations": result["iteration_count"]
        }

    except Exception as e:
        logger.error(f"Refinement failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/generations")
async def openai_compatible_generate(request: dict):
    """
    OpenAI-compatible image generation endpoint for Open WebUI
    Expects: {"prompt": "...", "n": 1, "size": "512x512"}
    Returns: {"created": timestamp, "data": [{"url": "data:image/png;base64,..."}]}
    """
    try:
        prompt = request.get("prompt", "")
        size = request.get("size", "512x512")
        width, height = map(int, size.split("x"))

        logger.info(f"OpenAI-compatible generate request: {prompt[:100]}...")

        # Run the workflow
        result = await workflow.run({
            "user_input": prompt,
            "task_type": "generate",
            "init_image_base64": None,
            "parameters": {
                "model": "sd15",
                "width": width,
                "height": height,
            },
            "enable_critique": True,
            "max_iterations": 2
        })

        # Return OpenAI-compatible format
        import time
        return {
            "created": int(time.time()),
            "data": [{
                "url": f"data:image/png;base64,{result['final_image_base64']}"
            }]
        }

    except Exception as e:
        logger.error(f"OpenAI-compatible generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/edits")
async def openai_compatible_edit(
    prompt: str = Form(...),
    image: UploadFile = File(...),
):
    """
    OpenAI-compatible image editing endpoint for Open WebUI.
    This endpoint handles image-to-image tasks.
    """
    try:
        logger.info(f"OpenAI-compatible edit request: {prompt[:100]}...")

        # Read uploaded image into memory
        image_bytes = await image.read()

        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Detect format
        if image_bytes.startswith(b'\xff\xd8'):
            mime_type = "image/jpeg"
        elif image_bytes.startswith(b'\x89PNG'):
            mime_type = "image/png"
        elif image_bytes.startswith(b'RIFF'):
            mime_type = "image/webp"
        else:
            mime_type = "image/png"

        # Create data URI
        image_data_uri = f"data:{mime_type};base64,{image_base64}"

        # Run the workflow
        result = await workflow.run({
            "user_input": prompt,
            "task_type": "refine",
            "init_image_base64": image_data_uri,
            "parameters": {
                "model": "sd15",
                "strength": 0.75, # Default strength for edits
            },
            "enable_critique": True,
            "max_iterations": 2
        })

        # Return OpenAI-compatible format
        import time
        return {
            "created": int(time.time()),
            "data": [{
                "url": f"data:image/png;base64,{result['final_image_base64']}"
            }]
        }

    except Exception as e:
        logger.error(f"OpenAI-compatible edit failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def openai_compatible_models():
    """
    OpenAI-compatible models endpoint for Open WebUI
    """
    return {
        "data": [
            {
                "id": "dall-e-3",
                "object": "model",
                "created": 1677610600,
                "owned_by": "openai"
            }
        ],
        "object": "list"
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "DiffuGen Agentic Orchestrator",
        "version": "2.0.0",
        "endpoints": {
            "generate": "/generate - Text-to-image generation",
            "refine": "/refine - Image refinement with upload",
            "health": "/health - Health check",
            "openai_compat": "/v1/images/generations - OpenAI-compatible endpoint"
        },
        "features": [
            "Self-refining prompts",
            "VRAM-optimized execution",
            "Base64 image handling",
            "Automatic critique loop",
            "Self-healing on errors"
        ]
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "agent_server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
