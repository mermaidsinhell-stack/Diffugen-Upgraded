import sys
import os
import logging
import subprocess
import uuid
import re
import argparse
import json
from pathlib import Path
import random
import time
import threading
import atexit
import base64
import tempfile

# Import new modules for character consistency and adetailer
from character_manager import CharacterManager
from adetailer import Adetailer
from preprocessor import Preprocessor

# Simplified logging setup - log only essential info
logging.basicConfig(
    filename='diffugen_debug.log',
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Queue management system
class GenerationQueue:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_busy = False
        self.lock_file = os.path.join(os.getcwd(), "diffugen.lock")
        # Clean up any stale lock on startup
        self._remove_lock_file()
        # Register cleanup on exit
        atexit.register(self._remove_lock_file)
    
    def acquire(self, timeout=0):
        """Try to acquire the lock for image generation.
        Returns True if successful, False if busy."""
        with self.lock:
            # First check local thread lock
            if self.is_busy:
                logging.info("Image generation already in progress (local lock)")
                return False
                
            # Then check file lock (for inter-process locking)
            if os.path.exists(self.lock_file):
                try:
                    # Check if the lock file is stale (older than 30 minutes)
                    lock_time = os.path.getmtime(self.lock_file)
                    if time.time() - lock_time > 1800:  # 30 minutes
                        logging.warning("Found stale lock file, removing it")
                        self._remove_lock_file()
                    else:
                        with open(self.lock_file, 'r') as f:
                            pid = f.read().strip()
                        logging.info(f"Image generation already in progress by process {pid}")
                        return False
                except Exception as e:
                    logging.error(f"Error checking lock file: {e}")
                    return False
            
            # If we got here, no active generation is running
            try:
                with open(self.lock_file, 'w') as f:
                    f.write(str(os.getpid()))
                self.is_busy = True
                logging.info(f"Acquired generation lock (PID: {os.getpid()})")
                return True
            except Exception as e:
                logging.error(f"Error creating lock file: {e}")
                return False
    
    def release(self):
        """Release the generation lock."""
        with self.lock:
            self.is_busy = False
            self._remove_lock_file()
            logging.info("Released generation lock")
    
    def _remove_lock_file(self):
        """Remove the lock file if it exists."""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except Exception as e:
            logging.error(f"Error removing lock file: {e}")

# Create global generation queue
generation_queue = GenerationQueue()

# Helper function to print to stderr
def log_to_stderr(message):
    print(message, file=sys.stderr, flush=True)

# Try to import real FastMCP with minimal error handling
mcp = None
try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("DiffuGen")
except ImportError as e:
    log_to_stderr(f"Error importing FastMCP: {e}")
    
    # Simple fallback MCP implementation
    class FallbackMCP:
        def __init__(self, name):
            self.name = name
            log_to_stderr(f"Using fallback MCP server: {name}")
        
        def tool(self, *args, **kwargs):
            def decorator(func): return func
            return decorator
        
        def start(self): return True
    
    mcp = FallbackMCP("DiffuGen")

if mcp is None:
    log_to_stderr("Failed to create MCP server")
    sys.exit(1)

# Function to load configuration
def _get_local_file_path(file_reference: str) -> str:
    """Smart file path resolution for images.

    Handles cases where LM Studio or other clients provide paths that don't exist
    in the Docker container. Searches multiple locations and falls back to filename matching.

    Args:
        file_reference: Path or filename to resolve

    Returns:
        Resolved absolute path, or None if not found
    """
    if not file_reference:
        return None

    # If the exact path exists, return it immediately
    if os.path.exists(file_reference):
        return os.path.abspath(file_reference)

    # Extract just the filename from the reference
    filename = os.path.basename(file_reference)

    # Search directories in priority order
    search_dirs = [
        '/app/inputs',      # Primary location for uploaded images
        '/app/outputs',     # Previously generated images
        '/app',             # Current working directory in Docker
        os.getcwd(),        # Fallback to actual cwd
    ]

    # Search for exact filename match
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        candidate_path = os.path.join(search_dir, filename)
        if os.path.exists(candidate_path):
            logging.info(f"Found image at: {candidate_path}")
            return os.path.abspath(candidate_path)

    # If no exact match, try fuzzy matching (in case of encoding issues with special chars)
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        try:
            for file in os.listdir(search_dir):
                # Skip hidden files and non-image files
                if file.startswith('.'):
                    continue
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    continue

                # Check if filenames are similar (handles encoding issues)
                if file.lower() == filename.lower():
                    candidate_path = os.path.join(search_dir, file)
                    logging.info(f"Found image with case-insensitive match: {candidate_path}")
                    return os.path.abspath(candidate_path)
        except Exception as e:
            logging.warning(f"Error searching directory {search_dir}: {e}")
            continue

    # Last resort: return the most recent image file in /app/inputs if no filename provided
    # or if we couldn't find a match (useful for "use the latest uploaded image" workflows)
    if file_reference.lower() in ['latest', 'last', 'recent'] or '/' not in file_reference:
        for search_dir in ['/app/inputs', '/app/outputs']:
            if not os.path.exists(search_dir):
                continue

            try:
                image_files = [
                    os.path.join(search_dir, f) for f in os.listdir(search_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))
                    and not f.startswith('.')
                ]

                if image_files:
                    # Sort by modification time, most recent first
                    most_recent = max(image_files, key=os.path.getmtime)
                    logging.info(f"Using most recent image: {most_recent}")
                    return os.path.abspath(most_recent)
            except Exception as e:
                logging.warning(f"Error finding recent images in {search_dir}: {e}")
                continue

    # Could not resolve the file reference
    logging.warning(f"Could not resolve file reference: {file_reference}")
    return None

def _save_base64_image(base64_data: str, prefix: str = "input") -> str:
    """Save a base64 encoded image to a temporary file.

    Args:
        base64_data: Base64 encoded image string (with or without data URI prefix)
        prefix: Prefix for the temp filename

    Returns:
        Absolute path to the saved temporary file
    """
    try:
        if not base64_data:
            raise ValueError("Empty base64 data provided")

        original_data = base64_data

        # Remove data URI prefix if present (e.g., "data:image/png;base64,")
        if ',' in base64_data:
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',', 1)[1]
            elif 'base64,' in base64_data:
                # Handle cases where it might be malformed
                base64_data = base64_data.split('base64,', 1)[1]

        # Strip whitespace, newlines, and any other non-base64 characters
        base64_data = base64_data.strip()
        base64_data = base64_data.replace('\n', '').replace('\r', '').replace(' ', '')

        # Validate base64 string (basic check)
        if not base64_data or len(base64_data) < 100:
            # Check if it's a common placeholder
            if base64_data in ["base64_encoded_image_data", "BASE64_DATA_HERE", "base64_data", "..."]:
                raise ValueError(
                    f"Received placeholder '{base64_data}' instead of actual base64 image data. "
                    "Your LM client doesn't support automatic image encoding yet. "
                    "Please use init_image_path instead by saving your image to the inputs/ folder."
                )
            raise ValueError(f"Base64 string too short or invalid (length: {len(base64_data)})")

        # Try to detect image format from base64 header BEFORE decoding
        ext = '.png'  # default
        if base64_data.startswith('/9j/'):
            ext = '.jpg'
        elif base64_data.startswith('iVBORw'):
            ext = '.png'
        elif base64_data.startswith('UklGR'):
            ext = '.webp'
        elif base64_data.startswith('Qk'):
            ext = '.bmp'

        # Decode base64 data with validation
        try:
            image_data = base64.b64decode(base64_data, validate=True)
        except Exception as decode_error:
            # Try without strict validation
            logging.warning(f"Strict base64 decode failed, trying lenient decode: {decode_error}")
            try:
                image_data = base64.b64decode(base64_data, validate=False)
            except Exception as e2:
                raise ValueError(f"Base64 decode failed even with lenient mode. Data starts with: {base64_data[:50]}...")

        # Verify we got actual image data
        if not image_data or len(image_data) < 100:
            raise ValueError(f"Decoded image data is too small ({len(image_data)} bytes), likely not a valid image")

        # Create temporary file in /app/inputs (so it's accessible and cleanable)
        temp_dir = '/app/inputs' if os.path.exists('/app/inputs') else tempfile.gettempdir()

        # Create temp file
        temp_file = os.path.join(temp_dir, f"{prefix}_{uuid.uuid4().hex}{ext}")

        with open(temp_file, 'wb') as f:
            f.write(image_data)

        # Verify file was created and has content
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 100:
            raise ValueError(f"Failed to write image file or file is too small")

        logging.info(f"Saved base64 image to: {temp_file} ({len(image_data)} bytes, format: {ext})")
        return temp_file

    except ValueError as ve:
        # Re-raise ValueError with context
        logging.error(f"Base64 image validation error: {ve}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error saving base64 image: {e}")
        logging.error(f"Base64 data length: {len(base64_data) if base64_data else 0}")
        if base64_data:
            logging.error(f"Base64 data starts with: {base64_data[:100]}...")
        raise ValueError(f"Failed to decode and save base64 image: {str(e)}")

def _encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        base64_data = base64.b64encode(image_data).decode('utf-8')

        # Add data URI prefix based on file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp'
        }.get(ext, 'image/png')

        return f"data:{mime_type};base64,{base64_data}"

    except Exception as e:
        logging.error(f"Error encoding image to base64: {e}")
        raise ValueError(f"Failed to encode image to base64: {e}")

def _handle_image_input(image_path: str = None, image_base64: str = None) -> str:
    """Handle image input from either file path or base64 data.

    Args:
        image_path: Path to image file (optional)
        image_base64: Base64 encoded image (optional)

    Returns:
        Resolved absolute path to the image file
    """
    if image_base64:
        # Base64 image provided - save it and return path
        return _save_base64_image(image_base64)
    elif image_path:
        # File path provided - resolve it
        return _get_local_file_path(image_path)
    else:
        return None

def load_config():
    config = {
        "sd_cpp_path": os.path.join(os.getcwd(), "stable-diffusion.cpp"),
        "models_dir": None,  # Will be set based on sd_cpp_path if not provided
        "lora_model_dir": "/app/loras",  # Default LoRA directory for Docker
        "output_dir": os.getcwd(),
        "default_model": None,  # No default model, will be determined by function
        "vram_usage": "adaptive",
        "gpu_layers": -1,
        "default_params": {
            "width": 512,
            "height": 512,
            "steps": {
                "flux-schnell": 8,
                "flux-dev": 20,
                "sdxl": 20,
                "sd3": 20,
                "sd15": 20
            },
            "cfg_scale": {
                "flux-schnell": 1.0,
                "flux-dev": 1.0,
                "sdxl": 7.0,
                "sd3": 7.0,
                "sd15": 7.0
            },
            "sampling_method": "euler"
        }
    }
    
    # Try to read from environment variables first (highest priority)
    if "SD_CPP_PATH" in os.environ:
        config["sd_cpp_path"] = os.path.normpath(os.environ.get("SD_CPP_PATH"))
        logging.info(f"Using SD_CPP_PATH from environment: {config['sd_cpp_path']}")
    
    if "DIFFUGEN_OUTPUT_DIR" in os.environ:
        config["output_dir"] = os.path.normpath(os.environ.get("DIFFUGEN_OUTPUT_DIR"))
        logging.info(f"Using output_dir from environment: {config['output_dir']}")
    
    if "DIFFUGEN_DEFAULT_MODEL" in os.environ:
        config["default_model"] = os.environ.get("DIFFUGEN_DEFAULT_MODEL")
        logging.info(f"Using default_model from environment: {config['default_model']}")
    
    if "DIFFUGEN_VRAM_USAGE" in os.environ:
        config["vram_usage"] = os.environ.get("DIFFUGEN_VRAM_USAGE")
        logging.info(f"Using vram_usage from environment: {config['vram_usage']}")

    if "DIFFUGEN_LORA_MODEL_DIR" in os.environ:
        config["lora_model_dir"] = os.path.normpath(os.environ.get("DIFFUGEN_LORA_MODEL_DIR"))
        logging.info(f"Using lora_model_dir from environment: {config['lora_model_dir']}")

    # Try to read from diffugen.json configuration (second priority)
    try:
        diffugen_json_path = os.path.join(os.getcwd(), "diffugen.json")
        if os.path.exists(diffugen_json_path):
            logging.info(f"Loading configuration from {diffugen_json_path}")
            with open(diffugen_json_path, 'r') as f:
                diffugen_config = json.load(f)
                
                # Extract values from the mcpServers.diffugen structure
                if 'mcpServers' in diffugen_config and 'diffugen' in diffugen_config.get('mcpServers', {}):
                    server_config = diffugen_config['mcpServers']['diffugen']
                    
                    # Extract environment variables
                    if 'env' in server_config:
                        env_vars = server_config['env']
                        if 'SD_CPP_PATH' in env_vars and 'SD_CPP_PATH' not in os.environ:
                            config['sd_cpp_path'] = os.path.normpath(env_vars['SD_CPP_PATH'])
                            logging.info(f"Using sd_cpp_path from diffugen.json: {config['sd_cpp_path']}")
                        
                        if 'default_model' in env_vars and 'DIFFUGEN_DEFAULT_MODEL' not in os.environ:
                            config['default_model'] = env_vars['default_model']
                            logging.info(f"Using default_model from diffugen.json: {config['default_model']}")
                    
                    # Extract resources
                    if 'resources' in server_config:
                        resources = server_config['resources']
                        if 'output_dir' in resources and 'DIFFUGEN_OUTPUT_DIR' not in os.environ:
                            config['output_dir'] = os.path.normpath(resources['output_dir'])
                            logging.info(f"Using output_dir from diffugen.json: {config['output_dir']}")
                        
                        if 'models_dir' in resources:
                            config['models_dir'] = os.path.normpath(resources['models_dir'])
                            logging.info(f"Using models_dir from diffugen.json: {config['models_dir']}")
                        
                        if 'vram_usage' in resources and 'DIFFUGEN_VRAM_USAGE' not in os.environ:
                            config['vram_usage'] = resources['vram_usage']
                            logging.info(f"Using vram_usage from diffugen.json: {config['vram_usage']}")

                        if 'lora_model_dir' in resources and 'DIFFUGEN_LORA_MODEL_DIR' not in os.environ:
                            config['lora_model_dir'] = os.path.normpath(resources['lora_model_dir'])
                            logging.info(f"Using lora_model_dir from diffugen.json: {config['lora_model_dir']}")

                        if 'gpu_layers' in resources:
                            config['gpu_layers'] = resources['gpu_layers']
                            logging.info(f"Using gpu_layers from diffugen.json: {config['gpu_layers']}")
                    
                    # Extract default_params
                    if 'default_params' in server_config:
                        config['default_params'] = server_config['default_params']
                        logging.info("Loaded default_params from diffugen.json")
    except Exception as e:
        logging.warning(f"Error loading diffugen.json configuration: {e}")
    
    # If models_dir wasn't set, use sd_cpp_path/models
    if not config["models_dir"]:
        config["models_dir"] = os.path.join(config["sd_cpp_path"], "models")
    
    return config

# Load the configuration
config = load_config()

# Core path initialization (using the config we loaded)
sd_cpp_path = os.path.normpath(config["sd_cpp_path"])
default_output_dir = os.path.normpath(config["output_dir"])

# Create output directory
os.makedirs(default_output_dir, exist_ok=True)

# Initialize character manager and adetailer
character_manager = CharacterManager()
adetailer_instance = Adetailer(
    sd_binary_path=os.path.join(sd_cpp_path, "build", "bin", "sd"),
    models_dir=config["models_dir"]
)
preprocessor_instance = Preprocessor(yolo_models_dir="yolo_models", output_dir=default_output_dir)

# Helper functions to get model-specific parameters from config
def get_default_steps(model):
    """Get default steps for a model"""
    model = model.lower()
    # Try to get from model-specific defaults, fall back to general default
    try:
        if isinstance(config["default_params"]["steps"], dict):
            return config["default_params"]["steps"].get(model, 20)
        else:
            return config["default_params"].get("steps", 20)
    except (KeyError, TypeError):
        logging.warning(f"Could not find default steps for model {model}, using fallback value of 20")
        return 20

def get_default_cfg_scale(model):
    """Get default CFG scale for a model"""
    model = model.lower()
    # Try to get from model-specific defaults, fall back to general default
    try:
        if isinstance(config["default_params"]["cfg_scale"], dict):
            # For Flux models, default to 1.0, for others default to 7.0
            default_value = 1.0 if model.startswith("flux-") else 7.0
            return config["default_params"]["cfg_scale"].get(model, default_value)
        else:
            return config["default_params"].get("cfg_scale", 7.0)
    except (KeyError, TypeError):
        # For Flux models, default to 1.0, for others default to 7.0
        default_value = 1.0 if model.startswith("flux-") else 7.0
        logging.warning(f"Could not find default cfg_scale for model {model}, using fallback value of {default_value}")
        return default_value

def get_default_sampling_method(model=None):
    """Get default sampling method, optionally model-specific"""
    try:
        # First try to get model-specific sampling method if provided
        if model and isinstance(config["default_params"]["sampling_method"], dict):
            model = model.lower()
            return config["default_params"]["sampling_method"].get(model, "euler")
        # Otherwise get general default
        elif isinstance(config["default_params"]["sampling_method"], dict):
            return config["default_params"]["sampling_method"].get("default", "euler")
        else:
            return config["default_params"].get("sampling_method", "euler")
    except (KeyError, TypeError):
        logging.warning("Could not find default sampling method, using fallback value of 'euler'")
        return "euler"

# Lazy-loaded model paths - only resolved when needed
_model_paths = {}
_supporting_files = {}

def get_model_path(model_name):
    """Lazy-load model paths only when needed"""
    if not _model_paths:
        # Initialize paths only on first access
        models_dir = config["models_dir"]
        _model_paths.update({
            "flux-schnell": os.path.join(models_dir, "flux", "flux1-schnell-q8_0.gguf"),
            "flux-dev": os.path.join(models_dir, "flux", "flux1-dev-q8_0.gguf"),
            "sdxl": os.path.join(models_dir, "sdxl-1.0-base.safetensors"),
            "sd3": os.path.join(models_dir, "sd3-medium.safetensors"),
            "sd15": os.path.join(models_dir, "sd15", "sd15.safetensors"),
            "revanimated": os.path.join(models_dir, "revAnimated_v2Rebirth.safetensors"),
            "oia": os.path.join(models_dir, "OIA Illustrator_0.10.safetensors"),
        })
    return _model_paths.get(model_name)

def get_supporting_file(file_name):
    """Lazy-load supporting file paths only when needed"""
    if not _supporting_files:
        # Initialize paths only on first access
        models_dir = config["models_dir"]
        _supporting_files.update({
            # Flux/SD3 supporting files
            "flux_vae": os.path.join(models_dir, "ae.sft"),
            "clip_l": os.path.join(models_dir, "clip_l.safetensors"),
            "t5xxl": os.path.join(models_dir, "t5xxl_fp16.safetensors"),
            # SDXL supporting files
            "sdxl_vae": os.path.join(models_dir, "sdxl_vae-fp16-fix.safetensors"),
            # SD1.5 supporting files (optional - model usually has built-in VAE)
            "sd15_vae": os.path.join(models_dir, "sd15_vae.safetensors"),
            "sd15_clip": os.path.join(models_dir, "sd15_clip.safetensors"),
        })
    return _supporting_files.get(file_name)

def get_model_defaults(model_name):
    """Return optimal settings for specific models based on creator recommendations"""
    defaults = {
        "oia": {
            "sampling_method": "euler_a",
            "steps": 25,
            "cfg_scale": 7.5,
            "clip_skip": 3,
        },
        "revanimated": {
            "sampling_method": "euler_a",
            "steps": 25,
            "cfg_scale": 7.0,
            "clip_skip": 2,
        },
        # Generic SD 1.5 defaults
        "sd15": {
            "sampling_method": "euler_a",
            "steps": 20,
            "cfg_scale": 7.0,
        },
        # SDXL defaults
        "sdxl": {
            "sampling_method": "euler_a",
            "steps": 20,
            "cfg_scale": 7.0,
        }
    }
    return defaults.get(model_name, {})

# Minimal ready message
log_to_stderr("DiffuGen ready")

@mcp.tool()
def generate_stable_diffusion_image(prompt: str, model: str = "sd15", output_dir: str = None,
                                   width: int = 512, height: int = 512, steps: int = None,
                                   cfg_scale: float = None, seed: int = -1,
                                   sampling_method: str = None, negative_prompt: str = "",
                                   lora_model_dir: str = None, clip_skip: int = None,
                                   character: str = None, character_merge_mode: str = "prepend",
                                   use_adetailer: bool = False, fix_faces: bool = True,
                                   fix_hands: bool = True, adetailer_strength: float = 0.4,
                                   controlnet_image_path: str = None, controlnet_image_base64: str = None,
                                   controlnet_model: str = "openpose",
                                   controlnet_weight: float = 0.7,
                                   ip_adapter_image_path: str = None, ip_adapter_image_base64: str = None,
                                   ip_adapter_weight: float = 0.5,
                                   ip_adapter_mask_path: str = None, ip_adapter_mask_base64: str = None,
                                   use_hires_fix: bool = False, hires_upscale_factor: float = 2.0,
                                   hires_denoising_strength: float = 0.4, run_preprocessor: bool = True,
                                   return_base64: bool = True) -> dict:
    """Generate an image using STABLE DIFFUSION models ONLY (SD1.5, SDXL, SD3)

    ** CRITICAL: DO NOT use this function for Flux models! Use generate_flux_image instead. **

    Supported models for THIS function:
    - sd15 (Stable Diffusion 1.5)
    - sdxl (Stable Diffusion XL)
    - sd3 (Stable Diffusion 3)
    - revanimated (revAnimated v2 Rebirth - SD1.5 based)
    - oia (OIA Illustrator - SD1.5 based)

    NEVER pass flux-schnell or flux-dev to this function!

    IMPORTANT - Auto-enable Adetailer:
    - ALWAYS set use_adetailer=True when generating images of people, characters, portraits, faces
    - Keywords that should trigger Adetailer: person, portrait, face, character, man, woman,
      child, selfie, headshot, hands, fingers, gesturing
    - When character parameter is used, strongly consider enabling Adetailer
    - Adetailer significantly improves face and hand quality with minimal overhead

    Args:
        prompt: The image description to generate (can include <lora:name:weight> syntax)
        model: SD model to use (sd15, sdxl, sd3, revanimated, oia - NEVER flux-schnell or flux-dev)
        output_dir: Directory to save the image (defaults to current directory)
        width: Image width in pixels
        height: Image height in pixels
        steps: Number of diffusion steps
        cfg_scale: CFG scale parameter
        seed: Seed for reproducibility (-1 for random)
        sampling_method: Sampling method (euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, lcm)
        negative_prompt: Negative prompt (for SD models ONLY)
        lora_model_dir: Directory containing LoRA models (optional)
        clip_skip: Number of CLIP layers to skip (1-10, model-specific defaults applied)
        character: Name of character to use for consistency (optional)
        character_merge_mode: How to merge character description ("prepend", "append", "replace")
        use_adetailer: Enable automatic face/hand refinement
        fix_faces: Fix faces with Adetailer (requires use_adetailer=True)
        fix_hands: Fix hands with Adetailer (requires use_adetailer=True)
        adetailer_strength: Denoising strength for Adetailer refinement (0.0-1.0)
        controlnet_image_path: Path to ROUGH DRAFT image for structural/pose control (optional)
        controlnet_image_base64: Base64 encoded ControlNet image (optional, alternative to path)
        controlnet_model: ControlNet model name (openpose, canny, depth, etc.)
        controlnet_weight: Strength of structural influence (0.0-1.0, default 0.7)
        ip_adapter_image_path: Path to CHARACTER PHOTO for identity/style transfer (optional)
        ip_adapter_image_base64: Base64 encoded IP-Adapter image (optional, alternative to path)
        ip_adapter_weight: Strength of identity influence (0.0-1.0, default 0.5)
        ip_adapter_mask_path: Path to binary mask for regional identity transfer (optional, for multi-character scenes)
        ip_adapter_mask_base64: Base64 encoded mask (optional, alternative to path)
        return_base64: Return the output image as base64 in the response (default True)

    Returns:
        A dictionary containing the path to the generated image, base64 data (if return_base64=True), and the command used
    """
    logging.info(f"Generate stable diffusion image request: prompt={prompt}, model={model}, character={character}, use_adetailer={use_adetailer}")
    
    # Use the generation queue to prevent concurrent generation
    if not generation_queue.acquire():
        return {
            "success": False,
            "error": "Another image generation is already in progress. Please try again when the current generation completes."
        }
    
    try:
        # Validate and fix boolean parameters (in case LLM sends strings)
        if isinstance(use_adetailer, str):
            use_adetailer = use_adetailer.lower() in ('true', '1', 'yes')
        if isinstance(fix_faces, str):
            fix_faces = fix_faces.lower() in ('true', '1', 'yes')
        if isinstance(fix_hands, str):
            fix_hands = fix_hands.lower() in ('true', '1', 'yes')

        # Ensure width and height are integers
        width = int(width) if width is not None else 512
        height = int(height) if height is not None else 512

        # Sanitize prompt and negative prompt (preserve <> for LoRA syntax)
        sanitized_prompt = re.sub(r'[^\w\s.,;:!?\'<>"()-]+', '', prompt).strip()
        sanitized_negative_prompt = negative_prompt
        if negative_prompt:
            sanitized_negative_prompt = re.sub(r'[^\w\s.,;:!?\'<>"()-]+', '', negative_prompt).strip()

        # Apply character consistency if specified
        if character:
            sanitized_prompt = character_manager.build_prompt_with_character(
                sanitized_prompt, character, character_merge_mode
            )
            # Also get character's negative prompt if not provided
            if not negative_prompt:
                char_neg_prompt = character_manager.get_character_negative_prompt(character)
                if char_neg_prompt:
                    sanitized_negative_prompt = char_neg_prompt
            
        # Select appropriate model
        if not model:
            model = "sd15"  # Default to SD1.5
        
        model = model.lower()
        
        # Only allow SD models in this function
        if model.startswith("flux-"):
            error_msg = f"Please use generate_flux_image for Flux models (received {model})"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
            
        # Normalize model name
        if model in ["sdxl", "sdxl-1.0", "sdxl1.0"]:
            model = "sdxl"
        elif model in ["sd3", "sd3-medium"]:
            model = "sd3"
        elif model in ["sd15", "sd1.5", "sd-1.5"]:
            model = "sd15"
        elif model and ("oia" in model.lower() or "illustrator" in model.lower()):
            model = "oia"
        elif model and ("revanimated" in model.lower() or "rebirth" in model.lower()):
            model = "revanimated"

        # Apply model-specific defaults first (only for unspecified params)
        model_defaults = get_model_defaults(model)
        if steps is None and "steps" in model_defaults:
            steps = model_defaults["steps"]
        if cfg_scale is None and "cfg_scale" in model_defaults:
            cfg_scale = model_defaults["cfg_scale"]
        if sampling_method is None and "sampling_method" in model_defaults:
            sampling_method = model_defaults["sampling_method"]
        if clip_skip is None and "clip_skip" in model_defaults:
            clip_skip = model_defaults["clip_skip"]

        # Use general default parameters if still not specified
        if width is None:
            width = config["default_params"]["width"]

        if height is None:
            height = config["default_params"]["height"]

        if steps is None:
            steps = get_default_steps(model)

        if cfg_scale is None:
            cfg_scale = get_default_cfg_scale(model)

        if sampling_method is None:
            sampling_method = get_default_sampling_method(model)

        # Always use configured output directory (ignore user-provided output_dir)
        # This prevents LLMs from passing relative paths like "./generated_images"
        output_dir = default_output_dir

        # Validate and fix LoRA model directory
        # LLMs sometimes pass LoRA names instead of the directory path
        if lora_model_dir is None:
            lora_model_dir = config.get("lora_model_dir")
        elif not lora_model_dir.startswith("/"):
            # If it's not an absolute path (e.g., "Lora REDNOSE"), use the default directory
            logging.warning(f"Invalid lora_model_dir '{lora_model_dir}', using default: {config.get('lora_model_dir')}")
            lora_model_dir = config.get("lora_model_dir")

        # Validate and auto-correct LoRA files if prompt contains LoRA tags
        lora_pattern = r'<lora:([^:>]+):([^>]+)>'
        lora_matches = re.findall(lora_pattern, sanitized_prompt)
        if lora_matches and lora_model_dir:
            # Check if the LoRA directory exists
            if not os.path.isdir(lora_model_dir):
                error_msg = f"LoRA directory does not exist: {lora_model_dir}"
                logging.error(error_msg)
                return {"success": False, "error": error_msg}

            # Validate and auto-correct LoRA names
            # stable-diffusion.cpp does exact case-sensitive filename matching
            lora_files = os.listdir(lora_model_dir)
            missing_loras = []
            lora_replacements = []  # Track corrections to apply to prompt

            for lora_name, lora_weight in lora_matches:
                # Step 1: Try exact match
                exact_safetensors = f"{lora_name}.safetensors"
                exact_ckpt = f"{lora_name}.ckpt"

                if exact_safetensors in lora_files:
                    logging.info(f"LoRA exact match: {exact_safetensors}")
                    continue  # No correction needed
                elif exact_ckpt in lora_files:
                    logging.info(f"LoRA exact match: {exact_ckpt}")
                    continue  # No correction needed

                # Step 2: Exact match failed - try fuzzy matching
                normalized_name = lora_name.lower().replace(" ", "").replace("_", "").replace("-", "")
                fuzzy_match = None

                for lora_file in lora_files:
                    if lora_file.endswith((".safetensors", ".ckpt")):
                        file_base = lora_file.rsplit(".", 1)[0]
                        file_normalized = file_base.lower().replace(" ", "").replace("_", "").replace("-", "")
                        if normalized_name == file_normalized:
                            fuzzy_match = file_base
                            break

                # Step 3: If fuzzy match found, auto-correct the prompt
                if fuzzy_match:
                    old_tag = f"<lora:{lora_name}:{lora_weight}>"
                    new_tag = f"<lora:{fuzzy_match}:{lora_weight}>"
                    lora_replacements.append((old_tag, new_tag))
                    logging.warning(f"LoRA auto-corrected: '{lora_name}' → '{fuzzy_match}'")
                    continue  # Fuzzy match successful, skip to next LoRA
                else:
                    # Step 4: No match at all - this is a real error
                    missing_loras.append(lora_name)

            # Apply all LoRA corrections to the prompt
            for old_tag, new_tag in lora_replacements:
                sanitized_prompt = sanitized_prompt.replace(old_tag, new_tag)

            # If any LoRAs genuinely don't exist, return error
            if missing_loras:
                available_loras = [f.rsplit(".", 1)[0] for f in lora_files if f.endswith((".safetensors", ".ckpt"))]
                error_msg = f"LoRA file(s) not found: {', '.join(missing_loras)}. Available LoRAs: {', '.join(available_loras)}"
                logging.error(error_msg)
                return {"success": False, "error": error_msg}

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
            
        # Get model path
        model_path = get_model_path(model)
        if not model_path:
            error_msg = f"Model not found: {model}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Generate a random seed if not provided
        if seed == -1:
            seed = random.randint(1, 1000000000)
            
        # Create output filename
        sanitized_prompt_for_filename = re.sub(r'[^\w\s]+', '', sanitized_prompt).strip()
        sanitized_prompt_for_filename = re.sub(r'\s+', '_', sanitized_prompt_for_filename)
        truncated_prompt = sanitized_prompt_for_filename[:20].lower()  # Limit to 20 chars
        unique_id = uuid.uuid4().hex[:8]
        output_filename = f"{model}_{truncated_prompt}_{unique_id}.png"
        output_path = os.path.join(output_dir, output_filename)
            
        # Prepare command for sd.cpp
        bin_path = os.path.join(sd_cpp_path, "build", "bin", "sd")
        
        base_command = [
            bin_path,
            "-p", sanitized_prompt
        ]
        
        # Add negative prompt if provided
        if sanitized_negative_prompt:
            base_command.extend(["--negative-prompt", sanitized_negative_prompt])
            
        # Add remaining parameters
        base_command.extend([
            "--cfg-scale", str(cfg_scale),
            "--sampling-method", sampling_method,
            "--steps", str(steps),
            "-H", str(height),
            "-W", str(width),
            "-o", output_path,
            "--seed", str(seed)
        ])

        # Add model-specific paths
        base_command.extend(["-m", model_path])

        # Add clip_skip if specified
        if clip_skip is not None:
            base_command.extend(["--clip-skip", str(clip_skip)])

        # Add model-specific supporting files
        if model == "sdxl":
            # SDXL needs VAE, CLIP, and T5XXL
            sdxl_vae_path = get_supporting_file("sdxl_vae")
            clip_l_path = get_supporting_file("clip_l")
            t5xxl_path = get_supporting_file("t5xxl")

            if sdxl_vae_path and os.path.exists(sdxl_vae_path):
                base_command.extend(["--vae", sdxl_vae_path])
            if clip_l_path and os.path.exists(clip_l_path):
                base_command.extend(["--clip_l", clip_l_path])
            if t5xxl_path and os.path.exists(t5xxl_path):
                base_command.extend(["--t5xxl", t5xxl_path])

        elif model == "sd15":
            # SD1.5 usually has built-in VAE, only use custom if explicitly provided
            sd15_vae_path = get_supporting_file("sd15_vae")
            if sd15_vae_path and os.path.exists(sd15_vae_path):
                base_command.extend(["--vae", sd15_vae_path])
            # SD1.5 doesn't typically need external CLIP

        elif model == "sd3":
            # SD3 may need supporting files similar to SDXL (if required)
            flux_vae_path = get_supporting_file("flux_vae")
            if flux_vae_path and os.path.exists(flux_vae_path):
                base_command.extend(["--vae", flux_vae_path])

        # Add LoRA model directory if provided
        if lora_model_dir:
            base_command.extend(["--lora-model-dir", lora_model_dir])

        # Auto-run preprocessor for ControlNet if enabled
        # Add ControlNet for structural/pose control (ROUGH DRAFT)
        # Handle image input from either file path or base64 data
        controlnet_path = _handle_image_input(controlnet_image_path, controlnet_image_base64)

        # Then run preprocessor if enabled (using resolved path)
        if run_preprocessor and controlnet_path and controlnet_model:
            logging.info(f"ControlNet preprocessor enabled for model: {controlnet_model}")
            processed_control_path = preprocessor_instance.run(
                input_path=controlnet_path,
                model_type=controlnet_model
            )

            if processed_control_path:
                logging.info(f"Preprocessor generated new control map: {processed_control_path}")
                controlnet_path = processed_control_path
            else:
                logging.warning(f"Preprocessor failed for model {controlnet_model}. Using original image for ControlNet.")
        if controlnet_path and os.path.exists(controlnet_path):
            logging.info(f"Adding ControlNet: model={controlnet_model}, weight={controlnet_weight}, image={controlnet_path}")
            # Get ControlNet model path
            controlnet_model_path = get_supporting_file(f"controlnet_{controlnet_model}")
            if controlnet_model_path and os.path.exists(controlnet_model_path):
                base_command.extend([
                    "--control-image", controlnet_path,
                    "--control-net", controlnet_model_path,
                    "--control-strength", str(controlnet_weight)
                ])
                logging.info(f"ControlNet model loaded: {controlnet_model_path}")
            else:
                logging.warning(f"ControlNet model not found: {controlnet_model}. Skipping ControlNet.")

        # Add IP-Adapter for identity/style transfer (CHARACTER PHOTO)
        ip_adapter_path = _handle_image_input(ip_adapter_image_path, ip_adapter_image_base64)
        if ip_adapter_path and os.path.exists(ip_adapter_path):
            logging.info(f"Adding IP-Adapter: weight={ip_adapter_weight}, image={ip_adapter_path}")
            # Get IP-Adapter model path
            ip_adapter_model_path = get_supporting_file("ip_adapter")
            ip_adapter_clip_path = get_supporting_file("ip_adapter_clip")

            if ip_adapter_model_path and os.path.exists(ip_adapter_model_path):
                base_command.extend([
                    "--image-prompts", ip_adapter_path,
                    "--style-ratio", str(ip_adapter_weight)
                ])

                # Add masking for multi-character consistency
                ip_adapter_mask = _handle_image_input(ip_adapter_mask_path, ip_adapter_mask_base64)
                if ip_adapter_mask and os.path.exists(ip_adapter_mask):
                    base_command.extend(["--control-image-mask", ip_adapter_mask])
                    logging.info(f"IP-Adapter mask applied: {ip_adapter_mask}")

                # Include supporting CLIP model if necessary
                if ip_adapter_clip_path and os.path.exists(ip_adapter_clip_path):
                    base_command.extend(["--clip-vision", ip_adapter_clip_path])

                logging.info(f"IP-Adapter model loaded: {ip_adapter_model_path}")
            else:
                logging.warning(f"IP-Adapter model not found. Skipping IP-Adapter.")

        # Add GPU and memory usage settings
        if config["vram_usage"] != "adaptive":
            base_command.append(f"--{config['vram_usage']}")

        if config["gpu_layers"] != -1:
            base_command.extend(["--gpu-layer", str(config["gpu_layers"])])

        try:
            # Run the command
            logging.info(f"Running command: {' '.join(base_command)}")

            result = subprocess.run(
                base_command,
                check=True,
                capture_output=True,
                text=True
            )

            logging.info(f"Successfully generated image at: {output_path} (size: {os.path.getsize(output_path)} bytes)")

            # Apply Hires Fix if requested (two-pass upscaling with img2img refinement)
            if use_hires_fix:
                logging.info(f"Applying Hires Fix: upscaling {hires_upscale_factor}x and refining with {hires_denoising_strength} strength")
                try:
                    from PIL import Image

                    # Step 1: Load and upscale the generated image
                    base_image = Image.open(output_path)
                    upscaled_width = int(width * hires_upscale_factor)
                    upscaled_height = int(height * hires_upscale_factor)
                    upscaled_image = base_image.resize((upscaled_width, upscaled_height), Image.Resampling.LANCZOS)

                    # Save upscaled version temporarily
                    upscaled_path = output_path.replace('.png', '_upscaled.png')
                    upscaled_image.save(upscaled_path)
                    logging.info(f"Upscaled image to {upscaled_width}x{upscaled_height}: {upscaled_path}")

                    # Step 2: Refine with img2img using low denoising strength
                    # Call the backend directly to stay within the same lock
                    refined_path = output_path.replace('.png', '_hires.png')
                    refine_command = [
                        bin_path, "-p", sanitized_prompt,
                        "--cfg-scale", str(cfg_scale),
                        "--sampling-method", sampling_method,
                        "--steps", str(steps),
                        "-H", str(upscaled_height), "-W", str(upscaled_width),
                        "-o", refined_path,
                        "--seed", str(seed),
                        "-m", model_path,
                        "--init-img", upscaled_path,
                        "--strength", str(hires_denoising_strength)
                    ]

                    # Add negative prompt if provided
                    if sanitized_negative_prompt:
                        refine_command.extend(["-n", sanitized_negative_prompt])

                    # Add VAE for all models that need it
                    if model == "sd15" or model == "revanimated" or model == "oia":
                        sd15_vae_path = get_supporting_file("sd15_vae")
                        if sd15_vae_path and os.path.exists(sd15_vae_path):
                            refine_command.extend(["--vae", sd15_vae_path])
                    elif model == "sdxl":
                        sdxl_vae_path = get_supporting_file("sdxl_vae")
                        if sdxl_vae_path and os.path.exists(sdxl_vae_path):
                            refine_command.extend(["--vae", sdxl_vae_path])
                    elif model == "sd3":
                        flux_vae_path = get_supporting_file("flux_vae")
                        if flux_vae_path and os.path.exists(flux_vae_path):
                            refine_command.extend(["--vae", flux_vae_path])
                    elif model in ["flux-schnell", "flux-dev"]:
                        flux_vae_path = get_supporting_file("flux_vae")
                        if flux_vae_path and os.path.exists(flux_vae_path):
                            refine_command.extend(["--vae", flux_vae_path])

                    # Add LoRA directory if provided
                    if lora_model_dir:
                        refine_command.extend(["--lora-model-dir", lora_model_dir])

                    logging.info(f"Running Hires Fix refinement: {' '.join(refine_command)}")
                    refine_result = subprocess.run(refine_command, capture_output=True, text=True)

                    if refine_result.returncode == 0:
                        output_path = refined_path
                        logging.info(f"Hires Fix completed: {output_path}")
                    else:
                        logging.warning(f"Hires Fix refinement failed: {refine_result.stderr}. Using upscaled image.")
                        output_path = upscaled_path

                except Exception as e:
                    logging.warning(f"Hires Fix failed: {e}. Using original image.")

            # Apply Adetailer if requested
            adetailer_results = None
            final_image_path = output_path
            if use_adetailer:
                logging.info("Applying Adetailer for face/hand refinement...")
                try:
                    adetailer_results = adetailer_instance.process_image(
                        image_path=output_path,
                        prompt=sanitized_prompt,
                        model_type=model,
                        fix_faces=fix_faces,
                        fix_hands=fix_hands,
                        face_strength=adetailer_strength,
                        hand_strength=adetailer_strength,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        output_dir=output_dir
                    )
                    if adetailer_results.get("success") and adetailer_results.get("final_image"):
                        final_image_path = adetailer_results["final_image"]
                        logging.info(f"Adetailer completed. Final image: {final_image_path}")
                except Exception as e:
                    logging.warning(f"Adetailer failed: {e}. Using original image.")

            # Format the response to match OpenAPI style
            image_description = "Image of " + sanitized_prompt[:50] + ("..." if len(sanitized_prompt) > 50 else "")

            markdown_response = f"Image generation complete.\n\n{image_description}\n\n**Generation Details:**\n\nModel: {model}\nResolution: {width}x{height} pixels\nSteps: {steps}\nCFG Scale: {cfg_scale}\nSampling Method: {sampling_method}\nSeed: {seed}"
            if character:
                markdown_response += f"\nCharacter: {character}"
            if use_adetailer and adetailer_results:
                markdown_response += f"\nAdetailer: {adetailer_results.get('faces_detected', 0)} faces, {adetailer_results.get('hands_detected', 0)} hands detected"
            markdown_response += f"\n\nPrompt: {sanitized_prompt}\n\nImage saved to: {final_image_path}"

            response = {
                "success": True,
                "image_path": final_image_path,
                "original_image": output_path if use_adetailer else None,
                "prompt": sanitized_prompt,
                "model": model,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "sampling_method": sampling_method,
                "command": " ".join(base_command),
                "output": result.stdout,
                "markdown_response": markdown_response
            }

            if character:
                response["character"] = character
            if adetailer_results:
                response["adetailer"] = adetailer_results

            # Add base64 encoded output image if requested
            if return_base64:
                try:
                    response["image_base64"] = _encode_image_to_base64(final_image_path)
                    logging.info("Added base64 encoded output image to response")
                except Exception as e:
                    logging.warning(f"Failed to encode output image to base64: {e}")
                    # Still return the response with the file path

            return response
        
        except subprocess.CalledProcessError as e:
            error_msg = f"Process error (exit code {e.returncode}): {str(e)}"
            logging.error(f"Image generation failed: {error_msg}")
            logging.error(f"Command: {' '.join(base_command)}")
            if e.stderr:
                logging.error(f"Process stderr: {e.stderr}")
            
            return {
                "success": False,
                "error": error_msg,
                "stderr": e.stderr,
                "command": " ".join(base_command),
                "exit_code": e.returncode
            }
        except FileNotFoundError as e:
            error_msg = f"Binary not found at {base_command[0]}"
            logging.error(f"Image generation failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "command": " ".join(base_command),
            }
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logging.error(f"Image generation failed with unexpected error: {error_msg}")
            logging.error(f"Command: {' '.join(base_command)}")
            
            return {
                "success": False,
                "error": error_msg,
                "command": " ".join(base_command),
            }
    finally:
        # Always release the lock when done
        generation_queue.release()

@mcp.tool()
def refine_image(prompt: str, init_image_path: str = None, init_image_base64: str = None,
                 model: str = None, output_dir: str = None,
                 strength: float = 0.5, width: int = None, height: int = None, steps: int = None,
                 cfg_scale: float = None, seed: int = -1, sampling_method: str = None,
                 negative_prompt: str = "", lora_model_dir: str = None, clip_skip: int = None,
                 character: str = None, character_merge_mode: str = "prepend",
                 return_base64: bool = True) -> dict:
    """Refine an existing image using img2img (iterative improvements based on feedback)

    This function is CRITICAL for iterative refinement workflows. Use it when:
    - User says "I like it but..." (e.g., "make the background darker", "fix the face")
    - User provides specific feedback about an existing image
    - Making targeted changes while preserving most of the original image

    THE MAGIC WORKFLOW (No file paths needed!):
    - User uploads image to LM Studio → LM Studio sends it as base64
    - User says "make the shirt magenta"
    - This function receives the image data directly
    - Returns the refined image as base64 directly to chat

    Args:
        prompt: Description of desired changes or the full scene description
        init_image_path: Path to the initial image to refine (optional if init_image_base64 provided)
        init_image_base64: Base64 encoded image data (optional if init_image_path provided)
        model: Model to use (sdxl, sd3, sd15, oia, revanimated, etc.)
        output_dir: Directory to save the refined image
        strength: How much to change (0.0=no change, 1.0=full regeneration, default 0.5)
                  - Use 0.2-0.4 for subtle changes (color, lighting, small tweaks)
                  - Use 0.5-0.6 for moderate changes (adding/removing elements)
                  - Use 0.7-0.9 for major changes (composition, style)
        width: Image width in pixels (defaults to init image size)
        height: Image height in pixels (defaults to init image size)
        steps: Number of diffusion steps
        cfg_scale: CFG scale parameter
        seed: Seed for reproducibility (-1 for random)
        sampling_method: Sampling method
        negative_prompt: Negative prompt
        lora_model_dir: Directory containing LoRA models
        clip_skip: Number of CLIP layers to skip
        character: Character name for consistency
        character_merge_mode: How to merge character description
        return_base64: Return the output image as base64 in the response (default True)

    Returns:
        Dictionary with refined image path, base64 data (if return_base64=True), and metadata
    """
    logging.info(f"Refine image request: init_image_path={init_image_path}, has_base64={bool(init_image_base64)}, prompt={prompt}, strength={strength}")

    if not generation_queue.acquire():
        return {
            "success": False,
            "error": "Another image generation is already in progress."
        }

    try:
        # Handle image input from either file path or base64 data
        resolved_init_image_path = _handle_image_input(init_image_path, init_image_base64)

        # Verify init image exists
        if not resolved_init_image_path or not os.path.exists(resolved_init_image_path):
            error_msg = "No input image provided. Please provide either init_image_path or init_image_base64."
            if init_image_path:
                error_msg = f"Initial image not found or invalid: {init_image_path}"
            return {"success": False, "error": error_msg}

        # Sanitize prompts
        sanitized_prompt = re.sub(r'[^\w\s.,;:!?\'<>"()-]+', '', prompt).strip()
        sanitized_negative_prompt = negative_prompt
        if negative_prompt:
            sanitized_negative_prompt = re.sub(r'[^\w\s.,;:!?\'<>"()-]+', '', negative_prompt).strip()

        # Apply character consistency if specified
        if character:
            sanitized_prompt = character_manager.build_prompt_with_character(
                sanitized_prompt, character, character_merge_mode
            )
            if not negative_prompt:
                char_neg_prompt = character_manager.get_character_negative_prompt(character)
                if char_neg_prompt:
                    sanitized_negative_prompt = char_neg_prompt

        # Select and normalize model
        if not model:
            model = "sd15"
        model = model.lower()

        if model in ["sdxl", "sdxl-1.0", "sdxl1.0"]:
            model = "sdxl"
        elif model in ["sd3", "sd3-medium"]:
            model = "sd3"
        elif model in ["sd15", "sd1.5", "sd-1.5"]:
            model = "sd15"

        # Apply model-specific defaults
        model_defaults = get_model_defaults(model)
        if steps is None and "steps" in model_defaults:
            steps = model_defaults["steps"]
        if cfg_scale is None and "cfg_scale" in model_defaults:
            cfg_scale = model_defaults["cfg_scale"]
        if sampling_method is None and "sampling_method" in model_defaults:
            sampling_method = model_defaults["sampling_method"]
        if clip_skip is None and "clip_skip" in model_defaults:
            clip_skip = model_defaults["clip_skip"]

        # Apply general defaults
        if steps is None:
            steps = get_default_steps(model)
        if cfg_scale is None:
            cfg_scale = get_default_cfg_scale(model)
        if sampling_method is None:
            sampling_method = get_default_sampling_method(model)
        if output_dir is None:
            output_dir = default_output_dir
        if lora_model_dir is None:
            lora_model_dir = config.get("lora_model_dir")

        os.makedirs(output_dir, exist_ok=True)

        # Get model path
        model_path = get_model_path(model)
        if not model_path:
            return {"success": False, "error": f"Model not found: {model}"}

        if not os.path.exists(model_path):
            return {"success": False, "error": f"Model file not found: {model_path}"}

        # Generate output filename
        safe_prompt = sanitized_prompt[:30].replace(" ", "_").replace("/", "_")
        unique_id = uuid.uuid4().hex[:8]
        output_filename = f"{model}_{safe_prompt}_{unique_id}_refined.png"
        output_path = os.path.join(output_dir, output_filename)

        # Build command
        bin_path = os.path.join(sd_cpp_path, "build", "bin", "sd")
        base_command = [
            bin_path,
            "-p", sanitized_prompt,
            "--init-img", resolved_init_image_path,
            "--strength", str(strength)
        ]

        if sanitized_negative_prompt:
            base_command.extend(["--negative-prompt", sanitized_negative_prompt])

        base_command.extend([
            "--cfg-scale", str(cfg_scale),
            "--sampling-method", sampling_method,
            "--steps", str(steps),
            "-o", output_path,
            "--seed", str(seed)
        ])

        # Add width/height if specified
        if width is not None:
            base_command.extend(["-W", str(width)])
        if height is not None:
            base_command.extend(["-H", str(height)])

        base_command.extend(["-m", model_path])

        if clip_skip is not None:
            base_command.extend(["--clip-skip", str(clip_skip)])

        if lora_model_dir:
            base_command.extend(["--lora-model-dir", lora_model_dir])

        # Run command
        logging.info(f"Running img2img command: {' '.join(base_command)}")

        result = subprocess.run(
            base_command,
            check=True,
            capture_output=True,
            text=True
        )

        logging.info(f"Successfully refined image at: {output_path}")

        # Prepare response
        response = {
            "success": True,
            "image_path": output_path,
            "init_image": init_image_path if init_image_path else "base64_input",
            "strength": strength,
            "model": model,
            "prompt": sanitized_prompt,
            "command": " ".join(base_command)
        }

        # Add base64 encoded output image if requested
        if return_base64:
            try:
                response["image_base64"] = _encode_image_to_base64(output_path)
                logging.info("Added base64 encoded output image to response")
            except Exception as e:
                logging.warning(f"Failed to encode output image to base64: {e}")
                # Still return the response with the file path

        return response

    except subprocess.CalledProcessError as e:
        error_msg = f"Process error (exit code {e.returncode}): {e.stderr if e.stderr else e.stdout}"
        logging.error(f"Image refinement failed: {error_msg}")
        return {"success": False, "error": error_msg}
    except Exception as e:
        logging.error(f"Failed to refine image: {str(e)}")
        return {"success": False, "error": str(e)}
    finally:
        generation_queue.release()

@mcp.tool()
def generate_flux_image(prompt: str, output_dir: str = None, cfg_scale: float = None,
                        sampling_method: str = None, steps: int = None,
                        model: str = None, width: int = None,
                        height: int = None, seed: int = -1, lora_model_dir: str = None,
                        character: str = None, character_merge_mode: str = "prepend",
                        use_adetailer: bool = False, fix_faces: bool = True,
                        fix_hands: bool = True, adetailer_strength: float = 0.4,
                        use_hires_fix: bool = False, hires_upscale_factor: float = 2.0,
                        hires_denoising_strength: float = 0.4, return_base64: bool = True) -> dict:
    """
    Generate an image using FLUX models ONLY (flux-schnell, flux-dev).

    ** CRITICAL: DO NOT use this function for Stable Diffusion models! Use generate_stable_diffusion_image instead. **

    This function is ONLY for:
    - flux-schnell (fast flux model)
    - flux-dev (high quality flux model)

    NEVER use this for: sd15, sdxl, sd3, revanimated, oia
    Those models require generate_stable_diffusion_image!

    Args:
        prompt: The image description to generate (can include <lora:name:weight> syntax)
        model: FLUX model to use (ONLY "flux-schnell" or "flux-dev" - NEVER sd15, oia, revanimated, etc.)
        output_dir: Directory to save the image
        cfg_scale: CFG scale parameter (default: 1.0 for all flux models)
        sampling_method: Sampling method to use (default: euler)
        steps: Number of diffusion steps (default: 8 for flux-schnell, 20 for flux-dev)
        width: Image width in pixels (default: 512)
        height: Image height in pixels (default: 512)
        seed: Seed for reproducibility (-1 for random)
        lora_model_dir: Directory containing LoRA models (optional)
        character: Name of character to use for consistency (optional)
        character_merge_mode: How to merge character description ("prepend", "append", "replace")
        use_adetailer: Enable automatic face/hand refinement
        fix_faces: Fix faces with Adetailer (requires use_adetailer=True)
        fix_hands: Fix hands with Adetailer (requires use_adetailer=True)
        adetailer_strength: Denoising strength for Adetailer refinement (0.0-1.0)

    Returns:
        A dictionary containing the path to the generated image and the command used
    """
    logging.info(f"Generate flux image request: prompt={prompt}, model={model}, character={character}, use_adetailer={use_adetailer}")

    # Use the generation queue to prevent concurrent generation
    if not generation_queue.acquire():
        return {
            "success": False,
            "error": "Another image generation is already in progress. Please try again when the current generation completes."
        }

    try:
        # Sanitize prompt (preserve <> for LoRA syntax)
        sanitized_prompt = re.sub(r'[^\w\s.,;:!?\'<>"()-]+', '', prompt).strip()

        # Apply character consistency if specified
        if character:
            sanitized_prompt = character_manager.build_prompt_with_character(
                sanitized_prompt, character, character_merge_mode
            )
            
        # Select appropriate model
        if not model:
            model = "flux-schnell"  # Default to flux-schnell
        
        model = model.lower()
        
        # Only allow Flux models in this function
        if not model.startswith("flux-"):
            # If the user specified an SD model, suggest using the other function
            if model in ["sdxl", "sd3", "sd15", "sd1.5", "revanimated", "oia", "illustrator"]:
                error_msg = f"WRONG FUNCTION! Please use generate_stable_diffusion_image for SD models like {model}. This function is ONLY for flux-schnell and flux-dev!"
            else:
                error_msg = f"Invalid model: {model}. For Flux image generation, use 'flux-schnell' or 'flux-dev'. For SD models, use generate_stable_diffusion_image!"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
            
        # Normalize model name
        if model in ["flux-schnell", "flux_schnell", "fluxschnell", "flux1-schnell"]:
            model = "flux-schnell"
        elif model in ["flux-dev", "flux_dev", "fluxdev", "flux1-dev"]:
            model = "flux-dev"
        
        # Use default parameters if not specified
        if width is None:
            width = config["default_params"]["width"]
        
        if height is None:
            height = config["default_params"]["height"]
            
        if steps is None:
            steps = get_default_steps(model)
            
        if cfg_scale is None:
            cfg_scale = get_default_cfg_scale(model)
            
        if sampling_method is None:
            sampling_method = get_default_sampling_method(model)

        # Always use configured output directory (ignore user-provided output_dir)
        # This prevents LLMs from passing relative paths like "./generated_images"
        output_dir = default_output_dir

        # Validate and fix LoRA model directory
        # LLMs sometimes pass LoRA names instead of the directory path
        if lora_model_dir is None:
            lora_model_dir = config.get("lora_model_dir")
        elif not lora_model_dir.startswith("/"):
            # If it's not an absolute path (e.g., "Lora REDNOSE"), use the default directory
            logging.warning(f"Invalid lora_model_dir '{lora_model_dir}', using default: {config.get('lora_model_dir')}")
            lora_model_dir = config.get("lora_model_dir")

        # Validate and auto-correct LoRA files if prompt contains LoRA tags
        lora_pattern = r'<lora:([^:>]+):([^>]+)>'
        lora_matches = re.findall(lora_pattern, sanitized_prompt)
        if lora_matches and lora_model_dir:
            # Check if the LoRA directory exists
            if not os.path.isdir(lora_model_dir):
                error_msg = f"LoRA directory does not exist: {lora_model_dir}"
                logging.error(error_msg)
                return {"success": False, "error": error_msg}

            # Validate and auto-correct LoRA names
            # stable-diffusion.cpp does exact case-sensitive filename matching
            lora_files = os.listdir(lora_model_dir)
            missing_loras = []
            lora_replacements = []  # Track corrections to apply to prompt

            for lora_name, lora_weight in lora_matches:
                # Step 1: Try exact match
                exact_safetensors = f"{lora_name}.safetensors"
                exact_ckpt = f"{lora_name}.ckpt"

                if exact_safetensors in lora_files:
                    logging.info(f"LoRA exact match: {exact_safetensors}")
                    continue  # No correction needed
                elif exact_ckpt in lora_files:
                    logging.info(f"LoRA exact match: {exact_ckpt}")
                    continue  # No correction needed

                # Step 2: Exact match failed - try fuzzy matching
                normalized_name = lora_name.lower().replace(" ", "").replace("_", "").replace("-", "")
                fuzzy_match = None

                for lora_file in lora_files:
                    if lora_file.endswith((".safetensors", ".ckpt")):
                        file_base = lora_file.rsplit(".", 1)[0]
                        file_normalized = file_base.lower().replace(" ", "").replace("_", "").replace("-", "")
                        if normalized_name == file_normalized:
                            fuzzy_match = file_base
                            break

                # Step 3: If fuzzy match found, auto-correct the prompt
                if fuzzy_match:
                    old_tag = f"<lora:{lora_name}:{lora_weight}>"
                    new_tag = f"<lora:{fuzzy_match}:{lora_weight}>"
                    lora_replacements.append((old_tag, new_tag))
                    logging.warning(f"LoRA auto-corrected: '{lora_name}' → '{fuzzy_match}'")
                    continue  # Fuzzy match successful, skip to next LoRA
                else:
                    # Step 4: No match at all - this is a real error
                    missing_loras.append(lora_name)

            # Apply all LoRA corrections to the prompt
            for old_tag, new_tag in lora_replacements:
                sanitized_prompt = sanitized_prompt.replace(old_tag, new_tag)

            # If any LoRAs genuinely don't exist, return error
            if missing_loras:
                available_loras = [f.rsplit(".", 1)[0] for f in lora_files if f.endswith((".safetensors", ".ckpt"))]
                error_msg = f"LoRA file(s) not found: {', '.join(missing_loras)}. Available LoRAs: {', '.join(available_loras)}"
                logging.error(error_msg)
                return {"success": False, "error": error_msg}

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
            
        # Get model path
        model_path = get_model_path(model)
        if not model_path:
            error_msg = f"Model not found: {model}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Generate a random seed if not provided
        if seed == -1:
            seed = random.randint(1, 1000000000)
            
        # Create output filename
        sanitized_prompt_for_filename = re.sub(r'[^\w\s]+', '', sanitized_prompt).strip()
        sanitized_prompt_for_filename = re.sub(r'\s+', '_', sanitized_prompt_for_filename)
        truncated_prompt = sanitized_prompt_for_filename[:20].lower()  # Limit to 20 chars
        unique_id = uuid.uuid4().hex[:8]
        output_filename = f"{model}_{truncated_prompt}_{unique_id}.png"
        output_path = os.path.join(output_dir, output_filename)
            
        # Prepare command for sd.cpp
        bin_path = os.path.join(sd_cpp_path, "build", "bin", "sd")
        
        base_command = [
            bin_path,
            "-p", sanitized_prompt,
            "--cfg-scale", str(cfg_scale),
            "--sampling-method", sampling_method,
            "--steps", str(steps),
            "-H", str(height),
            "-W", str(width),
            "-o", output_path,
            "--diffusion-fa",  # Add Flux-specific flag
            "--seed", str(seed)
        ]
        
        # Add model-specific paths
        base_command.extend(["--diffusion-model", model_path])

        # Get supporting files for Flux
        flux_vae_path = get_supporting_file("flux_vae")
        clip_l_path = get_supporting_file("clip_l")
        t5xxl_path = get_supporting_file("t5xxl")

        if flux_vae_path and os.path.exists(flux_vae_path):
            base_command.extend(["--vae", flux_vae_path])

        if clip_l_path and os.path.exists(clip_l_path):
            base_command.extend(["--clip_l", clip_l_path])

        if t5xxl_path and os.path.exists(t5xxl_path):
            base_command.extend(["--t5xxl", t5xxl_path])

        # Add LoRA model directory if provided
        if lora_model_dir:
            base_command.extend(["--lora-model-dir", lora_model_dir])

        # Add GPU and memory usage settings
        if config["vram_usage"] != "adaptive":
            base_command.append(f"--{config['vram_usage']}")

        if config["gpu_layers"] != -1:
            base_command.extend(["--gpu-layer", str(config["gpu_layers"])])

        try:
            # Run the command
            logging.info(f"Running command: {' '.join(base_command)}")

            result = subprocess.run(
                base_command,
                check=True,
                capture_output=True,
                text=True
            )

            logging.info(f"Successfully generated image at: {output_path} (size: {os.path.getsize(output_path)} bytes)")

            # Apply Hires Fix if requested (two-pass upscaling with img2img refinement)
            if use_hires_fix:
                logging.info(f"Applying Hires Fix: upscaling {hires_upscale_factor}x and refining with {hires_denoising_strength} strength")
                try:
                    from PIL import Image

                    # Step 1: Load and upscale the generated image
                    base_image = Image.open(output_path)
                    upscaled_width = int(width * hires_upscale_factor)
                    upscaled_height = int(height * hires_upscale_factor)
                    upscaled_image = base_image.resize((upscaled_width, upscaled_height), Image.Resampling.LANCZOS)

                    # Save upscaled version temporarily
                    upscaled_path = output_path.replace('.png', '_upscaled.png')
                    upscaled_image.save(upscaled_path)
                    logging.info(f"Upscaled image to {upscaled_width}x{upscaled_height}: {upscaled_path}")

                    # Step 2: Refine with img2img using low denoising strength
                    # Call the backend directly to stay within the same lock
                    refined_path = output_path.replace('.png', '_hires.png')
                    refine_command = [
                        bin_path, "-p", sanitized_prompt,
                        "--cfg-scale", str(cfg_scale),
                        "--sampling-method", sampling_method,
                        "--steps", str(steps),
                        "-H", str(upscaled_height), "-W", str(upscaled_width),
                        "-o", refined_path,
                        "--seed", str(seed),
                        "-m", model_path,
                        "--init-img", upscaled_path,
                        "--strength", str(hires_denoising_strength)
                    ]

                    # Add negative prompt if provided
                    if sanitized_negative_prompt:
                        refine_command.extend(["-n", sanitized_negative_prompt])

                    # Add VAE for all models that need it
                    if model == "sd15" or model == "revanimated" or model == "oia":
                        sd15_vae_path = get_supporting_file("sd15_vae")
                        if sd15_vae_path and os.path.exists(sd15_vae_path):
                            refine_command.extend(["--vae", sd15_vae_path])
                    elif model == "sdxl":
                        sdxl_vae_path = get_supporting_file("sdxl_vae")
                        if sdxl_vae_path and os.path.exists(sdxl_vae_path):
                            refine_command.extend(["--vae", sdxl_vae_path])
                    elif model == "sd3":
                        flux_vae_path = get_supporting_file("flux_vae")
                        if flux_vae_path and os.path.exists(flux_vae_path):
                            refine_command.extend(["--vae", flux_vae_path])
                    elif model in ["flux-schnell", "flux-dev"]:
                        flux_vae_path = get_supporting_file("flux_vae")
                        if flux_vae_path and os.path.exists(flux_vae_path):
                            refine_command.extend(["--vae", flux_vae_path])

                    # Add LoRA directory if provided
                    if lora_model_dir:
                        refine_command.extend(["--lora-model-dir", lora_model_dir])

                    logging.info(f"Running Hires Fix refinement: {' '.join(refine_command)}")
                    refine_result = subprocess.run(refine_command, capture_output=True, text=True)

                    if refine_result.returncode == 0:
                        output_path = refined_path
                        logging.info(f"Hires Fix completed: {output_path}")
                    else:
                        logging.warning(f"Hires Fix refinement failed: {refine_result.stderr}. Using upscaled image.")
                        output_path = upscaled_path

                except Exception as e:
                    logging.warning(f"Hires Fix failed: {e}. Using original image.")

            # Apply Adetailer if requested
            adetailer_results = None
            final_image_path = output_path
            if use_adetailer:
                logging.info("Applying Adetailer for face/hand refinement...")
                try:
                    adetailer_results = adetailer_instance.process_image(
                        image_path=output_path,
                        prompt=sanitized_prompt,
                        model_type=model,
                        fix_faces=fix_faces,
                        fix_hands=fix_hands,
                        face_strength=adetailer_strength,
                        hand_strength=adetailer_strength,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        output_dir=output_dir
                    )
                    if adetailer_results.get("success") and adetailer_results.get("final_image"):
                        final_image_path = adetailer_results["final_image"]
                        logging.info(f"Adetailer completed. Final image: {final_image_path}")
                except Exception as e:
                    logging.warning(f"Adetailer failed: {e}. Using original image.")

            # Format the response to match OpenAPI style
            image_description = "Image of " + sanitized_prompt[:50] + ("..." if len(sanitized_prompt) > 50 else "")

            markdown_response = f"Image generation complete.\n\n{image_description}\n\n**Generation Details:**\n\nModel: {model}\nResolution: {width}x{height} pixels\nSteps: {steps}\nCFG Scale: {cfg_scale}\nSampling Method: {sampling_method}\nSeed: {seed}"
            if character:
                markdown_response += f"\nCharacter: {character}"
            if use_adetailer and adetailer_results:
                markdown_response += f"\nAdetailer: {adetailer_results.get('faces_detected', 0)} faces, {adetailer_results.get('hands_detected', 0)} hands detected"
            markdown_response += f"\n\nPrompt: {sanitized_prompt}\n\nImage saved to: {final_image_path}"

            response = {
                "success": True,
                "image_path": final_image_path,
                "original_image": output_path if use_adetailer else None,
                "prompt": sanitized_prompt,
                "model": model,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "sampling_method": sampling_method,
                "command": " ".join(base_command),
                "output": result.stdout,
                "markdown_response": markdown_response
            }

            if character:
                response["character"] = character
            if adetailer_results:
                response["adetailer"] = adetailer_results

            # Add base64 encoded output image if requested
            if return_base64:
                try:
                    response["image_base64"] = _encode_image_to_base64(final_image_path)
                    logging.info("Added base64 encoded output image to response")
                except Exception as e:
                    logging.warning(f"Failed to encode output image to base64: {e}")
                    # Still return the response with the file path

            return response
        
        except subprocess.CalledProcessError as e:
            error_msg = f"Process error (exit code {e.returncode}): {str(e)}"
            logging.error(f"Image generation failed: {error_msg}")
            logging.error(f"Command: {' '.join(base_command)}")
            if e.stderr:
                logging.error(f"Process stderr: {e.stderr}")
            
            return {
                "success": False,
                "error": error_msg,
                "stderr": e.stderr,
                "command": " ".join(base_command),
                "exit_code": e.returncode
            }
        except FileNotFoundError as e:
            error_msg = f"Binary not found at {base_command[0]}"
            logging.error(f"Image generation failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "command": " ".join(base_command),
            }
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logging.error(f"Image generation failed with unexpected error: {error_msg}")
            logging.error(f"Command: {' '.join(base_command)}")
            
            return {
                "success": False,
                "error": error_msg,
                "command": " ".join(base_command),
            }
    finally:
        # Always release the lock when done
        generation_queue.release()

if __name__ == "__main__":
    try:
        # Check if command line arguments are provided for direct image generation
        if len(sys.argv) > 1:
            # Parse command line arguments
            parser = argparse.ArgumentParser(description="Generate SD\FLux Images")
            parser.add_argument("prompt", type=str, help="The image description to generate")
            parser.add_argument("--model", type=str, 
                                help="Model to use (flux-schnell, flux-dev, sdxl, sd3, sd15)")
            parser.add_argument("--width", type=int, default=config["default_params"]["width"], 
                                help="Image width in pixels")
            parser.add_argument("--height", type=int, default=config["default_params"]["height"], 
                                help="Image height in pixels")
            # For steps and cfg_scale, we'll determine the default based on the model after parsing
            parser.add_argument("--steps", type=int, default=None, 
                                help="Number of diffusion steps")
            parser.add_argument("--cfg-scale", type=float, dest="cfg_scale", default=None, 
                                help="CFG scale parameter")
            parser.add_argument("--seed", type=int, default=-1, 
                                help="Seed for reproducibility (-1 for random)")
            parser.add_argument("--sampling-method", type=str, dest="sampling_method", 
                                default=None, 
                                help="Sampling method")
            parser.add_argument("--negative-prompt", type=str, dest="negative_prompt", default="", 
                                help="Negative prompt")
            parser.add_argument("--output-dir", type=str, dest="output_dir", default=None, 
                                help="Directory to save the image")
            parser.add_argument("--use-adetailer", action="store_true", help="Enable Adetailer for face/hand refinement")
            parser.add_argument("--init-img", type=str, default=None, help="Path to the initial image for img2img")
            parser.add_argument("--strength", type=float, default=0.5, help="Strength for img2img")
            parser.add_argument("--controlnet-model", type=str, default=None, help="The ControlNet model to use (e.g., openpose, depth)")
            parser.add_argument("--controlnet-weight", type=float, default=0.7, help="The weight for the ControlNet model")
            parser.add_argument("--controlnet-image-path", type=str, default=None, help="Path to the ControlNet input image")
            
            # Parse arguments
            args, unknown = parser.parse_known_args()
            
            # Determine model - use from args, config, or choose appropriate default
            if args.model is None:
                args.model = config["default_model"]
                # If still None, prompt the user to specify a model
                if args.model is None:
                    log_to_stderr("Model not specified. Please specify a model using --model. Available options:")
                    log_to_stderr("  Flux models: flux-schnell, flux-dev")
                    log_to_stderr("  SD models: sdxl, sd3, sd15")
                    sys.exit(1)
                    
            # Set model-specific defaults if not provided
            if args.steps is None:
                args.steps = get_default_steps(args.model)
            if args.cfg_scale is None:
                args.cfg_scale = get_default_cfg_scale(args.model)
            if args.sampling_method is None:
                args.sampling_method = get_default_sampling_method(args.model)
            
            # Determine which generation function to use based on model
            if args.init_img:
                log_to_stderr(f"Refining image {args.init_img} with model: {args.model}")
                result = refine_image(
                    prompt=args.prompt,
                    init_image_path=args.init_img,
                    strength=args.strength,
                    model=args.model,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    cfg_scale=args.cfg_scale,
                    seed=args.seed,
                    sampling_method=args.sampling_method,
                    negative_prompt=args.negative_prompt,
                    output_dir=args.output_dir
                )
            elif args.model.lower().startswith("flux-"):
                log_to_stderr(f"Generating Flux image with model: {args.model}")
                result = generate_flux_image(
                    prompt=args.prompt,
                    model=args.model,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    cfg_scale=args.cfg_scale,
                    seed=args.seed,
                    sampling_method=args.sampling_method,
                    output_dir=args.output_dir,
                    use_adetailer=args.use_adetailer
                )
            else:
                log_to_stderr(f"Generating SD image with model: {args.model}")
                result = generate_stable_diffusion_image(
                    prompt=args.prompt,
                    model=args.model,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    cfg_scale=args.cfg_scale,
                    seed=args.seed,
                    sampling_method=args.sampling_method,
                    negative_prompt=args.negative_prompt,
                    output_dir=args.output_dir,
                    use_adetailer=args.use_adetailer,
                    controlnet_image_path=args.controlnet_image_path,
                    controlnet_model=args.controlnet_model,
                    controlnet_weight=args.controlnet_weight
                )
            
            # Print the result path
            if result.get("success", False):
                print(f"Image generated successfully: {result['image_path']}")
                sys.exit(0)
            else:
                log_to_stderr(f"Image generation failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        else:
            # No arguments provided, start the MCP server
            mcp.run()
    except Exception as e:
        logging.error(f"Error running DiffuGen: {e}")
        sys.exit(1)
