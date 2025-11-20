"""
DiffuGen Core Helper Functions
Common utilities for image generation, validation, and processing
"""

import os
import re
import uuid
import logging
import subprocess
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# File Path Resolution
# ============================================================================

def resolve_local_file_path(file_reference: str, search_dirs: List[str] = None) -> Optional[str]:
    """
    Smart file path resolution for images

    Args:
        file_reference: Path or filename to resolve
        search_dirs: Optional list of directories to search

    Returns:
        Resolved absolute path, or None if not found
    """
    if not file_reference:
        return None

    # If exact path exists, return it
    if os.path.exists(file_reference):
        return os.path.abspath(file_reference)

    # Extract filename
    filename = os.path.basename(file_reference)

    # Default search directories
    if search_dirs is None:
        search_dirs = [
            '/app/inputs',
            '/app/outputs',
            '/app',
            os.getcwd(),
        ]

    # Search for exact filename match
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        candidate_path = os.path.join(search_dir, filename)
        if os.path.exists(candidate_path):
            logger.info(f"Found image at: {candidate_path}")
            return os.path.abspath(candidate_path)

    # Try case-insensitive fuzzy matching
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        try:
            for file in os.listdir(search_dir):
                if file.startswith('.'):
                    continue
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    continue

                if file.lower() == filename.lower():
                    candidate_path = os.path.join(search_dir, file)
                    logger.info(f"Found image (case-insensitive): {candidate_path}")
                    return os.path.abspath(candidate_path)
        except Exception as e:
            logger.warning(f"Error searching directory {search_dir}: {e}")

    # Last resort: return most recent image if "latest" is requested
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
                    most_recent = max(image_files, key=os.path.getmtime)
                    logger.info(f"Using most recent image: {most_recent}")
                    return os.path.abspath(most_recent)
            except Exception as e:
                logger.warning(f"Error finding recent images in {search_dir}: {e}")

    logger.warning(f"Could not resolve file reference: {file_reference}")
    return None


# ============================================================================
# Base64 Image Handling
# ============================================================================

def save_base64_image(base64_data: str, prefix: str = "input") -> str:
    """
    Save a base64 encoded image to a temporary file

    Args:
        base64_data: Base64 encoded image string
        prefix: Prefix for temp filename

    Returns:
        Path to saved temporary file
    """
    if not base64_data:
        raise ValueError("Empty base64 data provided")

    # Remove data URI prefix if present
    if 'base64,' in base64_data:
        base64_data = base64_data.split('base64,', 1)[1]

    # Clean whitespace
    base64_data = base64_data.strip().replace('\n', '').replace('\r', '').replace(' ', '')

    # Validate base64 data
    if len(base64_data) < 100:
        if base64_data in ["base64_encoded_image_data", "BASE64_DATA_HERE", "base64_data", "..."]:
            raise ValueError(
                f"Received placeholder '{base64_data}' instead of actual base64 image data. "
                "Please use init_image_path instead."
            )
        raise ValueError(f"Base64 string too short ({len(base64_data)} bytes)")

    # Detect image format
    ext = '.png'
    if base64_data.startswith('/9j/'):
        ext = '.jpg'
    elif base64_data.startswith('iVBORw'):
        ext = '.png'
    elif base64_data.startswith('UklGR'):
        ext = '.webp'
    elif base64_data.startswith('Qk'):
        ext = '.bmp'

    # Decode base64
    try:
        image_data = base64.b64decode(base64_data, validate=True)
    except Exception as e:
        raise ValueError(f"Base64 decode failed: {e}")

    if len(image_data) < 100:
        raise ValueError(f"Decoded image data too small ({len(image_data)} bytes)")

    # Create temp file
    temp_dir = '/app/inputs' if os.path.exists('/app/inputs') else tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"{prefix}_{uuid.uuid4().hex}{ext}")

    with open(temp_file, 'wb') as f:
        f.write(image_data)

    if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 100:
        raise ValueError("Failed to write image file")

    logger.info(f"Saved base64 image to: {temp_file} ({len(image_data)} bytes)")
    return temp_file


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string with data URI prefix
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()

    base64_data = base64.b64encode(image_data).decode('utf-8')

    # Add data URI prefix based on extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    mime_type = mime_types.get(ext, 'image/png')

    return f"data:{mime_type};base64,{base64_data}"


def handle_image_input(image_path: str = None, image_base64: str = None) -> Optional[str]:
    """
    Handle image input from either file path or base64 data

    Args:
        image_path: Path to image file
        image_base64: Base64 encoded image

    Returns:
        Resolved absolute path to image file
    """
    if image_base64:
        return save_base64_image(image_base64)
    elif image_path:
        return resolve_local_file_path(image_path)
    return None


# ============================================================================
# LoRA Validation and Correction
# ============================================================================

def validate_and_correct_loras(
    prompt: str,
    lora_model_dir: str
) -> Tuple[str, Optional[str]]:
    """
    Validate and auto-correct LoRA references in prompt

    Args:
        prompt: Prompt containing LoRA tags
        lora_model_dir: Directory containing LoRA files

    Returns:
        Tuple of (corrected_prompt, error_message)
        error_message is None if all LoRAs are valid
    """
    if not lora_model_dir or not os.path.isdir(lora_model_dir):
        return prompt, f"LoRA directory does not exist: {lora_model_dir}"

    # Find all LoRA tags in prompt
    lora_pattern = r'<lora:([^:>]+):([^>]+)>'
    lora_matches = re.findall(lora_pattern, prompt)

    if not lora_matches:
        return prompt, None

    # Get available LoRA files
    lora_files = os.listdir(lora_model_dir)
    missing_loras = []
    lora_replacements = []

    for lora_name, lora_weight in lora_matches:
        # Try exact match
        exact_safetensors = f"{lora_name}.safetensors"
        exact_ckpt = f"{lora_name}.ckpt"

        if exact_safetensors in lora_files or exact_ckpt in lora_files:
            logger.info(f"LoRA exact match: {lora_name}")
            continue

        # Try fuzzy match
        normalized_name = lora_name.lower().replace(" ", "").replace("_", "").replace("-", "")
        fuzzy_match = None

        for lora_file in lora_files:
            if lora_file.endswith((".safetensors", ".ckpt")):
                file_base = lora_file.rsplit(".", 1)[0]
                file_normalized = file_base.lower().replace(" ", "").replace("_", "").replace("-", "")
                if normalized_name == file_normalized:
                    fuzzy_match = file_base
                    break

        if fuzzy_match:
            old_tag = f"<lora:{lora_name}:{lora_weight}>"
            new_tag = f"<lora:{fuzzy_match}:{lora_weight}>"
            lora_replacements.append((old_tag, new_tag))
            logger.warning(f"LoRA auto-corrected: '{lora_name}' → '{fuzzy_match}'")
        else:
            missing_loras.append(lora_name)

    # Apply corrections
    corrected_prompt = prompt
    for old_tag, new_tag in lora_replacements:
        corrected_prompt = corrected_prompt.replace(old_tag, new_tag)

    # Return error if any LoRAs are missing
    if missing_loras:
        available_loras = [f.rsplit(".", 1)[0] for f in lora_files if f.endswith((".safetensors", ".ckpt"))]
        error_msg = f"LoRA file(s) not found: {', '.join(missing_loras)}. Available: {', '.join(available_loras)}"
        return corrected_prompt, error_msg

    return corrected_prompt, None


# ============================================================================
# Command Building Helpers
# ============================================================================

def build_base_command(
    bin_path: str,
    prompt: str,
    output_path: str,
    model_path: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampling_method: str,
    seed: int,
    negative_prompt: str = None,
    clip_skip: int = None
) -> List[str]:
    """
    Build base command for stable-diffusion.cpp

    Returns:
        Command as list of strings
    """
    cmd = [
        bin_path,
        "-p", prompt,
        "-m", model_path,
        "-H", str(height),
        "-W", str(width),
        "-o", output_path,
        "--steps", str(steps),
        "--cfg-scale", str(cfg_scale),
        "--sampling-method", sampling_method,
        "--seed", str(seed)
    ]

    if negative_prompt:
        cmd.extend(["--negative-prompt", negative_prompt])

    if clip_skip is not None:
        cmd.extend(["--clip-skip", str(clip_skip)])

    return cmd


def add_supporting_files(
    cmd: List[str],
    model_type: str,
    supporting_files: Dict[str, str]
) -> List[str]:
    """
    Add model-specific supporting files to command

    Args:
        cmd: Base command list
        model_type: Model type (sdxl, sd15, sd3, flux, etc.)
        supporting_files: Dict mapping file keys to paths

    Returns:
        Updated command list
    """
    if model_type == "sdxl":
        if "sdxl_vae" in supporting_files and os.path.exists(supporting_files["sdxl_vae"]):
            cmd.extend(["--vae", supporting_files["sdxl_vae"]])
        if "clip_l" in supporting_files and os.path.exists(supporting_files["clip_l"]):
            cmd.extend(["--clip_l", supporting_files["clip_l"]])
        if "t5xxl" in supporting_files and os.path.exists(supporting_files["t5xxl"]):
            cmd.extend(["--t5xxl", supporting_files["t5xxl"]])

    elif model_type == "sd15":
        if "sd15_vae" in supporting_files and os.path.exists(supporting_files["sd15_vae"]):
            cmd.extend(["--vae", supporting_files["sd15_vae"]])

    elif model_type == "sd3":
        if "flux_vae" in supporting_files and os.path.exists(supporting_files["flux_vae"]):
            cmd.extend(["--vae", supporting_files["flux_vae"]])

    elif model_type in ["flux-schnell", "flux-dev"]:
        if "flux_vae" in supporting_files and os.path.exists(supporting_files["flux_vae"]):
            cmd.extend(["--vae", supporting_files["flux_vae"]])
        if "clip_l" in supporting_files and os.path.exists(supporting_files["clip_l"]):
            cmd.extend(["--clip_l", supporting_files["clip_l"]])
        if "t5xxl" in supporting_files and os.path.exists(supporting_files["t5xxl"]):
            cmd.extend(["--t5xxl", supporting_files["t5xxl"]])

    return cmd


# ============================================================================
# Hires Fix
# ============================================================================

def apply_hires_fix(
    bin_path: str,
    original_image_path: str,
    prompt: str,
    model_path: str,
    supporting_files: Dict[str, str],
    upscale_factor: float = 2.0,
    denoising_strength: float = 0.4,
    steps: int = 20,
    cfg_scale: float = 7.0,
    sampling_method: str = "euler",
    seed: int = -1,
    negative_prompt: str = None,
    lora_model_dir: str = None,
    model_type: str = "sd15"
) -> Optional[str]:
    """
    Apply Hires Fix (two-pass upscaling with img2img refinement)

    Args:
        bin_path: Path to sd.cpp binary
        original_image_path: Path to original generated image
        prompt: Generation prompt
        model_path: Path to model file
        supporting_files: Dict of supporting file paths
        upscale_factor: Upscale factor
        denoising_strength: Denoising strength for refinement
        steps: Number of steps
        cfg_scale: CFG scale
        sampling_method: Sampling method
        seed: Random seed
        negative_prompt: Negative prompt
        lora_model_dir: LoRA directory
        model_type: Model type for VAE selection

    Returns:
        Path to refined image, or None if failed
    """
    try:
        # Load and upscale
        base_image = Image.open(original_image_path)
        width, height = base_image.size
        upscaled_width = int(width * upscale_factor)
        upscaled_height = int(height * upscale_factor)
        upscaled_image = base_image.resize((upscaled_width, upscaled_height), Image.Resampling.LANCZOS)

        # Save upscaled version
        upscaled_path = original_image_path.replace('.png', '_upscaled.png')
        upscaled_image.save(upscaled_path)
        logger.info(f"Upscaled to {upscaled_width}x{upscaled_height}: {upscaled_path}")

        # Build refinement command
        refined_path = original_image_path.replace('.png', '_hires.png')
        refine_cmd = [
            bin_path,
            "-p", prompt,
            "-m", model_path,
            "-H", str(upscaled_height),
            "-W", str(upscaled_width),
            "-o", refined_path,
            "--steps", str(steps),
            "--cfg-scale", str(cfg_scale),
            "--sampling-method", sampling_method,
            "--seed", str(seed),
            "--init-img", upscaled_path,
            "--strength", str(denoising_strength)
        ]

        if negative_prompt:
            refine_cmd.extend(["--negative-prompt", negative_prompt])

        # Add VAE based on model type
        if model_type in ["sd15", "revanimated", "oia"] and "sd15_vae" in supporting_files:
            if os.path.exists(supporting_files["sd15_vae"]):
                refine_cmd.extend(["--vae", supporting_files["sd15_vae"]])
        elif model_type == "sdxl" and "sdxl_vae" in supporting_files:
            if os.path.exists(supporting_files["sdxl_vae"]):
                refine_cmd.extend(["--vae", supporting_files["sdxl_vae"]])
        elif "flux_vae" in supporting_files and os.path.exists(supporting_files["flux_vae"]):
            refine_cmd.extend(["--vae", supporting_files["flux_vae"]])

        if lora_model_dir:
            refine_cmd.extend(["--lora-model-dir", lora_model_dir])

        logger.info(f"Running Hires Fix: {' '.join(refine_cmd)}")
        result = subprocess.run(refine_cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            logger.info(f"Hires Fix completed: {refined_path}")
            return refined_path
        else:
            logger.warning(f"Hires Fix refinement failed: {result.stderr}")
            return upscaled_path

    except Exception as e:
        logger.warning(f"Hires Fix failed: {e}")
        return None


# ============================================================================
# Prompt Sanitization
# ============================================================================

def sanitize_prompt(prompt: str, preserve_lora_tags: bool = True) -> str:
    """
    Sanitize prompt for safe command-line usage

    Args:
        prompt: Original prompt
        preserve_lora_tags: Whether to preserve <lora:...> tags

    Returns:
        Sanitized prompt
    """
    if preserve_lora_tags:
        # Preserve <> for LoRA syntax
        return re.sub(r'[^\w\s.,;:!?\'<>"()-]+', '', prompt).strip()
    else:
        return re.sub(r'[^\w\s.,;:!?\'\"()-]+', '', prompt).strip()


def create_output_filename(model: str, prompt: str, output_dir: str, suffix: str = "") -> str:
    """
    Create unique output filename

    Args:
        model: Model name
        prompt: Generation prompt
        output_dir: Output directory
        suffix: Optional suffix (e.g., "_refined")

    Returns:
        Full path to output file
    """
    # Sanitize prompt for filename
    safe_prompt = re.sub(r'[^\w\s]+', '', prompt).strip()
    safe_prompt = re.sub(r'\s+', '_', safe_prompt)
    truncated = safe_prompt[:20].lower()

    # Generate unique ID
    unique_id = uuid.uuid4().hex[:8]

    # Build filename
    filename = f"{model}_{truncated}_{unique_id}{suffix}.png"
    return os.path.join(output_dir, filename)
