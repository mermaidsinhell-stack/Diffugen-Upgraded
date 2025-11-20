"""
Preprocessor Module for DiffuGen (Async Refactored)
Handles ControlNet preprocessing: Canny edge detection, depth maps, pose/segmentation
"""

import os
import logging
import uuid
import asyncio
from pathlib import Path
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Attempt to import transformers and torch for depth estimation
try:
    import torch
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not found. Depth preprocessing unavailable. Install: pip install transformers torch")


class Preprocessor:
    """
    Async-capable preprocessor for ControlNet image preprocessing

    Supports:
    - Canny edge detection
    - Depth estimation (DPT model)
    - Pose detection (YOLO)
    - Segmentation (YOLO)
    """

    def __init__(
        self,
        yolo_models_dir: str = "yolo_models",
        output_dir: str = "outputs",
        max_workers: int = 2
    ):
        """
        Initialize the Preprocessor

        Args:
            yolo_models_dir: Directory containing YOLO model files
            output_dir: Directory for processed images
            max_workers: Number of threads for CPU-bound operations
        """
        self.yolo_models_dir = Path(yolo_models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.yolo_models: Dict[str, YOLO] = {}
        self.depth_model = None
        self.depth_processor = None

        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self._discover_yolo_models()
        self._load_depth_model()

    def _discover_yolo_models(self):
        """Discover and load YOLO models from the specified directory"""
        if not self.yolo_models_dir.is_dir():
            logger.warning(f"YOLO models directory not found: {self.yolo_models_dir}")
            return

        for model_file in self.yolo_models_dir.glob("*.pt"):
            model_name = model_file.stem
            try:
                self.yolo_models[model_name] = YOLO(str(model_file))
                logger.info(f"Loaded YOLO model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model {model_file}: {e}")

    def _load_depth_model(self):
        """Load the DPT depth estimation model"""
        if not TRANSFORMERS_AVAILABLE:
            return

        try:
            model_name = "Intel/dpt-large"
            self.depth_processor = DPTImageProcessor.from_pretrained(model_name)
            self.depth_model = DPTForDepthEstimation.from_pretrained(model_name)
            logger.info(f"Loaded depth model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            self.depth_model = None
            self.depth_processor = None

    def _generate_output_path(self, input_path: str, model_type: str) -> str:
        """Generate unique output path for processed image"""
        unique_id = uuid.uuid4().hex[:8]
        input_name = Path(input_path).stem
        output_filename = f"{input_name}_{model_type}_{unique_id}.png"
        return str(self.output_dir / output_filename)

    def _process_canny(self, input_path: str, output_path: str) -> Optional[str]:
        """Process image with Canny edge detection (CPU-bound)"""
        try:
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.error(f"Failed to read image: {input_path}")
                return None

            edges = cv2.Canny(image, 100, 200)
            cv2.imwrite(output_path, edges)

            logger.info(f"Canny edge map saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Canny processing failed: {e}")
            return None

    def _process_depth(self, input_path: str, output_path: str) -> Optional[str]:
        """Process image with depth estimation (GPU-bound)"""
        if not self.depth_model or not self.depth_processor:
            logger.error("Depth model not available")
            return None

        try:
            image = Image.open(input_path).convert("RGB")
            inputs = self.depth_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth_map = Image.fromarray(formatted)
            depth_map.save(output_path)

            logger.info(f"Depth map saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Depth processing failed: {e}")
            if 'torch' in str(e).lower() and not TRANSFORMERS_AVAILABLE:
                logger.error("PyTorch required. Install: pip install torch")
            return None

    def _process_yolo(
        self,
        input_path: str,
        output_path: str,
        model_type: str
    ) -> Optional[str]:
        """Process image with YOLO (pose/segmentation)"""
        # Find appropriate YOLO model
        yolo_model = None
        search_term = 'pose' if 'pose' in model_type else 'seg'

        for name, model in self.yolo_models.items():
            if search_term in name.lower():
                yolo_model = model
                break

        if not yolo_model:
            logger.error(f"No YOLO model found for: {model_type}")
            return None

        try:
            results = yolo_model(input_path, verbose=False)
            if results and len(results) > 0:
                annotated_image = results[0].plot()
                cv2.imwrite(output_path, annotated_image)
                logger.info(f"YOLO control map saved: {output_path}")
                return output_path
            else:
                logger.warning("YOLO returned no results")
                return None
        except Exception as e:
            logger.error(f"YOLO processing failed: {e}")
            return None

    async def run(self, input_path: str, model_type: str) -> Optional[str]:
        """
        Run preprocessor on input image (async)

        Args:
            input_path: Path to input image
            model_type: Type of preprocessing ('canny', 'depth', 'pose', 'seg')

        Returns:
            Path to processed image, or None if failed
        """
        if not os.path.exists(input_path):
            logger.error(f"Input image not found: {input_path}")
            return None

        logger.info(f"Running preprocessor '{model_type}' on: {input_path}")
        output_path = self._generate_output_path(input_path, model_type)

        try:
            if model_type == "canny":
                # CPU-bound, run in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor,
                    self._process_canny,
                    input_path,
                    output_path
                )

            elif model_type == "depth":
                # GPU-bound, run in thread pool (PyTorch handles GPU internally)
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor,
                    self._process_depth,
                    input_path,
                    output_path
                )

            elif 'pose' in model_type or 'seg' in model_type:
                # YOLO processing, run in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor,
                    self._process_yolo,
                    input_path,
                    output_path,
                    model_type
                )

            else:
                logger.error(f"Unknown preprocessor type: {model_type}")
                return None

        except Exception as e:
            logger.error(f"Preprocessing failed for '{model_type}': {e}")
            return None

    def list_available_models(self) -> Dict[str, bool]:
        """List available preprocessing models"""
        return {
            "canny": True,  # Always available
            "depth": self.depth_model is not None,
            "pose": any('pose' in name.lower() for name in self.yolo_models.keys()),
            "segmentation": any('seg' in name.lower() for name in self.yolo_models.keys())
        }

    async def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("Preprocessor resources released")


# For backward compatibility, keep synchronous wrapper
class PreprocessorSync:
    """Synchronous wrapper for backward compatibility"""

    def __init__(self, yolo_models_dir: str = "yolo_models", output_dir: str = "outputs"):
        self._preprocessor = Preprocessor(yolo_models_dir, output_dir)

    def run(self, input_path: str, model_type: str) -> Optional[str]:
        """Synchronous run method"""
        return asyncio.run(self._preprocessor.run(input_path, model_type))

    def list_available_models(self) -> Dict[str, bool]:
        return self._preprocessor.list_available_models()


# CLI interface for testing
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="DiffuGen Preprocessor")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--type", choices=["canny", "depth", "pose", "seg"],
                       default="canny", help="Preprocessing type")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--yolo-dir", default="yolo_models", help="YOLO models directory")

    args = parser.parse_args()

    async def main():
        preprocessor = Preprocessor(args.yolo_dir, args.output_dir)
        result = await preprocessor.run(args.image, args.type)

        if result:
            print(f"Success: {result}")
        else:
            print("Preprocessing failed")

        await preprocessor.close()

    asyncio.run(main())
