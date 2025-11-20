import os
import logging
import uuid
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Attempt to import transformers and torch for depth estimation
try:
    import torch
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not found. Depth preprocessing will be unavailable. Install with: pip install transformers torch")

class Preprocessor:
    def __init__(self, yolo_models_dir: str = "yolo_models", output_dir: str = "outputs"):
        """
        Initializes the Preprocessor with paths to models and output directory.
        """
        self.yolo_models_dir = yolo_models_dir
        self.output_dir = output_dir
        self.yolo_models = {}
        self.depth_model = None
        self.depth_processor = None

        os.makedirs(self.output_dir, exist_ok=True)
        self._discover_yolo_models()
        self._load_depth_model()

    def _discover_yolo_models(self):
        """Discovers and loads YOLO models from the specified directory."""
        if not os.path.isdir(self.yolo_models_dir):
            logging.warning(f"YOLO models directory not found: {self.yolo_models_dir}")
            return
        for model_file in os.listdir(self.yolo_models_dir):
            if model_file.endswith(".pt"):
                model_name = os.path.splitext(model_file)[0]
                model_path = os.path.join(self.yolo_models_dir, model_file)
                try:
                    self.yolo_models[model_name] = YOLO(model_path)
                    logging.info(f"Successfully loaded YOLO model: {model_name} from {model_path}")
                except Exception as e:
                    logging.error(f"Failed to load YOLO model {model_file}: {e}")

    def _load_depth_model(self):
        """Loads the DPT depth estimation model from Hugging Face."""
        if TRANSFORMERS_AVAILABLE:
            try:
                model_name = "Intel/dpt-large"
                self.depth_processor = DPTImageProcessor.from_pretrained(model_name)
                self.depth_model = DPTForDepthEstimation.from_pretrained(model_name)
                logging.info(f"Successfully loaded Depth model: {model_name}")
            except Exception as e:
                logging.error(f"Failed to load depth estimation model: {e}")
                self.depth_model = None
                self.depth_processor = None

    def _get_output_path(self, input_path: str, model_type: str) -> str:
        """Generates a unique output path for the processed image."""
        unique_id = uuid.uuid4().hex[:8]
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{base_name}_{model_type}_{unique_id}.png"
        return os.path.join(self.output_dir, output_filename)

    def run(self, input_path: str, model_type: str) -> str:
        """
        Runs the specified preprocessor model on the input image.
        """
        if not os.path.exists(input_path):
            logging.error(f"Input image not found for preprocessing: {input_path}")
            return None

        logging.info(f"Running preprocessor '{model_type}' on image: {input_path}")
        output_path = self._get_output_path(input_path, model_type)

        try:
            if model_type == "canny":
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    logging.error(f"Failed to read image for Canny: {input_path}")
                    return None
                edges = cv2.Canny(image, 100, 200)
                cv2.imwrite(output_path, edges)
                logging.info(f"Successfully saved Canny edge map to: {output_path}")
                return output_path

            elif model_type == "depth":
                if not self.depth_model or not self.depth_processor:
                    logging.error("Depth model is not available.")
                    return None
                
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
                logging.info(f"Successfully saved depth map to: {output_path}")
                return output_path

            elif 'pose' in model_type or 'seg' in model_type:
                yolo_model = None
                if 'pose' in model_type:
                    for name, model in self.yolo_models.items():
                        if 'pose' in name:
                            yolo_model = model
                            break
                elif 'seg' in model_type:
                    for name, model in self.yolo_models.items():
                        if 'seg' in name:
                            yolo_model = model
                            break
                
                if not yolo_model:
                    logging.error(f"No suitable YOLO model found for preprocessor type: {model_type}")
                    return None

                results = yolo_model(input_path, verbose=False)
                if results and len(results) > 0:
                    annotated_image = results[0].plot()
                    cv2.imwrite(output_path, annotated_image)
                    logging.info(f"Successfully saved YOLO control map to: {output_path}")
                    return output_path
                else:
                    logging.warning("YOLO model did not return any results.")
                    return None
            
            else:
                logging.error(f"Unknown or unsupported preprocessor model type: {model_type}")
                return None

        except Exception as e:
            logging.error(f"An error occurred during '{model_type}' preprocessing: {e}")
            # For torch/transformers errors, provide more specific guidance
            if 'torch' in str(e).lower() and not TRANSFORMERS_AVAILABLE:
                 logging.error("PyTorch is required for depth estimation. Install with: pip install torch")
            return None

if __name__ == '__main__':
    # This block is for testing and will not run in production.
    # Assumes you have model files and test images in the correct directories.
    pass

