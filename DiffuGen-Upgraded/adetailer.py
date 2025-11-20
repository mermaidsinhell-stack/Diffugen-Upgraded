"""
Adetailer Module for DiffuGen
Automatically detects and refines faces and hands in generated images.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics YOLO not available. Install with: pip install ultralytics")


class BoundingBox:
    """Represents a bounding box for detected objects."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float, class_name: str):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_name = class_name
        self.width = x2 - x1
        self.height = y2 - y1

    def expand(self, padding: int = 32) -> 'BoundingBox':
        """
        Expand the bounding box by padding pixels.

        Args:
            padding: Pixels to add on each side

        Returns:
            New expanded BoundingBox
        """
        return BoundingBox(
            max(0, self.x1 - padding),
            max(0, self.y1 - padding),
            self.x2 + padding,
            self.y2 + padding,
            self.confidence,
            self.class_name
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "class": self.class_name
        }


class Adetailer:
    """Automatic detail enhancement for faces and hands."""

    def __init__(self,
                 sd_binary_path: str,
                 models_dir: str,
                 detection_model: str = "yolov8n.pt"):
        """
        Initialize Adetailer.

        Args:
            sd_binary_path: Path to stable-diffusion.cpp binary
            models_dir: Path to models directory
            detection_model: YOLO model to use for detection
        """
        self.sd_binary = sd_binary_path
        self.models_dir = models_dir
        self.detection_model_name = detection_model

        # Initialize YOLO model if available
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(detection_model)
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}")

    def detect_faces_and_hands(self,
                               image_path: str,
                               face_confidence: float = 0.3,
                               hand_confidence: float = 0.25) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        """
        Detect faces and hands in an image.

        Args:
            image_path: Path to the image
            face_confidence: Minimum confidence for face detection
            hand_confidence: Minimum confidence for hand detection

        Returns:
            Tuple of (faces, hands) as lists of BoundingBox objects
        """
        if not YOLO_AVAILABLE or self.yolo_model is None:
            print("Warning: YOLO not available, skipping detection")
            return [], []

        if not PIL_AVAILABLE:
            print("Warning: PIL not available, skipping detection")
            return [], []

        try:
            # Run detection
            results = self.yolo_model(image_path, verbose=False)

            faces = []
            hands = []

            # Extract bounding boxes
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]

                    # Check if it's a person (class 0 in COCO)
                    # For face detection, we'll use person bounding box upper portion
                    if class_id == 0 and confidence >= face_confidence:
                        # Estimate face region (upper 30% of person bbox)
                        face_height = int((y2 - y1) * 0.3)
                        face_bbox = BoundingBox(
                            int(x1), int(y1),
                            int(x2), int(y1 + face_height),
                            confidence, "face"
                        )
                        faces.append(face_bbox)

            # For better hand and face detection, use specialized models
            # You can replace this with dedicated face/hand detection models
            # like MediaPipe or RetinaFace for faces, and hand detection models

            return faces, hands

        except Exception as e:
            print(f"Error during detection: {e}")
            return [], []

    def detect_with_mediapipe(self, image_path: str) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        """
        Alternative detection using MediaPipe (more accurate for faces/hands).

        Args:
            image_path: Path to the image

        Returns:
            Tuple of (faces, hands) as lists of BoundingBox objects
        """
        try:
            import cv2
            import mediapipe as mp

            mp_face_detection = mp.solutions.face_detection
            mp_hands = mp.solutions.hands

            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape

            faces = []
            hands = []

            # Detect faces
            with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
                results = face_detection.process(image_rgb)
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x1 = int(bbox.xmin * width)
                        y1 = int(bbox.ymin * height)
                        x2 = int((bbox.xmin + bbox.width) * width)
                        y2 = int((bbox.ymin + bbox.height) * height)
                        confidence = detection.score[0]

                        faces.append(BoundingBox(x1, y1, x2, y2, confidence, "face"))

            # Detect hands
            with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.25) as hands_detector:
                results = hands_detector.process(image_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get bounding box from landmarks
                        x_coords = [lm.x * width for lm in hand_landmarks.landmark]
                        y_coords = [lm.y * height for lm in hand_landmarks.landmark]

                        x1 = int(min(x_coords))
                        y1 = int(min(y_coords))
                        x2 = int(max(x_coords))
                        y2 = int(max(y_coords))

                        hands.append(BoundingBox(x1, y1, x2, y2, 0.9, "hand"))

            return faces, hands

        except ImportError:
            print("Warning: MediaPipe not available. Install with: pip install mediapipe opencv-python")
            return self.detect_faces_and_hands(image_path)
        except Exception as e:
            print(f"Error with MediaPipe detection: {e}")
            return self.detect_faces_and_hands(image_path)

    def create_inpaint_mask(self,
                           image_path: str,
                           bboxes: List[BoundingBox],
                           output_path: str,
                           padding: int = 32) -> bool:
        """
        Create an inpainting mask for the detected regions.

        Args:
            image_path: Original image path
            bboxes: List of bounding boxes to mask
            output_path: Where to save the mask
            padding: Padding around bounding boxes

        Returns:
            True if successful
        """
        if not PIL_AVAILABLE:
            return False

        try:
            # Load image
            img = Image.open(image_path)
            width, height = img.size

            # Create white mask
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)

            # Draw white rectangles for each bbox
            for bbox in bboxes:
                expanded = bbox.expand(padding)
                draw.rectangle(
                    [
                        max(0, expanded.x1),
                        max(0, expanded.y1),
                        min(width, expanded.x2),
                        min(height, expanded.y2)
                    ],
                    fill=255
                )

            # Save mask
            mask.save(output_path)
            return True

        except Exception as e:
            print(f"Error creating mask: {e}")
            return False

    def inpaint_region(self,
                      original_image: str,
                      mask_image: str,
                      output_path: str,
                      prompt: str,
                      model_type: str = "sdxl",
                      strength: float = 0.4,
                      steps: int = 20,
                      cfg_scale: float = 7.0,
                      seed: int = -1) -> bool:
        """
        Inpaint a region using stable-diffusion.cpp.

        Args:
            original_image: Path to original image
            mask_image: Path to mask image (white = inpaint area)
            output_path: Where to save result
            prompt: Prompt for inpainting
            model_type: Model to use
            strength: Denoising strength (0.0-1.0)
            steps: Number of steps
            cfg_scale: CFG scale
            seed: Random seed

        Returns:
            True if successful
        """
        try:
            # Build command for img2img with mask
            cmd = [
                self.sd_binary,
                "--mode", "img2img",
                "-i", original_image,
                "--control-image", mask_image,
                "-p", prompt,
                "-o", output_path,
                "--steps", str(steps),
                "--cfg-scale", str(cfg_scale),
                "--strength", str(strength),
                "--sampling-method", "euler_a"
            ]

            if seed >= 0:
                cmd.extend(["--seed", str(seed)])

            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            return result.returncode == 0

        except Exception as e:
            print(f"Error during inpainting: {e}")
            return False

    def process_image(self,
                     image_path: str,
                     prompt: str,
                     model_type: str = "sdxl",
                     fix_faces: bool = True,
                     fix_hands: bool = True,
                     face_strength: float = 0.4,
                     hand_strength: float = 0.5,
                     face_prompt: Optional[str] = None,
                     hand_prompt: Optional[str] = None,
                     steps: int = 20,
                     cfg_scale: float = 7.0,
                     output_dir: Optional[str] = None,
                     detection_method: str = "yolo") -> Dict:
        """
        Process an image with Adetailer (detect and refine faces/hands).

        Args:
            image_path: Path to the generated image
            prompt: Original prompt
            model_type: Model to use for refinement
            fix_faces: Whether to fix faces
            fix_hands: Whether to fix hands
            face_strength: Denoising strength for faces
            hand_strength: Denoising strength for hands
            face_prompt: Custom prompt for face refinement (uses original if None)
            hand_prompt: Custom prompt for hand refinement (uses original if None)
            steps: Number of steps for refinement
            cfg_scale: CFG scale
            output_dir: Output directory
            detection_method: "yolo" or "mediapipe"

        Returns:
            Dictionary with results
        """
        if not os.path.exists(image_path):
            return {"success": False, "error": "Image not found"}

        # Detect faces and hands
        if detection_method == "mediapipe":
            faces, hands = self.detect_with_mediapipe(image_path)
        else:
            faces, hands = self.detect_faces_and_hands(image_path)

        results = {
            "success": True,
            "original_image": image_path,
            "faces_detected": len(faces),
            "hands_detected": len(hands),
            "faces": [f.to_dict() for f in faces],
            "hands": [h.to_dict() for h in hands],
            "refined_images": []
        }

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(image_path)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        current_image = image_path

        # Fix faces
        if fix_faces and faces:
            print(f"Detected {len(faces)} face(s), refining...")
            mask_path = os.path.join(output_dir, f"{base_name}_face_mask.png")
            output_path = os.path.join(output_dir, f"{base_name}_adetailer_faces.png")

            if self.create_inpaint_mask(current_image, faces, mask_path):
                inpaint_prompt = face_prompt or f"{prompt}, detailed face, high quality face"
                if self.inpaint_region(
                    current_image, mask_path, output_path,
                    inpaint_prompt, model_type,
                    face_strength, steps, cfg_scale
                ):
                    current_image = output_path
                    results["refined_images"].append(output_path)
                    results["face_refined"] = True
                else:
                    results["face_refined"] = False
                    results["face_error"] = "Inpainting failed"

        # Fix hands
        if fix_hands and hands:
            print(f"Detected {len(hands)} hand(s), refining...")
            mask_path = os.path.join(output_dir, f"{base_name}_hand_mask.png")
            output_path = os.path.join(output_dir, f"{base_name}_adetailer_hands.png")

            if self.create_inpaint_mask(current_image, hands, mask_path):
                inpaint_prompt = hand_prompt or f"{prompt}, detailed hands, perfect hands, anatomically correct hands"
                if self.inpaint_region(
                    current_image, mask_path, output_path,
                    inpaint_prompt, model_type,
                    hand_strength, steps, cfg_scale
                ):
                    current_image = output_path
                    results["refined_images"].append(output_path)
                    results["hand_refined"] = True
                else:
                    results["hand_refined"] = False
                    results["hand_error"] = "Inpainting failed"

        results["final_image"] = current_image

        return results


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adetailer - Automatic face and hand refinement")
    parser.add_argument("image", help="Image to process")
    parser.add_argument("--prompt", required=True, help="Original prompt")
    parser.add_argument("--sd-binary", required=True, help="Path to stable-diffusion.cpp binary")
    parser.add_argument("--models-dir", required=True, help="Path to models directory")
    parser.add_argument("--no-faces", action="store_true", help="Don't fix faces")
    parser.add_argument("--no-hands", action="store_true", help="Don't fix hands")
    parser.add_argument("--face-strength", type=float, default=0.4, help="Face denoising strength")
    parser.add_argument("--hand-strength", type=float, default=0.5, help="Hand denoising strength")
    parser.add_argument("--steps", type=int, default=20, help="Refinement steps")
    parser.add_argument("--cfg-scale", type=float, default=7.0, help="CFG scale")
    parser.add_argument("--detection", choices=["yolo", "mediapipe"], default="mediapipe",
                       help="Detection method")
    parser.add_argument("--output-dir", help="Output directory")

    args = parser.parse_args()

    adetailer = Adetailer(args.sd_binary, args.models_dir)

    results = adetailer.process_image(
        image_path=args.image,
        prompt=args.prompt,
        fix_faces=not args.no_faces,
        fix_hands=not args.no_hands,
        face_strength=args.face_strength,
        hand_strength=args.hand_strength,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        output_dir=args.output_dir,
        detection_method=args.detection
    )

    print(json.dumps(results, indent=2))
