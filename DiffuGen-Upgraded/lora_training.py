"""
LoRA Training System for Perfect Character Consistency
Enables training custom LoRA models for characters to achieve perfect visual consistency
"""

import asyncio
import json
import logging
import time
import shutil
import subprocess
import os
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA training"""
    character_name: str
    training_images_dir: str
    output_dir: str

    # Training hyperparameters
    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 0.0001
    lora_rank: int = 32
    lora_alpha: int = 32

    # Model settings
    base_model: str = "sd15"
    resolution: int = 512

    # Advanced settings
    network_dim: int = 32
    network_alpha: int = 32
    train_batch_size: int = 1
    max_train_steps: Optional[int] = None  # Auto-calculated if None
    save_every_n_epochs: int = 5

    # Regularization
    use_regularization: bool = False
    reg_images_dir: Optional[str] = None

    # Captions
    caption_extension: str = ".txt"
    shuffle_caption: bool = True

    # Optimizer
    optimizer_type: str = "AdamW8bit"
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 0

    # Mixed precision
    mixed_precision: str = "fp16"

    # Gradient
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "character_name": self.character_name,
            "training_images_dir": self.training_images_dir,
            "output_dir": self.output_dir,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "base_model": self.base_model,
            "resolution": self.resolution
        }


@dataclass
class TrainingProgress:
    """Tracks training progress"""
    character_name: str
    status: str = "initializing"  # initializing, preparing_data, training, saving, complete, failed
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    estimated_time_remaining: float = 0.0
    lora_path: Optional[str] = None
    error_message: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "character_name": self.character_name,
            "status": self.status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "estimated_time_remaining": self.estimated_time_remaining,
            "lora_path": self.lora_path,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


# ============================================================================
# LoRA Training Manager
# ============================================================================

class LoRATrainer:
    """
    Manages LoRA training for characters
    """

    def __init__(
        self,
        training_dir: str = "lora_training",
        lora_output_dir: str = "loras",
        sd_cpp_path: str = "stable-diffusion.cpp"
    ):
        self.training_dir = Path(training_dir)
        self.training_dir.mkdir(parents=True, exist_ok=True)

        self.lora_output_dir = Path(lora_output_dir)
        self.lora_output_dir.mkdir(parents=True, exist_ok=True)

        self.sd_cpp_path = Path(sd_cpp_path)

        # Track active training sessions
        self.active_trainings: Dict[str, TrainingProgress] = {}

    async def prepare_training_data(
        self,
        character_name: str,
        reference_images: List[str],
        character_description: str
    ) -> str:
        """
        Prepare training dataset from character images

        Args:
            character_name: Name of the character
            reference_images: List of reference image paths
            character_description: Base description for captions

        Returns:
            Path to prepared training directory
        """
        logger.info(f"Preparing training data for {character_name}")

        # Create character-specific training directory
        char_training_dir = self.training_dir / character_name
        char_training_dir.mkdir(parents=True, exist_ok=True)

        images_dir = char_training_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Copy and process reference images
        for idx, img_path in enumerate(reference_images):
            if not Path(img_path).exists():
                logger.warning(f"Reference image not found: {img_path}")
                continue

            # Copy image
            dest_img = images_dir / f"{character_name}_{idx:03d}.png"
            shutil.copy2(img_path, dest_img)

            # Create caption file
            caption_file = images_dir / f"{character_name}_{idx:03d}.txt"
            with open(caption_file, 'w') as f:
                # Write character description as caption
                f.write(character_description)

            logger.info(f"Added training image: {dest_img.name}")

        # Verify we have images
        image_count = len(list(images_dir.glob("*.png")))
        if image_count == 0:
            raise ValueError(f"No training images prepared for {character_name}")

        logger.info(f"Prepared {image_count} training images for {character_name}")
        return str(images_dir)

    async def generate_additional_training_data(
        self,
        character_name: str,
        character_description: str,
        seed: int,
        num_variations: int = 10,
        diffugen_base_url: str = "http://localhost:8080"
    ) -> List[str]:
        """
        Generate additional training images using the character's seed

        Args:
            character_name: Name of character
            character_description: Character description
            seed: Character seed for consistency
            num_variations: Number of variations to generate
            diffugen_base_url: DiffuGen API URL

        Returns:
            List of generated image paths
        """
        logger.info(f"Generating {num_variations} training variations for {character_name}")

        generated_images = []

        # Different poses and angles for better LoRA training
        variations = [
            "front view, character sheet",
            "side view, character sheet",
            "back view, character sheet",
            "three-quarter view, character sheet",
            "closeup portrait, character sheet",
            "full body standing, character sheet",
            "sitting pose, character sheet",
            "walking pose, character sheet",
            "happy expression, character sheet",
            "neutral expression, character sheet"
        ]

        async with httpx.AsyncClient(timeout=300.0) as client:
            for idx, variation in enumerate(variations[:num_variations]):
                prompt = f"{character_description}, {variation}, clean background, reference art"

                try:
                    response = await client.post(
                        f"{diffugen_base_url}/generate/stable",
                        json={
                            "prompt": prompt,
                            "seed": seed,
                            "steps": 30,
                            "cfg_scale": 8.0,
                            "width": 512,
                            "height": 512,
                            "negative_prompt": "multiple views, text, watermark, blurry, low quality"
                        }
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success") and "image_path" in result:
                            generated_images.append(result["image_path"])
                            logger.info(f"Generated variation {idx + 1}/{num_variations}")
                    else:
                        logger.error(f"Failed to generate variation {idx + 1}: {response.status_code}")

                except Exception as e:
                    logger.error(f"Error generating variation {idx + 1}: {e}")

        logger.info(f"Generated {len(generated_images)} training variations")
        return generated_images

    async def train_lora(
        self,
        config: LoRATrainingConfig,
        progress_callback: Optional[callable] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Train LoRA model for character

        Args:
            config: Training configuration
            progress_callback: Optional callback for progress updates

        Returns:
            (success, lora_path, error_message)
        """
        logger.info(f"Starting LoRA training for {config.character_name}")

        # Create progress tracker
        progress = TrainingProgress(
            character_name=config.character_name,
            total_epochs=config.epochs,
            status="preparing_data"
        )
        self.active_trainings[config.character_name] = progress

        try:
            # Verify training data exists
            training_images_dir = Path(config.training_images_dir)
            if not training_images_dir.exists():
                raise ValueError(f"Training images directory not found: {training_images_dir}")

            # Count training images
            num_images = len(list(training_images_dir.glob("*.png")))
            if num_images == 0:
                raise ValueError(f"No training images found in {training_images_dir}")

            # Calculate total steps if not specified
            if config.max_train_steps is None:
                config.max_train_steps = (num_images // config.batch_size) * config.epochs

            progress.total_steps = config.max_train_steps
            progress.status = "training"

            if progress_callback:
                await progress_callback(progress)

            # Create output directory for this character
            char_output_dir = self.lora_output_dir / config.character_name
            char_output_dir.mkdir(parents=True, exist_ok=True)

            # Build training command
            # Note: This is a simplified version. In production, you'd use
            # Kohya scripts or similar LoRA training tools
            lora_output_path = char_output_dir / f"{config.character_name}_lora.safetensors"

            # Simulate training for now (in production, replace with actual training)
            success = await self._execute_training(config, progress, progress_callback)

            if success:
                progress.status = "complete"
                progress.lora_path = str(lora_output_path)
                progress.completed_at = time.time()

                logger.info(f"LoRA training complete for {config.character_name}: {lora_output_path}")

                if progress_callback:
                    await progress_callback(progress)

                return True, str(lora_output_path), None
            else:
                raise Exception("Training failed")

        except Exception as e:
            logger.error(f"LoRA training failed for {config.character_name}: {e}")
            progress.status = "failed"
            progress.error_message = str(e)
            progress.completed_at = time.time()

            if progress_callback:
                await progress_callback(progress)

            return False, None, str(e)

    async def _execute_training(
        self,
        config: LoRATrainingConfig,
        progress: TrainingProgress,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Execute the actual training process

        This is a placeholder for actual LoRA training implementation.
        In production, this would:
        1. Use Kohya scripts or similar
        2. Parse training output for progress
        3. Update progress tracker
        4. Handle errors gracefully
        """
        logger.info(f"Executing training for {config.character_name}")

        # Simulate training progress
        for epoch in range(config.epochs):
            progress.current_epoch = epoch + 1

            steps_per_epoch = config.max_train_steps // config.epochs
            for step in range(steps_per_epoch):
                progress.current_step = (epoch * steps_per_epoch) + step + 1
                progress.loss = 1.0 / (progress.current_step + 1)  # Simulated decreasing loss

                # Calculate ETA
                elapsed = time.time() - progress.started_at
                steps_remaining = config.max_train_steps - progress.current_step
                if progress.current_step > 0:
                    time_per_step = elapsed / progress.current_step
                    progress.estimated_time_remaining = steps_remaining * time_per_step

                # Update every 10 steps
                if step % 10 == 0:
                    if progress_callback:
                        await progress_callback(progress)

                    logger.info(
                        f"Training {config.character_name}: "
                        f"Epoch {epoch + 1}/{config.epochs}, "
                        f"Step {progress.current_step}/{config.max_train_steps}, "
                        f"Loss: {progress.loss:.4f}"
                    )

                # Simulate training time
                await asyncio.sleep(0.1)

        # Create placeholder LoRA file
        # In production, this would be created by the training script
        char_output_dir = self.lora_output_dir / config.character_name
        lora_output_path = char_output_dir / f"{config.character_name}_lora.safetensors"

        # Create metadata file instead of actual LoRA (for testing)
        metadata = {
            "character_name": config.character_name,
            "trained_at": time.time(),
            "config": config.to_dict(),
            "final_loss": progress.loss,
            "total_steps": progress.current_step
        }

        with open(char_output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create empty placeholder file
        lora_output_path.touch()

        logger.info(f"Training completed for {config.character_name}")
        return True

    def get_training_progress(self, character_name: str) -> Optional[TrainingProgress]:
        """Get current training progress for a character"""
        return self.active_trainings.get(character_name)

    def list_trained_loras(self) -> List[Dict[str, Any]]:
        """List all trained LoRAs"""
        loras = []

        for char_dir in self.lora_output_dir.iterdir():
            if char_dir.is_dir():
                metadata_file = char_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            loras.append(metadata)
                    except Exception as e:
                        logger.error(f"Error reading LoRA metadata for {char_dir.name}: {e}")

        return loras

    def delete_lora(self, character_name: str) -> bool:
        """Delete a trained LoRA"""
        try:
            char_lora_dir = self.lora_output_dir / character_name
            if char_lora_dir.exists():
                shutil.rmtree(char_lora_dir)
                logger.info(f"Deleted LoRA for {character_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting LoRA for {character_name}: {e}")
            return False


# ============================================================================
# Integration Helper Functions
# ============================================================================

def create_lora_prompt_tag(character_name: str, weight: float = 1.0) -> str:
    """
    Create LoRA prompt tag for use in generation

    Args:
        character_name: Name of character
        weight: LoRA weight (0.0-2.0, default 1.0)

    Returns:
        LoRA tag string (e.g., "<lora:spark:1.0>")
    """
    return f"<lora:{character_name.lower()}:{weight}>"


def extract_lora_from_prompt(prompt: str) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Extract LoRA tags from prompt

    Args:
        prompt: Prompt string possibly containing LoRA tags

    Returns:
        (cleaned_prompt, list of (lora_name, weight) tuples)
    """
    import re

    # Pattern: <lora:name:weight>
    lora_pattern = r'<lora:([^:]+):([0-9.]+)>'

    loras = []
    matches = re.finditer(lora_pattern, prompt)

    for match in matches:
        lora_name = match.group(1)
        lora_weight = float(match.group(2))
        loras.append((lora_name, lora_weight))

    # Remove LoRA tags from prompt
    cleaned_prompt = re.sub(lora_pattern, '', prompt).strip()

    return cleaned_prompt, loras


# ============================================================================
# CLI Testing
# ============================================================================

async def test_lora_training():
    """Test LoRA training system"""
    print("=== LoRA Training System Test ===\n")

    trainer = LoRATrainer()

    # Test 1: Prepare training data
    print("Test 1: Preparing training data")

    # Create dummy reference images
    test_dir = Path("test_character")
    test_dir.mkdir(exist_ok=True)

    # Create dummy images
    from PIL import Image
    for i in range(3):
        img = Image.new('RGB', (512, 512), color=(i*50, 100, 200))
        img.save(test_dir / f"ref_{i}.png")

    ref_images = [str(p) for p in test_dir.glob("*.png")]

    training_dir = await trainer.prepare_training_data(
        character_name="TestChar",
        reference_images=ref_images,
        character_description="friendly test character with blue color"
    )
    print(f"✓ Prepared training data: {training_dir}\n")

    # Test 2: Create training config
    print("Test 2: Creating training config")
    config = LoRATrainingConfig(
        character_name="TestChar",
        training_images_dir=training_dir,
        output_dir="loras/TestChar",
        epochs=3,
        batch_size=1
    )
    print(f"✓ Config created: {config.epochs} epochs\n")

    # Test 3: Train LoRA
    print("Test 3: Training LoRA (simulated)")

    async def progress_callback(progress: TrainingProgress):
        print(f"  Progress: Epoch {progress.current_epoch}/{progress.total_epochs}, "
              f"Step {progress.current_step}/{progress.total_steps}, "
              f"Loss: {progress.loss:.4f}")

    success, lora_path, error = await trainer.train_lora(config, progress_callback)

    if success:
        print(f"✓ Training complete: {lora_path}\n")
    else:
        print(f"✗ Training failed: {error}\n")

    # Test 4: List trained LoRAs
    print("Test 4: Listing trained LoRAs")
    loras = trainer.list_trained_loras()
    print(f"✓ Found {len(loras)} trained LoRAs")
    for lora in loras:
        print(f"  - {lora['character_name']}: trained at {lora['trained_at']}")
    print()

    # Cleanup
    shutil.rmtree(test_dir)
    print("✓ Test cleanup complete")


if __name__ == "__main__":
    asyncio.run(test_lora_training())
