"""
Character Consistency System for Storybooks
Enables reusing the same character across multiple scenes with visual consistency
"""

import asyncio
import json
import logging
import time
import shutil
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import base64
import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Character:
    """Character definition with reference images"""
    name: str
    description: str  # Base prompt description
    reference_image: str  # Path to reference image
    seed: int  # Seed used for generation
    style_notes: str = ""  # Additional style notes
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    # Multiple reference images for different poses/angles
    reference_images: Dict[str, str] = field(default_factory=dict)  # pose -> image_path

    # Generation parameters used
    parameters: Dict[str, Any] = field(default_factory=dict)

    # LoRA model support for perfect consistency
    lora_path: Optional[str] = None  # Path to trained LoRA model
    lora_weight: float = 1.0  # LoRA weight (0.0-2.0)
    has_lora: bool = False  # Whether LoRA is trained and available

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Character':
        """Create from dictionary"""
        return cls(**data)


# ============================================================================
# Character Library Manager
# ============================================================================

class CharacterLibrary:
    """
    Manages character storage and retrieval
    """

    def __init__(self, library_dir: str = "characters"):
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.characters: Dict[str, Character] = {}

        # Load existing characters
        self._load_characters()

    def _load_characters(self):
        """Load characters from disk"""
        try:
            for char_file in self.library_dir.glob("*.json"):
                with open(char_file, 'r') as f:
                    data = json.load(f)
                    character = Character.from_dict(data)
                    self.characters[character.name] = character
                    logger.info(f"Loaded character: {character.name}")
        except Exception as e:
            logger.error(f"Error loading characters: {e}")

    def save_character(self, character: Character):
        """Save character to library"""
        try:
            # Save JSON metadata
            char_file = self.library_dir / f"{character.name}.json"
            with open(char_file, 'w') as f:
                json.dump(character.to_dict(), f, indent=2)

            # Update cache
            self.characters[character.name] = character

            logger.info(f"Saved character: {character.name}")

        except Exception as e:
            logger.error(f"Error saving character: {e}")
            raise

    def get_character(self, name: str) -> Optional[Character]:
        """Get character by name"""
        return self.characters.get(name)

    def list_characters(self) -> List[Character]:
        """List all characters"""
        return list(self.characters.values())

    def delete_character(self, name: str) -> bool:
        """Delete character"""
        try:
            if name in self.characters:
                # Delete JSON file
                char_file = self.library_dir / f"{name}.json"
                if char_file.exists():
                    char_file.unlink()

                # Delete reference images
                character = self.characters[name]
                if character.reference_image and Path(character.reference_image).exists():
                    Path(character.reference_image).unlink()

                for img_path in character.reference_images.values():
                    if Path(img_path).exists():
                        Path(img_path).unlink()

                # Remove from cache
                del self.characters[name]

                logger.info(f"Deleted character: {name}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting character: {e}")
            return False

    def search_characters(self, query: str) -> List[Character]:
        """Search characters by name or tags"""
        query_lower = query.lower()
        results = []

        for character in self.characters.values():
            if query_lower in character.name.lower():
                results.append(character)
            elif any(query_lower in tag.lower() for tag in character.tags):
                results.append(character)

        return results


# ============================================================================
# Character Consistency Engine
# ============================================================================

class CharacterConsistencyEngine:
    """
    Provides character consistency across generations
    """

    def __init__(
        self,
        diffugen_base_url: str = "http://localhost:8080",
        library: Optional[CharacterLibrary] = None
    ):
        self.diffugen_base_url = diffugen_base_url
        self.library = library or CharacterLibrary()
        self.client = httpx.AsyncClient(timeout=300.0)

    async def create_character(
        self,
        name: str,
        description: str,
        tags: Optional[List[str]] = None,
        **generation_params
    ) -> Tuple[Character, Dict[str, Any]]:
        """
        Create new character with reference image

        Args:
            name: Character name
            description: Character description (becomes base prompt)
            tags: Optional tags for searching
            **generation_params: Additional generation parameters

        Returns:
            (Character object, generation result)
        """
        logger.info(f"Creating character: {name}")

        # Build prompt for character sheet generation
        prompt = self._build_character_prompt(description)

        # Set consistent parameters for character generation
        params = {
            "prompt": prompt,
            "model": generation_params.get("model", "sd15"),
            "width": generation_params.get("width", 512),
            "height": generation_params.get("height", 512),
            "steps": generation_params.get("steps", 30),  # Higher quality for reference
            "cfg_scale": generation_params.get("cfg_scale", 8.0),
            "seed": generation_params.get("seed", -1),
            "sampling_method": generation_params.get("sampling_method", "euler_a"),
            "negative_prompt": generation_params.get("negative_prompt", self._get_default_negative())
        }

        # Generate reference image
        result = await self._generate_image(params)

        if not result.get("success"):
            raise Exception(f"Failed to generate character: {result.get('error')}")

        # Extract actual seed used
        actual_seed = result.get("parameters", {}).get("seed", -1)

        # Create character object
        character = Character(
            name=name,
            description=description,
            reference_image=result["image_path"],
            seed=actual_seed,
            tags=tags or [],
            parameters=params
        )

        # Save to library
        self.library.save_character(character)

        return character, result

    async def generate_with_character(
        self,
        character: Character,
        scene_description: str,
        consistency_strength: float = 0.7,
        **generation_params
    ) -> Dict[str, Any]:
        """
        Generate image with character in new scene

        Args:
            character: Character to use
            scene_description: Description of the scene
            consistency_strength: How much to preserve character (0.0-1.0)
                - 0.8-1.0: Very consistent (character dominates)
                - 0.5-0.7: Balanced (character + scene)
                - 0.0-0.4: Less consistent (scene dominates)
            **generation_params: Additional parameters

        Returns:
            Generation result
        """
        logger.info(f"Generating scene with character: {character.name}")

        # Build combined prompt
        combined_prompt = self._build_scene_prompt(character, scene_description)

        # Use img2img with reference image for consistency
        params = {
            "prompt": combined_prompt,
            "model": generation_params.get("model", character.parameters.get("model", "sd15")),
            "init_image_path": character.reference_image,
            "strength": 1.0 - consistency_strength,  # Lower strength = more like reference
            "width": generation_params.get("width", character.parameters.get("width", 512)),
            "height": generation_params.get("height", character.parameters.get("height", 512)),
            "steps": generation_params.get("steps", character.parameters.get("steps", 25)),
            "cfg_scale": generation_params.get("cfg_scale", character.parameters.get("cfg_scale", 7.5)),
            "seed": character.seed if generation_params.get("use_character_seed", False) else -1,
            "sampling_method": generation_params.get("sampling_method", character.parameters.get("sampling_method", "euler_a")),
            "negative_prompt": generation_params.get("negative_prompt", self._get_default_negative())
        }

        # Generate with img2img
        result = await self._generate_image_img2img(params)

        return result

    async def generate_character_sheet(
        self,
        character: Character,
        poses: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate character sheet with multiple poses

        Args:
            character: Character to generate sheet for
            poses: List of poses to generate (default: front, side, back)

        Returns:
            Dictionary of pose -> image_path
        """
        logger.info(f"Generating character sheet for: {character.name}")

        if poses is None:
            poses = ["front view", "side view", "back view", "three-quarter view"]

        results = {}

        for pose in poses:
            # Build prompt for this pose
            prompt = f"{character.description}, {pose}, character sheet, reference art, clean background"

            params = {
                "prompt": prompt,
                "model": character.parameters.get("model", "sd15"),
                "width": character.parameters.get("width", 512),
                "height": character.parameters.get("height", 512),
                "steps": 30,
                "cfg_scale": 8.0,
                "seed": character.seed,  # Use same seed for consistency
                "sampling_method": character.parameters.get("sampling_method", "euler_a"),
                "negative_prompt": self._get_default_negative()
            }

            result = await self._generate_image(params)

            if result.get("success"):
                results[pose] = result["image_path"]

                # Add to character's reference images
                character.reference_images[pose] = result["image_path"]
            else:
                logger.error(f"Failed to generate {pose}: {result.get('error')}")

        # Save updated character
        self.library.save_character(character)

        return results

    async def train_character_lora(
        self,
        character: Character,
        num_additional_images: int = 10,
        epochs: int = 10,
        progress_callback: Optional[callable] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Train a LoRA model for perfect character consistency

        Args:
            character: Character to train LoRA for
            num_additional_images: Number of additional training images to generate
            epochs: Training epochs
            progress_callback: Optional callback for progress updates

        Returns:
            (success, lora_path, error_message)
        """
        from lora_training import LoRATrainer, LoRATrainingConfig

        logger.info(f"Training LoRA for character: {character.name}")

        try:
            # Initialize LoRA trainer
            trainer = LoRATrainer()

            # Collect all reference images
            reference_images = [character.reference_image]
            reference_images.extend(character.reference_images.values())

            # Generate additional training images for variety
            if num_additional_images > 0:
                logger.info(f"Generating {num_additional_images} additional training images")
                additional_images = await trainer.generate_additional_training_data(
                    character_name=character.name,
                    character_description=character.description,
                    seed=character.seed,
                    num_variations=num_additional_images,
                    diffugen_base_url=self.diffugen_base_url
                )
                reference_images.extend(additional_images)

            # Prepare training dataset
            training_dir = await trainer.prepare_training_data(
                character_name=character.name,
                reference_images=reference_images,
                character_description=character.description
            )

            # Create training configuration
            config = LoRATrainingConfig(
                character_name=character.name,
                training_images_dir=training_dir,
                output_dir=f"loras/{character.name}",
                epochs=epochs,
                base_model=character.parameters.get("model", "sd15"),
                resolution=character.parameters.get("width", 512)
            )

            # Train LoRA
            success, lora_path, error = await trainer.train_lora(config, progress_callback)

            if success:
                # Update character with LoRA information
                character.lora_path = lora_path
                character.has_lora = True
                self.library.save_character(character)

                logger.info(f"LoRA training complete for {character.name}: {lora_path}")
                return True, lora_path, None
            else:
                logger.error(f"LoRA training failed for {character.name}: {error}")
                return False, None, error

        except Exception as e:
            logger.error(f"Error training LoRA for {character.name}: {e}")
            return False, None, str(e)

    async def generate_with_lora(
        self,
        character: Character,
        scene_description: str,
        lora_weight: float = 1.0,
        **generation_params
    ) -> Dict[str, Any]:
        """
        Generate image using character's LoRA model for perfect consistency

        Args:
            character: Character with trained LoRA
            scene_description: Description of the scene
            lora_weight: LoRA weight (0.0-2.0, default 1.0)
            **generation_params: Additional parameters

        Returns:
            Generation result
        """
        if not character.has_lora or not character.lora_path:
            raise ValueError(f"Character {character.name} does not have a trained LoRA")

        logger.info(f"Generating with LoRA for character: {character.name}")

        # Build prompt with LoRA tag
        from lora_training import create_lora_prompt_tag

        lora_tag = create_lora_prompt_tag(character.name, lora_weight)
        combined_prompt = f"{lora_tag} {character.description} {scene_description}, "
        combined_prompt += "children's book illustration, cute, friendly, wholesome"

        if character.style_notes:
            combined_prompt += f", {character.style_notes}"

        # Generate with LoRA
        params = {
            "prompt": combined_prompt,
            "model": generation_params.get("model", character.parameters.get("model", "sd15")),
            "width": generation_params.get("width", character.parameters.get("width", 512)),
            "height": generation_params.get("height", character.parameters.get("height", 512)),
            "steps": generation_params.get("steps", 25),
            "cfg_scale": generation_params.get("cfg_scale", 7.5),
            "seed": character.seed if generation_params.get("use_character_seed", True) else -1,
            "sampling_method": generation_params.get("sampling_method", "euler_a"),
            "negative_prompt": generation_params.get("negative_prompt", self._get_default_negative()),
            "lora_model_dir": str(Path(character.lora_path).parent),
        }

        result = await self._generate_image(params)
        return result

    def _build_character_prompt(self, description: str) -> str:
        """Build prompt for character generation"""
        base = f"{description}, character design, reference art, clean background, full body, "
        base += "children's book illustration, cute, friendly, wholesome, "
        base += "simple design, clear features, vibrant colors"
        return base

    def _build_scene_prompt(self, character: Character, scene: str) -> str:
        """Build prompt for scene with character"""
        # Combine character description with scene
        prompt = f"{character.description} {scene}, "
        prompt += "children's book illustration, cute, friendly, wholesome"

        if character.style_notes:
            prompt += f", {character.style_notes}"

        return prompt

    def _get_default_negative(self) -> str:
        """Get default negative prompt for children's books"""
        return """scary, frightening, dark, horror, violent, weapon, blood,
        realistic, photorealistic, uncanny, creepy, disturbing,
        inappropriate, adult themes, multiple characters, crowd,
        cluttered, messy background"""

    async def _generate_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image via DiffuGen API"""
        try:
            response = await self.client.post(
                f"{self.diffugen_base_url}/generate/stable",
                json=params
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }

            return response.json()

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _generate_image_img2img(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image via img2img for consistency"""
        try:
            # Read reference image
            with open(params["init_image_path"], "rb") as f:
                image_data = base64.b64encode(f.read()).decode()

            # Build request
            request_params = params.copy()
            request_params["init_image_base64"] = image_data
            del request_params["init_image_path"]

            response = await self.client.post(
                f"{self.diffugen_base_url}/generate/stable",
                json=request_params
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }

            return response.json()

        except Exception as e:
            logger.error(f"Error in img2img generation: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def close(self):
        """Cleanup resources"""
        await self.client.aclose()


# ============================================================================
# Seed-Based Consistency Helper
# ============================================================================

class SeedConsistency:
    """
    Helper for maintaining consistency using seeds and prompt templates
    """

    @staticmethod
    def generate_consistent_prompts(
        base_character: str,
        scenes: List[str],
        style: str = "children's book illustration"
    ) -> List[Dict[str, str]]:
        """
        Generate consistent prompts for multiple scenes

        Args:
            base_character: Base character description
            scenes: List of scene descriptions
            style: Art style to maintain

        Returns:
            List of prompt dictionaries
        """
        prompts = []

        for scene in scenes:
            prompt = f"{base_character}, {scene}, {style}"
            prompts.append({
                "prompt": prompt,
                "scene": scene
            })

        return prompts

    @staticmethod
    def extract_character_features(prompt: str) -> str:
        """
        Extract consistent character features from prompt
        """
        # Remove scene-specific details
        keywords_to_preserve = [
            "friendly", "cute", "dragon", "character",
            "green", "blue", "red", "yellow",  # Colors
            "big eyes", "small wings", "long tail",  # Features
        ]

        words = prompt.lower().split()
        preserved = [w for w in words if any(k in w for k in keywords_to_preserve)]

        return " ".join(preserved)


# ============================================================================
# CLI Testing
# ============================================================================

async def test_character_consistency():
    """Test character consistency system"""
    print("=== Character Consistency System Test ===\n")

    engine = CharacterConsistencyEngine()

    # Test 1: Create character
    print("Test 1: Creating character 'Spark the Dragon'")
    character, result = await engine.create_character(
        name="Spark",
        description="friendly green dragon with big eyes and small wings",
        tags=["dragon", "main_character", "friendly"]
    )
    print(f"✓ Created character: {character.name}")
    print(f"  Reference image: {character.reference_image}")
    print(f"  Seed: {character.seed}")
    print()

    # Test 2: Generate scene with character
    print("Test 2: Generating scene with Spark")
    scene_result = await engine.generate_with_character(
        character,
        "sitting in a colorful castle",
        consistency_strength=0.7
    )
    print(f"✓ Generated scene: {scene_result.get('image_path')}")
    print()

    # Test 3: Generate character sheet
    print("Test 3: Generating character sheet")
    sheet_results = await engine.generate_character_sheet(character)
    print(f"✓ Generated {len(sheet_results)} poses:")
    for pose, path in sheet_results.items():
        print(f"  - {pose}: {path}")
    print()

    # Test 4: List characters
    print("Test 4: Listing all characters")
    characters = engine.library.list_characters()
    print(f"✓ Total characters: {len(characters)}")
    for char in characters:
        print(f"  - {char.name}: {char.description}")
    print()

    await engine.close()


if __name__ == "__main__":
    asyncio.run(test_character_consistency())
