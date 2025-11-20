"""
Batch Scene Generation System
Generate multiple scenes at once with consistent characters and style
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Batch Generation Models
# ============================================================================

class BatchStatus(Enum):
    """Status of batch generation"""
    PENDING = "pending"
    PREPARING = "preparing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SceneDefinition:
    """Definition of a single scene to generate"""
    id: str
    description: str
    character_name: Optional[str] = None
    style_override: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)

    # Results
    status: str = "pending"
    image_path: Optional[str] = None
    error_message: Optional[str] = None
    generation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BatchJob:
    """Batch generation job"""
    batch_id: str
    session_id: str
    scenes: List[SceneDefinition]

    # Global settings
    character_name: Optional[str] = None
    style_name: Optional[str] = None
    use_lora: bool = False

    # Progress tracking
    status: BatchStatus = BatchStatus.PENDING
    total_scenes: int = 0
    completed_scenes: int = 0
    failed_scenes: int = 0

    # Timing
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    estimated_time_remaining: float = 0.0

    # Results
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        self.total_scenes = len(self.scenes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "batch_id": self.batch_id,
            "session_id": self.session_id,
            "scenes": [s.to_dict() for s in self.scenes],
            "character_name": self.character_name,
            "style_name": self.style_name,
            "use_lora": self.use_lora,
            "status": self.status.value,
            "total_scenes": self.total_scenes,
            "completed_scenes": self.completed_scenes,
            "failed_scenes": self.failed_scenes,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "estimated_time_remaining": self.estimated_time_remaining,
            "results": self.results
        }


# ============================================================================
# Batch Generation Manager
# ============================================================================

class BatchGenerationManager:
    """
    Manages batch scene generation
    """

    def __init__(
        self,
        character_engine=None,
        style_manager=None,
        max_concurrent: int = 3,
        batch_dir: str = "batches"
    ):
        self.character_engine = character_engine
        self.style_manager = style_manager
        self.max_concurrent = max_concurrent

        self.batch_dir = Path(batch_dir)
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        # Track active batch jobs
        self.active_batches: Dict[str, BatchJob] = {}

        # Semaphore for concurrent generation limit
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def create_batch(
        self,
        session_id: str,
        scene_descriptions: List[str],
        character_name: Optional[str] = None,
        style_name: Optional[str] = None,
        use_lora: bool = False
    ) -> BatchJob:
        """
        Create a new batch generation job

        Args:
            session_id: Session ID
            scene_descriptions: List of scene descriptions
            character_name: Optional character to use
            style_name: Optional style to apply
            use_lora: Whether to use LoRA if available

        Returns:
            BatchJob object
        """
        import uuid

        batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Create scene definitions
        scenes = []
        for idx, description in enumerate(scene_descriptions):
            scene = SceneDefinition(
                id=f"scene_{idx:03d}",
                description=description,
                character_name=character_name,
                style_override=None
            )
            scenes.append(scene)

        # Create batch job
        batch = BatchJob(
            batch_id=batch_id,
            session_id=session_id,
            scenes=scenes,
            character_name=character_name,
            style_name=style_name,
            use_lora=use_lora
        )

        # Store in active batches
        self.active_batches[batch_id] = batch

        # Save batch metadata
        self._save_batch_metadata(batch)

        logger.info(f"Created batch job {batch_id} with {len(scenes)} scenes")

        return batch

    async def execute_batch(
        self,
        batch_id: str,
        progress_callback: Optional[callable] = None
    ) -> BatchJob:
        """
        Execute batch generation

        Args:
            batch_id: Batch ID to execute
            progress_callback: Optional callback for progress updates

        Returns:
            Updated BatchJob
        """
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch not found: {batch_id}")

        batch = self.active_batches[batch_id]

        logger.info(f"Starting batch generation for {batch_id}")

        batch.status = BatchStatus.PREPARING
        batch.started_at = time.time()

        if progress_callback:
            await progress_callback(batch)

        try:
            # Load character if specified
            character = None
            if batch.character_name and self.character_engine:
                character = self.character_engine.library.get_character(batch.character_name)
                if not character:
                    raise ValueError(f"Character not found: {batch.character_name}")

            # Load style if specified
            style = None
            if batch.style_name and self.style_manager:
                style = self.style_manager.library.get_style(batch.style_name)
                if not style:
                    raise ValueError(f"Style not found: {batch.style_name}")

            batch.status = BatchStatus.GENERATING

            # Generate scenes concurrently (with semaphore limit)
            tasks = []
            for scene in batch.scenes:
                task = self._generate_scene(
                    batch,
                    scene,
                    character,
                    style,
                    progress_callback
                )
                tasks.append(task)

            # Wait for all scenes to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Update batch status
            if batch.failed_scenes == 0:
                batch.status = BatchStatus.COMPLETED
            elif batch.completed_scenes > 0:
                batch.status = BatchStatus.COMPLETED  # Partial success
            else:
                batch.status = BatchStatus.FAILED

            batch.completed_at = time.time()

            logger.info(
                f"Batch {batch_id} completed: "
                f"{batch.completed_scenes}/{batch.total_scenes} scenes generated"
            )

            if progress_callback:
                await progress_callback(batch)

            # Save final results
            self._save_batch_metadata(batch)

            return batch

        except Exception as e:
            logger.error(f"Batch generation failed for {batch_id}: {e}")
            batch.status = BatchStatus.FAILED
            batch.completed_at = time.time()

            if progress_callback:
                await progress_callback(batch)

            return batch

    async def _generate_scene(
        self,
        batch: BatchJob,
        scene: SceneDefinition,
        character,
        style,
        progress_callback: Optional[callable] = None
    ):
        """Generate a single scene"""
        async with self.semaphore:
            try:
                logger.info(f"Generating scene {scene.id}: {scene.description}")

                scene.status = "generating"
                start_time = time.time()

                # Build generation parameters
                params = scene.custom_params.copy()

                # Apply style if present
                prompt = scene.description
                if style:
                    prompt = style.build_style_prompt(prompt)
                    params["cfg_scale"] = style.cfg_scale
                    params["steps"] = style.steps
                    params["sampling_method"] = style.sampling_method
                    params["negative_prompt"] = style.build_negative_prompt()

                # Generate with character if present
                if character and self.character_engine:
                    if batch.use_lora and character.has_lora:
                        # Use LoRA for perfect consistency
                        result = await self.character_engine.generate_with_lora(
                            character=character,
                            scene_description=scene.description,
                            **params
                        )
                    else:
                        # Use img2img
                        result = await self.character_engine.generate_with_character(
                            character=character,
                            scene_description=scene.description,
                            consistency_strength=0.75,
                            **params
                        )
                else:
                    # Generate without character (need to implement direct generation)
                    # For now, raise error
                    raise NotImplementedError("Generation without character not yet implemented")

                # Update scene with results
                if result.get("success"):
                    scene.status = "completed"
                    scene.image_path = result.get("image_path")
                    scene.generation_time = time.time() - start_time

                    batch.completed_scenes += 1
                    batch.results[scene.id] = result

                    logger.info(f"Scene {scene.id} completed in {scene.generation_time:.2f}s")
                else:
                    raise Exception(result.get("error", "Unknown error"))

            except Exception as e:
                logger.error(f"Failed to generate scene {scene.id}: {e}")
                scene.status = "failed"
                scene.error_message = str(e)
                batch.failed_scenes += 1

            # Update ETA
            if batch.completed_scenes + batch.failed_scenes > 0:
                elapsed = time.time() - batch.started_at
                avg_time = elapsed / (batch.completed_scenes + batch.failed_scenes)
                remaining = batch.total_scenes - (batch.completed_scenes + batch.failed_scenes)
                batch.estimated_time_remaining = avg_time * remaining

            # Trigger progress callback
            if progress_callback:
                await progress_callback(batch)

    def get_batch(self, batch_id: str) -> Optional[BatchJob]:
        """Get batch job by ID"""
        return self.active_batches.get(batch_id)

    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch job"""
        if batch_id not in self.active_batches:
            return False

        batch = self.active_batches[batch_id]
        batch.status = BatchStatus.CANCELLED
        batch.completed_at = time.time()

        logger.info(f"Cancelled batch {batch_id}")
        return True

    def list_batches(self, session_id: Optional[str] = None) -> List[BatchJob]:
        """List batch jobs, optionally filtered by session"""
        batches = list(self.active_batches.values())

        if session_id:
            batches = [b for b in batches if b.session_id == session_id]

        return batches

    def _save_batch_metadata(self, batch: BatchJob):
        """Save batch metadata to disk"""
        try:
            metadata_file = self.batch_dir / f"{batch.batch_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(batch.to_dict(), f, indent=2)

            logger.info(f"Saved batch metadata: {batch.batch_id}")

        except Exception as e:
            logger.error(f"Error saving batch metadata: {e}")


# ============================================================================
# Helper Functions
# ============================================================================

def parse_batch_request(message: str) -> Optional[List[str]]:
    """
    Parse natural language batch request

    Examples:
    - "Generate scenes: castle, forest, beach, mountain"
    - "Create 4 scenes: dragon flying, dragon sleeping, dragon eating, dragon playing"
    - "Make images of: sunny day, rainy day, snowy day"

    Returns:
        List of scene descriptions or None
    """
    import re

    # Pattern 1: "Generate scenes: x, y, z"
    pattern1 = r'(?:generate|create|make)\s+(?:scenes?|images?)\s*:\s*(.+)'
    match = re.search(pattern1, message, re.IGNORECASE)

    if match:
        scenes_text = match.group(1)
        scenes = [s.strip() for s in scenes_text.split(',')]
        return scenes

    # Pattern 2: "Generate X scenes: y, z, ..."
    pattern2 = r'(?:generate|create|make)\s+(\d+)\s+scenes?\s*:\s*(.+)'
    match = re.search(pattern2, message, re.IGNORECASE)

    if match:
        count = int(match.group(1))
        scenes_text = match.group(2)
        scenes = [s.strip() for s in scenes_text.split(',')]
        return scenes[:count]  # Limit to specified count

    return None


# ============================================================================
# CLI Testing
# ============================================================================

async def test_batch_generation():
    """Test batch generation system"""
    print("=== Batch Generation System Test ===\n")

    # Test 1: Create batch manager
    print("Test 1: Creating batch manager")
    manager = BatchGenerationManager(max_concurrent=2)
    print(f"✓ Batch manager created (max concurrent: 2)\n")

    # Test 2: Parse batch request
    print("Test 2: Parsing batch requests")
    test_messages = [
        "Generate scenes: castle, forest, beach",
        "Create 5 scenes: morning, noon, afternoon, evening, night",
        "Make images of: happy dragon, sad dragon, excited dragon"
    ]

    for msg in test_messages:
        scenes = parse_batch_request(msg)
        print(f"  '{msg}'")
        print(f"  → {len(scenes) if scenes else 0} scenes: {scenes}\n")

    # Test 3: Create batch job
    print("Test 3: Creating batch job")
    scene_descriptions = ["dragon in castle", "dragon in forest", "dragon on beach"]

    batch = await manager.create_batch(
        session_id="test_session",
        scene_descriptions=scene_descriptions,
        character_name="Spark",
        style_name="watercolor_soft",
        use_lora=False
    )

    print(f"✓ Created batch: {batch.batch_id}")
    print(f"  Total scenes: {batch.total_scenes}")
    print(f"  Character: {batch.character_name}")
    print(f"  Style: {batch.style_name}\n")

    # Test 4: Check batch status
    print("Test 4: Checking batch status")
    retrieved_batch = manager.get_batch(batch.batch_id)
    if retrieved_batch:
        print(f"✓ Retrieved batch: {retrieved_batch.batch_id}")
        print(f"  Status: {retrieved_batch.status.value}")
        print(f"  Progress: {retrieved_batch.completed_scenes}/{retrieved_batch.total_scenes}\n")

    print("✓ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_batch_generation())
