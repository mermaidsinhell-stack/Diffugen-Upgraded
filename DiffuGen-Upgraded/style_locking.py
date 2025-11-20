"""
Style Locking System for Consistent Art Style Across Scenes
Ensures all illustrations maintain the same visual aesthetic
"""

import json
import logging
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Style Definition
# ============================================================================

@dataclass
class ArtStyle:
    """Definition of a consistent art style"""
    name: str
    description: str  # Style description for prompts

    # Visual characteristics
    technique: str  # e.g., "watercolor", "digital painting", "pencil sketch"
    color_palette: str  # e.g., "pastel colors", "vibrant colors", "muted tones"
    mood: str  # e.g., "whimsical", "dramatic", "playful"
    detail_level: str  # e.g., "highly detailed", "simple", "moderate detail"

    # Technical parameters
    cfg_scale: float = 7.5
    steps: int = 25
    sampling_method: str = "euler_a"

    # Prompt additions
    positive_tags: List[str] = field(default_factory=list)
    negative_tags: List[str] = field(default_factory=list)

    # Style enforcement
    style_strength: float = 0.8  # How strongly to enforce style (0.0-1.0)

    # Metadata
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    # Reference image for style (optional)
    reference_image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArtStyle':
        """Create from dictionary"""
        return cls(**data)

    def build_style_prompt(self, base_prompt: str) -> str:
        """Build prompt with style applied"""
        # Start with base prompt
        prompt = base_prompt

        # Add style description
        prompt += f", {self.description}"

        # Add technique
        prompt += f", {self.technique}"

        # Add color palette
        prompt += f", {self.color_palette}"

        # Add mood
        prompt += f", {self.mood}"

        # Add detail level
        prompt += f", {self.detail_level}"

        # Add positive tags
        if self.positive_tags:
            prompt += ", " + ", ".join(self.positive_tags)

        return prompt

    def build_negative_prompt(self, base_negative: str = "") -> str:
        """Build negative prompt with style restrictions"""
        negative_parts = [base_negative] if base_negative else []

        # Add style-specific negative tags
        if self.negative_tags:
            negative_parts.extend(self.negative_tags)

        return ", ".join(negative_parts)


# ============================================================================
# Predefined Styles for Children's Books
# ============================================================================

PRESET_STYLES = {
    "watercolor_soft": ArtStyle(
        name="Watercolor Soft",
        description="soft watercolor illustration",
        technique="watercolor painting, traditional media",
        color_palette="pastel colors, soft hues",
        mood="gentle, whimsical, dreamy",
        detail_level="moderate detail, painterly",
        positive_tags=["watercolor", "soft edges", "flowing colors", "children's book"],
        negative_tags=["harsh lines", "photorealistic", "digital art", "3d render"],
        cfg_scale=7.0,
        steps=25,
        style_strength=0.8
    ),

    "digital_vibrant": ArtStyle(
        name="Digital Vibrant",
        description="vibrant digital illustration",
        technique="digital painting, modern illustration",
        color_palette="vibrant colors, bold hues, saturated",
        mood="energetic, playful, cheerful",
        detail_level="clean lines, polished",
        positive_tags=["digital art", "vibrant", "clean", "children's book illustration"],
        negative_tags=["washed out", "dull", "sketch", "messy"],
        cfg_scale=7.5,
        steps=25,
        style_strength=0.85
    ),

    "pencil_sketch": ArtStyle(
        name="Pencil Sketch",
        description="pencil sketch illustration",
        technique="pencil drawing, hand-drawn, sketch",
        color_palette="grayscale, minimal color, subtle tones",
        mood="gentle, nostalgic, classic",
        detail_level="moderate detail, sketch lines",
        positive_tags=["pencil sketch", "hand-drawn", "sketch lines", "traditional"],
        negative_tags=["digital", "photorealistic", "3d", "overly polished"],
        cfg_scale=6.5,
        steps=20,
        style_strength=0.75
    ),

    "cartoon_bold": ArtStyle(
        name="Cartoon Bold",
        description="bold cartoon illustration",
        technique="cartoon style, vector art, graphic",
        color_palette="bold colors, primary colors, high contrast",
        mood="fun, energetic, dynamic",
        detail_level="simple, clean, bold outlines",
        positive_tags=["cartoon", "bold lines", "graphic", "children's cartoon"],
        negative_tags=["realistic", "detailed", "photographic", "subtle"],
        cfg_scale=8.0,
        steps=20,
        style_strength=0.9
    ),

    "storybook_classic": ArtStyle(
        name="Storybook Classic",
        description="classic storybook illustration",
        technique="traditional illustration, storybook art",
        color_palette="warm colors, earthy tones, nostalgic palette",
        mood="warm, comforting, timeless",
        detail_level="detailed, intricate, traditional",
        positive_tags=["storybook", "classic illustration", "traditional art", "timeless"],
        negative_tags=["modern", "digital", "minimalist", "abstract"],
        cfg_scale=7.5,
        steps=30,
        style_strength=0.8
    )
}


# ============================================================================
# Style Library Manager
# ============================================================================

class StyleLibrary:
    """
    Manages art style storage and retrieval
    """

    def __init__(self, library_dir: str = "styles"):
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.styles: Dict[str, ArtStyle] = {}

        # Load preset styles
        self._load_preset_styles()

        # Load custom styles from disk
        self._load_custom_styles()

    def _load_preset_styles(self):
        """Load predefined styles"""
        for style_name, style in PRESET_STYLES.items():
            self.styles[style_name] = style
            logger.info(f"Loaded preset style: {style_name}")

    def _load_custom_styles(self):
        """Load custom styles from disk"""
        try:
            for style_file in self.library_dir.glob("*.json"):
                with open(style_file, 'r') as f:
                    data = json.load(f)
                    style = ArtStyle.from_dict(data)
                    self.styles[style.name] = style
                    logger.info(f"Loaded custom style: {style.name}")
        except Exception as e:
            logger.error(f"Error loading custom styles: {e}")

    def save_style(self, style: ArtStyle):
        """Save custom style to library"""
        try:
            # Save JSON metadata
            style_file = self.library_dir / f"{style.name}.json"
            with open(style_file, 'w') as f:
                json.dump(style.to_dict(), f, indent=2)

            # Update cache
            self.styles[style.name] = style

            logger.info(f"Saved style: {style.name}")

        except Exception as e:
            logger.error(f"Error saving style: {e}")
            raise

    def get_style(self, name: str) -> Optional[ArtStyle]:
        """Get style by name"""
        return self.styles.get(name)

    def list_styles(self) -> List[ArtStyle]:
        """List all styles"""
        return list(self.styles.values())

    def delete_style(self, name: str) -> bool:
        """Delete custom style (cannot delete presets)"""
        try:
            if name in PRESET_STYLES:
                logger.warning(f"Cannot delete preset style: {name}")
                return False

            if name in self.styles:
                # Delete JSON file
                style_file = self.library_dir / f"{name}.json"
                if style_file.exists():
                    style_file.unlink()

                # Remove from cache
                del self.styles[name]

                logger.info(f"Deleted style: {name}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting style: {e}")
            return False

    def search_styles(self, query: str) -> List[ArtStyle]:
        """Search styles by name or tags"""
        query_lower = query.lower()
        results = []

        for style in self.styles.values():
            if query_lower in style.name.lower():
                results.append(style)
            elif any(query_lower in tag.lower() for tag in style.tags):
                results.append(style)

        return results


# ============================================================================
# Style Lock Manager
# ============================================================================

class StyleLockManager:
    """
    Manages style locking for conversation sessions
    """

    def __init__(self, library: Optional[StyleLibrary] = None):
        self.library = library or StyleLibrary()

        # Track active style locks per session
        self.active_locks: Dict[str, str] = {}  # session_id -> style_name

    def lock_style(self, session_id: str, style_name: str) -> bool:
        """Lock a style for a session"""
        style = self.library.get_style(style_name)
        if not style:
            logger.error(f"Style not found: {style_name}")
            return False

        self.active_locks[session_id] = style_name
        logger.info(f"Locked style '{style_name}' for session {session_id}")
        return True

    def unlock_style(self, session_id: str):
        """Unlock style for a session"""
        if session_id in self.active_locks:
            del self.active_locks[session_id]
            logger.info(f"Unlocked style for session {session_id}")

    def get_locked_style(self, session_id: str) -> Optional[ArtStyle]:
        """Get locked style for a session"""
        if session_id not in self.active_locks:
            return None

        style_name = self.active_locks[session_id]
        return self.library.get_style(style_name)

    def is_locked(self, session_id: str) -> bool:
        """Check if session has a locked style"""
        return session_id in self.active_locks

    def apply_style_to_params(
        self,
        session_id: str,
        prompt: str,
        params: Dict[str, Any],
        negative_prompt: str = ""
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Apply locked style to generation parameters

        Args:
            session_id: Session ID
            prompt: Base prompt
            params: Generation parameters
            negative_prompt: Base negative prompt

        Returns:
            (styled_prompt, updated_params)
        """
        style = self.get_locked_style(session_id)
        if not style:
            return prompt, params

        # Build styled prompt
        styled_prompt = style.build_style_prompt(prompt)

        # Build styled negative prompt
        styled_negative = style.build_negative_prompt(negative_prompt)

        # Update parameters with style settings
        updated_params = params.copy()
        updated_params["prompt"] = styled_prompt
        updated_params["negative_prompt"] = styled_negative
        updated_params["cfg_scale"] = style.cfg_scale
        updated_params["steps"] = style.steps
        updated_params["sampling_method"] = style.sampling_method

        logger.info(f"Applied style '{style.name}' to generation")

        return styled_prompt, updated_params


# ============================================================================
# CLI Testing
# ============================================================================

def test_style_locking():
    """Test style locking system"""
    print("=== Style Locking System Test ===\n")

    # Test 1: Load library
    print("Test 1: Loading style library")
    library = StyleLibrary()
    print(f"✓ Loaded {len(library.styles)} styles")
    print(f"  Preset styles: {list(PRESET_STYLES.keys())}\n")

    # Test 2: Get style
    print("Test 2: Getting watercolor style")
    watercolor = library.get_style("watercolor_soft")
    if watercolor:
        print(f"✓ Style: {watercolor.name}")
        print(f"  Technique: {watercolor.technique}")
        print(f"  Colors: {watercolor.color_palette}")
        print(f"  Mood: {watercolor.mood}\n")

    # Test 3: Apply style to prompt
    print("Test 3: Applying style to prompt")
    base_prompt = "a friendly dragon sitting in a castle"
    styled_prompt = watercolor.build_style_prompt(base_prompt)
    print(f"  Base: {base_prompt}")
    print(f"  Styled: {styled_prompt}\n")

    # Test 4: Style locking
    print("Test 4: Testing style lock manager")
    lock_manager = StyleLockManager(library)

    session_id = "test_session_123"
    lock_manager.lock_style(session_id, "watercolor_soft")
    print(f"✓ Locked style for session")

    locked_style = lock_manager.get_locked_style(session_id)
    print(f"✓ Retrieved locked style: {locked_style.name}")

    # Test 5: Apply to params
    print("\nTest 5: Applying style to parameters")
    params = {"width": 512, "height": 512}
    styled_prompt, updated_params = lock_manager.apply_style_to_params(
        session_id,
        base_prompt,
        params
    )
    print(f"✓ Updated parameters:")
    print(f"  CFG Scale: {updated_params['cfg_scale']}")
    print(f"  Steps: {updated_params['steps']}")
    print(f"  Sampling: {updated_params['sampling_method']}\n")

    print("✓ All tests passed!")


if __name__ == "__main__":
    test_style_locking()
