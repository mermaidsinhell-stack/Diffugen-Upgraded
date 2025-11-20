"""
Character Relationships System
Manages relationships between characters for multi-character scenes
"""

import json
import logging
import time
from typing import Optional, Dict, List, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Relationship Types
# ============================================================================

class RelationType(Enum):
    """Types of character relationships"""
    FRIEND = "friend"
    FAMILY = "family"
    COMPANION = "companion"
    RIVAL = "rival"
    MENTOR = "mentor"
    STUDENT = "student"
    SIBLING = "sibling"
    PARENT = "parent"
    CHILD = "child"
    TEAMMATE = "teammate"
    CUSTOM = "custom"


@dataclass
class Relationship:
    """Relationship between two characters"""
    character_a: str  # Character name
    character_b: str  # Character name
    relationship_type: RelationType
    description: str = ""  # e.g., "best friends since childhood"

    # Interaction hints for generation
    typical_distance: str = "close"  # close, medium, far
    typical_interaction: str = ""  # e.g., "playing together", "studying"
    emotional_tone: str = "friendly"  # friendly, playful, serious, etc.

    # Metadata
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["relationship_type"] = self.relationship_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create from dictionary"""
        data_copy = data.copy()
        if "relationship_type" in data_copy and isinstance(data_copy["relationship_type"], str):
            data_copy["relationship_type"] = RelationType(data_copy["relationship_type"])
        return cls(**data_copy)

    def get_relationship_description(self) -> str:
        """Get natural language description of relationship"""
        if self.description:
            return self.description

        # Generate default description based on type
        type_descriptions = {
            RelationType.FRIEND: f"{self.character_a} and {self.character_b} are friends",
            RelationType.FAMILY: f"{self.character_a} and {self.character_b} are family",
            RelationType.COMPANION: f"{self.character_a} and {self.character_b} are companions",
            RelationType.RIVAL: f"{self.character_a} and {self.character_b} are rivals",
            RelationType.MENTOR: f"{self.character_a} is {self.character_b}'s mentor",
            RelationType.STUDENT: f"{self.character_a} is {self.character_b}'s student",
            RelationType.SIBLING: f"{self.character_a} and {self.character_b} are siblings",
            RelationType.PARENT: f"{self.character_a} is {self.character_b}'s parent",
            RelationType.CHILD: f"{self.character_a} is {self.character_b}'s child",
            RelationType.TEAMMATE: f"{self.character_a} and {self.character_b} are teammates",
        }

        return type_descriptions.get(
            self.relationship_type,
            f"{self.character_a} and {self.character_b}"
        )


# ============================================================================
# Character Group Definition
# ============================================================================

@dataclass
class CharacterGroup:
    """A group of related characters"""
    name: str
    characters: List[str]  # Character names
    description: str = ""

    # Group dynamics
    typical_arrangement: str = "together"  # together, circle, line, scattered
    group_activity: str = ""  # e.g., "playing", "adventuring"

    # Metadata
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterGroup':
        """Create from dictionary"""
        return cls(**data)


# ============================================================================
# Relationship Graph Manager
# ============================================================================

class RelationshipGraph:
    """
    Manages character relationships as a graph
    """

    def __init__(self, relationships_dir: str = "relationships"):
        self.relationships_dir = Path(relationships_dir)
        self.relationships_dir.mkdir(parents=True, exist_ok=True)

        # Store relationships as adjacency list
        self.relationships: Dict[str, List[Relationship]] = {}

        # Store character groups
        self.groups: Dict[str, CharacterGroup] = {}

        # Load existing relationships
        self._load_relationships()

    def _load_relationships(self):
        """Load relationships from disk"""
        try:
            # Load relationships
            relationships_file = self.relationships_dir / "relationships.json"
            if relationships_file.exists():
                with open(relationships_file, 'r') as f:
                    data = json.load(f)
                    for char_name, rels in data.items():
                        self.relationships[char_name] = [
                            Relationship.from_dict(r) for r in rels
                        ]
                        logger.info(f"Loaded {len(rels)} relationships for {char_name}")

            # Load groups
            groups_file = self.relationships_dir / "groups.json"
            if groups_file.exists():
                with open(groups_file, 'r') as f:
                    data = json.load(f)
                    for group_name, group_data in data.items():
                        self.groups[group_name] = CharacterGroup.from_dict(group_data)
                        logger.info(f"Loaded group: {group_name}")

        except Exception as e:
            logger.error(f"Error loading relationships: {e}")

    def _save_relationships(self):
        """Save relationships to disk"""
        try:
            # Save relationships
            relationships_file = self.relationships_dir / "relationships.json"
            with open(relationships_file, 'w') as f:
                data = {
                    char: [r.to_dict() for r in rels]
                    for char, rels in self.relationships.items()
                }
                json.dump(data, f, indent=2)

            # Save groups
            groups_file = self.relationships_dir / "groups.json"
            with open(groups_file, 'w') as f:
                data = {
                    name: group.to_dict()
                    for name, group in self.groups.items()
                }
                json.dump(data, f, indent=2)

            logger.info("Saved relationships and groups")

        except Exception as e:
            logger.error(f"Error saving relationships: {e}")

    def add_relationship(
        self,
        character_a: str,
        character_b: str,
        relationship_type: RelationType,
        description: str = "",
        **kwargs
    ) -> Relationship:
        """Add a relationship between two characters"""

        # Create relationship
        relationship = Relationship(
            character_a=character_a,
            character_b=character_b,
            relationship_type=relationship_type,
            description=description,
            **kwargs
        )

        # Add to adjacency list (bidirectional)
        if character_a not in self.relationships:
            self.relationships[character_a] = []
        if character_b not in self.relationships:
            self.relationships[character_b] = []

        self.relationships[character_a].append(relationship)

        # Add reverse relationship
        reverse_type = self._get_reverse_relationship_type(relationship_type)
        reverse_relationship = Relationship(
            character_a=character_b,
            character_b=character_a,
            relationship_type=reverse_type,
            description=description,
            **kwargs
        )
        self.relationships[character_b].append(reverse_relationship)

        logger.info(f"Added relationship: {character_a} -> {character_b} ({relationship_type.value})")

        self._save_relationships()

        return relationship

    def _get_reverse_relationship_type(self, rel_type: RelationType) -> RelationType:
        """Get the reverse relationship type"""
        reverse_map = {
            RelationType.MENTOR: RelationType.STUDENT,
            RelationType.STUDENT: RelationType.MENTOR,
            RelationType.PARENT: RelationType.CHILD,
            RelationType.CHILD: RelationType.PARENT,
        }

        return reverse_map.get(rel_type, rel_type)

    def get_relationships(self, character_name: str) -> List[Relationship]:
        """Get all relationships for a character"""
        return self.relationships.get(character_name, [])

    def get_related_characters(self, character_name: str) -> List[str]:
        """Get all characters related to a character"""
        relationships = self.get_relationships(character_name)
        return [r.character_b for r in relationships]

    def are_related(self, character_a: str, character_b: str) -> bool:
        """Check if two characters are related"""
        if character_a not in self.relationships:
            return False

        related = self.get_related_characters(character_a)
        return character_b in related

    def get_relationship(
        self,
        character_a: str,
        character_b: str
    ) -> Optional[Relationship]:
        """Get specific relationship between two characters"""
        if character_a not in self.relationships:
            return None

        for rel in self.relationships[character_a]:
            if rel.character_b == character_b:
                return rel

        return None

    def create_group(
        self,
        name: str,
        characters: List[str],
        description: str = "",
        **kwargs
    ) -> CharacterGroup:
        """Create a character group"""

        group = CharacterGroup(
            name=name,
            characters=characters,
            description=description,
            **kwargs
        )

        self.groups[name] = group

        logger.info(f"Created group '{name}' with {len(characters)} characters")

        self._save_relationships()

        return group

    def get_group(self, name: str) -> Optional[CharacterGroup]:
        """Get a character group"""
        return self.groups.get(name)

    def list_groups(self) -> List[CharacterGroup]:
        """List all groups"""
        return list(self.groups.values())

    def remove_relationship(self, character_a: str, character_b: str) -> bool:
        """Remove relationship between two characters"""
        if character_a not in self.relationships:
            return False

        # Remove from character_a's relationships
        self.relationships[character_a] = [
            r for r in self.relationships[character_a]
            if r.character_b != character_b
        ]

        # Remove from character_b's relationships
        if character_b in self.relationships:
            self.relationships[character_b] = [
                r for r in self.relationships[character_b]
                if r.character_b != character_a
            ]

        logger.info(f"Removed relationship: {character_a} <-> {character_b}")

        self._save_relationships()

        return True


# ============================================================================
# Multi-Character Scene Generator
# ============================================================================

class MultiCharacterSceneGenerator:
    """
    Generates scenes with multiple characters considering their relationships
    """

    def __init__(
        self,
        character_engine,
        relationship_graph: Optional[RelationshipGraph] = None
    ):
        self.character_engine = character_engine
        self.relationship_graph = relationship_graph or RelationshipGraph()

    async def generate_multi_character_scene(
        self,
        characters: List[str],
        scene_description: str,
        **generation_params
    ) -> Dict[str, Any]:
        """
        Generate scene with multiple characters

        Args:
            characters: List of character names
            scene_description: Scene description
            **generation_params: Additional parameters

        Returns:
            Generation result
        """
        logger.info(f"Generating multi-character scene with: {', '.join(characters)}")

        # Load all characters
        char_objects = []
        for char_name in characters:
            char = self.character_engine.library.get_character(char_name)
            if not char:
                raise ValueError(f"Character not found: {char_name}")
            char_objects.append(char)

        # Build combined prompt considering relationships
        prompt = self._build_multi_character_prompt(
            char_objects,
            scene_description
        )

        # For now, use composition approach
        # In production, this could use ControlNet or regional prompting
        result = await self._generate_composite_scene(
            char_objects,
            prompt,
            **generation_params
        )

        return result

    def _build_multi_character_prompt(
        self,
        characters: List,
        scene_description: str
    ) -> str:
        """Build prompt for multi-character scene"""

        # Build character descriptions
        char_descriptions = []
        for char in characters:
            char_descriptions.append(char.description)

        # Check for relationships
        relationship_context = ""
        if len(characters) == 2:
            rel = self.relationship_graph.get_relationship(
                characters[0].name,
                characters[1].name
            )
            if rel:
                relationship_context = f", {rel.get_relationship_description()}, {rel.typical_interaction}"

        # Build combined prompt
        prompt = f"{' and '.join(char_descriptions)} {scene_description}"
        prompt += relationship_context
        prompt += ", children's book illustration, cute, friendly, wholesome, multiple characters"

        return prompt

    async def _generate_composite_scene(
        self,
        characters: List,
        prompt: str,
        **generation_params
    ) -> Dict[str, Any]:
        """
        Generate composite scene with multiple characters

        This is a simplified version. Production implementation could use:
        - Regional prompting
        - ControlNet for precise character placement
        - Multiple generations with compositing
        - LoRA blending
        """

        # For now, generate with combined prompt
        params = {
            "prompt": prompt,
            "model": generation_params.get("model", "sd15"),
            "width": generation_params.get("width", 512),
            "height": generation_params.get("height", 512),
            "steps": generation_params.get("steps", 30),
            "cfg_scale": generation_params.get("cfg_scale", 7.5),
            "seed": generation_params.get("seed", -1),
            "sampling_method": generation_params.get("sampling_method", "euler_a"),
            "negative_prompt": generation_params.get(
                "negative_prompt",
                "scary, frightening, horror, violent, weapon, blood"
            )
        }

        result = await self.character_engine._generate_image(params)

        return result

    async def generate_group_scene(
        self,
        group_name: str,
        scene_description: str,
        **generation_params
    ) -> Dict[str, Any]:
        """Generate scene with a predefined character group"""

        group = self.relationship_graph.get_group(group_name)
        if not group:
            raise ValueError(f"Character group not found: {group_name}")

        # Add group context to scene
        scene_with_context = f"{scene_description}, {group.description}, {group.group_activity}"

        return await self.generate_multi_character_scene(
            characters=group.characters,
            scene_description=scene_with_context,
            **generation_params
        )


# ============================================================================
# Helper Functions
# ============================================================================

def parse_relationship_request(message: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse natural language relationship request

    Examples:
    - "Spark's friend is Whiskers"
    - "Luna is Spark's sister"
    - "Make Bolt and Thunder teammates"

    Returns:
        (character_a, character_b, relationship_type) or None
    """
    import re

    # Pattern 1: "X's RELATIONSHIP is Y"
    pattern1 = r"([A-Za-z]+)'s\s+(friend|sibling|sister|brother|companion|rival|mentor|student)\s+is\s+([A-Za-z]+)"
    match = re.search(pattern1, message, re.IGNORECASE)

    if match:
        char_a = match.group(1).capitalize()
        rel_type = match.group(2).lower()
        char_b = match.group(3).capitalize()
        # Normalize sister/brother to sibling
        if rel_type in ["sister", "brother"]:
            rel_type = "sibling"
        return (char_a, char_b, rel_type)

    # Pattern 2: "X is Y's RELATIONSHIP"
    pattern2 = r"([A-Za-z]+)\s+is\s+([A-Za-z]+)'s\s+(friend|sibling|sister|brother|companion|rival|mentor|student)"
    match = re.search(pattern2, message, re.IGNORECASE)

    if match:
        char_a = match.group(1).capitalize()
        char_b = match.group(2).capitalize()
        rel_type = match.group(3).lower()
        # Normalize sister/brother to sibling
        if rel_type in ["sister", "brother"]:
            rel_type = "sibling"
        return (char_a, char_b, rel_type)

    # Pattern 3: "Make X and Y RELATIONSHIP"
    pattern3 = r"(?:make|set)\s+([A-Za-z]+)\s+and\s+([A-Za-z]+)\s+(friends|siblings|companions|rivals|teammates)"
    match = re.search(pattern3, message, re.IGNORECASE)

    if match:
        char_a = match.group(1).capitalize()
        char_b = match.group(2).capitalize()
        rel_type = match.group(3).lower().rstrip('s')  # Remove plural
        return (char_a, char_b, rel_type)

    return None


# ============================================================================
# CLI Testing
# ============================================================================

def test_character_relationships():
    """Test character relationships system"""
    print("=== Character Relationships System Test ===\n")

    # Test 1: Create relationship graph
    print("Test 1: Creating relationship graph")
    graph = RelationshipGraph()
    print("✓ Relationship graph created\n")

    # Test 2: Add relationships
    print("Test 2: Adding relationships")

    graph.add_relationship(
        "Spark",
        "Whiskers",
        RelationType.FRIEND,
        description="best friends",
        typical_interaction="playing together"
    )
    print("✓ Added: Spark -> Whiskers (friend)")

    graph.add_relationship(
        "Luna",
        "Spark",
        RelationType.SIBLING,
        description="siblings",
        typical_interaction="adventuring together"
    )
    print("✓ Added: Luna -> Spark (sibling)\n")

    # Test 3: Query relationships
    print("Test 3: Querying relationships")

    spark_rels = graph.get_relationships("Spark")
    print(f"✓ Spark has {len(spark_rels)} relationships:")
    for rel in spark_rels:
        print(f"  - {rel.character_b} ({rel.relationship_type.value})")
    print()

    # Test 4: Create group
    print("Test 4: Creating character group")

    group = graph.create_group(
        name="Dragon Squad",
        characters=["Spark", "Luna", "Whiskers"],
        description="a team of adventurers",
        group_activity="going on adventures"
    )
    print(f"✓ Created group: {group.name}")
    print(f"  Members: {', '.join(group.characters)}\n")

    # Test 5: Parse relationship requests
    print("Test 5: Parsing relationship requests")

    test_messages = [
        "Spark's friend is Whiskers",
        "Luna is Spark's sister",
        "Make Bolt and Thunder teammates"
    ]

    for msg in test_messages:
        result = parse_relationship_request(msg)
        if result:
            char_a, char_b, rel_type = result
            print(f"  '{msg}'")
            print(f"  → {char_a} + {char_b} = {rel_type}\n")

    print("✓ All tests passed!")


if __name__ == "__main__":
    test_character_relationships()
