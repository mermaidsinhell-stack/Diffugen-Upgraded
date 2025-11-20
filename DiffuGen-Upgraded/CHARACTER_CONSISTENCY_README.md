# Character Consistency System

## Overview

The Character Consistency System enables you to create characters once and reuse them across multiple scenes in your storybook with visual consistency.

**Status**: ✅ Core system complete, API endpoints coming next

---

## How It Works

```
Step 1: Create Character
You: "Create a friendly dragon named Spark"
→ System generates reference image
→ Stores character with seed for consistency

Step 2: Reuse in Scenes
You: "Show Spark in a castle"
→ Uses img2img with reference image
→ Maintains character appearance
→ Changes only the scene/background

Step 3: Character Sheet (Optional)
You: "Generate character sheet for Spark"
→ Creates multiple poses (front, side, back, 3/4 view)
→ All use same seed for consistency
```

---

## Quick Start

### 1. Create Your Character

```python
# Via Conversational API
POST /conversational/generate
{
  "session_id": "your-session",
  "message": "Create a friendly green dragon named Spark with big eyes"
}

# Response
{
  "success": true,
  "character_name": "Spark",
  "character_seed": 12345,
  "image_url": "http://localhost:5199/images/spark_ref.png",
  "explanation": "Created character 'Spark': friendly green dragon..."
}
```

### 2. Use Character in Scenes

```python
# Scene 1: Castle
POST /conversational/generate
{
  "session_id": "your-session",
  "message": "Show Spark in a colorful castle"
}

# Scene 2: Forest
POST /conversational/generate
{
  "session_id": "your-session",
  "message": "Show Spark flying through a forest"
}

# Scene 3: Beach
POST /conversational/generate
{
  "session_id": "your-session",
  "message": "Show Spark playing on a beach"
}
```

All scenes will maintain Spark's appearance!

---

## Technical Implementation

### Components

**character_consistency.py** (523 lines):
- `Character` - Character dataclass with reference images
- `CharacterLibrary` - Storage and retrieval system
- `CharacterConsistencyEngine` - Generation with consistency
- `SeedConsistency` - Seed-based helper methods

**Integration** (intelligent_workflow.py):
- Character intent detection (CREATE, USE, SHEET)
- Automatic character tracking per session
- Context-aware character selection

---

## How Consistency Works

###Method 1: Reference Image (img2img)

```python
# Generate base character
character = create_character("Spark", "friendly green dragon")

# Use in scene with img2img
generate_with_character(
    character=character,
    scene="in a castle",
    consistency_strength=0.75  # 0.0-1.0
)
```

- **High strength (0.8-1.0)**: Very consistent, character dominates
- **Medium strength (0.5-0.7)**: Balanced character + scene
- **Low strength (0.0-0.4)**: Scene dominates, less consistent

### Method 2: Seed Locking

```python
# All generations use same seed
character.seed = 12345

# Consistent results across scenes
scene1 = generate(prompt="Spark in castle", seed=12345)
scene2 = generate(prompt="Spark in forest", seed=12345)
```

### Method 3: Character Sheets

```python
# Generate multiple poses with same seed
poses = [
    "front view",
    "side view",
    "back view",
    "three-quarter view"
]

for pose in poses:
    generate(prompt=f"{character.description}, {pose}", seed=character.seed)
```

---

## Features

### Character Library

```python
# Save character
library.save_character(character)

# Load character
spark = library.get_character("Spark")

# List all characters
all_chars = library.list_characters()

# Search characters
dragons = library.search_characters("dragon")

# Delete character
library.delete_character("Spark")
```

### Character Sheet Generation

```python
# Generate character sheet with multiple poses
sheet = await engine.generate_character_sheet(
    character=spark,
    poses=["front view", "side view", "back view"]
)

# Returns
{
    "front view": "/path/to/front.png",
    "side view": "/path/to/side.png",
    "back view": "/path/to/back.png"
}
```

### Conversational Integration

The system automatically detects character-related requests:

```
You: "Create a dragon named Spark"
→ Intent: CHARACTER_CREATE
→ Creates character and stores in library

You: "Show Spark in a forest"
→ Intent: CHARACTER_USE
→ Uses stored character in new scene

You: "Generate a character sheet for Spark"
→ Intent: CHARACTER_SHEET
→ Generates multiple poses
```

---

## Storage

Characters are stored in:
- **Directory**: `characters/`
- **Format**: JSON metadata + reference images
- **Structure**:
  ```
  characters/
    ├── Spark.json           # Metadata
    ├── spark_ref.png        # Reference image
    ├── spark_front.png      # Pose: front
    ├── spark_side.png       # Pose: side
    └── spark_back.png       # Pose: back
  ```

---

## Character JSON Format

```json
{
  "name": "Spark",
  "description": "friendly green dragon with big eyes and small wings",
  "reference_image": "/path/to/spark_ref.png",
  "seed": 12345,
  "style_notes": "children's book illustration, watercolor style",
  "created_at": 1704326400.0,
  "tags": ["dragon", "main_character", "storybook"],
  "reference_images": {
    "front view": "/path/to/spark_front.png",
    "side view": "/path/to/spark_side.png"
  },
  "parameters": {
    "model": "sd15",
    "steps": 30,
    "cfg_scale": 8.0,
    "width": 512,
    "height": 512
  }
}
```

---

## Examples

### Example 1: Creating a Dragon Character

```python
# Create character
POST /conversational/generate
{
  "message": "Create a friendly green dragon named Spark with big eyes"
}

# Result
{
  "character_name": "Spark",
  "character_seed": 54321,
  "image_url": "/images/spark.png"
}

# Use in scenes
POST /conversational/generate
{
  "message": "Show Spark sitting in a castle"
}

POST /conversational/generate
{
  "message": "Show Spark flying in the sky"
}
```

### Example 2: Character Sheet

```python
# Create character first
POST /conversational/generate
{
  "message": "Create a cute cat named Whiskers"
}

# Generate character sheet
POST /conversational/generate
{
  "message": "Generate a character sheet for Whiskers"
}

# Result
{
  "character_sheet": {
    "front view": "/images/whiskers_front.png",
    "side view": "/images/whiskers_side.png",
    "back view": "/images/whiskers_back.png",
    "three-quarter view": "/images/whiskers_3q.png"
  }
}
```

---

## Limitations & Future Improvements

### Current Limitations:

1. **Img2img dependency**: Relies on reference image which can drift slightly
2. **Manual consistency tuning**: May need to adjust `consistency_strength`
3. **Single reference**: Best results from one primary reference image

### Coming Soon:

1. **LoRA Training** ⭐⭐⭐⭐⭐
   - Train custom LoRA for perfect character consistency
   - One-time training, infinite consistent generations

2. **IP-Adapter Support** ⭐⭐⭐⭐
   - Better reference-based consistency
   - More control over character vs scene balance

3. **Multi-Reference Blending** ⭐⭐⭐
   - Use multiple reference images
   - Better pose and angle variation

4. **Automatic Pose Detection** ⭐⭐⭐
   - Analyze scene and suggest best character pose
   - Auto-select from character sheet

---

## Tips for Best Results

### 1. Create Detailed Characters

```
Good: "friendly green dragon with big round eyes, small purple wings, orange belly scales"
Not: "a dragon"
```

More detail = better consistency

### 2. Use High Consistency Strength

```python
consistency_strength=0.75  # Good for storybooks
consistency_strength=0.85  # Even more consistent
```

### 3. Generate Character Sheet First

Having multiple poses helps:
- Front view: dialogue scenes
- Side view: walking/movement
- Back view: leaving scenes
- 3/4 view: dynamic angles

### 4. Keep Style Notes

```python
character.style_notes = "watercolor style, soft colors, children's book"
```

Maintains consistent art style across all scenes.

---

## Troubleshooting

**Q: Character looks different in each scene**
A: Increase `consistency_strength` to 0.8-0.9

**Q: Scene is too similar to reference image**
A: Decrease `consistency_strength` to 0.5-0.6

**Q: Character not found**
A: Check if character was created in this session, or specify character name explicitly

**Q: Can I use a character from a previous session?**
A: Yes! Characters are stored in the library and persist across sessions

---

## Summary

**Character Consistency System provides:**
✅ Create characters once, reuse everywhere
✅ Reference image-based consistency (img2img)
✅ Seed-based generation for stability
✅ Character sheet generation (multiple poses)
✅ Character library for persistence
✅ Conversational interface integration
✅ Perfect for storybook illustration

**Next steps:**
- LoRA training for perfect consistency
- IP-Adapter integration
- API endpoints (coming next)
- Complete documentation

**Ready to use!** 🎨📚
