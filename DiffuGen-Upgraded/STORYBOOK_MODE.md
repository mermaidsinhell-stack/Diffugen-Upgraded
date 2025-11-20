# Conversational Image Generation for Storybooks

## Overview

**Storybook Mode** provides an intelligent, conversation-based interface for creating children's book illustrations. Instead of manually tweaking parameters, you simply describe what you want in natural language, and the system intelligently adjusts everything for you.

Perfect for: **Children's storybooks, character consistency, iterative refinement**

---

## How It Works

```
You: "Create a friendly dragon in a castle"
AI: [Analyzes request with Qwen LLM]
    - Detects: "children's book" intent
    - Applies: child-safe filters
    - Sets: optimal parameters for illustrations
    - Generates: image with friendly, cute style

You: "Make it brighter and more colorful"
AI: [Understands refinement request]
    - Increases: cfg_scale by 2.0
    - Adds to prompt: "vibrant colors, saturated"
    - Regenerates: adjusted image

You: "Less detailed, more cartoon-like"
AI: [Adjusts for simplicity]
    - Decreases: steps by 10
    - Adds to prompt: "simple shapes, cartoon style"
    - Regenerates: simplified version
```

The system:
- 🧠 **Analyzes natural language** using Qwen LLM
- 🎯 **Maps requests to parameters** intelligently
- 💬 **Remembers conversation** for context
- 👶 **Ensures child-appropriate** content
- 📈 **Learns from feedback** iteratively

---

## Quick Start

### 1. Create a Session

```bash
curl -X POST http://localhost:5199/conversational/session
```

**Response:**
```json
{
  "session_id": "abc123-def456-ghi789",
  "created_at": 1704326400.0,
  "message": "New conversation session created. Start by describing what you want to generate!"
}
```

### 2. Generate Your First Image

```bash
curl -X POST http://localhost:5199/conversational/generate \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123-def456-ghi789",
    "message": "Create a friendly dragon in a colorful castle"
  }'
```

**Response:**
```json
{
  "success": true,
  "image_url": "http://localhost:5199/images/dragon_castle_001.png",
  "explanation": "Generated new image: cute, friendly dragon in a colorful castle, children's book illustration",
  "parameters": {
    "model": "sd15",
    "steps": 25,
    "cfg_scale": 7.5,
    "width": 512,
    "height": 512
  },
  "safety_warnings": [],
  "intent": "User wants character for children's book"
}
```

### 3. Refine Iteratively

```bash
curl -X POST http://localhost:5199/conversational/generate \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123-def456-ghi789",
    "message": "Make it brighter and more colorful"
  }'
```

**Response:**
```json
{
  "success": true,
  "image_url": "http://localhost:5199/images/dragon_castle_002.png",
  "explanation": "Applied adjustments, steps: 25, cfg_scale: 9.5",
  "adjustments_applied": {
    "cfg_scale_delta": 2.0,
    "prompt_add": ["bright", "well-lit", "vibrant colors"]
  }
}
```

---

## Natural Language Commands

### Brightness/Lighting

| Command | Effect | Parameters Changed |
|---------|--------|-------------------|
| "Make it brighter" | Increases brightness | cfg_scale +2.0, adds "bright, well-lit" |
| "Make it darker" | Decreases brightness | cfg_scale -1.5, adds "dim lighting" |
| "More light" | Adds more light | cfg_scale +1.5, adds "sunny, bright lighting" |
| "Softer lighting" | Gentler lighting | cfg_scale -1.0, adds "soft light" |

### Detail Level

| Command | Effect | Parameters Changed |
|---------|--------|-------------------|
| "More detailed" | Increases detail | steps +10, adds "highly detailed, intricate" |
| "Less detailed" | Simplifies | steps -10, adds "simple, minimalist" |
| "Simpler" | Much simpler | steps -10, cfg_scale -1.0, adds "simple shapes" |
| "More complex" | Adds complexity | steps +10, cfg_scale +1.0, adds "intricate" |

### Color Adjustments

| Command | Effect | Parameters Changed |
|---------|--------|-------------------|
| "More colorful" | Saturates colors | cfg_scale +1.0, adds "vibrant, saturated" |
| "Less colorful" | Mutes colors | cfg_scale -0.5, adds "soft colors, muted" |
| "Vibrant colors" | Very saturated | cfg_scale +1.5, adds "bold colors" |
| "Pastel colors" | Soft pastels | adds "pastel", removes "saturated" |
| "Warm colors" | Red/orange/yellow | adds "warm tones" |
| "Cool colors" | Blue/green/purple | adds "cool tones" |

### Style Changes

| Command | Effect | Parameters Changed |
|---------|--------|-------------------|
| "More cartoonish" | Cartoon style | adds "cartoon, animated, stylized" |
| "More realistic" | Realistic (but safe) | steps +10, adds "realistic, detailed" |
| "Watercolor style" | Watercolor painting | adds "watercolor painting, soft edges" |
| "Sketch style" | Hand-drawn sketch | adds "pencil sketch, hand-drawn" |
| "Digital art" | Digital illustration | adds "digital illustration" |

### Quality/Sharpness

| Command | Effect | Parameters Changed |
|---------|--------|-------------------|
| "Sharper" | Increases sharpness | cfg_scale +1.0, adds "sharp, crisp" |
| "Softer" | Softens image | cfg_scale -1.0, adds "soft, gentle" |
| "More contrast" | Increases contrast | cfg_scale +1.5, adds "high contrast" |
| "Less contrast" | Reduces contrast | cfg_scale -1.0, adds "low contrast" |

---

## Child Safety Features

### Automatic Filtering

The system automatically ensures all content is appropriate for children:

```python
# Forbidden keywords (automatically removed)
scary, frightening, horror, violent, weapon, blood,
realistic gore, inappropriate, adult, nsfw

# Automatic replacements
"dragon" → "friendly dragon"
"monster" → "friendly creature"
"scary" → "exciting"
"fight" → "play"
```

### Safety Warnings

If you accidentally use inappropriate keywords, you'll get warnings:

```json
{
  "safety_warnings": [
    "Removed inappropriate keyword: 'scary'",
    "Replaced 'monster' with 'friendly creature'"
  ]
}
```

### Base Negative Prompt

All generations automatically include:
```
scary, frightening, dark, horror, violent, weapon, blood, gore,
realistic, photorealistic, uncanny, creepy, disturbing,
inappropriate, adult themes, nsfw, mature content
```

---

## API Reference

### POST /conversational/session

Create a new conversation session.

**Response:**
```json
{
  "session_id": "string",
  "created_at": 1704326400.0,
  "message": "string"
}
```

---

### POST /conversational/generate

Generate or refine image using natural language.

**Request:**
```json
{
  "session_id": "string",
  "message": "string"
}
```

**Response:**
```json
{
  "success": boolean,
  "image_path": "string",
  "image_url": "string",
  "explanation": "string",
  "parameters": {
    "model": "string",
    "steps": number,
    "cfg_scale": number,
    "width": number,
    "height": number,
    "prompt": "string",
    "negative_prompt": "string"
  },
  "adjustments_applied": {
    "steps_delta": number,
    "cfg_scale_delta": number,
    "prompt_add": ["string"],
    "negative_add": ["string"]
  },
  "safety_warnings": ["string"],
  "intent": "string",
  "error": "string"
}
```

---

### GET /conversational/session/{session_id}

Get session history and context.

**Response:**
```json
{
  "session_id": "string",
  "created_at": 1704326400.0,
  "message_count": number,
  "generation_count": number,
  "current_parameters": {...},
  "recent_messages": [
    {
      "role": "user",
      "content": "string",
      "timestamp": 1704326400.0
    }
  ],
  "style_preferences": {},
  "character_references": {}
}
```

---

### GET /conversational/sessions

List all active sessions.

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "string",
      "created_at": 1704326400.0,
      "message_count": number,
      "generation_count": number,
      "last_activity": 1704326400.0
    }
  ],
  "total": number
}
```

---

### DELETE /conversational/session/{session_id}

Delete a conversation session.

**Response:**
```json
{
  "success": boolean,
  "message": "string"
}
```

---

## Example Workflow: Creating a Dragon Character

```bash
# Step 1: Create session
SESSION=$(curl -s -X POST http://localhost:5199/conversational/session | jq -r '.session_id')

# Step 2: Initial generation
curl -X POST http://localhost:5199/conversational/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"Create a friendly green dragon with big eyes\"
  }"

# Step 3: Adjust colors
curl -X POST http://localhost:5199/conversational/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"Make it more colorful, with vibrant scales\"
  }"

# Step 4: Simplify for children's book
curl -X POST http://localhost:5199/conversational/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"Less detailed, more cartoon-like\"
  }"

# Step 5: Add watercolor style
curl -X POST http://localhost:5199/conversational/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"Make it look like a watercolor painting\"
  }"

# Step 6: Check session history
curl -s "http://localhost:5199/conversational/session/$SESSION" | jq
```

---

## Python Client Example

```python
import httpx
import asyncio

class StorybookClient:
    def __init__(self, base_url="http://localhost:5199"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
        self.session_id = None

    async def start_session(self):
        """Create new conversation session"""
        response = await self.client.post(f"{self.base_url}/conversational/session")
        data = response.json()
        self.session_id = data["session_id"]
        print(f"Session started: {self.session_id}")
        return self.session_id

    async def generate(self, message):
        """Generate or refine image"""
        response = await self.client.post(
            f"{self.base_url}/conversational/generate",
            json={
                "session_id": self.session_id,
                "message": message
            }
        )
        return response.json()

    async def refine_until_happy(self):
        """Interactive refinement loop"""
        print("Creating your storybook illustration...")
        print("Describe what you want, or type 'done' when satisfied\n")

        while True:
            user_input = input("You: ")

            if user_input.lower() == "done":
                print("Great! Your illustration is ready.")
                break

            result = await self.generate(user_input)

            if result["success"]:
                print(f"AI: {result['explanation']}")
                print(f"Image: {result['image_url']}")

                if result.get("safety_warnings"):
                    print(f"⚠️  Safety: {', '.join(result['safety_warnings'])}")

                if result.get("adjustments_applied"):
                    adj = result["adjustments_applied"]
                    if adj.get("cfg_scale_delta"):
                        print(f"   Adjusted brightness: {adj['cfg_scale_delta']:+.1f}")
                    if adj.get("steps_delta"):
                        print(f"   Adjusted detail: {adj['steps_delta']:+d} steps")
            else:
                print(f"Error: {result.get('error')}")

            print()


# Usage
async def main():
    client = StorybookClient()
    await client.start_session()
    await client.refine_until_happy()

asyncio.run(main())
```

**Example Session:**
```
Session started: abc123-def456

You: Create a friendly dragon
AI: Generated new image: cute, friendly dragon, children's book illustration
Image: http://localhost:5199/images/dragon_001.png

You: Make it brighter and more colorful
AI: Applied adjustments, cfg_scale: 9.5
   Adjusted brightness: +2.0
Image: http://localhost:5199/images/dragon_002.png

You: Perfect! But make the wings bigger
AI: Applied adjustments
Image: http://localhost:5199/images/dragon_003.png

You: done
Great! Your illustration is ready.
```

---

## Technical Details

### Intent Analysis

The system uses Qwen LLM to analyze your natural language:

```python
System Prompt:
"You are an AI assistant helping create children's storybook illustrations.
 Analyze user requests and determine:
 1. Intent type (new generation, refinement, parameter adjustment)
 2. What technical parameters should change
 3. How to modify the prompt"

User: "Make it brighter"

LLM Response:
{
  "intent_type": "parameter_adjustment",
  "confidence": 0.95,
  "adjustments": {
    "cfg_scale_delta": 2.0
  },
  "prompt_changes": {
    "add": ["bright", "well-lit"]
  }
}
```

### Parameter Mapping

Natural language → technical parameters:

```python
"brighter" → {
    "cfg_scale_delta": +2.0,
    "prompt_add": ["bright", "well-lit", "luminous"]
}

"simpler" → {
    "steps_delta": -10,
    "cfg_scale_delta": -1.0,
    "prompt_add": ["simple shapes", "clean design"]
}

"more colorful" → {
    "cfg_scale_delta": +1.0,
    "prompt_add": ["vibrant colors", "saturated"],
    "negative_add": ["muted", "desaturated"]
}
```

### Conversation Context

Each session maintains:
- Message history (last 3-5 messages)
- Generation history (all images created)
- Current parameters (for iterative refinement)
- Style preferences (learned from feedback)
- Character references (for consistency)

---

## Tips for Best Results

### 1. Start Simple
```
Good: "Create a friendly dragon"
Not: "Create a highly detailed photorealistic dragon with scales reflecting light..."
```

The system will add child-appropriate details automatically.

### 2. Iterate Gradually
```
Step 1: "Create a dragon"
Step 2: "Make it brighter"
Step 3: "More colorful"
Step 4: "Bigger wings"
```

Small adjustments work better than trying to fix everything at once.

### 3. Use Natural Language
```
Good: "Make it more cartoon-like"
Good: "Brighter colors please"
Good: "Less detailed"

Also works: "Can you make this brighter?"
Also works: "I want more vibrant colors"
```

The system understands conversational tone.

### 4. Character Consistency
```
Session 1: "Create a friendly dragon named Sparky"
[Save the image]

Session 2: "Show Sparky in a forest"
[Use same seed or reference image - coming soon!]
```

For now, keep the same session and similar prompt structure.

---

## Troubleshooting

### "Could not understand request"

**Problem**: LLM couldn't parse your request

**Solution**: Try rephrasing more explicitly:
- Instead of: "Fix it"
- Try: "Make it brighter and more colorful"

### "Session not found"

**Problem**: Session expired or invalid ID

**Solution**: Create a new session:
```bash
curl -X POST http://localhost:5199/conversational/session
```

### "No previous image to refine"

**Problem**: Trying to refine before generating

**Solution**: Generate first image:
```bash
curl -X POST http://localhost:5199/conversational/generate \
  -d '{"session_id": "...", "message": "Create a dragon"}'
```

### Images too realistic/scary

**Problem**: Child-safety filter needs tuning

**Solution**: This shouldn't happen, but if it does:
- Use phrases like "friendly", "cute", "for kids"
- Report the issue for filter improvement

---

## Limitations

### Current Limitations:

1. **Character Consistency**: Not perfect across sessions yet
   - *Coming soon*: Reference image support, LoRA training

2. **LLM Dependency**: Requires Ollama with Qwen model
   - Falls back to keyword matching if LLM unavailable

3. **Single Model**: Currently uses SD1.5 only
   - *Coming soon*: Model selection per style

4. **Session Persistence**: Sessions stored in memory
   - *Coming soon*: Database persistence

---

## Roadmap

### Phase 2 (Next):
- ✅ Character sheet generation
- ✅ Reference image support for consistency
- ✅ LoRA training integration
- ✅ Style locking across scenes

### Phase 3 (Future):
- ✅ Batch scene generation
- ✅ Multi-character support
- ✅ Story timeline management
- ✅ Export as PDF/ebook

---

## FAQ

**Q: Do I need to know technical parameters?**
A: No! Just describe what you want in natural language.

**Q: How does it know I want children's book style?**
A: It automatically detects this and applies appropriate filters and styles.

**Q: Can I use specific technical parameters?**
A: Yes, but not recommended. The system is designed to handle that for you.

**Q: How do I maintain character consistency?**
A: Stay in the same session and use similar descriptions. Reference image support coming soon!

**Q: Is it safe for my kids to use?**
A: Yes! All content is automatically filtered for child-appropriateness.

**Q: Can I save my sessions?**
A: Sessions are currently in-memory. Export your images and note the session_id for reference.

---

## Support

- **Documentation**: See `/docs` endpoint on the API server
- **Issues**: Report on GitHub
- **Examples**: Check `STORYBOOK_MODE.md` (this file)

---

## Summary

**Storybook Mode gives you:**

✅ Natural language interface
✅ Intelligent parameter tuning
✅ Conversation memory
✅ Child-safe content
✅ Iterative refinement
✅ Professional quality illustrations

**Perfect for:**

📚 Children's storybooks
🎨 Character illustrations
🖼️ Scene generation
👶 Age-appropriate content

**Get started:**

```bash
# Create session
curl -X POST http://localhost:5199/conversational/session

# Start creating!
curl -X POST http://localhost:5199/conversational/generate \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID", "message": "Create a friendly dragon"}'
```

**Happy storytelling! 🐉📖**
