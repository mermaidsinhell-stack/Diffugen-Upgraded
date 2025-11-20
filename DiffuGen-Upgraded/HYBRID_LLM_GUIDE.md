# Hybrid LLM System - Gemini + Qwen-VL

## Overview

The hybrid LLM system intelligently routes tasks between Gemini API (cloud) and Qwen-VL (local) to maximize your free tier usage while maintaining high-quality natural language understanding.

**Key Benefits:**
- ✅ **6x more capacity**: 300 storybooks/day vs 50 with Gemini-only
- ✅ **Smart routing**: Complex tasks → Gemini, Simple tasks → Qwen-VL
- ✅ **Unlimited vision analysis**: All image checks run locally
- ✅ **Automatic fallback**: System works even if Gemini is unavailable
- ✅ **Usage tracking**: Monitor Gemini API usage in real-time

---

## Quick Start

### 1. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your free API key

### 2. Configure Environment

```bash
cd DiffuGen-Upgraded
cp .env.example .env
nano .env
```

Add your Gemini API key:
```
GEMINI_API_KEY=your_actual_key_here
USE_HYBRID_ROUTING=true
```

### 3. Install Dependencies

```bash
pip install google-generativeai
```

### 4. Ensure Qwen-VL is Running

```bash
# Check if Qwen-VL is available
ollama list | grep qwen2-vl

# If not, pull it
ollama pull qwen2-vl:latest
```

### 5. Start the System

```bash
python diffugen_openapi.py
```

You'll see:
```
INFO - Gemini gemini-1.5-flash initialized successfully
INFO - Qwen-VL qwen2-vl:latest detected and ready
INFO - Hybrid LLM routing enabled (Gemini + Qwen-VL)
```

---

## How It Works

### Routing Decision Flow

```
User Message
     |
     ▼
┌─────────────────┐
│ Message Router  │
│ (Analyzes task) │
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
Simple/Vision  Complex
    │          │
    ▼          ▼
┌────────┐  ┌────────┐
│Qwen-VL │  │ Gemini │
│ FREE   │  │ 1,500  │
│        │  │ /day   │
└────────┘  └────────┘
```

### What Goes to Qwen-VL (FREE)

- ✅ Simple keyword matching: "make it brighter", "train LoRA"
- ✅ Pattern-based tasks: "Generate scenes: x, y, z"
- ✅ Image analysis: quality checks, consistency verification
- ✅ Simple refinements: parameter adjustments
- ✅ Relationship parsing: "Spark's friend is Whiskers"

### What Goes to Gemini (Limited)

- ✅ Complex character descriptions: "Create a wise old dragon mentor"
- ✅ Nuanced requests: "Make it more adventurous but keep it cozy"
- ✅ Contradictory requirements: "Detailed but simple"
- ✅ Emotional/subjective: "Make it feel magical"
- ✅ Ambiguous requests needing clarification

---

## Usage Statistics

### Monitor Your Gemini Usage

```python
# In Python
from intelligent_workflow import IntelligentWorkflow

workflow = IntelligentWorkflow()

# Get usage stats
stats = workflow.intent_analyzer.router.get_usage_stats()

print(f"Gemini calls today: {stats['gemini_calls_today']}")
print(f"Remaining: {stats['gemini_calls_remaining']}")
print(f"Qwen-VL calls today: {stats['qwen_vl_calls_today']}")
print(f"Percentage on Gemini: {stats['percentage_on_gemini']:.1f}%")
```

### Via API

```bash
curl http://localhost:5199/llm/usage-stats
```

Response:
```json
{
  "gemini_calls_today": 23,
  "gemini_calls_remaining": 1477,
  "gemini_calls_total": 156,
  "qwen_vl_calls_today": 142,
  "qwen_vl_calls_total": 1893,
  "percentage_on_gemini": 13.9
}
```

---

## Routing Examples

| User Message | LLM Used | Reason | Cost |
|-------------|----------|--------|------|
| "Make it brighter" | **Qwen-VL** | Simple keyword | Free |
| "Train LoRA for Spark" | **Qwen-VL** | Pattern match | Free |
| "Generate scenes: castle, forest" | **Qwen-VL** | Pattern match | Free |
| "Create wise old dragon mentor" | **Gemini** | Complex character | 1 call |
| "Make it feel more magical but less scary" | **Gemini** | Nuanced request | 1 call |
| "Check if image is child-appropriate" | **Qwen-VL** | Vision task | Free |
| "Does Spark look consistent?" | **Qwen-VL** | Vision comparison | Free |
| "Spark's friend is Whiskers" | **Qwen-VL** | Regex extraction | Free |

---

## Daily Capacity

### Gemini 1.5 Flash Free Tier: 1,500 calls/day

**With Hybrid Routing:**
- Simple storybook: ~5 Gemini calls
- **Capacity: 300 storybooks/day** (6,000 images)

**Without Routing (Gemini-only):**
- Simple storybook: ~30 Gemini calls
- **Capacity: 50 storybooks/day** (1,000 images)

**Improvement: 6x more capacity!**

---

## Configuration Options

### Enable/Disable Hybrid Routing

```python
# Enable (default)
workflow = IntelligentWorkflow()  # Hybrid enabled by default

# Disable (Qwen-only fallback)
workflow = IntelligentWorkflow()
workflow.intent_analyzer.use_router = False
```

### Change Gemini Model

```python
# Use Gemini 2.0 Flash (newer, same limits)
from llm_router import GeminiAnalyzer

gemini = GeminiAnalyzer(model="gemini-2.0-flash-exp")
```

### Adjust Qwen-VL Model

```python
from llm_router import QwenVLAnalyzer

# Use different Qwen-VL version
qwen_vl = QwenVLAnalyzer(
    base_url="http://localhost:11434",
    model="qwen2-vl:7b"  # or qwen2-vl:72b for better quality
)
```

---

## Vision Analysis Features

### Automatic Quality Checks

```python
# Generate image
result = await workflow.process_message(session, "Create dragon Spark")

# Auto-check quality with Qwen-VL (FREE)
quality = await workflow.intent_analyzer.qwen_vl.analyze_image_quality(
    result['image_path']
)

if quality['child_appropriate'] < 7:
    print("⚠️  Warning: Image may not be child-appropriate")

if quality['overall'] < 7:
    # Auto-regenerate
    await workflow.process_message(session, "Regenerate with better quality")
```

### Character Consistency Verification

```python
# Compare new generation with reference
consistency = await qwen_vl.verify_character_consistency(
    reference_image="characters/spark_ref.png",
    new_image="output/scene_001.png",
    character_name="Spark"
)

if consistency['overall_match'] < 8:
    print(f"⚠️  Consistency issues: {consistency['differences']}")
    print(f"Recommendation: {consistency['recommendation']}")
```

---

## Troubleshooting

### "Gemini not enabled - check API key"

**Solution:**
```bash
# Check .env file
cat .env | grep GEMINI_API_KEY

# Ensure key is set
export GEMINI_API_KEY=your_key_here

# Restart server
python diffugen_openapi.py
```

### "Qwen-VL not available"

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull Qwen-VL model
ollama pull qwen2-vl:latest

# Verify it's running
ollama list
```

### High Gemini Usage

**Check routing**:
```python
stats = workflow.intent_analyzer.router.get_usage_stats()
print(f"Percentage on Gemini: {stats['percentage_on_gemini']}%")
```

**Expected**: 5-15% of calls should go to Gemini
**If higher**: Check if complex messages are being sent

### "Approaching Gemini limit"

System automatically routes to Qwen-VL when < 100 calls remaining.

Check logs:
```
WARNING - Low on Gemini calls (95 remaining) - routing to Qwen-VL
```

---

## Performance Metrics

### Routing Efficiency

**Ideal distribution** (300 storybook sessions/day):
- Gemini: ~1,500 calls (5 per storybook)
- Qwen-VL: ~4,500 calls (15 per storybook)
- Vision analysis: ~6,000 calls (all Qwen-VL, free)

### Response Times

- **Qwen-VL (local)**: 0.5-2s
- **Gemini (API)**: 1-3s
- **Vision analysis**: 2-5s

---

## API Endpoints

### Get Usage Stats

```bash
GET /llm/usage-stats
```

Response:
```json
{
  "gemini_calls_today": 45,
  "gemini_calls_remaining": 1455,
  "qwen_vl_calls_today": 234,
  "percentage_on_gemini": 16.1
}
```

### Force Routing Choice

```bash
POST /conversational/generate
{
  "session_id": "session-123",
  "message": "Create dragon",
  "force_llm": "gemini"  // or "qwen-vl"
}
```

---

## Best Practices

### 1. Let the Router Decide

Don't manually force routing unless testing. The router is optimized.

### 2. Use Vision Analysis Liberally

It's free! Check every image:
```python
for scene in scenes:
    quality = await qwen_vl.analyze_image_quality(scene)
    if quality['overall'] < 7:
        regenerate(scene)
```

### 3. Batch Simple Operations

```python
# Good: All handled by Qwen-VL (free)
scenes = ["castle", "forest", "beach", "mountain"]
await workflow.process_message(session, f"Generate scenes: {', '.join(scenes)}")

# Bad: 4 separate Gemini calls
for scene in scenes:
    await workflow.process_message(session, f"Generate {scene}")
```

### 4. Monitor Usage Weekly

```bash
# Add to cron
0 0 * * 0 curl http://localhost:5199/llm/usage-stats >> usage_log.txt
```

---

## Summary

✅ **Setup**: Add Gemini API key to `.env`
✅ **Routing**: Automatic (Gemini for complex, Qwen-VL for simple/vision)
✅ **Capacity**: 300 storybooks/day (vs 50 Gemini-only)
✅ **Vision**: Unlimited quality checks and consistency verification
✅ **Monitoring**: Real-time usage statistics

**Result**: Professional storybook system with 6x capacity! 🎨📚
