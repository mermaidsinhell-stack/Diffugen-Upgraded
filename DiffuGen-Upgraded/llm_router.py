"""
Intelligent LLM Router for Hybrid Gemini + Qwen-VL System
Routes tasks to Gemini (complex reasoning) or Qwen-VL (vision + simple text)
"""

import os
import re
import logging
import asyncio
import httpx
import base64
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class LLMType(Enum):
    """Which LLM to use"""
    GEMINI = "gemini"
    QWEN_VL = "qwen-vl"


@dataclass
class RoutingDecision:
    """Decision about which LLM to use"""
    llm_type: LLMType
    reason: str
    confidence: float = 1.0


@dataclass
class UsageStats:
    """Track LLM usage"""
    gemini_calls_today: int = 0
    gemini_calls_total: int = 0
    qwen_vl_calls_today: int = 0
    qwen_vl_calls_total: int = 0
    last_reset: float = field(default_factory=time.time)

    def reset_daily(self):
        """Reset daily counters"""
        current_day = time.time() // 86400
        last_day = self.last_reset // 86400

        if current_day > last_day:
            self.gemini_calls_today = 0
            self.qwen_vl_calls_today = 0
            self.last_reset = time.time()

    def record_gemini_call(self):
        """Record a Gemini API call"""
        self.reset_daily()
        self.gemini_calls_today += 1
        self.gemini_calls_total += 1

    def record_qwen_vl_call(self):
        """Record a Qwen-VL call"""
        self.reset_daily()
        self.qwen_vl_calls_today += 1
        self.qwen_vl_calls_total += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        self.reset_daily()
        return {
            "gemini_calls_today": self.gemini_calls_today,
            "gemini_calls_remaining": max(0, 1500 - self.gemini_calls_today),
            "gemini_calls_total": self.gemini_calls_total,
            "qwen_vl_calls_today": self.qwen_vl_calls_today,
            "qwen_vl_calls_total": self.qwen_vl_calls_total,
            "percentage_on_gemini": (
                (self.gemini_calls_today / (self.gemini_calls_today + self.qwen_vl_calls_today) * 100)
                if (self.gemini_calls_today + self.qwen_vl_calls_today) > 0 else 0
            )
        }


# ============================================================================
# Gemini Analyzer
# ============================================================================

class GeminiAnalyzer:
    """Handles complex text reasoning with Gemini API"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.enabled = bool(self.api_key)

        if self.enabled:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.genai = genai
                self.model_instance = genai.GenerativeModel(model)
                logger.info(f"Gemini {model} initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.enabled = False
        else:
            logger.warning("Gemini API key not found - Gemini disabled")

    async def analyze_intent(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze complex user intent with Gemini"""

        if not self.enabled:
            raise RuntimeError("Gemini not enabled - check API key")

        # Build context-aware prompt
        context_info = ""
        if context:
            if context.get("active_character"):
                context_info += f"\nActive character: {context['active_character']}"
            if context.get("locked_style"):
                context_info += f"\nLocked style: {context['locked_style']}"
            if context.get("last_generation"):
                context_info += f"\nLast generation: {context['last_generation']}"

        prompt = f"""
You are analyzing a user's request for a children's storybook illustration system.
{context_info}

User message: "{message}"

Analyze this message and respond in JSON format with:
{{
    "intent_type": "one of: new_generation, refine_previous, train_lora, style_lock, style_unlock, batch_generate, add_relationship, multi_character_scene, character_create, character_use, character_sheet",
    "confidence": 0.0-1.0,
    "character_names": ["list", "of", "characters"],
    "style_preference": "watercolor/digital/pencil/cartoon/classic or null",
    "adjustments": {{
        "brightness": "brighter/darker or null",
        "detail": "more/less or null",
        "color": "more/less or null"
    }},
    "prompt_additions": ["keywords", "to", "add"],
    "prompt_removals": ["keywords", "to", "remove"],
    "explanation": "brief explanation of what user wants"
}}

Focus on understanding nuanced requests like "make it more adventurous but keep it cozy".
"""

        try:
            response = await self.model_instance.generate_content_async(prompt)

            # Parse JSON from response
            response_text = response.text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = re.sub(r'```json\n?', '', response_text)
                response_text = re.sub(r'```\n?', '', response_text)

            result = json.loads(response_text)

            logger.info(f"Gemini analyzed intent: {result.get('intent_type')} (confidence: {result.get('confidence')})")

            return result

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            raise

    async def analyze_complexity(self, message: str) -> bool:
        """Quick check: is this message complex?"""

        if not self.enabled:
            return False

        prompt = f"""
Is this message simple (keyword-based) or complex (nuanced reasoning)?

Message: "{message}"

Respond with just one word: SIMPLE or COMPLEX
"""

        try:
            response = await self.model_instance.generate_content_async(prompt)
            result = response.text.strip().upper()
            return "COMPLEX" in result
        except Exception as e:
            logger.error(f"Complexity check failed: {e}")
            return False


# ============================================================================
# Qwen-VL Analyzer
# ============================================================================

class QwenVLAnalyzer:
    """Handles vision tasks and simple text with local Qwen-VL"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2-vl:latest"
    ):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=120.0)
        self.enabled = False

        # Test connection
        asyncio.create_task(self._test_connection())

    async def _test_connection(self):
        """Test if Qwen-VL is available"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]

                if any(self.model in name for name in model_names):
                    self.enabled = True
                    logger.info(f"Qwen-VL {self.model} detected and ready")
                else:
                    logger.warning(f"Qwen-VL model '{self.model}' not found. Available: {model_names}")
            else:
                logger.warning(f"Ollama server responded with status {response.status_code}")
        except Exception as e:
            logger.warning(f"Qwen-VL not available: {e}")

    async def analyze_intent(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze simple intent with keyword matching + Qwen-VL"""

        # Try fast keyword-based analysis first
        keyword_result = self._keyword_based_analysis(message)
        if keyword_result:
            logger.info(f"Qwen-VL: keyword match for {keyword_result['intent_type']}")
            return keyword_result

        # Fall back to Qwen-VL LLM if enabled
        if self.enabled:
            return await self._qwen_vl_analysis(message, context)
        else:
            # Ultra-simple fallback
            return {
                "intent_type": "new_generation",
                "confidence": 0.5,
                "character_names": [],
                "style_preference": None,
                "adjustments": {},
                "prompt_additions": [],
                "prompt_removals": [],
                "explanation": "Simple generation request"
            }

    def _keyword_based_analysis(self, message: str) -> Optional[Dict[str, Any]]:
        """Fast keyword-based intent detection"""

        msg_lower = message.lower()

        # Train LoRA
        if re.search(r'train\s+(?:lora|LoRA)', msg_lower):
            return {
                "intent_type": "train_lora",
                "confidence": 0.95,
                "character_names": self._extract_character_names(message),
                "explanation": "LoRA training request"
            }

        # Style lock/unlock
        if re.search(r'lock\s+style|use\s+style|set\s+style', msg_lower):
            style = self._extract_style(message)
            return {
                "intent_type": "style_lock",
                "confidence": 0.95,
                "style_preference": style,
                "explanation": f"Lock style to {style}"
            }

        if "unlock style" in msg_lower:
            return {
                "intent_type": "style_unlock",
                "confidence": 0.95,
                "explanation": "Unlock style"
            }

        # Batch generation
        if re.search(r'generate\s+scenes?:', msg_lower):
            return {
                "intent_type": "batch_generate",
                "confidence": 0.95,
                "explanation": "Batch scene generation"
            }

        # Relationships
        if re.search(r"'s\s+(friend|sibling|companion|rival)", msg_lower):
            return {
                "intent_type": "add_relationship",
                "confidence": 0.95,
                "character_names": self._extract_character_names(message),
                "explanation": "Add character relationship"
            }

        # Multi-character
        if re.search(r'\band\b.*\b(with|together)', msg_lower):
            chars = self._extract_character_names(message)
            if len(chars) >= 2:
                return {
                    "intent_type": "multi_character_scene",
                    "confidence": 0.85,
                    "character_names": chars,
                    "explanation": "Multi-character scene"
                }

        # Simple refinements
        adjustments = {}
        if "brighter" in msg_lower:
            adjustments["brightness"] = "brighter"
        if "darker" in msg_lower:
            adjustments["brightness"] = "darker"
        if "more detailed" in msg_lower:
            adjustments["detail"] = "more"
        if "less detailed" in msg_lower or "simpler" in msg_lower:
            adjustments["detail"] = "less"
        if "more colorful" in msg_lower or "more vibrant" in msg_lower:
            adjustments["color"] = "more"

        if adjustments:
            return {
                "intent_type": "refine_previous",
                "confidence": 0.9,
                "adjustments": adjustments,
                "explanation": "Simple parameter adjustment"
            }

        return None

    def _extract_character_names(self, message: str) -> List[str]:
        """Extract capitalized character names"""
        # Find capitalized words (likely character names)
        matches = re.findall(r'\b([A-Z][a-z]+)\b', message)
        return list(set(matches))

    def _extract_style(self, message: str) -> Optional[str]:
        """Extract style name from message"""
        styles = ["watercolor", "digital", "pencil", "cartoon", "storybook"]
        for style in styles:
            if style in message.lower():
                return f"{style}_soft" if style == "watercolor" else f"{style}_vibrant"
        return None

    async def _qwen_vl_analysis(self, message: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Use Qwen-VL LLM for intent analysis"""

        prompt = f"""
Analyze this storybook generation request and respond in JSON:

Message: "{message}"

{{
    "intent_type": "new_generation or refine_previous or character_create",
    "confidence": 0.0-1.0,
    "character_names": [],
    "explanation": "brief explanation"
}}
"""

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
            )

            result = response.json()
            response_text = result.get("response", "{}")

            return json.loads(response_text)

        except Exception as e:
            logger.error(f"Qwen-VL analysis failed: {e}")
            raise

    async def analyze_image_quality(self, image_path: str) -> Dict[str, Any]:
        """Analyze generated image quality"""

        if not self.enabled:
            return {"error": "Qwen-VL not available"}

        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode()

        prompt = """
Analyze this children's storybook illustration.

Rate (1-10):
1. Child-appropriate (no scary/violent content)
2. Visual quality (colors, clarity, composition)
3. Child-friendly aesthetic (cute, friendly, warm)

Respond in JSON:
{
    "child_appropriate": 1-10,
    "visual_quality": 1-10,
    "aesthetic": 1-10,
    "overall": 1-10,
    "issues": ["list of issues if any"],
    "suggestions": ["improvements"]
}
"""

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "format": "json"
                }
            )

            result = response.json()
            return json.loads(result.get("response", "{}"))

        except Exception as e:
            logger.error(f"Image quality analysis failed: {e}")
            return {"error": str(e)}

    async def verify_character_consistency(
        self,
        reference_image: str,
        new_image: str,
        character_name: str
    ) -> Dict[str, Any]:
        """Compare two images for character consistency"""

        if not self.enabled:
            return {"error": "Qwen-VL not available"}

        images = []
        for img_path in [reference_image, new_image]:
            with open(img_path, 'rb') as f:
                images.append(base64.b64encode(f.read()).decode())

        prompt = f"""
Compare these two images of {character_name}.

Image 1: Reference
Image 2: New generation

Rate consistency (1-10):
1. Color scheme
2. Character features
3. Art style
4. Overall match

Respond in JSON:
{{
    "color_consistency": 1-10,
    "feature_consistency": 1-10,
    "style_consistency": 1-10,
    "overall_match": 1-10,
    "is_consistent": true/false,
    "differences": ["list of differences"],
    "recommendation": "keep or regenerate"
}}
"""

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": images,
                    "stream": False,
                    "format": "json"
                }
            )

            result = response.json()
            return json.loads(result.get("response", "{}"))

        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return {"error": str(e)}


# ============================================================================
# Intelligent Message Router
# ============================================================================

class MessageRouter:
    """Routes messages to Gemini (complex) or Qwen-VL (simple/vision)"""

    def __init__(
        self,
        gemini_analyzer: GeminiAnalyzer,
        qwen_vl_analyzer: QwenVLAnalyzer
    ):
        self.gemini = gemini_analyzer
        self.qwen_vl = qwen_vl_analyzer
        self.stats = UsageStats()

        # Cache recent decisions to avoid redundant routing
        self.decision_cache: Dict[str, RoutingDecision] = {}

    async def route_message(
        self,
        message: str,
        context: Optional[Dict] = None,
        has_image: bool = False
    ) -> RoutingDecision:
        """Decide which LLM to use for this message"""

        # Check cache
        cache_key = self._make_cache_key(message)
        if cache_key in self.decision_cache:
            logger.info("Router: using cached decision")
            return self.decision_cache[cache_key]

        # Rule 1: ANY vision task → Qwen-VL
        if has_image or self._needs_vision(message):
            decision = RoutingDecision(
                llm_type=LLMType.QWEN_VL,
                reason="vision_task",
                confidence=1.0
            )
            self.decision_cache[cache_key] = decision
            return decision

        # Rule 2: Simple keyword patterns → Qwen-VL
        if self._is_simple_pattern(message):
            decision = RoutingDecision(
                llm_type=LLMType.QWEN_VL,
                reason="simple_keyword_pattern",
                confidence=0.95
            )
            self.decision_cache[cache_key] = decision
            return decision

        # Rule 3: Approaching Gemini limit → Qwen-VL
        stats = self.stats.get_stats()
        if stats["gemini_calls_remaining"] < 100:
            logger.warning(f"Low on Gemini calls ({stats['gemini_calls_remaining']} remaining) - routing to Qwen-VL")
            decision = RoutingDecision(
                llm_type=LLMType.QWEN_VL,
                reason="gemini_quota_low",
                confidence=0.8
            )
            return decision

        # Rule 4: Complex reasoning needed → Gemini
        if self._needs_complex_reasoning(message):
            if self.gemini.enabled:
                decision = RoutingDecision(
                    llm_type=LLMType.GEMINI,
                    reason="complex_reasoning_required",
                    confidence=0.9
                )
                self.decision_cache[cache_key] = decision
                return decision
            else:
                logger.warning("Gemini needed but not enabled - falling back to Qwen-VL")
                decision = RoutingDecision(
                    llm_type=LLMType.QWEN_VL,
                    reason="gemini_unavailable_fallback",
                    confidence=0.6
                )
                return decision

        # Default: Qwen-VL (free, local, fast)
        decision = RoutingDecision(
            llm_type=LLMType.QWEN_VL,
            reason="default_to_local",
            confidence=0.8
        )
        self.decision_cache[cache_key] = decision
        return decision

    def _make_cache_key(self, message: str) -> str:
        """Create cache key from normalized message"""
        normalized = message.lower().strip().rstrip('!?.')
        return normalized[:100]  # Limit key length

    def _needs_vision(self, message: str) -> bool:
        """Does this require image analysis?"""
        vision_keywords = [
            "look", "see", "check", "analyze image", "is this",
            "does this look", "consistency", "quality", "compare"
        ]
        return any(kw in message.lower() for kw in vision_keywords)

    def _is_simple_pattern(self, message: str) -> bool:
        """Can be handled with regex/keyword matching?"""
        simple_patterns = [
            r'train\s+lora',
            r'lock\s+style',
            r'unlock\s+style',
            r'generate\s+scenes?:',
            r"'s\s+(friend|sibling)",
            r'\b(brighter|darker|more detailed|simpler)\b'
        ]
        return any(re.search(p, message, re.IGNORECASE) for p in simple_patterns)

    def _needs_complex_reasoning(self, message: str) -> bool:
        """Needs deep understanding?"""

        # Long messages with nuance
        if len(message.split()) > 15:
            return True

        # Contradictory requirements
        if any(word in message.lower() for word in [" but ", " however ", " while ", " although "]):
            return True

        # New character creation
        if any(phrase in message.lower() for phrase in ["create character", "create a", "make a character"]):
            return True

        # Emotional/subjective requests
        emotional_words = ["feel", "mood", "atmosphere", "vibe", "emotion", "cozy", "adventurous"]
        if any(word in message.lower() for word in emotional_words):
            return True

        return False

    async def analyze_with_routing(
        self,
        message: str,
        context: Optional[Dict] = None,
        has_image: bool = False
    ) -> Tuple[Dict[str, Any], RoutingDecision]:
        """Route and analyze message"""

        decision = await self.route_message(message, context, has_image)

        if decision.llm_type == LLMType.GEMINI:
            self.stats.record_gemini_call()
            result = await self.gemini.analyze_intent(message, context)
        else:
            self.stats.record_qwen_vl_call()
            result = await self.qwen_vl.analyze_intent(message, context)

        return result, decision

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return self.stats.get_stats()


# ============================================================================
# CLI Testing
# ============================================================================

async def test_router():
    """Test the routing system"""
    print("=== LLM Router Test ===\n")

    # Initialize
    gemini = GeminiAnalyzer(model="gemini-1.5-flash")
    qwen_vl = QwenVLAnalyzer()
    router = MessageRouter(gemini, qwen_vl)

    # Wait for Qwen-VL to connect
    await asyncio.sleep(2)

    # Test messages
    test_cases = [
        ("Make it brighter", False),
        ("Create a wise old dragon mentor named Merlin", False),
        ("Train LoRA for Spark", False),
        ("Make it more adventurous but keep it cozy and friendly", False),
        ("Generate scenes: castle, forest, beach", False),
        ("Check if this image is appropriate", True),
    ]

    for message, has_image in test_cases:
        print(f"\nMessage: '{message}'")
        decision = await router.route_message(message, has_image=has_image)
        print(f"  → {decision.llm_type.value} ({decision.reason}, confidence: {decision.confidence:.2f})")

    # Show stats
    print("\n" + "="*50)
    stats = router.get_usage_stats()
    print(f"\nUsage Statistics:")
    print(f"  Gemini calls today: {stats['gemini_calls_today']}")
    print(f"  Gemini remaining: {stats['gemini_calls_remaining']}")
    print(f"  Qwen-VL calls today: {stats['qwen_vl_calls_today']}")
    print(f"  Percentage on Gemini: {stats['percentage_on_gemini']:.1f}%")


if __name__ == "__main__":
    asyncio.run(test_router())
