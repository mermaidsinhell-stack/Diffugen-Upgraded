"""
Intelligent Workflow System for Storybook Illustration
Provides natural language interface for iterative image generation with LLM-powered parameter tuning
"""

import asyncio
import json
import logging
import time
import re
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class IntentType(Enum):
    """Types of user intents"""
    NEW_GENERATION = "new_generation"
    REFINE_PREVIOUS = "refine_previous"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    STYLE_CHANGE = "style_change"
    CHARACTER_CONSISTENCY = "character_consistency"
    CLARIFICATION_NEEDED = "clarification_needed"


@dataclass
class GenerationParameters:
    """Image generation parameters"""
    prompt: str
    model: str = "sd15"  # SD1.5 better for illustrations
    width: int = 512
    height: int = 512
    steps: int = 25
    cfg_scale: float = 7.5
    seed: int = -1
    sampling_method: str = "euler_a"
    negative_prompt: str = ""
    lora: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def copy(self) -> 'GenerationParameters':
        """Create a copy"""
        return GenerationParameters(**self.to_dict())


@dataclass
class Intent:
    """Parsed user intent"""
    type: IntentType
    confidence: float  # 0-1
    adjustments: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    prompt_modifications: Dict[str, List[str]] = field(default_factory=dict)  # add, remove, replace


@dataclass
class GenerationResult:
    """Result from generation"""
    success: bool
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    parameters: Optional[GenerationParameters] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationContext:
    """Conversation context for memory"""
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    generation_history: List[GenerationResult] = field(default_factory=list)
    current_parameters: Optional[GenerationParameters] = None
    style_preferences: Dict[str, Any] = field(default_factory=dict)
    character_references: Dict[str, str] = field(default_factory=dict)  # name -> image_path
    created_at: float = field(default_factory=time.time)


# ============================================================================
# Parameter Mapping
# ============================================================================

class ParameterMapper:
    """
    Maps natural language descriptions to parameter adjustments
    """

    # Brightness/lighting adjustments
    BRIGHTNESS_KEYWORDS = {
        "brighter": {"cfg_scale_delta": +2.0, "prompt_add": ["bright", "well-lit", "luminous"]},
        "darker": {"cfg_scale_delta": -1.5, "prompt_add": ["dim lighting", "shadowy"], "negative_add": ["bright", "overexposed"]},
        "more light": {"cfg_scale_delta": +1.5, "prompt_add": ["bright lighting", "sunny"]},
        "less light": {"cfg_scale_delta": -1.0, "prompt_add": ["soft lighting", "gentle light"]},
    }

    # Detail level adjustments
    DETAIL_KEYWORDS = {
        "more detailed": {"steps_delta": +10, "prompt_add": ["highly detailed", "intricate"]},
        "less detailed": {"steps_delta": -10, "prompt_add": ["simple", "minimalist", "clean"]},
        "simpler": {"steps_delta": -10, "cfg_scale_delta": -1.0, "prompt_add": ["simple shapes", "clean design"]},
        "more complex": {"steps_delta": +10, "cfg_scale_delta": +1.0, "prompt_add": ["detailed", "intricate"]},
    }

    # Color adjustments
    COLOR_KEYWORDS = {
        "more colorful": {"cfg_scale_delta": +1.0, "prompt_add": ["vibrant colors", "saturated", "colorful"], "negative_add": ["muted", "desaturated"]},
        "less colorful": {"cfg_scale_delta": -0.5, "prompt_add": ["soft colors", "muted palette"], "negative_add": ["saturated", "vibrant"]},
        "vibrant": {"cfg_scale_delta": +1.5, "prompt_add": ["vibrant", "saturated colors", "bold colors"]},
        "pastel": {"prompt_add": ["pastel colors", "soft palette"], "negative_add": ["dark", "saturated"]},
        "warm colors": {"prompt_add": ["warm tones", "red", "orange", "yellow hues"]},
        "cool colors": {"prompt_add": ["cool tones", "blue", "green", "purple hues"]},
    }

    # Style adjustments
    STYLE_KEYWORDS = {
        "more cartoonish": {"prompt_add": ["cartoon style", "animated", "stylized"], "negative_add": ["realistic", "photorealistic"]},
        "more realistic": {"steps_delta": +10, "prompt_add": ["realistic", "detailed"], "negative_add": ["cartoon", "stylized"]},
        "watercolor": {"prompt_add": ["watercolor painting", "watercolor style", "soft edges"]},
        "sketch": {"prompt_add": ["pencil sketch", "hand-drawn", "sketchy"]},
        "digital art": {"prompt_add": ["digital illustration", "digital art"]},
    }

    # Quality/sharpness
    QUALITY_KEYWORDS = {
        "sharper": {"cfg_scale_delta": +1.0, "prompt_add": ["sharp", "crisp", "clear"]},
        "softer": {"cfg_scale_delta": -1.0, "prompt_add": ["soft", "gentle", "smooth"]},
        "more contrast": {"cfg_scale_delta": +1.5, "prompt_add": ["high contrast"]},
        "less contrast": {"cfg_scale_delta": -1.0, "prompt_add": ["low contrast", "soft"]},
    }

    @classmethod
    def analyze_adjustments(cls, text: str) -> Dict[str, Any]:
        """
        Analyze text for parameter adjustment keywords

        Returns:
            Dictionary of adjustments to apply
        """
        text_lower = text.lower()
        adjustments = {
            "steps_delta": 0,
            "cfg_scale_delta": 0.0,
            "prompt_add": [],
            "prompt_remove": [],
            "negative_add": [],
            "negative_remove": []
        }

        # Check all keyword categories
        all_keywords = {
            **cls.BRIGHTNESS_KEYWORDS,
            **cls.DETAIL_KEYWORDS,
            **cls.COLOR_KEYWORDS,
            **cls.STYLE_KEYWORDS,
            **cls.QUALITY_KEYWORDS
        }

        for keyword, mods in all_keywords.items():
            if keyword in text_lower:
                logger.info(f"Found keyword '{keyword}' in request")

                # Apply deltas
                if "steps_delta" in mods:
                    adjustments["steps_delta"] += mods["steps_delta"]
                if "cfg_scale_delta" in mods:
                    adjustments["cfg_scale_delta"] += mods["cfg_scale_delta"]

                # Add prompt modifications
                if "prompt_add" in mods:
                    adjustments["prompt_add"].extend(mods["prompt_add"])
                if "prompt_remove" in mods:
                    adjustments["prompt_remove"].extend(mods["prompt_remove"])
                if "negative_add" in mods:
                    adjustments["negative_add"].extend(mods["negative_add"])
                if "negative_remove" in mods:
                    adjustments["negative_remove"].extend(mods["negative_remove"])

        return adjustments


# ============================================================================
# Child Safety Filter
# ============================================================================

class ChildSafetyFilter:
    """
    Ensures all content is appropriate for children's storybooks
    """

    FORBIDDEN_KEYWORDS = [
        "scary", "frightening", "horror", "terrifying", "creepy",
        "violent", "blood", "weapon", "gun", "knife", "sword",
        "monster", "demon", "evil", "dark magic",
        "realistic gore", "photorealistic violence",
        "inappropriate", "adult"
    ]

    REQUIRED_SAFE_WORDS = [
        "friendly", "cute", "wholesome", "appropriate for children",
        "children's book illustration", "safe for kids"
    ]

    BASE_NEGATIVE_PROMPT = """
    scary, frightening, dark, horror, violent, weapon, blood, gore,
    realistic, photorealistic, uncanny, creepy, disturbing,
    inappropriate, adult themes, nsfw, mature content
    """

    CHILD_FRIENDLY_REPLACEMENTS = {
        "dragon": "friendly dragon",
        "monster": "friendly creature",
        "beast": "cute animal",
        "dark": "cozy dim",
        "scary": "exciting",
        "fight": "play",
        "battle": "adventure"
    }

    @classmethod
    def filter_prompt(cls, prompt: str) -> Tuple[str, List[str]]:
        """
        Filter prompt for child-appropriate content

        Returns:
            (filtered_prompt, warnings)
        """
        filtered = prompt
        warnings = []

        # Check for forbidden keywords
        for forbidden in cls.FORBIDDEN_KEYWORDS:
            if forbidden in prompt.lower():
                warnings.append(f"Removed inappropriate keyword: '{forbidden}'")
                # Remove the keyword
                filtered = re.sub(
                    rf'\b{re.escape(forbidden)}\b',
                    '',
                    filtered,
                    flags=re.IGNORECASE
                )

        # Apply friendly replacements
        for word, replacement in cls.CHILD_FRIENDLY_REPLACEMENTS.items():
            if word in filtered.lower() and replacement.split()[0] not in filtered.lower():
                filtered = re.sub(
                    rf'\b{re.escape(word)}\b',
                    replacement,
                    filtered,
                    flags=re.IGNORECASE
                )

        # Ensure child-friendly descriptors
        if not any(safe in filtered.lower() for safe in ["friendly", "cute", "adorable"]):
            filtered = f"cute, friendly {filtered}"

        # Add children's book style if not present
        if "children's book" not in filtered.lower() and "kids book" not in filtered.lower():
            filtered += ", children's book illustration"

        return filtered.strip(), warnings

    @classmethod
    def get_child_safe_negative_prompt(cls, additional: str = "") -> str:
        """
        Get child-safe negative prompt
        """
        return f"{cls.BASE_NEGATIVE_PROMPT}, {additional}".strip(", ")


# ============================================================================
# Intent Analyzer (LLM-powered)
# ============================================================================

class IntentAnalyzer:
    """
    Analyzes user intent using LLM (Qwen)
    """

    def __init__(self, llm_base_url: str = "http://localhost:11434/v1"):
        self.llm_base_url = llm_base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    SYSTEM_PROMPT = """You are an AI assistant helping create children's storybook illustrations.

Your job is to analyze user requests and determine:
1. Intent type (new generation, refinement, parameter adjustment, style change)
2. What technical parameters should change
3. How to modify the prompt

Rules for children's storybooks:
- Always ensure content is appropriate for young children (ages 3-8)
- Use friendly, cute, wholesome themes
- Prefer simple, clear illustrations over complex realistic ones
- Use soft colors and friendly characters
- NO scary, violent, or inappropriate content

When user says:
- "make it brighter" → increase cfg_scale by 2.0, add "bright, well-lit" to prompt
- "less detailed" → decrease steps by 10, add "simple, minimalist" to prompt
- "more colorful" → increase cfg_scale by 1.0, add "vibrant, colorful" to prompt
- "make the [object] bigger" → modify prompt to emphasize that object

Respond in JSON format:
{
    "intent_type": "new_generation|refine_previous|parameter_adjustment",
    "confidence": 0.95,
    "adjustments": {
        "steps_delta": 0,
        "cfg_scale_delta": 0.0,
        "width": null,
        "height": null
    },
    "prompt_changes": {
        "add": ["list", "of", "things", "to", "add"],
        "remove": ["list", "to", "remove"],
        "emphasis": ["things", "to", "emphasize"]
    },
    "explanation": "Brief explanation of what you're doing"
}"""

    async def analyze(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        current_params: Optional[GenerationParameters] = None
    ) -> Intent:
        """
        Analyze user intent using LLM

        Args:
            user_message: Current user message
            conversation_history: Previous conversation
            current_params: Current generation parameters

        Returns:
            Parsed intent
        """
        try:
            # Build context
            context = self._build_context(conversation_history, current_params)

            # Call LLM
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": context + "\n\nUser request: " + user_message}
            ]

            response = await self.client.post(
                f"{self.llm_base_url}/chat/completions",
                json={
                    "model": "qwen2.5:latest",
                    "messages": messages,
                    "temperature": 0.3,  # Low temperature for consistent analysis
                    "max_tokens": 500
                }
            )

            if response.status_code != 200:
                logger.error(f"LLM request failed: {response.status_code}")
                return self._fallback_analysis(user_message)

            result = response.json()
            llm_response = result["choices"][0]["message"]["content"]

            # Parse JSON response
            intent = self._parse_llm_response(llm_response)

            return intent

        except Exception as e:
            logger.error(f"Error analyzing intent: {e}", exc_info=True)
            return self._fallback_analysis(user_message)

    def _build_context(
        self,
        history: List[Dict[str, str]],
        params: Optional[GenerationParameters]
    ) -> str:
        """Build context string for LLM"""
        context_parts = []

        if params:
            context_parts.append(f"Current parameters:")
            context_parts.append(f"- Model: {params.model}")
            context_parts.append(f"- Steps: {params.steps}")
            context_parts.append(f"- CFG Scale: {params.cfg_scale}")
            context_parts.append(f"- Size: {params.width}x{params.height}")
            context_parts.append(f"- Prompt: {params.prompt[:200]}")

        if history:
            context_parts.append("\nRecent conversation:")
            for msg in history[-3:]:  # Last 3 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]
                context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def _parse_llm_response(self, response: str) -> Intent:
        """Parse LLM JSON response into Intent"""
        try:
            # Extract JSON from response (might have text before/after)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            # Map intent type
            intent_type_str = data.get("intent_type", "new_generation")
            intent_type = IntentType.NEW_GENERATION

            if "refine" in intent_type_str:
                intent_type = IntentType.REFINE_PREVIOUS
            elif "parameter" in intent_type_str or "adjustment" in intent_type_str:
                intent_type = IntentType.PARAMETER_ADJUSTMENT
            elif "style" in intent_type_str:
                intent_type = IntentType.STYLE_CHANGE

            return Intent(
                type=intent_type,
                confidence=data.get("confidence", 0.8),
                adjustments=data.get("adjustments", {}),
                explanation=data.get("explanation", ""),
                prompt_modifications=data.get("prompt_changes", {})
            )

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._fallback_analysis(response)

    def _fallback_analysis(self, text: str) -> Intent:
        """
        Fallback analysis using keyword matching
        """
        text_lower = text.lower()

        # Use parameter mapper for adjustments
        adjustments = ParameterMapper.analyze_adjustments(text)

        # Determine intent type
        if any(word in text_lower for word in ["generate", "create", "make", "draw", "new"]):
            intent_type = IntentType.NEW_GENERATION
        elif any(word in text_lower for word in ["adjust", "change", "modify", "fix", "improve"]):
            intent_type = IntentType.PARAMETER_ADJUSTMENT
        else:
            intent_type = IntentType.REFINE_PREVIOUS

        return Intent(
            type=intent_type,
            confidence=0.6,  # Lower confidence for fallback
            adjustments=adjustments,
            explanation="Using keyword-based analysis (LLM unavailable)",
            prompt_modifications={}
        )

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# ============================================================================
# Intelligent Workflow Manager
# ============================================================================

class IntelligentWorkflow:
    """
    Main workflow manager for conversational image generation
    """

    def __init__(
        self,
        diffugen_base_url: str = "http://localhost:8080",
        llm_base_url: str = "http://localhost:11434/v1",
        vram_manager=None
    ):
        self.diffugen_base_url = diffugen_base_url
        self.llm_base_url = llm_base_url
        self.vram_manager = vram_manager

        self.intent_analyzer = IntentAnalyzer(llm_base_url)
        self.client = httpx.AsyncClient(timeout=300.0)

        # Active conversations
        self.conversations: Dict[str, ConversationContext] = {}

        # Default parameters for children's books
        self.default_params = GenerationParameters(
            prompt="",
            model="sd15",  # Better for illustrations
            width=512,
            height=512,
            steps=25,
            cfg_scale=7.5,
            sampling_method="euler_a",
            negative_prompt=ChildSafetyFilter.get_child_safe_negative_prompt()
        )

    def create_session(self, session_id: str) -> ConversationContext:
        """Create new conversation session"""
        context = ConversationContext(
            session_id=session_id,
            current_parameters=self.default_params.copy()
        )
        self.conversations[session_id] = context
        return context

    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """Get existing session"""
        return self.conversations.get(session_id)

    async def process_message(
        self,
        session_id: str,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Process user message and generate/refine image

        Args:
            session_id: Conversation session ID
            user_message: User's natural language request

        Returns:
            Response with image and explanation
        """
        # Get or create session
        context = self.get_session(session_id)
        if not context:
            context = self.create_session(session_id)

        # Add message to history
        context.messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": time.time()
        })

        try:
            # Analyze intent
            logger.info(f"Analyzing intent for: {user_message[:100]}")
            intent = await self.intent_analyzer.analyze(
                user_message,
                context.messages,
                context.current_parameters
            )

            logger.info(f"Intent: {intent.type.value}, confidence: {intent.confidence}")

            # Route to appropriate handler
            if intent.type == IntentType.NEW_GENERATION:
                result = await self._handle_new_generation(context, user_message, intent)
            elif intent.type in [IntentType.REFINE_PREVIOUS, IntentType.PARAMETER_ADJUSTMENT]:
                result = await self._handle_refinement(context, user_message, intent)
            elif intent.type == IntentType.STYLE_CHANGE:
                result = await self._handle_style_change(context, user_message, intent)
            else:
                result = {
                    "success": False,
                    "error": "Could not understand request. Please try rephrasing."
                }

            # Add assistant response
            context.messages.append({
                "role": "assistant",
                "content": result.get("explanation", "Generated image"),
                "timestamp": time.time()
            })

            return result

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_new_generation(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle new image generation"""
        logger.info("Handling new generation")

        # Build prompt from user message
        base_prompt = message

        # Apply child safety filter
        safe_prompt, warnings = ChildSafetyFilter.filter_prompt(base_prompt)

        if warnings:
            logger.warning(f"Safety warnings: {warnings}")

        # Create parameters
        params = self.default_params.copy()
        params.prompt = safe_prompt

        # Apply any adjustments from intent
        params = self._apply_adjustments(params, intent)

        # Store parameters
        context.current_parameters = params

        # Generate image
        result = await self._generate_image(params)

        # Store result
        if result.success:
            context.generation_history.append(result)

        # Build response
        return {
            "success": result.success,
            "image_path": result.image_path,
            "image_url": result.image_url,
            "parameters": params.to_dict(),
            "explanation": f"Generated new image: {safe_prompt[:100]}...",
            "safety_warnings": warnings,
            "intent": intent.explanation,
            "error": result.error
        }

    async def _handle_refinement(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle refinement of previous image"""
        logger.info("Handling refinement")

        if not context.current_parameters:
            return {
                "success": False,
                "error": "No previous image to refine. Please generate an image first."
            }

        # Copy current parameters
        params = context.current_parameters.copy()

        # Apply adjustments
        params = self._apply_adjustments(params, intent)

        # Store updated parameters
        context.current_parameters = params

        # Generate image
        result = await self._generate_image(params)

        # Store result
        if result.success:
            context.generation_history.append(result)

        # Build explanation
        explanation_parts = [intent.explanation or "Applied adjustments"]

        if intent.adjustments.get("steps_delta"):
            explanation_parts.append(f"steps: {params.steps}")
        if intent.adjustments.get("cfg_scale_delta"):
            explanation_parts.append(f"cfg_scale: {params.cfg_scale}")

        return {
            "success": result.success,
            "image_path": result.image_path,
            "image_url": result.image_url,
            "parameters": params.to_dict(),
            "explanation": ", ".join(explanation_parts),
            "adjustments_applied": intent.adjustments,
            "error": result.error
        }

    async def _handle_style_change(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle style change request"""
        # Similar to refinement but focuses on style
        return await self._handle_refinement(context, message, intent)

    def _apply_adjustments(
        self,
        params: GenerationParameters,
        intent: Intent
    ) -> GenerationParameters:
        """
        Apply intent adjustments to parameters
        """
        adjustments = intent.adjustments

        # Apply deltas
        if "steps_delta" in adjustments:
            params.steps = max(10, min(50, params.steps + adjustments["steps_delta"]))

        if "cfg_scale_delta" in adjustments:
            params.cfg_scale = max(1.0, min(20.0, params.cfg_scale + adjustments["cfg_scale_delta"]))

        # Apply absolute values
        if "width" in adjustments and adjustments["width"]:
            params.width = adjustments["width"]
        if "height" in adjustments and adjustments["height"]:
            params.height = adjustments["height"]
        if "steps" in adjustments and adjustments["steps"]:
            params.steps = adjustments["steps"]
        if "cfg_scale" in adjustments and adjustments["cfg_scale"]:
            params.cfg_scale = adjustments["cfg_scale"]

        # Apply prompt modifications
        if "prompt_add" in adjustments and adjustments["prompt_add"]:
            additions = ", ".join(adjustments["prompt_add"])
            params.prompt = f"{params.prompt}, {additions}"

        if "negative_add" in adjustments and adjustments["negative_add"]:
            additions = ", ".join(adjustments["negative_add"])
            params.negative_prompt = f"{params.negative_prompt}, {additions}"

        # Apply LLM prompt modifications
        prompt_mods = intent.prompt_modifications
        if "add" in prompt_mods and prompt_mods["add"]:
            additions = ", ".join(prompt_mods["add"])
            params.prompt = f"{params.prompt}, {additions}"

        if "emphasis" in prompt_mods and prompt_mods["emphasis"]:
            # Emphasize certain words
            for word in prompt_mods["emphasis"]:
                params.prompt = params.prompt.replace(word, f"({word}:1.2)")

        return params

    async def _generate_image(self, params: GenerationParameters) -> GenerationResult:
        """
        Generate image using DiffuGen
        """
        try:
            # Switch to diffusion phase if VRAM manager available
            if self.vram_manager:
                await self.vram_manager.prepare_for_diffusion_phase()

            # Call DiffuGen API
            response = await self.client.post(
                f"{self.diffugen_base_url}/generate/stable",
                json=params.to_dict()
            )

            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                return GenerationResult(
                    success=False,
                    error=f"Generation failed: {error_detail}"
                )

            result_data = response.json()

            return GenerationResult(
                success=result_data.get("success", False),
                image_path=result_data.get("image_path"),
                image_url=result_data.get("image_url"),
                parameters=params,
                error=result_data.get("error")
            )

        except Exception as e:
            logger.error(f"Error generating image: {e}", exc_info=True)
            return GenerationResult(
                success=False,
                error=str(e)
            )

    async def close(self):
        """Cleanup resources"""
        await self.intent_analyzer.close()
        await self.client.aclose()


# ============================================================================
# CLI Testing
# ============================================================================

async def test_workflow():
    """Test the intelligent workflow"""
    workflow = IntelligentWorkflow()
    session_id = "test_session"

    print("=== Intelligent Workflow Test ===\n")

    # Test 1: New generation
    print("Test 1: New generation")
    print("User: Create a friendly dragon in a castle")
    result = await workflow.process_message(
        session_id,
        "Create a friendly dragon in a castle"
    )
    print(f"Result: {result.get('explanation')}")
    print(f"Parameters: steps={result['parameters']['steps']}, cfg={result['parameters']['cfg_scale']}")
    print()

    # Test 2: Refinement
    print("Test 2: Make it brighter")
    print("User: Make it brighter and more colorful")
    result = await workflow.process_message(
        session_id,
        "Make it brighter and more colorful"
    )
    print(f"Result: {result.get('explanation')}")
    print(f"Adjustments: {result.get('adjustments_applied')}")
    print()

    # Test 3: Style change
    print("Test 3: Style change")
    print("User: Make it look like a watercolor painting")
    result = await workflow.process_message(
        session_id,
        "Make it look like a watercolor painting"
    )
    print(f"Result: {result.get('explanation')}")
    print()

    await workflow.close()


if __name__ == "__main__":
    asyncio.run(test_workflow())
