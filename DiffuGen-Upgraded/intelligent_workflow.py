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

from character_consistency import (
    CharacterConsistencyEngine,
    Character,
    CharacterLibrary
)
from style_locking import StyleLockManager, StyleLibrary, ArtStyle
from batch_generation import BatchGenerationManager, parse_batch_request
from character_relationships import (
    RelationshipGraph,
    MultiCharacterSceneGenerator,
    parse_relationship_request,
    RelationType
)

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
    CHARACTER_CREATE = "character_create"
    CHARACTER_USE = "character_use"
    CHARACTER_SHEET = "character_sheet"
    TRAIN_LORA = "train_lora"  # Train LoRA for character
    STYLE_LOCK = "style_lock"  # Lock art style
    STYLE_UNLOCK = "style_unlock"  # Unlock art style
    BATCH_GENERATE = "batch_generate"  # Generate multiple scenes
    ADD_RELATIONSHIP = "add_relationship"  # Add character relationship
    MULTI_CHARACTER_SCENE = "multi_character_scene"  # Scene with multiple characters
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
    active_character: Optional[str] = None  # Currently active character name
    locked_style: Optional[str] = None  # Locked style name
    active_batch_id: Optional[str] = None  # Current batch job ID
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
# Intent Analyzer (Hybrid Gemini + Qwen-VL)
# ============================================================================

class IntentAnalyzer:
    """
    Analyzes user intent using hybrid Gemini + Qwen-VL system
    """

    def __init__(self, llm_base_url: str = "http://localhost:11434/v1", use_router: bool = True):
        self.llm_base_url = llm_base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.use_router = use_router

        # Initialize hybrid router if enabled
        if self.use_router:
            from llm_router import GeminiAnalyzer, QwenVLAnalyzer, MessageRouter

            self.gemini = GeminiAnalyzer(model="gemini-1.5-flash")
            self.qwen_vl = QwenVLAnalyzer(base_url="http://localhost:11434", model="qwen2-vl:latest")
            self.router = MessageRouter(self.gemini, self.qwen_vl)
            logger.info("Hybrid LLM routing enabled (Gemini + Qwen-VL)")
        else:
            self.router = None
            logger.info("Using legacy Qwen-only mode")

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
        Analyze user intent using hybrid LLM routing

        Args:
            user_message: Current user message
            conversation_history: Previous conversation
            current_params: Current generation parameters

        Returns:
            Parsed intent
        """
        try:
            # Use hybrid router if enabled
            if self.use_router and self.router:
                return await self._analyze_with_router(user_message, conversation_history, current_params)
            else:
                return await self._analyze_legacy(user_message, conversation_history, current_params)

        except Exception as e:
            logger.error(f"Error analyzing intent: {e}", exc_info=True)
            return self._fallback_analysis(user_message)

    async def _analyze_with_router(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        current_params: Optional[GenerationParameters] = None
    ) -> Intent:
        """Analyze using hybrid Gemini + Qwen-VL router"""

        # Build context for router
        context_dict = {
            "conversation_history": conversation_history[-5:] if conversation_history else [],
            "current_params": current_params.to_dict() if current_params else None
        }

        # Route and analyze
        result, decision = await self.router.analyze_with_routing(
            message=user_message,
            context=context_dict
        )

        logger.info(f"Routed to {decision.llm_type.value}: {decision.reason}")

        # Convert to Intent object
        return self._convert_router_result_to_intent(result)

    def _convert_router_result_to_intent(self, result: Dict[str, Any]) -> Intent:
        """Convert router result to Intent object"""

        # Map intent_type string to IntentType enum
        intent_type_map = {
            "new_generation": IntentType.NEW_GENERATION,
            "refine_previous": IntentType.REFINE_PREVIOUS,
            "parameter_adjustment": IntentType.PARAMETER_ADJUSTMENT,
            "style_change": IntentType.STYLE_CHANGE,
            "train_lora": IntentType.TRAIN_LORA,
            "style_lock": IntentType.STYLE_LOCK,
            "style_unlock": IntentType.STYLE_UNLOCK,
            "batch_generate": IntentType.BATCH_GENERATE,
            "add_relationship": IntentType.ADD_RELATIONSHIP,
            "multi_character_scene": IntentType.MULTI_CHARACTER_SCENE,
            "character_create": IntentType.CHARACTER_CREATE,
            "character_use": IntentType.CHARACTER_USE,
            "character_sheet": IntentType.CHARACTER_SHEET
        }

        intent_type_str = result.get("intent_type", "new_generation")
        intent_type = intent_type_map.get(intent_type_str, IntentType.NEW_GENERATION)

        # Build adjustments dict
        adjustments = result.get("adjustments", {})

        # Map brightness/detail/color to parameter deltas
        if adjustments.get("brightness") == "brighter":
            adjustments["cfg_scale_delta"] = 2.0
            adjustments["prompt_add"] = adjustments.get("prompt_add", []) + ["bright", "well-lit"]
        elif adjustments.get("brightness") == "darker":
            adjustments["cfg_scale_delta"] = -1.5
            adjustments["prompt_add"] = adjustments.get("prompt_add", []) + ["dim lighting"]

        if adjustments.get("detail") == "more":
            adjustments["steps_delta"] = 10
            adjustments["prompt_add"] = adjustments.get("prompt_add", []) + ["detailed", "intricate"]
        elif adjustments.get("detail") == "less":
            adjustments["steps_delta"] = -10
            adjustments["prompt_add"] = adjustments.get("prompt_add", []) + ["simple", "minimalist"]

        if adjustments.get("color") == "more":
            adjustments["cfg_scale_delta"] = adjustments.get("cfg_scale_delta", 0) + 1.0
            adjustments["prompt_add"] = adjustments.get("prompt_add", []) + ["colorful", "vibrant"]

        # Build prompt modifications
        prompt_modifications = {
            "add": result.get("prompt_additions", []),
            "remove": result.get("prompt_removals", []),
            "emphasis": []
        }

        return Intent(
            type=intent_type,
            confidence=result.get("confidence", 0.8),
            adjustments=adjustments,
            explanation=result.get("explanation", ""),
            prompt_modifications=prompt_modifications
        )

    async def _analyze_legacy(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        current_params: Optional[GenerationParameters] = None
    ) -> Intent:
        """Legacy Qwen-only analysis (fallback)"""

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
                "temperature": 0.3,
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

        # Character consistency engine
        self.character_engine = CharacterConsistencyEngine(
            diffugen_base_url=diffugen_base_url
        )

        # Style management
        self.style_manager = StyleLockManager()

        # Batch generation
        self.batch_manager = BatchGenerationManager(
            character_engine=self.character_engine,
            style_manager=self.style_manager,
            max_concurrent=3
        )

        # Character relationships
        self.relationship_graph = RelationshipGraph()
        self.multi_char_generator = MultiCharacterSceneGenerator(
            character_engine=self.character_engine,
            relationship_graph=self.relationship_graph
        )

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
            elif intent.type == IntentType.CHARACTER_CREATE:
                result = await self._handle_character_create(context, user_message, intent)
            elif intent.type == IntentType.CHARACTER_USE:
                result = await self._handle_character_use(context, user_message, intent)
            elif intent.type == IntentType.CHARACTER_SHEET:
                result = await self._handle_character_sheet(context, user_message, intent)
            elif intent.type == IntentType.TRAIN_LORA:
                result = await self._handle_train_lora(context, user_message, intent)
            elif intent.type == IntentType.STYLE_LOCK:
                result = await self._handle_style_lock(context, user_message, intent)
            elif intent.type == IntentType.STYLE_UNLOCK:
                result = await self._handle_style_unlock(context, user_message, intent)
            elif intent.type == IntentType.BATCH_GENERATE:
                result = await self._handle_batch_generate(context, user_message, intent)
            elif intent.type == IntentType.ADD_RELATIONSHIP:
                result = await self._handle_add_relationship(context, user_message, intent)
            elif intent.type == IntentType.MULTI_CHARACTER_SCENE:
                result = await self._handle_multi_character_scene(context, user_message, intent)
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

    async def _handle_character_create(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle character creation request"""
        logger.info("Handling character creation")

        # Extract character name and description
        # Simple parsing - look for patterns like "Create [name]" or "character named [name]"
        name_match = re.search(r'(?:named?|called)\s+([A-Za-z]+)', message, re.IGNORECASE)
        if name_match:
            character_name = name_match.group(1).capitalize()
        else:
            # Use first word after "create/make"
            create_match = re.search(r'(?:create|make)\s+(?:a\s+)?(.+?)(?:\s+named|\s+called|$)', message, re.IGNORECASE)
            if create_match:
                character_name = create_match.group(1).strip().split()[0].capitalize()
            else:
                character_name = f"Character_{int(time.time())}"

        # Use full message as description
        description = message

        # Apply child safety filter
        safe_description, warnings = ChildSafetyFilter.filter_prompt(description)

        try:
            # Create character
            character, result = await self.character_engine.create_character(
                name=character_name,
                description=safe_description,
                tags=["storybook", "custom"]
            )

            # Store in context
            context.character_references[character_name] = character.reference_image
            context.active_character = character_name

            return {
                "success": True,
                "image_path": character.reference_image,
                "image_url": result.get("image_url"),
                "explanation": f"Created character '{character_name}': {safe_description[:100]}...",
                "character_name": character_name,
                "character_seed": character.seed,
                "safety_warnings": warnings,
                "parameters": character.parameters
            }

        except Exception as e:
            logger.error(f"Error creating character: {e}")
            return {
                "success": False,
                "error": f"Failed to create character: {str(e)}"
            }

    async def _handle_character_use(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle using existing character in new scene"""
        logger.info("Handling character use in scene")

        # Get active character
        if not context.active_character:
            # Try to find character name in message
            for char_name in context.character_references.keys():
                if char_name.lower() in message.lower():
                    context.active_character = char_name
                    break

        if not context.active_character:
            return {
                "success": False,
                "error": "No active character. Please create a character first or specify which character to use."
            }

        # Get character from library
        character = self.character_engine.library.get_character(context.active_character)

        if not character:
            return {
                "success": False,
                "error": f"Character '{context.active_character}' not found in library."
            }

        # Extract scene description
        scene_description = message

        try:
            # Generate with character
            result = await self.character_engine.generate_with_character(
                character=character,
                scene_description=scene_description,
                consistency_strength=0.75  # High consistency
            )

            return {
                "success": result.get("success", False),
                "image_path": result.get("image_path"),
                "image_url": result.get("image_url"),
                "explanation": f"Generated scene with {character.name}: {scene_description[:100]}...",
                "character_name": character.name,
                "parameters": result.get("parameters"),
                "error": result.get("error")
            }

        except Exception as e:
            logger.error(f"Error generating with character: {e}")
            return {
                "success": False,
                "error": f"Failed to generate scene: {str(e)}"
            }

    async def _handle_character_sheet(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle character sheet generation request"""
        logger.info("Handling character sheet generation")

        # Get active character
        if not context.active_character:
            return {
                "success": False,
                "error": "No active character. Please create a character first."
            }

        # Get character from library
        character = self.character_engine.library.get_character(context.active_character)

        if not character:
            return {
                "success": False,
                "error": f"Character '{context.active_character}' not found in library."
            }

        try:
            # Generate character sheet
            poses = ["front view", "side view", "back view", "three-quarter view"]
            sheet_results = await self.character_engine.generate_character_sheet(
                character=character,
                poses=poses
            )

            return {
                "success": True,
                "explanation": f"Generated character sheet for {character.name} with {len(sheet_results)} poses",
                "character_name": character.name,
                "character_sheet": sheet_results,
                "poses_generated": list(sheet_results.keys())
            }

        except Exception as e:
            logger.error(f"Error generating character sheet: {e}")
            return {
                "success": False,
                "error": f"Failed to generate character sheet: {str(e)}"
            }

    async def _handle_train_lora(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle LoRA training request"""
        try:
            # Extract character name from message
            character_name = context.active_character
            if not character_name:
                # Try to extract from message
                name_match = re.search(r'(?:train|lora for)\s+([A-Za-z]+)', message, re.IGNORECASE)
                if name_match:
                    character_name = name_match.group(1).capitalize()
                else:
                    return {
                        "success": False,
                        "error": "Please specify which character to train LoRA for"
                    }

            # Get character
            character = self.character_engine.library.get_character(character_name)
            if not character:
                return {
                    "success": False,
                    "error": f"Character '{character_name}' not found"
                }

            # Train LoRA
            logger.info(f"Starting LoRA training for {character_name}")

            success, lora_path, error = await self.character_engine.train_character_lora(
                character=character,
                num_additional_images=10,
                epochs=10
            )

            if success:
                return {
                    "success": True,
                    "explanation": f"Successfully trained LoRA for {character_name}. You can now use it for perfect character consistency!",
                    "character_name": character_name,
                    "lora_path": lora_path
                }
            else:
                return {
                    "success": False,
                    "error": f"LoRA training failed: {error}"
                }

        except Exception as e:
            logger.error(f"Error training LoRA: {e}")
            return {
                "success": False,
                "error": f"Failed to train LoRA: {str(e)}"
            }

    async def _handle_style_lock(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle style locking request"""
        try:
            # Extract style name from message
            style_match = re.search(
                r'(?:lock|use|set)\s+(?:style\s+)?([a-z_]+)(?:\s+style)?',
                message,
                re.IGNORECASE
            )

            if style_match:
                style_name = style_match.group(1).lower().replace(' ', '_')
            else:
                # Default to watercolor
                style_name = "watercolor_soft"

            # Lock style
            success = self.style_manager.lock_style(context.session_id, style_name)

            if success:
                context.locked_style = style_name
                style = self.style_manager.get_locked_style(context.session_id)

                return {
                    "success": True,
                    "explanation": f"Locked art style to '{style.name}'. All future generations will use this consistent style.",
                    "style_name": style.name,
                    "style_description": style.description
                }
            else:
                return {
                    "success": False,
                    "error": f"Style '{style_name}' not found. Available styles: watercolor_soft, digital_vibrant, pencil_sketch, cartoon_bold, storybook_classic"
                }

        except Exception as e:
            logger.error(f"Error locking style: {e}")
            return {
                "success": False,
                "error": f"Failed to lock style: {str(e)}"
            }

    async def _handle_style_unlock(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle style unlocking request"""
        try:
            self.style_manager.unlock_style(context.session_id)
            context.locked_style = None

            return {
                "success": True,
                "explanation": "Style lock removed. You can now use different styles or let me choose automatically."
            }

        except Exception as e:
            logger.error(f"Error unlocking style: {e}")
            return {
                "success": False,
                "error": f"Failed to unlock style: {str(e)}"
            }

    async def _handle_batch_generate(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle batch scene generation request"""
        try:
            # Parse scene descriptions from message
            scene_descriptions = parse_batch_request(message)

            if not scene_descriptions:
                return {
                    "success": False,
                    "error": "Could not parse scene descriptions. Try: 'Generate scenes: castle, forest, beach'"
                }

            # Create batch job
            batch = await self.batch_manager.create_batch(
                session_id=context.session_id,
                scene_descriptions=scene_descriptions,
                character_name=context.active_character,
                style_name=context.locked_style,
                use_lora=True  # Use LoRA if available
            )

            context.active_batch_id = batch.batch_id

            # Start execution in background
            asyncio.create_task(self.batch_manager.execute_batch(batch.batch_id))

            return {
                "success": True,
                "explanation": f"Started batch generation of {len(scene_descriptions)} scenes. Batch ID: {batch.batch_id}",
                "batch_id": batch.batch_id,
                "total_scenes": batch.total_scenes,
                "scene_descriptions": scene_descriptions
            }

        except Exception as e:
            logger.error(f"Error creating batch: {e}")
            return {
                "success": False,
                "error": f"Failed to create batch: {str(e)}"
            }

    async def _handle_add_relationship(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle adding character relationship"""
        try:
            # Parse relationship from message
            relationship_data = parse_relationship_request(message)

            if not relationship_data:
                return {
                    "success": False,
                    "error": "Could not parse relationship. Try: 'Spark's friend is Whiskers'"
                }

            char_a, char_b, rel_type_str = relationship_data

            # Convert string to RelationType
            rel_type_map = {
                "friend": RelationType.FRIEND,
                "sibling": RelationType.SIBLING,
                "companion": RelationType.COMPANION,
                "rival": RelationType.RIVAL,
                "mentor": RelationType.MENTOR,
                "student": RelationType.STUDENT,
                "teammate": RelationType.TEAMMATE,
            }

            rel_type = rel_type_map.get(rel_type_str, RelationType.FRIEND)

            # Add relationship
            relationship = self.relationship_graph.add_relationship(
                character_a=char_a,
                character_b=char_b,
                relationship_type=rel_type
            )

            return {
                "success": True,
                "explanation": f"Added relationship: {relationship.get_relationship_description()}",
                "character_a": char_a,
                "character_b": char_b,
                "relationship_type": rel_type.value
            }

        except Exception as e:
            logger.error(f"Error adding relationship: {e}")
            return {
                "success": False,
                "error": f"Failed to add relationship: {str(e)}"
            }

    async def _handle_multi_character_scene(
        self,
        context: ConversationContext,
        message: str,
        intent: Intent
    ) -> Dict[str, Any]:
        """Handle multi-character scene generation"""
        try:
            # Extract character names from message
            # Simple approach: look for character names in message
            characters = []

            # Check for "with" pattern: "Spark with Whiskers"
            with_match = re.search(r'([A-Za-z]+)\s+with\s+([A-Za-z]+)', message, re.IGNORECASE)
            if with_match:
                characters = [
                    with_match.group(1).capitalize(),
                    with_match.group(2).capitalize()
                ]

            # Check for "and" pattern: "Spark and Whiskers"
            elif re.search(r'([A-Za-z]+)\s+and\s+([A-Za-z]+)', message, re.IGNORECASE):
                and_match = re.findall(r'\b([A-Z][a-z]+)\b', message)
                characters = and_match[:3]  # Limit to 3 characters

            if len(characters) < 2:
                return {
                    "success": False,
                    "error": "Please specify at least 2 characters for a multi-character scene"
                }

            # Extract scene description (remove character names)
            scene_description = message
            for char in characters:
                scene_description = scene_description.replace(char, "")
            scene_description = scene_description.replace("with", "").replace("and", "")
            scene_description = scene_description.strip()

            # Generate multi-character scene
            result = await self.multi_char_generator.generate_multi_character_scene(
                characters=characters,
                scene_description=scene_description
            )

            return {
                "success": result.get("success", False),
                "image_path": result.get("image_path"),
                "image_url": result.get("image_url"),
                "explanation": f"Generated scene with {', '.join(characters)}",
                "characters": characters
            }

        except Exception as e:
            logger.error(f"Error generating multi-character scene: {e}")
            return {
                "success": False,
                "error": f"Failed to generate multi-character scene: {str(e)}"
            }

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
        await self.character_engine.close()
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
